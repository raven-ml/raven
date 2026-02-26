(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let invalid_argf_fn fn fmt =
  Printf.ksprintf (fun msg -> invalid_argf "Loss.%s: %s" fn msg) fmt

let check_logits_shape ~fn logits =
  let logits_shape = Rune.shape logits in
  let logits_rank = Array.length logits_shape in
  if logits_rank < 1 then invalid_argf_fn fn "logits must have rank >= 1";
  let class_axis = logits_rank - 1 in
  let num_classes = logits_shape.(class_axis) in
  if num_classes <= 0 then
    invalid_argf_fn fn "logits class dimension must be positive (got %d)"
      num_classes;
  logits_shape

let check_same_shape ~fn ~rhs_name lhs rhs =
  let lhs_rank = Array.length lhs in
  let rhs_rank = Array.length rhs in
  if rhs_rank <> lhs_rank then
    invalid_argf_fn fn "%s rank mismatch (got %d, expected %d)" rhs_name
      rhs_rank lhs_rank;
  for i = 0 to lhs_rank - 1 do
    if rhs.(i) <> lhs.(i) then
      invalid_argf_fn fn "%s shape mismatch at axis %d (got %d, expected %d)"
        rhs_name i rhs.(i) lhs.(i)
  done

let check_cross_entropy_shapes logits labels =
  let fn = "cross_entropy" in
  let logits_shape = check_logits_shape ~fn logits in
  let labels_shape = Rune.shape labels in
  check_same_shape ~fn ~rhs_name:"labels" logits_shape labels_shape

let cross_entropy logits labels =
  check_cross_entropy_shapes logits labels;
  let max_logits = Rune.max logits ~axes:[ -1 ] ~keepdims:true in
  let shifted = Rune.sub logits max_logits in
  let log_sum_exp =
    Rune.log (Rune.sum (Rune.exp shifted) ~axes:[ -1 ] ~keepdims:true)
  in
  let log_softmax = Rune.sub shifted log_sum_exp in
  let per_example =
    Rune.neg (Rune.sum (Rune.mul labels log_softmax) ~axes:[ -1 ])
  in
  Rune.mean per_example

let check_sparse_indices_dtype indices =
  let fn = "cross_entropy_sparse" in
  let dtype = Rune.dtype indices in
  if not (Nx_core.Dtype.is_int dtype) then
    invalid_argf_fn fn "expected integer labels, got %s"
      (Nx_core.Dtype.to_string dtype)

let check_sparse_shapes logits indices =
  let fn = "cross_entropy_sparse" in
  let logits_shape = check_logits_shape ~fn logits in
  let indices_shape = Rune.shape indices in
  let logits_rank = Array.length logits_shape in
  let indices_rank = Array.length indices_shape in
  if indices_rank <> logits_rank - 1 then
    invalid_argf_fn fn "labels rank mismatch (got %d, expected %d)" indices_rank
      (logits_rank - 1);
  for i = 0 to indices_rank - 1 do
    if indices_shape.(i) <> logits_shape.(i) then
      invalid_argf_fn fn
        "labels shape mismatch at axis %d (got %d, expected %d)" i
        indices_shape.(i) logits_shape.(i)
  done;
  let class_axis = logits_rank - 1 in
  logits_shape.(class_axis)

let cross_entropy_sparse logits indices =
  check_sparse_indices_dtype indices;
  ignore (check_sparse_shapes logits indices : int);
  let indices_int = Rune.cast Rune.int32 indices in
  (* Numerically stable log-softmax *)
  let max_logits = Rune.max logits ~axes:[ -1 ] ~keepdims:true in
  let shifted = Rune.sub logits max_logits in
  let log_sum_exp =
    Rune.log (Rune.sum (Rune.exp shifted) ~axes:[ -1 ] ~keepdims:true)
  in
  (* Gather true-class logits: [...] → [...; 1] for take_along_axis *)
  let indices_expanded = Rune.expand_dims [ -1 ] indices_int in
  let true_logits = Rune.take_along_axis ~axis:(-1) indices_expanded shifted in
  (* loss = -(true_logit - log_sum_exp) *)
  let per_example =
    Rune.neg
      (Rune.sub
         (Rune.squeeze ~axes:[ -1 ] true_logits)
         (Rune.squeeze ~axes:[ -1 ] log_sum_exp))
  in
  Rune.mean per_example

let binary_cross_entropy logits labels =
  let fn = "binary_cross_entropy" in
  let logits_shape = Rune.shape logits in
  let labels_shape = Rune.shape labels in
  check_same_shape ~fn ~rhs_name:"labels" logits_shape labels_shape;
  let dtype = Rune.dtype logits in
  let one = Rune.scalar dtype 1.0 in
  let log_p = Activation.log_sigmoid logits in
  let log_1_minus_p = Activation.log_sigmoid (Rune.neg logits) in
  let per_element =
    Rune.neg
      (Rune.add (Rune.mul labels log_p)
         (Rune.mul (Rune.sub one labels) log_1_minus_p))
  in
  Rune.mean per_element

let mse predictions targets =
  let diff = Rune.sub predictions targets in
  Rune.mean (Rune.mul diff diff)

let mae predictions targets =
  Rune.mean (Rune.abs (Rune.sub predictions targets))
