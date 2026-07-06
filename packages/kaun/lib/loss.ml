(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type reduction = [ `Mean | `Sum ]

let invalid_argf fn fmt =
  Printf.ksprintf (fun msg -> invalid_arg ("Loss." ^ fn ^ ": " ^ msg)) fmt

let shape_str s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

let reduce reduction t =
  match reduction with `Mean -> Nx.mean t | `Sum -> Nx.sum t

(* Regression *)

let mse ?(reduction = `Mean) predictions targets =
  let d = Nx.sub predictions targets in
  reduce reduction (Nx.mul d d)

let mae ?(reduction = `Mean) predictions targets =
  reduce reduction (Nx.abs (Nx.sub predictions targets))

let huber ?(delta = 1.0) ?(reduction = `Mean) predictions targets =
  if not (delta > 0.0) then
    invalid_argf "huber" "delta must be positive (got %g)" delta;
  let d = Nx.sub predictions targets in
  let abs_d = Nx.abs d in
  let quadratic = Nx.mul_s (Nx.mul d d) 0.5 in
  let linear = Nx.mul_s (Nx.sub_s abs_d (0.5 *. delta)) delta in
  let in_quadratic = Nx.less_equal abs_d (Nx.full_like abs_d delta) in
  reduce reduction (Nx.where in_quadratic quadratic linear)

(* Classification *)

let sigmoid_bce ?(reduction = `Mean) logits targets =
  (* Numerically stable: max(z,0) - z*y + log(1 + exp(-|z|)). *)
  let z = logits and y = targets in
  let relu_z = Nx.maximum z (Nx.zeros_like z) in
  let softplus = Nx.log (Nx.add_s (Nx.exp (Nx.neg (Nx.abs z))) 1.0) in
  reduce reduction (Nx.add (Nx.sub relu_z (Nx.mul z y)) softplus)

let check_logits ~fn logits =
  let shape = Nx.shape logits in
  let rank = Array.length shape in
  if rank < 1 then invalid_argf fn "logits must have rank >= 1";
  if shape.(rank - 1) <= 0 then
    invalid_argf fn "logits class dimension must be positive (got %d)"
      shape.(rank - 1);
  shape

let softmax_cross_entropy ?(reduction = `Mean) logits targets =
  let fn = "softmax_cross_entropy" in
  let logits_shape = check_logits ~fn logits in
  let targets_shape = Nx.shape targets in
  if targets_shape <> logits_shape then
    invalid_argf fn "targets shape %s does not match logits shape %s"
      (shape_str targets_shape) (shape_str logits_shape);
  let log_probs = Nx.log_softmax logits in
  reduce reduction (Nx.neg (Nx.sum ~axes:[ -1 ] (Nx.mul targets log_probs)))

let softmax_cross_entropy_sparse ?(reduction = `Mean) logits labels =
  let fn = "softmax_cross_entropy_sparse" in
  let logits_shape = check_logits ~fn logits in
  let labels_shape = Nx.shape labels in
  let batch_shape = Array.sub logits_shape 0 (Array.length logits_shape - 1) in
  if labels_shape <> batch_shape then
    invalid_argf fn "labels shape %s does not match logits batch shape %s"
      (shape_str labels_shape) (shape_str batch_shape);
  let log_probs = Nx.log_softmax logits in
  let picked =
    Nx.take_along_axis ~axis:(-1) (Nx.expand_dims [ -1 ] labels) log_probs
  in
  reduce reduction (Nx.neg (Nx.squeeze ~axes:[ -1 ] picked))
