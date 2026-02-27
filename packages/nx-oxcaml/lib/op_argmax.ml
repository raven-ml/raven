(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let parallel_threshold = 62500

(* --- argmax --- *)

let argmax_all_float64 (out_arr : int32# array) out_offset a_arr va in_numel =
  if in_numel = 0 then
    invalid_arg "argmax: empty input";
  let a_offset = View.offset va in
  if View.is_c_contiguous va then (
    let best_idx = ref 0 in
    let acc = Array.make_float64 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      let v = Array.unsafe_get a_arr (a_offset + i) in
      if Float_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx))
  else
    let in_shape = shape va in
    let in_strides = View.strides va in
    let md_idx = Array.make (Array.length in_shape) 0 in
    let best_idx = ref 0 in
    let acc = Array.make_float64 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      Shape.unravel_index_into i in_shape md_idx;
      let lin = Shape.ravel_index md_idx in_strides in
      let v = Array.unsafe_get a_arr (a_offset + lin) in
      if Float_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx)

let argmax_all_float32 (out_arr : int32# array) out_offset a_arr va in_numel =
  if in_numel = 0 then
    invalid_arg "argmax: empty input";
  let a_offset = View.offset va in
  if View.is_c_contiguous va then (
    let best_idx = ref 0 in
    let acc = Array.make_float32 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      let v = Array.unsafe_get a_arr (a_offset + i) in
      if Float32_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx))
  else
    let in_shape = shape va in
    let in_strides = View.strides va in
    let md_idx = Array.make (Array.length in_shape) 0 in
    let best_idx = ref 0 in
    let acc = Array.make_float32 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      Shape.unravel_index_into i in_shape md_idx;
      let lin = Shape.ravel_index md_idx in_strides in
      let v = Array.unsafe_get a_arr (a_offset + lin) in
      if Float32_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx)

let argmax_all_int32 (out_arr : int32# array) out_offset (a_arr : int32# array)
    va in_numel =
  if in_numel = 0 then
    invalid_arg "argmax: empty input";
  let a_offset = View.offset va in
  if View.is_c_contiguous va then (
    let best_idx = ref 0 in
    let acc = Array.make_int32 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      let v = Array.unsafe_get a_arr (a_offset + i) in
      if Int32_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx))
  else
    let in_shape = shape va in
    let in_strides = View.strides va in
    let md_idx = Array.make (Array.length in_shape) 0 in
    let best_idx = ref 0 in
    let acc = Array.make_int32 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      Shape.unravel_index_into i in_shape md_idx;
      let lin = Shape.ravel_index md_idx in_strides in
      let v = Array.unsafe_get a_arr (a_offset + lin) in
      if Int32_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx)

let argmax_all_int64 (out_arr : int32# array) out_offset (a_arr : int64# array)
    va in_numel =
  if in_numel = 0 then
    invalid_arg "argmax: empty input";
  let a_offset = View.offset va in
  if View.is_c_contiguous va then (
    let best_idx = ref 0 in
    let acc = Array.make_int64 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      let v = Array.unsafe_get a_arr (a_offset + i) in
      if Int64_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx))
  else
    let in_shape = shape va in
    let in_strides = View.strides va in
    let md_idx = Array.make (Array.length in_shape) 0 in
    let best_idx = ref 0 in
    let acc = Array.make_int64 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      Shape.unravel_index_into i in_shape md_idx;
      let lin = Shape.ravel_index md_idx in_strides in
      let v = Array.unsafe_get a_arr (a_offset + lin) in
      if Int64_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx)

(* Axis-based argmax *)

let argmax_axis_float64 (out_arr : int32# array) a_arr va vout axis keepdims
    start_idx end_idx =
  let plan = Reduce_ops.make_plan [| axis |] keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    Reduce_ops.init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let best_idx = ref 0 in
    let idx = ref 1 in
    let continue = ref (Reduce_ops.increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      if Float_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := !idx);
      incr idx;
      continue := Reduce_ops.increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Int32_u.of_int !best_idx)
  done

let argmax_axis_float32 (out_arr : int32# array) a_arr va vout axis keepdims
    start_idx end_idx =
  let plan = Reduce_ops.make_plan [| axis |] keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    Reduce_ops.init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let best_idx = ref 0 in
    let idx = ref 1 in
    let continue = ref (Reduce_ops.increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      if Float32_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := !idx);
      incr idx;
      continue := Reduce_ops.increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Int32_u.of_int !best_idx)
  done

let argmax_axis_int32 (out_arr : int32# array) (a_arr : int32# array) va vout
    axis keepdims start_idx end_idx =
  let plan = Reduce_ops.make_plan [| axis |] keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    Reduce_ops.init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let best_idx = ref 0 in
    let idx = ref 1 in
    let continue = ref (Reduce_ops.increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      if Int32_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := !idx);
      incr idx;
      continue := Reduce_ops.increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Int32_u.of_int !best_idx)
  done

let argmax_axis_int64 (out_arr : int32# array) (a_arr : int64# array) va vout
    axis keepdims start_idx end_idx =
  let plan = Reduce_ops.make_plan [| axis |] keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    Reduce_ops.init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let best_idx = ref 0 in
    let idx = ref 1 in
    let continue = ref (Reduce_ops.increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      if Int64_u.compare v (Array.unsafe_get acc 0) > 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := !idx);
      incr idx;
      continue := Reduce_ops.increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Int32_u.of_int !best_idx)
  done

(* Entry points *)

let argmax_float64 pool ~(out_arr : int32# array) ~a_arr ~va ~vout ~axis
    ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if in_numel = 0 then
    invalid_arg "argmax: empty input"
  else if out_numel = 1 then
    argmax_all_float64 out_arr (View.offset vout) a_arr va in_numel
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        argmax_axis_float64 out_arr a_arr va vout axis keepdims s e)
  else argmax_axis_float64 out_arr a_arr va vout axis keepdims 0 out_numel

let argmax_float32 pool ~(out_arr : int32# array) ~a_arr ~va ~vout ~axis
    ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if in_numel = 0 then
    invalid_arg "argmax: empty input"
  else if out_numel = 1 then
    argmax_all_float32 out_arr (View.offset vout) a_arr va in_numel
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        argmax_axis_float32 out_arr a_arr va vout axis keepdims s e)
  else argmax_axis_float32 out_arr a_arr va vout axis keepdims 0 out_numel

let argmax_int32 pool ~(out_arr : int32# array) ~a_arr ~va ~vout ~axis
    ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if in_numel = 0 then
    invalid_arg "argmax: empty input"
  else if out_numel = 1 then
    argmax_all_int32 out_arr (View.offset vout) a_arr va in_numel
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        argmax_axis_int32 out_arr a_arr va vout axis keepdims s e)
  else argmax_axis_int32 out_arr a_arr va vout axis keepdims 0 out_numel

let argmax_int64 pool ~(out_arr : int32# array) ~a_arr ~va ~vout ~axis
    ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if in_numel = 0 then
    invalid_arg "argmax: empty input"
  else if out_numel = 1 then
    argmax_all_int64 out_arr (View.offset vout) a_arr va in_numel
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        argmax_axis_int64 out_arr a_arr va vout axis keepdims s e)
  else argmax_axis_int64 out_arr a_arr va vout axis keepdims 0 out_numel

(* --- argmin --- *)

let argmin_all_float64 (out_arr : int32# array) out_offset a_arr va in_numel =
  if in_numel = 0 then
    invalid_arg "argmin: empty input";
  let a_offset = View.offset va in
  if View.is_c_contiguous va then (
    let best_idx = ref 0 in
    let acc = Array.make_float64 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      let v = Array.unsafe_get a_arr (a_offset + i) in
      if Float_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx))
  else
    let in_shape = shape va in
    let in_strides = View.strides va in
    let md_idx = Array.make (Array.length in_shape) 0 in
    let best_idx = ref 0 in
    let acc = Array.make_float64 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      Shape.unravel_index_into i in_shape md_idx;
      let lin = Shape.ravel_index md_idx in_strides in
      let v = Array.unsafe_get a_arr (a_offset + lin) in
      if Float_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx)

let argmin_all_float32 (out_arr : int32# array) out_offset a_arr va in_numel =
  if in_numel = 0 then
    invalid_arg "argmin: empty input";
  let a_offset = View.offset va in
  if View.is_c_contiguous va then (
    let best_idx = ref 0 in
    let acc = Array.make_float32 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      let v = Array.unsafe_get a_arr (a_offset + i) in
      if Float32_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx))
  else
    let in_shape = shape va in
    let in_strides = View.strides va in
    let md_idx = Array.make (Array.length in_shape) 0 in
    let best_idx = ref 0 in
    let acc = Array.make_float32 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      Shape.unravel_index_into i in_shape md_idx;
      let lin = Shape.ravel_index md_idx in_strides in
      let v = Array.unsafe_get a_arr (a_offset + lin) in
      if Float32_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx)

let argmin_all_int32 (out_arr : int32# array) out_offset (a_arr : int32# array)
    va in_numel =
  if in_numel = 0 then
    invalid_arg "argmin: empty input";
  let a_offset = View.offset va in
  if View.is_c_contiguous va then (
    let best_idx = ref 0 in
    let acc = Array.make_int32 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      let v = Array.unsafe_get a_arr (a_offset + i) in
      if Int32_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx))
  else
    let in_shape = shape va in
    let in_strides = View.strides va in
    let md_idx = Array.make (Array.length in_shape) 0 in
    let best_idx = ref 0 in
    let acc = Array.make_int32 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      Shape.unravel_index_into i in_shape md_idx;
      let lin = Shape.ravel_index md_idx in_strides in
      let v = Array.unsafe_get a_arr (a_offset + lin) in
      if Int32_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx)

let argmin_all_int64 (out_arr : int32# array) out_offset (a_arr : int64# array)
    va in_numel =
  if in_numel = 0 then
    invalid_arg "argmin: empty input";
  let a_offset = View.offset va in
  if View.is_c_contiguous va then (
    let best_idx = ref 0 in
    let acc = Array.make_int64 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      let v = Array.unsafe_get a_arr (a_offset + i) in
      if Int64_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx))
  else
    let in_shape = shape va in
    let in_strides = View.strides va in
    let md_idx = Array.make (Array.length in_shape) 0 in
    let best_idx = ref 0 in
    let acc = Array.make_int64 1 in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr a_offset);
    for i = 1 to in_numel - 1 do
      Shape.unravel_index_into i in_shape md_idx;
      let lin = Shape.ravel_index md_idx in_strides in
      let v = Array.unsafe_get a_arr (a_offset + lin) in
      if Int64_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := i)
    done;
    Array.unsafe_set out_arr out_offset (Int32_u.of_int !best_idx)

(* Axis-based argmin *)

let argmin_axis_float64 (out_arr : int32# array) a_arr va vout axis keepdims
    start_idx end_idx =
  let plan = Reduce_ops.make_plan [| axis |] keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    Reduce_ops.init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let best_idx = ref 0 in
    let idx = ref 1 in
    let continue = ref (Reduce_ops.increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      if Float_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := !idx);
      incr idx;
      continue := Reduce_ops.increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Int32_u.of_int !best_idx)
  done

let argmin_axis_float32 (out_arr : int32# array) a_arr va vout axis keepdims
    start_idx end_idx =
  let plan = Reduce_ops.make_plan [| axis |] keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    Reduce_ops.init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let best_idx = ref 0 in
    let idx = ref 1 in
    let continue = ref (Reduce_ops.increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      if Float32_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := !idx);
      incr idx;
      continue := Reduce_ops.increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Int32_u.of_int !best_idx)
  done

let argmin_axis_int32 (out_arr : int32# array) (a_arr : int32# array) va vout
    axis keepdims start_idx end_idx =
  let plan = Reduce_ops.make_plan [| axis |] keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    Reduce_ops.init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let best_idx = ref 0 in
    let idx = ref 1 in
    let continue = ref (Reduce_ops.increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      if Int32_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := !idx);
      incr idx;
      continue := Reduce_ops.increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Int32_u.of_int !best_idx)
  done

let argmin_axis_int64 (out_arr : int32# array) (a_arr : int64# array) va vout
    axis keepdims start_idx end_idx =
  let plan = Reduce_ops.make_plan [| axis |] keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    Reduce_ops.init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let best_idx = ref 0 in
    let idx = ref 1 in
    let continue = ref (Reduce_ops.increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      if Int64_u.compare v (Array.unsafe_get acc 0) < 0 then (
        Array.unsafe_set acc 0 v;
        best_idx := !idx);
      incr idx;
      continue := Reduce_ops.increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Int32_u.of_int !best_idx)
  done

(* Entry points *)

let argmin_float64 pool ~(out_arr : int32# array) ~a_arr ~va ~vout ~axis
    ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if in_numel = 0 then
    invalid_arg "argmin: empty input"
  else if out_numel = 1 then
    argmin_all_float64 out_arr (View.offset vout) a_arr va in_numel
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        argmin_axis_float64 out_arr a_arr va vout axis keepdims s e)
  else argmin_axis_float64 out_arr a_arr va vout axis keepdims 0 out_numel

let argmin_float32 pool ~(out_arr : int32# array) ~a_arr ~va ~vout ~axis
    ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if in_numel = 0 then
    invalid_arg "argmin: empty input"
  else if out_numel = 1 then
    argmin_all_float32 out_arr (View.offset vout) a_arr va in_numel
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        argmin_axis_float32 out_arr a_arr va vout axis keepdims s e)
  else argmin_axis_float32 out_arr a_arr va vout axis keepdims 0 out_numel

let argmin_int32 pool ~(out_arr : int32# array) ~a_arr ~va ~vout ~axis
    ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if in_numel = 0 then
    invalid_arg "argmin: empty input"
  else if out_numel = 1 then
    argmin_all_int32 out_arr (View.offset vout) a_arr va in_numel
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        argmin_axis_int32 out_arr a_arr va vout axis keepdims s e)
  else argmin_axis_int32 out_arr a_arr va vout axis keepdims 0 out_numel

let argmin_int64 pool ~(out_arr : int32# array) ~a_arr ~va ~vout ~axis
    ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if in_numel = 0 then
    invalid_arg "argmin: empty input"
  else if out_numel = 1 then
    argmin_all_int64 out_arr (View.offset vout) a_arr va in_numel
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        argmin_axis_int64 out_arr a_arr va vout axis keepdims s e)
  else argmin_axis_int64 out_arr a_arr va vout axis keepdims 0 out_numel
