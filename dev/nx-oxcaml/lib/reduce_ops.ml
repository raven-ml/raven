(*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

type plan = {
  axes_mask : bool array;
  in_shape : int array;
  in_strides : int array;
  in_offset : int;
  out_shape : int array;
  out_offset : int;
  rank : int;
  out_rank : int;
  keepdims : bool;
}

let make_plan axes keepdims va vout =
  let in_shape = shape va in
  let rank = Array.length in_shape in
  let axes_mask = Array.make rank false in
  Array.iter
    (fun ax ->
      let ax' = if ax < 0 then ax + rank else ax in
      axes_mask.(ax') <- true)
    axes;
  let out_shape = shape vout in
  {
    axes_mask;
    in_shape;
    in_strides = View.strides va;
    in_offset = View.offset va;
    out_shape;
    out_offset = View.offset vout;
    rank;
    out_rank = Array.length out_shape;
    keepdims;
  }

let init_input_index plan out_md_index in_md_index =
  if plan.keepdims then
    for d = 0 to plan.rank - 1 do
      if plan.axes_mask.(d) then in_md_index.(d) <- 0
      else in_md_index.(d) <- out_md_index.(d)
    done
  else
    let out_pos = ref 0 in
    for d = 0 to plan.rank - 1 do
      if plan.axes_mask.(d) then in_md_index.(d) <- 0
      else (
        in_md_index.(d) <- out_md_index.(!out_pos);
        incr out_pos)
    done

let increment_input_index plan in_md_index =
  let rec carry d =
    if d < 0 then false
    else if not plan.axes_mask.(d) then carry (d - 1)
    else
      let next = in_md_index.(d) + 1 in
      if next < plan.in_shape.(d) then (
        in_md_index.(d) <- next;
        true)
      else (
        in_md_index.(d) <- 0;
        carry (d - 1))
  in
  carry (plan.rank - 1)

let parallel_threshold = 62500

let copy_float64 a_arr out_arr va vout start_idx end_idx =
  let out_offset = View.offset vout in
  let in_offset = View.offset va in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let out_base = out_offset + start_idx in
    let in_base = in_offset + start_idx in
    let n = end_idx - start_idx in
    for i = 0 to n - 1 do
      Array.unsafe_set out_arr (out_base + i)
        (Array.unsafe_get a_arr (in_base + i))
    done)
  else
    let out_shape = shape vout in
    let a_strides = View.strides va in
    let md_index = Array.make (Array.length out_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (in_offset + a_lin) in
      Array.unsafe_set out_arr (out_offset + k) v
    done

let copy_float32 a_arr out_arr va vout start_idx end_idx =
  let out_offset = View.offset vout in
  let in_offset = View.offset va in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let out_base = out_offset + start_idx in
    let in_base = in_offset + start_idx in
    let n = end_idx - start_idx in
    for i = 0 to n - 1 do
      Array.unsafe_set out_arr (out_base + i)
        (Array.unsafe_get a_arr (in_base + i))
    done)
  else
    let out_shape = shape vout in
    let a_strides = View.strides va in
    let md_index = Array.make (Array.length out_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (in_offset + a_lin) in
      Array.unsafe_set out_arr (out_offset + k) v
    done

let copy_int32 a_arr out_arr va vout start_idx end_idx =
  let out_offset = View.offset vout in
  let in_offset = View.offset va in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let out_base = out_offset + start_idx in
    let in_base = in_offset + start_idx in
    let n = end_idx - start_idx in
    for i = 0 to n - 1 do
      Array.unsafe_set out_arr (out_base + i)
        (Array.unsafe_get a_arr (in_base + i))
    done)
  else
    let out_shape = shape vout in
    let a_strides = View.strides va in
    let md_index = Array.make (Array.length out_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (in_offset + a_lin) in
      Array.unsafe_set out_arr (out_offset + k) v
    done

let copy_int64 a_arr out_arr va vout start_idx end_idx =
  let out_offset = View.offset vout in
  let in_offset = View.offset va in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let out_base = out_offset + start_idx in
    let in_base = in_offset + start_idx in
    let n = end_idx - start_idx in
    for i = 0 to n - 1 do
      Array.unsafe_set out_arr (out_base + i)
        (Array.unsafe_get a_arr (in_base + i))
    done)
  else
    let out_shape = shape vout in
    let a_strides = View.strides va in
    let md_index = Array.make (Array.length out_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (in_offset + a_lin) in
      Array.unsafe_set out_arr (out_offset + k) v
    done

let fill_float64 out_arr vout value =
  let out_offset = View.offset vout in
  let out_numel = numel vout in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr (out_offset + i) value
  done

let fill_float32 out_arr vout value =
  let out_offset = View.offset vout in
  let out_numel = numel vout in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr (out_offset + i) value
  done

let fill_int32 out_arr vout value =
  let out_offset = View.offset vout in
  let out_numel = numel vout in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr (out_offset + i) value
  done

let fill_int64 out_arr vout value =
  let out_offset = View.offset vout in
  let out_numel = numel vout in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr (out_offset + i) value
  done

let sum_axis_float64 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 (Float_u.of_int 0);
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float_u.add cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let sum_axis_float32 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 (Float32_u.of_int 0);
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float32_u.add cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let sum_axis_int32 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 #0l;
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int32_u.add cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let sum_axis_int64 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 #0L;
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int64_u.add cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let sum_all_partial_float64 a_arr va start_idx end_idx =
  if start_idx >= end_idx then Float_u.of_int 0
  else
    let acc = Array.make_float64 1 in
    Array.unsafe_set acc 0 (Float_u.of_int 0);
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Float_u.add cur (Array.unsafe_get a_arr i))
      done;
      Array.unsafe_get acc 0)
    else
      let a_shape = shape va in
      let a_strides = View.strides va in
      let a_offset = View.offset va in
      let md_index = Array.make (Array.length a_shape) 0 in
      for k = start_idx to end_idx - 1 do
        Shape.unravel_index_into k a_shape md_index;
        let a_lin = Shape.ravel_index md_index a_strides in
        let v = Array.unsafe_get a_arr (a_offset + a_lin) in
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Float_u.add cur v)
      done;
      Array.unsafe_get acc 0

let sum_all_partial_float32 a_arr va start_idx end_idx =
  if start_idx >= end_idx then Float32_u.of_int 0
  else
    let acc = Array.make_float32 1 in
    Array.unsafe_set acc 0 (Float32_u.of_int 0);
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Float32_u.add cur (Array.unsafe_get a_arr i))
      done;
      Array.unsafe_get acc 0)
    else
      let a_shape = shape va in
      let a_strides = View.strides va in
      let a_offset = View.offset va in
      let md_index = Array.make (Array.length a_shape) 0 in
      for k = start_idx to end_idx - 1 do
        Shape.unravel_index_into k a_shape md_index;
        let a_lin = Shape.ravel_index md_index a_strides in
        let v = Array.unsafe_get a_arr (a_offset + a_lin) in
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Float32_u.add cur v)
      done;
      Array.unsafe_get acc 0

let sum_all_partial_int32 a_arr va start_idx end_idx =
  if start_idx >= end_idx then #0l
  else
    let acc = Array.make_int32 1 in
    Array.unsafe_set acc 0 #0l;
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int32_u.add cur (Array.unsafe_get a_arr i))
      done;
      Array.unsafe_get acc 0)
    else
      let a_shape = shape va in
      let a_strides = View.strides va in
      let a_offset = View.offset va in
      let md_index = Array.make (Array.length a_shape) 0 in
      for k = start_idx to end_idx - 1 do
        Shape.unravel_index_into k a_shape md_index;
        let a_lin = Shape.ravel_index md_index a_strides in
        let v = Array.unsafe_get a_arr (a_offset + a_lin) in
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int32_u.add cur v)
      done;
      Array.unsafe_get acc 0

let sum_all_partial_int64 a_arr va start_idx end_idx =
  if start_idx >= end_idx then #0L
  else
    let acc = Array.make_int64 1 in
    Array.unsafe_set acc 0 #0L;
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int64_u.add cur (Array.unsafe_get a_arr i))
      done;
      Array.unsafe_get acc 0)
    else
      let a_shape = shape va in
      let a_strides = View.strides va in
      let a_offset = View.offset va in
      let md_index = Array.make (Array.length a_shape) 0 in
      for k = start_idx to end_idx - 1 do
        Shape.unravel_index_into k a_shape md_index;
        let a_lin = Shape.ravel_index md_index a_strides in
        let v = Array.unsafe_get a_arr (a_offset + a_lin) in
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int64_u.add cur v)
      done;
      Array.unsafe_get acc 0

let prod_axis_float64 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 (Float_u.of_int 1);
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float_u.mul cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let prod_axis_float32 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 (Float32_u.of_int 1);
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float32_u.mul cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let prod_axis_int32 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 #1l;
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int32_u.mul cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let prod_axis_int64 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 #1L;
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int64_u.mul cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let prod_all_partial_float64 a_arr va start_idx end_idx =
  if start_idx >= end_idx then Float_u.of_int 1
  else
    let acc = Array.make_float64 1 in
    Array.unsafe_set acc 0 (Float_u.of_int 1);
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Float_u.mul cur (Array.unsafe_get a_arr i))
      done;
      Array.unsafe_get acc 0)
    else
      let a_shape = shape va in
      let a_strides = View.strides va in
      let a_offset = View.offset va in
      let md_index = Array.make (Array.length a_shape) 0 in
      for k = start_idx to end_idx - 1 do
        Shape.unravel_index_into k a_shape md_index;
        let a_lin = Shape.ravel_index md_index a_strides in
        let v = Array.unsafe_get a_arr (a_offset + a_lin) in
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Float_u.mul cur v)
      done;
      Array.unsafe_get acc 0

let prod_all_partial_float32 a_arr va start_idx end_idx =
  if start_idx >= end_idx then Float32_u.of_int 1
  else
    let acc = Array.make_float32 1 in
    Array.unsafe_set acc 0 (Float32_u.of_int 1);
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Float32_u.mul cur (Array.unsafe_get a_arr i))
      done;
      Array.unsafe_get acc 0)
    else
      let a_shape = shape va in
      let a_strides = View.strides va in
      let a_offset = View.offset va in
      let md_index = Array.make (Array.length a_shape) 0 in
      for k = start_idx to end_idx - 1 do
        Shape.unravel_index_into k a_shape md_index;
        let a_lin = Shape.ravel_index md_index a_strides in
        let v = Array.unsafe_get a_arr (a_offset + a_lin) in
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Float32_u.mul cur v)
      done;
      Array.unsafe_get acc 0

let prod_all_partial_int32 a_arr va start_idx end_idx =
  if start_idx >= end_idx then #1l
  else
    let acc = Array.make_int32 1 in
    Array.unsafe_set acc 0 #1l;
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int32_u.mul cur (Array.unsafe_get a_arr i))
      done;
      Array.unsafe_get acc 0)
    else
      let a_shape = shape va in
      let a_strides = View.strides va in
      let a_offset = View.offset va in
      let md_index = Array.make (Array.length a_shape) 0 in
      for k = start_idx to end_idx - 1 do
        Shape.unravel_index_into k a_shape md_index;
        let a_lin = Shape.ravel_index md_index a_strides in
        let v = Array.unsafe_get a_arr (a_offset + a_lin) in
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int32_u.mul cur v)
      done;
      Array.unsafe_get acc 0

let prod_all_partial_int64 a_arr va start_idx end_idx =
  if start_idx >= end_idx then #1L
  else
    let acc = Array.make_int64 1 in
    Array.unsafe_set acc 0 #1L;
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int64_u.mul cur (Array.unsafe_get a_arr i))
      done;
      Array.unsafe_get acc 0)
    else
      let a_shape = shape va in
      let a_strides = View.strides va in
      let a_offset = View.offset va in
      let md_index = Array.make (Array.length a_shape) 0 in
      for k = start_idx to end_idx - 1 do
        Shape.unravel_index_into k a_shape md_index;
        let a_lin = Shape.ravel_index md_index a_strides in
        let v = Array.unsafe_get a_arr (a_offset + a_lin) in
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int64_u.mul cur v)
      done;
      Array.unsafe_get acc 0

let min_axis_float64 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let continue = ref (increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float_u.min cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let min_axis_float32 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let continue = ref (increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float32_u.min cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let min_axis_int32 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let continue = ref (increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int32_u.min cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let min_axis_int64 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let continue = ref (increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int64_u.min cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let min_all_float64 a_arr va start_idx end_idx =
  let acc = Array.make_float64 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float_u.min cur (Array.unsafe_get a_arr i))
    done;
    Array.unsafe_get acc 0)
  else
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let md_index = Array.make (Array.length a_shape) 0 in
    Shape.unravel_index_into start_idx a_shape md_index;
    let first_lin = Shape.ravel_index md_index a_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (a_offset + first_lin));
    for k = start_idx + 1 to end_idx - 1 do
      Shape.unravel_index_into k a_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (a_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float_u.min cur v)
    done;
    Array.unsafe_get acc 0

let min_all_float32 a_arr va start_idx end_idx =
  let acc = Array.make_float32 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float32_u.min cur (Array.unsafe_get a_arr i))
    done;
    Array.unsafe_get acc 0)
  else
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let md_index = Array.make (Array.length a_shape) 0 in
    Shape.unravel_index_into start_idx a_shape md_index;
    let first_lin = Shape.ravel_index md_index a_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (a_offset + first_lin));
    for k = start_idx + 1 to end_idx - 1 do
      Shape.unravel_index_into k a_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (a_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float32_u.min cur v)
    done;
    Array.unsafe_get acc 0

let min_all_int32 a_arr va start_idx end_idx =
  let acc = Array.make_int32 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int32_u.min cur (Array.unsafe_get a_arr i))
    done;
    Array.unsafe_get acc 0)
  else
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let md_index = Array.make (Array.length a_shape) 0 in
    Shape.unravel_index_into start_idx a_shape md_index;
    let first_lin = Shape.ravel_index md_index a_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (a_offset + first_lin));
    for k = start_idx + 1 to end_idx - 1 do
      Shape.unravel_index_into k a_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (a_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int32_u.min cur v)
    done;
    Array.unsafe_get acc 0

let min_all_int64 a_arr va start_idx end_idx =
  let acc = Array.make_int64 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int64_u.min cur (Array.unsafe_get a_arr i))
    done;
    Array.unsafe_get acc 0)
  else
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let md_index = Array.make (Array.length a_shape) 0 in
    Shape.unravel_index_into start_idx a_shape md_index;
    let first_lin = Shape.ravel_index md_index a_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (a_offset + first_lin));
    for k = start_idx + 1 to end_idx - 1 do
      Shape.unravel_index_into k a_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (a_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int64_u.min cur v)
    done;
    Array.unsafe_get acc 0

let max_axis_float64 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let continue = ref (increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float_u.max cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let max_axis_float32 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_float32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let continue = ref (increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float32_u.max cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let max_axis_int32 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int32 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let continue = ref (increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int32_u.max cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let max_axis_int64 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int64 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    let a_lin = Shape.ravel_index in_md_index plan.in_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (plan.in_offset + a_lin));
    let continue = ref (increment_input_index plan in_md_index) in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int64_u.max cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let max_all_float64 a_arr va start_idx end_idx =
  let acc = Array.make_float64 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float_u.max cur (Array.unsafe_get a_arr i))
    done;
    Array.unsafe_get acc 0)
  else
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let md_index = Array.make (Array.length a_shape) 0 in
    Shape.unravel_index_into start_idx a_shape md_index;
    let first_lin = Shape.ravel_index md_index a_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (a_offset + first_lin));
    for k = start_idx + 1 to end_idx - 1 do
      Shape.unravel_index_into k a_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (a_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float_u.max cur v)
    done;
    Array.unsafe_get acc 0

let max_all_float32 a_arr va start_idx end_idx =
  let acc = Array.make_float32 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float32_u.max cur (Array.unsafe_get a_arr i))
    done;
    Array.unsafe_get acc 0)
  else
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let md_index = Array.make (Array.length a_shape) 0 in
    Shape.unravel_index_into start_idx a_shape md_index;
    let first_lin = Shape.ravel_index md_index a_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (a_offset + first_lin));
    for k = start_idx + 1 to end_idx - 1 do
      Shape.unravel_index_into k a_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (a_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Float32_u.max cur v)
    done;
    Array.unsafe_get acc 0

let max_all_int32 a_arr va start_idx end_idx =
  let acc = Array.make_int32 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int32_u.max cur (Array.unsafe_get a_arr i))
    done;
    Array.unsafe_get acc 0)
  else
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let md_index = Array.make (Array.length a_shape) 0 in
    Shape.unravel_index_into start_idx a_shape md_index;
    let first_lin = Shape.ravel_index md_index a_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (a_offset + first_lin));
    for k = start_idx + 1 to end_idx - 1 do
      Shape.unravel_index_into k a_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (a_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int32_u.max cur v)
    done;
    Array.unsafe_get acc 0

let max_all_int64 a_arr va start_idx end_idx =
  let acc = Array.make_int64 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int64_u.max cur (Array.unsafe_get a_arr i))
    done;
    Array.unsafe_get acc 0)
  else
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let md_index = Array.make (Array.length a_shape) 0 in
    Shape.unravel_index_into start_idx a_shape md_index;
    let first_lin = Shape.ravel_index md_index a_strides in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr (a_offset + first_lin));
    for k = start_idx + 1 to end_idx - 1 do
      Shape.unravel_index_into k a_shape md_index;
      let a_lin = Shape.ravel_index md_index a_strides in
      let v = Array.unsafe_get a_arr (a_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int64_u.max cur v)
    done;
    Array.unsafe_get acc 0

let reduce_sum_float64 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_float64 a_arr out_arr va vout s e)
    else copy_float64 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (
    if out_numel > 0 then fill_float64 out_arr vout (Float_u.of_int 0))
  else if out_numel = 1 then
    let total = sum_all_partial_float64 a_arr va 0 in_numel in
    fill_float64 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        sum_axis_float64 a_arr out_arr va vout axes keepdims s e)
  else sum_axis_float64 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_sum_float32 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_float32 a_arr out_arr va vout s e)
    else copy_float32 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (
    if out_numel > 0 then fill_float32 out_arr vout (Float32_u.of_int 0))
  else if out_numel = 1 then
    let total = sum_all_partial_float32 a_arr va 0 in_numel in
    fill_float32 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        sum_axis_float32 a_arr out_arr va vout axes keepdims s e)
  else sum_axis_float32 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_sum_int32 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int32 a_arr out_arr va vout s e)
    else copy_int32 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (if out_numel > 0 then fill_int32 out_arr vout #0l)
  else if out_numel = 1 then
    let total = sum_all_partial_int32 a_arr va 0 in_numel in
    fill_int32 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        sum_axis_int32 a_arr out_arr va vout axes keepdims s e)
  else sum_axis_int32 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_sum_int64 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int64 a_arr out_arr va vout s e)
    else copy_int64 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (if out_numel > 0 then fill_int64 out_arr vout #0L)
  else if out_numel = 1 then
    let total = sum_all_partial_int64 a_arr va 0 in_numel in
    fill_int64 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        sum_axis_int64 a_arr out_arr va vout axes keepdims s e)
  else sum_axis_int64 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_prod_float64 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_float64 a_arr out_arr va vout s e)
    else copy_float64 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (
    if out_numel > 0 then fill_float64 out_arr vout (Float_u.of_int 1))
  else if out_numel = 1 then
    let total = prod_all_partial_float64 a_arr va 0 in_numel in
    fill_float64 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        prod_axis_float64 a_arr out_arr va vout axes keepdims s e)
  else prod_axis_float64 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_prod_float32 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_float32 a_arr out_arr va vout s e)
    else copy_float32 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (
    if out_numel > 0 then fill_float32 out_arr vout (Float32_u.of_int 1))
  else if out_numel = 1 then
    let total = prod_all_partial_float32 a_arr va 0 in_numel in
    fill_float32 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        prod_axis_float32 a_arr out_arr va vout axes keepdims s e)
  else prod_axis_float32 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_prod_int32 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int32 a_arr out_arr va vout s e)
    else copy_int32 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (if out_numel > 0 then fill_int32 out_arr vout #1l)
  else if out_numel = 1 then
    let total = prod_all_partial_int32 a_arr va 0 in_numel in
    fill_int32 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        prod_axis_int32 a_arr out_arr va vout axes keepdims s e)
  else prod_axis_int32 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_prod_int64 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int64 a_arr out_arr va vout s e)
    else copy_int64 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (if out_numel > 0 then fill_int64 out_arr vout #1L)
  else if out_numel = 1 then
    let total = prod_all_partial_int64 a_arr va 0 in_numel in
    fill_int64 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        prod_axis_int64 a_arr out_arr va vout axes keepdims s e)
  else prod_axis_int64 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_min_float64 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_float64 a_arr out_arr va vout s e)
    else copy_float64 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_min" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = min_all_float64 a_arr va 0 in_numel in
    fill_float64 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        min_axis_float64 a_arr out_arr va vout axes keepdims s e)
  else min_axis_float64 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_min_float32 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_float32 a_arr out_arr va vout s e)
    else copy_float32 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_min" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = min_all_float32 a_arr va 0 in_numel in
    fill_float32 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        min_axis_float32 a_arr out_arr va vout axes keepdims s e)
  else min_axis_float32 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_min_int32 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int32 a_arr out_arr va vout s e)
    else copy_int32 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_min" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = min_all_int32 a_arr va 0 in_numel in
    fill_int32 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        min_axis_int32 a_arr out_arr va vout axes keepdims s e)
  else min_axis_int32 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_min_int64 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int64 a_arr out_arr va vout s e)
    else copy_int64 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_min" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = min_all_int64 a_arr va 0 in_numel in
    fill_int64 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        min_axis_int64 a_arr out_arr va vout axes keepdims s e)
  else min_axis_int64 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_max_float64 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_float64 a_arr out_arr va vout s e)
    else copy_float64 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_max" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = max_all_float64 a_arr va 0 in_numel in
    fill_float64 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        max_axis_float64 a_arr out_arr va vout axes keepdims s e)
  else max_axis_float64 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_max_float32 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_float32 a_arr out_arr va vout s e)
    else copy_float32 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_max" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = max_all_float32 a_arr va 0 in_numel in
    fill_float32 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        max_axis_float32 a_arr out_arr va vout axes keepdims s e)
  else max_axis_float32 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_max_int32 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int32 a_arr out_arr va vout s e)
    else copy_int32 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_max" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = max_all_int32 a_arr va 0 in_numel in
    fill_int32 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        max_axis_int32 a_arr out_arr va vout axes keepdims s e)
  else max_axis_int32 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_max_int64 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int64 a_arr out_arr va vout s e)
    else copy_int64 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_max" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = max_all_int64 a_arr va 0 in_numel in
    fill_int64 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        max_axis_int64 a_arr out_arr va vout axes keepdims s e)
  else max_axis_int64 a_arr out_arr va vout axes keepdims 0 out_numel
