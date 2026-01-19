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

let copy_int8 a_arr out_arr va vout start_idx end_idx =
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

let copy_int16 a_arr out_arr va vout start_idx end_idx =
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

let fill_int8 out_arr vout value =
  let out_offset = View.offset vout in
  let out_numel = numel vout in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr (out_offset + i) value
  done

let fill_int16 out_arr vout value =
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

let sum_axis_int8 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int8 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 #0s;
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int8_u.add cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let sum_axis_int16 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int16 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 #0S;
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int16_u.add cur v);
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
  else if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let n = end_idx - start_idx in
    (* 4x unrolled: process 8 elements (4 vectors of 2) per iteration *)
    let n8 = n - 7 in
    let rec unrolled_loop i (acc0 : float64x2#) (acc1 : float64x2#)
        (acc2 : float64x2#) (acc3 : float64x2#) =
      if i < n8 then
        let v0 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i) in
        let v1 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 2) in
        let v2 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 4) in
        let v3 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 6) in
        unrolled_loop (i + 8) (Float64x2.add acc0 v0) (Float64x2.add acc1 v1)
          (Float64x2.add acc2 v2) (Float64x2.add acc3 v3)
      else #(acc0, acc1, acc2, acc3, i)
    in
    let #(acc0, acc1, acc2, acc3, i) =
      unrolled_loop 0 (Float64x2.zero ()) (Float64x2.zero ()) (Float64x2.zero ())
        (Float64x2.zero ())
    in
    let acc01 = Float64x2.add acc0 acc1 in
    let acc23 = Float64x2.add acc2 acc3 in
    let acc_vec = Float64x2.add acc01 acc23 in
    (* Handle remaining 2-element chunks *)
    let n2 = n - 1 in
    let rec simd_loop j (acc : float64x2#) =
      if j < n2 then
        let vec = Float64x2.Array.unsafe_get a_arr ~idx:(base + j) in
        simd_loop (j + 2) (Float64x2.add acc vec)
      else acc
    in
    let acc_vec = simd_loop i acc_vec in
    let h = Float64x2.horizontal_add acc_vec acc_vec in
    let simd_result = Float64x2.extract0 h in
    let start_remainder = (n / 2) * 2 in
    let rec scalar_loop k (acc : float#) =
      if k < n then
        scalar_loop (k + 1) (Float_u.add acc (Array.unsafe_get a_arr (base + k)))
      else acc
    in
    scalar_loop start_remainder simd_result)
  else
    let acc = Array.make_float64 1 in
    Array.unsafe_set acc 0 (Float_u.of_int 0);
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
  else if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let n = end_idx - start_idx in
    (* 4x unrolled: process 16 elements (4 vectors of 4) per iteration *)
    let n16 = n - 15 in
    let rec unrolled_loop i (acc0 : float32x4#) (acc1 : float32x4#)
        (acc2 : float32x4#) (acc3 : float32x4#) =
      if i < n16 then
        let v0 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i) in
        let v1 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 4) in
        let v2 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 8) in
        let v3 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 12) in
        unrolled_loop (i + 16) (Float32x4.add acc0 v0) (Float32x4.add acc1 v1)
          (Float32x4.add acc2 v2) (Float32x4.add acc3 v3)
      else #(acc0, acc1, acc2, acc3, i)
    in
    let #(acc0, acc1, acc2, acc3, i) =
      unrolled_loop 0 (Float32x4.zero ()) (Float32x4.zero ()) (Float32x4.zero ())
        (Float32x4.zero ())
    in
    let acc01 = Float32x4.add acc0 acc1 in
    let acc23 = Float32x4.add acc2 acc3 in
    let acc_vec = Float32x4.add acc01 acc23 in
    (* Handle remaining 4-element chunks *)
    let n4 = n - 3 in
    let rec simd_loop j (acc : float32x4#) =
      if j < n4 then
        let vec = Float32x4.Array.unsafe_get a_arr ~idx:(base + j) in
        simd_loop (j + 4) (Float32x4.add acc vec)
      else acc
    in
    let acc_vec = simd_loop i acc_vec in
    let h1 = Float32x4.horizontal_add acc_vec acc_vec in
    let h2 = Float32x4.horizontal_add h1 h1 in
    let simd_result = Float32x4.extract0 h2 in
    let start_remainder = (n / 4) * 4 in
    let rec scalar_loop k (acc : float32#) =
      if k < n then
        scalar_loop (k + 1) (Float32_u.add acc (Array.unsafe_get a_arr (base + k)))
      else acc
    in
    scalar_loop start_remainder simd_result)
  else
    let acc = Array.make_float32 1 in
    Array.unsafe_set acc 0 (Float32_u.of_int 0);
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

let sum_all_partial_int8 a_arr va start_idx end_idx =
  if start_idx >= end_idx then #0s
  else
    let acc = Array.make_int8 1 in
    Array.unsafe_set acc 0 #0s;
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int8_u.add cur (Array.unsafe_get a_arr i))
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
        Array.unsafe_set acc 0 (Int8_u.add cur v)
      done;
      Array.unsafe_get acc 0

let sum_all_partial_int16 a_arr va start_idx end_idx =
  if start_idx >= end_idx then #0S
  else
    let acc = Array.make_int16 1 in
    Array.unsafe_set acc 0 #0S;
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int16_u.add cur (Array.unsafe_get a_arr i))
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
        Array.unsafe_set acc 0 (Int16_u.add cur v)
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

let prod_axis_int8 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int8 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 #1s;
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int8_u.mul cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let prod_axis_int16 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int16 1 in
  for k = start_idx to end_idx - 1 do
    Shape.unravel_index_into k plan.out_shape out_md_index;
    init_input_index plan out_md_index in_md_index;
    Array.unsafe_set acc 0 #1S;
    let continue = ref true in
    while !continue do
      let a_lin = Shape.ravel_index in_md_index plan.in_strides in
      let v = Array.unsafe_get a_arr (plan.in_offset + a_lin) in
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int16_u.mul cur v);
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
  else if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let n = end_idx - start_idx in
    (* 4x unrolled: process 8 elements (4 vectors of 2) per iteration *)
    let n8 = n - 7 in
    let rec unrolled_loop i (acc0 : float64x2#) (acc1 : float64x2#)
        (acc2 : float64x2#) (acc3 : float64x2#) =
      if i < n8 then
        let v0 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i) in
        let v1 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 2) in
        let v2 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 4) in
        let v3 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 6) in
        unrolled_loop (i + 8) (Float64x2.mul acc0 v0) (Float64x2.mul acc1 v1)
          (Float64x2.mul acc2 v2) (Float64x2.mul acc3 v3)
      else #(acc0, acc1, acc2, acc3, i)
    in
    let #(acc0, acc1, acc2, acc3, i) =
      unrolled_loop 0 (Float64x2.one ()) (Float64x2.one ()) (Float64x2.one ())
        (Float64x2.one ())
    in
    let acc01 = Float64x2.mul acc0 acc1 in
    let acc23 = Float64x2.mul acc2 acc3 in
    let acc_vec = Float64x2.mul acc01 acc23 in
    (* Handle remaining 2-element chunks *)
    let n2 = n - 1 in
    let rec simd_loop j (acc : float64x2#) =
      if j < n2 then
        let vec = Float64x2.Array.unsafe_get a_arr ~idx:(base + j) in
        simd_loop (j + 2) (Float64x2.mul acc vec)
      else acc
    in
    let acc_vec = simd_loop i acc_vec in
    let #(v0, v1) = Float64x2.splat acc_vec in
    let simd_result = Float_u.mul v0 v1 in
    let start_remainder = (n / 2) * 2 in
    let rec scalar_loop k (acc : float#) =
      if k < n then
        scalar_loop (k + 1) (Float_u.mul acc (Array.unsafe_get a_arr (base + k)))
      else acc
    in
    scalar_loop start_remainder simd_result)
  else
    let acc = Array.make_float64 1 in
    Array.unsafe_set acc 0 (Float_u.of_int 1);
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
  else if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let n = end_idx - start_idx in
    (* 4x unrolled: process 16 elements (4 vectors of 4) per iteration *)
    let n16 = n - 15 in
    let rec unrolled_loop i (acc0 : float32x4#) (acc1 : float32x4#)
        (acc2 : float32x4#) (acc3 : float32x4#) =
      if i < n16 then
        let v0 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i) in
        let v1 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 4) in
        let v2 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 8) in
        let v3 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 12) in
        unrolled_loop (i + 16) (Float32x4.mul acc0 v0) (Float32x4.mul acc1 v1)
          (Float32x4.mul acc2 v2) (Float32x4.mul acc3 v3)
      else #(acc0, acc1, acc2, acc3, i)
    in
    let #(acc0, acc1, acc2, acc3, i) =
      unrolled_loop 0 (Float32x4.one ()) (Float32x4.one ()) (Float32x4.one ())
        (Float32x4.one ())
    in
    let acc01 = Float32x4.mul acc0 acc1 in
    let acc23 = Float32x4.mul acc2 acc3 in
    let acc_vec = Float32x4.mul acc01 acc23 in
    (* Handle remaining 4-element chunks *)
    let n4 = n - 3 in
    let rec simd_loop j (acc : float32x4#) =
      if j < n4 then
        let vec = Float32x4.Array.unsafe_get a_arr ~idx:(base + j) in
        simd_loop (j + 4) (Float32x4.mul acc vec)
      else acc
    in
    let acc_vec = simd_loop i acc_vec in
    let #(v0, v1, v2, v3) = Float32x4.splat acc_vec in
    let simd_result = Float32_u.mul (Float32_u.mul v0 v1) (Float32_u.mul v2 v3) in
    let start_remainder = (n / 4) * 4 in
    let rec scalar_loop k (acc : float32#) =
      if k < n then
        scalar_loop (k + 1) (Float32_u.mul acc (Array.unsafe_get a_arr (base + k)))
      else acc
    in
    scalar_loop start_remainder simd_result)
  else
    let acc = Array.make_float32 1 in
    Array.unsafe_set acc 0 (Float32_u.of_int 1);
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

let prod_all_partial_int8 a_arr va start_idx end_idx =
  if start_idx >= end_idx then #1s
  else
    let acc = Array.make_int8 1 in
    Array.unsafe_set acc 0 #1s;
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int8_u.mul cur (Array.unsafe_get a_arr i))
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
        Array.unsafe_set acc 0 (Int8_u.mul cur v)
      done;
      Array.unsafe_get acc 0

let prod_all_partial_int16 a_arr va start_idx end_idx =
  if start_idx >= end_idx then #1S
  else
    let acc = Array.make_int16 1 in
    Array.unsafe_set acc 0 #1S;
    if View.is_c_contiguous va then (
      let base = View.offset va + start_idx in
      let last = View.offset va + end_idx in
      for i = base to last - 1 do
        let cur = Array.unsafe_get acc 0 in
        Array.unsafe_set acc 0 (Int16_u.mul cur (Array.unsafe_get a_arr i))
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
        Array.unsafe_set acc 0 (Int16_u.mul cur v)
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

let min_axis_int8 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int8 1 in
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
      Array.unsafe_set acc 0 (Int8_u.min cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let min_axis_int16 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int16 1 in
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
      Array.unsafe_set acc 0 (Int16_u.min cur v);
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
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let n = end_idx - start_idx in
    if n < 2 then Array.unsafe_get a_arr base
    else
      (* 4x unrolled: process 8 elements (4 vectors of 2) per iteration *)
      let n8 = n - 7 in
      let first_vec = Float64x2.Array.unsafe_get a_arr ~idx:base in
      let rec unrolled_loop i (acc0 : float64x2#) (acc1 : float64x2#)
          (acc2 : float64x2#) (acc3 : float64x2#) =
        if i < n8 then
          let v0 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i) in
          let v1 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 2) in
          let v2 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 4) in
          let v3 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 6) in
          unrolled_loop (i + 8) (Float64x2.min acc0 v0) (Float64x2.min acc1 v1)
            (Float64x2.min acc2 v2) (Float64x2.min acc3 v3)
        else #(acc0, acc1, acc2, acc3, i)
      in
      let #(acc0, acc1, acc2, acc3, i) =
        unrolled_loop 2 first_vec first_vec first_vec first_vec
      in
      let acc01 = Float64x2.min acc0 acc1 in
      let acc23 = Float64x2.min acc2 acc3 in
      let acc_vec = Float64x2.min acc01 acc23 in
      (* Handle remaining 2-element chunks *)
      let n2 = n - 1 in
      let rec simd_loop j (acc : float64x2#) =
        if j < n2 then
          let vec = Float64x2.Array.unsafe_get a_arr ~idx:(base + j) in
          simd_loop (j + 2) (Float64x2.min acc vec)
        else acc
      in
      let acc_vec = simd_loop i acc_vec in
      let #(v0, v1) = Float64x2.splat acc_vec in
      let simd_result = Float_u.min v0 v1 in
      let start_remainder = (n / 2) * 2 in
      let rec scalar_loop k (acc : float#) =
        if k < n then
          scalar_loop (k + 1) (Float_u.min acc (Array.unsafe_get a_arr (base + k)))
        else acc
      in
      scalar_loop start_remainder simd_result)
  else
    let acc = Array.make_float64 1 in
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
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let n = end_idx - start_idx in
    if n < 4 then (
      let rec scalar_loop i (acc : float32#) =
        if i < n then
          scalar_loop (i + 1) (Float32_u.min acc (Array.unsafe_get a_arr (base + i)))
        else acc
      in
      scalar_loop 1 (Array.unsafe_get a_arr base))
    else
      (* 4x unrolled: process 16 elements (4 vectors of 4) per iteration *)
      let n16 = n - 15 in
      let first_vec = Float32x4.Array.unsafe_get a_arr ~idx:base in
      let rec unrolled_loop i (acc0 : float32x4#) (acc1 : float32x4#)
          (acc2 : float32x4#) (acc3 : float32x4#) =
        if i < n16 then
          let v0 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i) in
          let v1 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 4) in
          let v2 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 8) in
          let v3 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 12) in
          unrolled_loop (i + 16) (Float32x4.min acc0 v0) (Float32x4.min acc1 v1)
            (Float32x4.min acc2 v2) (Float32x4.min acc3 v3)
        else #(acc0, acc1, acc2, acc3, i)
      in
      let #(acc0, acc1, acc2, acc3, i) =
        unrolled_loop 4 first_vec first_vec first_vec first_vec
      in
      let acc01 = Float32x4.min acc0 acc1 in
      let acc23 = Float32x4.min acc2 acc3 in
      let acc_vec = Float32x4.min acc01 acc23 in
      (* Handle remaining 4-element chunks *)
      let n4 = n - 3 in
      let rec simd_loop j (acc : float32x4#) =
        if j < n4 then
          let vec = Float32x4.Array.unsafe_get a_arr ~idx:(base + j) in
          simd_loop (j + 4) (Float32x4.min acc vec)
        else acc
      in
      let acc_vec = simd_loop i acc_vec in
      let #(v0, v1, v2, v3) = Float32x4.splat acc_vec in
      let simd_result = Float32_u.min (Float32_u.min v0 v1) (Float32_u.min v2 v3) in
      let start_remainder = (n / 4) * 4 in
      let rec scalar_loop k (acc : float32#) =
        if k < n then
          scalar_loop (k + 1) (Float32_u.min acc (Array.unsafe_get a_arr (base + k)))
        else acc
      in
      scalar_loop start_remainder simd_result)
  else
    let acc = Array.make_float32 1 in
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

let min_all_int8 a_arr va start_idx end_idx =
  let acc = Array.make_int8 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int8_u.min cur (Array.unsafe_get a_arr i))
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
      Array.unsafe_set acc 0 (Int8_u.min cur v)
    done;
    Array.unsafe_get acc 0

let min_all_int16 a_arr va start_idx end_idx =
  let acc = Array.make_int16 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int16_u.min cur (Array.unsafe_get a_arr i))
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
      Array.unsafe_set acc 0 (Int16_u.min cur v)
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

let max_axis_int8 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int8 1 in
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
      Array.unsafe_set acc 0 (Int8_u.max cur v);
      continue := increment_input_index plan in_md_index
    done;
    Array.unsafe_set out_arr (plan.out_offset + k) (Array.unsafe_get acc 0)
  done

let max_axis_int16 a_arr out_arr va vout axes keepdims start_idx end_idx =
  let plan = make_plan axes keepdims va vout in
  let out_md_index = Array.make plan.out_rank 0 in
  let in_md_index = Array.make plan.rank 0 in
  let acc = Array.make_int16 1 in
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
      Array.unsafe_set acc 0 (Int16_u.max cur v);
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
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let n = end_idx - start_idx in
    if n < 2 then Array.unsafe_get a_arr base
    else
      (* 4x unrolled: process 8 elements (4 vectors of 2) per iteration *)
      let n8 = n - 7 in
      let first_vec = Float64x2.Array.unsafe_get a_arr ~idx:base in
      let rec unrolled_loop i (acc0 : float64x2#) (acc1 : float64x2#)
          (acc2 : float64x2#) (acc3 : float64x2#) =
        if i < n8 then
          let v0 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i) in
          let v1 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 2) in
          let v2 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 4) in
          let v3 = Float64x2.Array.unsafe_get a_arr ~idx:(base + i + 6) in
          unrolled_loop (i + 8) (Float64x2.max acc0 v0) (Float64x2.max acc1 v1)
            (Float64x2.max acc2 v2) (Float64x2.max acc3 v3)
        else #(acc0, acc1, acc2, acc3, i)
      in
      let #(acc0, acc1, acc2, acc3, i) =
        unrolled_loop 2 first_vec first_vec first_vec first_vec
      in
      let acc01 = Float64x2.max acc0 acc1 in
      let acc23 = Float64x2.max acc2 acc3 in
      let acc_vec = Float64x2.max acc01 acc23 in
      (* Handle remaining 2-element chunks *)
      let n2 = n - 1 in
      let rec simd_loop j (acc : float64x2#) =
        if j < n2 then
          let vec = Float64x2.Array.unsafe_get a_arr ~idx:(base + j) in
          simd_loop (j + 2) (Float64x2.max acc vec)
        else acc
      in
      let acc_vec = simd_loop i acc_vec in
      let #(v0, v1) = Float64x2.splat acc_vec in
      let simd_result = Float_u.max v0 v1 in
      let start_remainder = (n / 2) * 2 in
      let rec scalar_loop k (acc : float#) =
        if k < n then
          scalar_loop (k + 1) (Float_u.max acc (Array.unsafe_get a_arr (base + k)))
        else acc
      in
      scalar_loop start_remainder simd_result)
  else
    let acc = Array.make_float64 1 in
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
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let n = end_idx - start_idx in
    if n < 4 then (
      let rec scalar_loop i (acc : float32#) =
        if i < n then
          scalar_loop (i + 1) (Float32_u.max acc (Array.unsafe_get a_arr (base + i)))
        else acc
      in
      scalar_loop 1 (Array.unsafe_get a_arr base))
    else
      (* 4x unrolled: process 16 elements (4 vectors of 4) per iteration *)
      let n16 = n - 15 in
      let first_vec = Float32x4.Array.unsafe_get a_arr ~idx:base in
      let rec unrolled_loop i (acc0 : float32x4#) (acc1 : float32x4#)
          (acc2 : float32x4#) (acc3 : float32x4#) =
        if i < n16 then
          let v0 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i) in
          let v1 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 4) in
          let v2 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 8) in
          let v3 = Float32x4.Array.unsafe_get a_arr ~idx:(base + i + 12) in
          unrolled_loop (i + 16) (Float32x4.max acc0 v0) (Float32x4.max acc1 v1)
            (Float32x4.max acc2 v2) (Float32x4.max acc3 v3)
        else #(acc0, acc1, acc2, acc3, i)
      in
      let #(acc0, acc1, acc2, acc3, i) =
        unrolled_loop 4 first_vec first_vec first_vec first_vec
      in
      let acc01 = Float32x4.max acc0 acc1 in
      let acc23 = Float32x4.max acc2 acc3 in
      let acc_vec = Float32x4.max acc01 acc23 in
      (* Handle remaining 4-element chunks *)
      let n4 = n - 3 in
      let rec simd_loop j (acc : float32x4#) =
        if j < n4 then
          let vec = Float32x4.Array.unsafe_get a_arr ~idx:(base + j) in
          simd_loop (j + 4) (Float32x4.max acc vec)
        else acc
      in
      let acc_vec = simd_loop i acc_vec in
      let #(v0, v1, v2, v3) = Float32x4.splat acc_vec in
      let simd_result = Float32_u.max (Float32_u.max v0 v1) (Float32_u.max v2 v3) in
      let start_remainder = (n / 4) * 4 in
      let rec scalar_loop k (acc : float32#) =
        if k < n then
          scalar_loop (k + 1) (Float32_u.max acc (Array.unsafe_get a_arr (base + k)))
        else acc
      in
      scalar_loop start_remainder simd_result)
  else
    let acc = Array.make_float32 1 in
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

let max_all_int8 a_arr va start_idx end_idx =
  let acc = Array.make_int8 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int8_u.max cur (Array.unsafe_get a_arr i))
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
      Array.unsafe_set acc 0 (Int8_u.max cur v)
    done;
    Array.unsafe_get acc 0

let max_all_int16 a_arr va start_idx end_idx =
  let acc = Array.make_int16 1 in
  if View.is_c_contiguous va then (
    let base = View.offset va + start_idx in
    let last = View.offset va + end_idx in
    Array.unsafe_set acc 0 (Array.unsafe_get a_arr base);
    for i = base + 1 to last - 1 do
      let cur = Array.unsafe_get acc 0 in
      Array.unsafe_set acc 0 (Int16_u.max cur (Array.unsafe_get a_arr i))
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
      Array.unsafe_set acc 0 (Int16_u.max cur v)
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

let reduce_sum_int8 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int8 a_arr out_arr va vout s e)
    else copy_int8 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (if out_numel > 0 then fill_int8 out_arr vout #0s)
  else if out_numel = 1 then
    let total = sum_all_partial_int8 a_arr va 0 in_numel in
    fill_int8 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        sum_axis_int8 a_arr out_arr va vout axes keepdims s e)
  else sum_axis_int8 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_sum_int16 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int16 a_arr out_arr va vout s e)
    else copy_int16 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (if out_numel > 0 then fill_int16 out_arr vout #0S)
  else if out_numel = 1 then
    let total = sum_all_partial_int16 a_arr va 0 in_numel in
    fill_int16 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        sum_axis_int16 a_arr out_arr va vout axes keepdims s e)
  else sum_axis_int16 a_arr out_arr va vout axes keepdims 0 out_numel

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

let reduce_prod_int8 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int8 a_arr out_arr va vout s e)
    else copy_int8 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (if out_numel > 0 then fill_int8 out_arr vout #1s)
  else if out_numel = 1 then
    let total = prod_all_partial_int8 a_arr va 0 in_numel in
    fill_int8 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        prod_axis_int8 a_arr out_arr va vout axes keepdims s e)
  else prod_axis_int8 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_prod_int16 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int16 a_arr out_arr va vout s e)
    else copy_int16 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then (if out_numel > 0 then fill_int16 out_arr vout #1S)
  else if out_numel = 1 then
    let total = prod_all_partial_int16 a_arr va 0 in_numel in
    fill_int16 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        prod_axis_int16 a_arr out_arr va vout axes keepdims s e)
  else prod_axis_int16 a_arr out_arr va vout axes keepdims 0 out_numel

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

let reduce_min_int8 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int8 a_arr out_arr va vout s e)
    else copy_int8 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_min" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = min_all_int8 a_arr va 0 in_numel in
    fill_int8 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        min_axis_int8 a_arr out_arr va vout axes keepdims s e)
  else min_axis_int8 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_min_int16 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int16 a_arr out_arr va vout s e)
    else copy_int16 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_min" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = min_all_int16 a_arr va 0 in_numel in
    fill_int16 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        min_axis_int16 a_arr out_arr va vout axes keepdims s e)
  else min_axis_int16 a_arr out_arr va vout axes keepdims 0 out_numel

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

let reduce_max_int8 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int8 a_arr out_arr va vout s e)
    else copy_int8 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_max" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = max_all_int8 a_arr va 0 in_numel in
    fill_int8 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        max_axis_int8 a_arr out_arr va vout axes keepdims s e)
  else max_axis_int8 a_arr out_arr va vout axes keepdims 0 out_numel

let reduce_max_int16 pool ~out_arr ~a_arr ~va ~vout ~axes ~keepdims =
  let in_numel = numel va in
  let out_numel = numel vout in
  if Array.length axes = 0 then
    if out_numel = 0 then ()
    else if out_numel > parallel_threshold then
      Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
          copy_int16 a_arr out_arr va vout s e)
    else copy_int16 a_arr out_arr va vout 0 out_numel
  else if in_numel = 0 then
    Error.invalid ~op:"reduce_max" ~what:"input" ~reason:"empty" ()
  else if out_numel = 1 then
    let total = max_all_int16 a_arr va 0 in_numel in
    fill_int16 out_arr vout total
  else if out_numel > parallel_threshold then
    Parallel.parallel_for pool 0 (out_numel - 1) (fun s e ->
        max_axis_int16 a_arr out_arr va vout axes keepdims s e)
  else max_axis_int16 a_arr out_arr va vout axes keepdims 0 out_numel

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
