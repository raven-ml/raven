(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let neg_float64 a_arr out_arr va vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Float_u.neg a0);
      Array.unsafe_set out_arr (out_base + i1) (Float_u.neg a1);
      Array.unsafe_set out_arr (out_base + i2) (Float_u.neg a2);
      Array.unsafe_set out_arr (out_base + i3) (Float_u.neg a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Float_u.neg a_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Float_u.neg a_val)
    done

let neg_float32 a_arr out_arr va vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Float32_u.neg a0);
      Array.unsafe_set out_arr (out_base + i1) (Float32_u.neg a1);
      Array.unsafe_set out_arr (out_base + i2) (Float32_u.neg a2);
      Array.unsafe_set out_arr (out_base + i3) (Float32_u.neg a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Float32_u.neg a_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Float32_u.neg a_val)
    done

let neg_int8 a_arr out_arr va vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Int8_u.neg a0);
      Array.unsafe_set out_arr (out_base + i1) (Int8_u.neg a1);
      Array.unsafe_set out_arr (out_base + i2) (Int8_u.neg a2);
      Array.unsafe_set out_arr (out_base + i3) (Int8_u.neg a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int8_u.neg a_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Int8_u.neg a_val)
    done

let neg_int16 a_arr out_arr va vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Int16_u.neg a0);
      Array.unsafe_set out_arr (out_base + i1) (Int16_u.neg a1);
      Array.unsafe_set out_arr (out_base + i2) (Int16_u.neg a2);
      Array.unsafe_set out_arr (out_base + i3) (Int16_u.neg a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int16_u.neg a_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Int16_u.neg a_val)
    done

let neg_int32 a_arr out_arr va vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Int32_u.neg a0);
      Array.unsafe_set out_arr (out_base + i1) (Int32_u.neg a1);
      Array.unsafe_set out_arr (out_base + i2) (Int32_u.neg a2);
      Array.unsafe_set out_arr (out_base + i3) (Int32_u.neg a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int32_u.neg a_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Int32_u.neg a_val)
    done

let neg_int64 a_arr out_arr va vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Int64_u.neg a0);
      Array.unsafe_set out_arr (out_base + i1) (Int64_u.neg a1);
      Array.unsafe_set out_arr (out_base + i2) (Int64_u.neg a2);
      Array.unsafe_set out_arr (out_base + i3) (Int64_u.neg a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int64_u.neg a_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let a_shape = shape va in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Int64_u.neg a_val)
    done
