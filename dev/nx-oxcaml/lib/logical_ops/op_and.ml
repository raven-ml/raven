(*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let and_int8 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let b0 = Array.unsafe_get b_arr (b_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let b1 = Array.unsafe_get b_arr (b_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let b2 = Array.unsafe_get b_arr (b_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      let b3 = Array.unsafe_get b_arr (b_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Int8_u.logand a0 b0);
      Array.unsafe_set out_arr (out_base + i1) (Int8_u.logand a1 b1);
      Array.unsafe_set out_arr (out_base + i2) (Int8_u.logand a2 b2);
      Array.unsafe_set out_arr (out_base + i3) (Int8_u.logand a3 b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int8_u.logand a_val b_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let a_shape = shape va in
    let b_shape = shape vb in
    let a_strides = View.strides va in
    let b_strides = View.strides vb in
    let a_offset = View.offset va in
    let b_offset = View.offset vb in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      let b_val = Array.unsafe_get b_arr (b_offset + b_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Int8_u.logand a_val b_val)
    done

let and_int16 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let b0 = Array.unsafe_get b_arr (b_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let b1 = Array.unsafe_get b_arr (b_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let b2 = Array.unsafe_get b_arr (b_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      let b3 = Array.unsafe_get b_arr (b_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Int16_u.logand a0 b0);
      Array.unsafe_set out_arr (out_base + i1) (Int16_u.logand a1 b1);
      Array.unsafe_set out_arr (out_base + i2) (Int16_u.logand a2 b2);
      Array.unsafe_set out_arr (out_base + i3) (Int16_u.logand a3 b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int16_u.logand a_val b_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let a_shape = shape va in
    let b_shape = shape vb in
    let a_strides = View.strides va in
    let b_strides = View.strides vb in
    let a_offset = View.offset va in
    let b_offset = View.offset vb in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      let b_val = Array.unsafe_get b_arr (b_offset + b_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Int16_u.logand a_val b_val)
    done

let and_int32 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let b0 = Array.unsafe_get b_arr (b_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let b1 = Array.unsafe_get b_arr (b_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let b2 = Array.unsafe_get b_arr (b_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      let b3 = Array.unsafe_get b_arr (b_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Int32_u.logand a0 b0);
      Array.unsafe_set out_arr (out_base + i1) (Int32_u.logand a1 b1);
      Array.unsafe_set out_arr (out_base + i2) (Int32_u.logand a2 b2);
      Array.unsafe_set out_arr (out_base + i3) (Int32_u.logand a3 b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int32_u.logand a_val b_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let a_shape = shape va in
    let b_shape = shape vb in
    let a_strides = View.strides va in
    let b_strides = View.strides vb in
    let a_offset = View.offset va in
    let b_offset = View.offset vb in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      let b_val = Array.unsafe_get b_arr (b_offset + b_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Int32_u.logand a_val b_val)
    done

let and_int64 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let b0 = Array.unsafe_get b_arr (b_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let b1 = Array.unsafe_get b_arr (b_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let b2 = Array.unsafe_get b_arr (b_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      let b3 = Array.unsafe_get b_arr (b_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Int64_u.logand a0 b0);
      Array.unsafe_set out_arr (out_base + i1) (Int64_u.logand a1 b1);
      Array.unsafe_set out_arr (out_base + i2) (Int64_u.logand a2 b2);
      Array.unsafe_set out_arr (out_base + i3) (Int64_u.logand a3 b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int64_u.logand a_val b_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let a_shape = shape va in
    let b_shape = shape vb in
    let a_strides = View.strides va in
    let b_strides = View.strides vb in
    let a_offset = View.offset va in
    let b_offset = View.offset vb in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      let b_val = Array.unsafe_get b_arr (b_offset + b_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Int64_u.logand a_val b_val)
    done
