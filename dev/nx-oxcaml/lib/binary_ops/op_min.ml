(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let min_float64 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n8 = n - 7 in
    while !i < n8 do
      let idx = !i in
      let a0 = Float64x2.Array.unsafe_get a_arr ~idx:(a_base + idx) in
      let b0 = Float64x2.Array.unsafe_get b_arr ~idx:(b_base + idx) in
      let a1 = Float64x2.Array.unsafe_get a_arr ~idx:(a_base + idx + 2) in
      let b1 = Float64x2.Array.unsafe_get b_arr ~idx:(b_base + idx + 2) in
      let a2 = Float64x2.Array.unsafe_get a_arr ~idx:(a_base + idx + 4) in
      let b2 = Float64x2.Array.unsafe_get b_arr ~idx:(b_base + idx + 4) in
      let a3 = Float64x2.Array.unsafe_get a_arr ~idx:(a_base + idx + 6) in
      let b3 = Float64x2.Array.unsafe_get b_arr ~idx:(b_base + idx + 6) in
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx) (Float64x2.min a0 b0);
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 2) (Float64x2.min a1 b1);
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 4) (Float64x2.min a2 b2);
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 6) (Float64x2.min a3 b3);
      i := idx + 8
    done;
    let n2 = n - 1 in
    while !i < n2 do
      let idx = !i in
      let a_vec = Float64x2.Array.unsafe_get a_arr ~idx:(a_base + idx) in
      let b_vec = Float64x2.Array.unsafe_get b_arr ~idx:(b_base + idx) in
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx) (Float64x2.min a_vec b_vec);
      i := idx + 2
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Float_u.min a_val b_val);
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
      Array.unsafe_set out_arr (out_offset + k) (Float_u.min a_val b_val)
    done

let min_float32 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n16 = n - 15 in
    while !i < n16 do
      let idx = !i in
      let a0 = Float32x4.Array.unsafe_get a_arr ~idx:(a_base + idx) in
      let b0 = Float32x4.Array.unsafe_get b_arr ~idx:(b_base + idx) in
      let a1 = Float32x4.Array.unsafe_get a_arr ~idx:(a_base + idx + 4) in
      let b1 = Float32x4.Array.unsafe_get b_arr ~idx:(b_base + idx + 4) in
      let a2 = Float32x4.Array.unsafe_get a_arr ~idx:(a_base + idx + 8) in
      let b2 = Float32x4.Array.unsafe_get b_arr ~idx:(b_base + idx + 8) in
      let a3 = Float32x4.Array.unsafe_get a_arr ~idx:(a_base + idx + 12) in
      let b3 = Float32x4.Array.unsafe_get b_arr ~idx:(b_base + idx + 12) in
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx) (Float32x4.min a0 b0);
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 4) (Float32x4.min a1 b1);
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 8) (Float32x4.min a2 b2);
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 12) (Float32x4.min a3 b3);
      i := idx + 16
    done;
    let n4 = n - 3 in
    while !i < n4 do
      let idx = !i in
      let a_vec = Float32x4.Array.unsafe_get a_arr ~idx:(a_base + idx) in
      let b_vec = Float32x4.Array.unsafe_get b_arr ~idx:(b_base + idx) in
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx) (Float32x4.min a_vec b_vec);
      i := idx + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Float32_u.min a_val b_val);
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
      Array.unsafe_set out_arr (out_offset + k) (Float32_u.min a_val b_val)
    done

let min_int8 a_arr b_arr out_arr va vb vout start_idx end_idx =
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
      Array.unsafe_set out_arr (out_base + i0) (Int8_u.min a0 b0);
      Array.unsafe_set out_arr (out_base + i1) (Int8_u.min a1 b1);
      Array.unsafe_set out_arr (out_base + i2) (Int8_u.min a2 b2);
      Array.unsafe_set out_arr (out_base + i3) (Int8_u.min a3 b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int8_u.min a_val b_val);
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
      Array.unsafe_set out_arr (out_offset + k) (Int8_u.min a_val b_val)
    done

let min_int16 a_arr b_arr out_arr va vb vout start_idx end_idx =
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
      Array.unsafe_set out_arr (out_base + i0) (Int16_u.min a0 b0);
      Array.unsafe_set out_arr (out_base + i1) (Int16_u.min a1 b1);
      Array.unsafe_set out_arr (out_base + i2) (Int16_u.min a2 b2);
      Array.unsafe_set out_arr (out_base + i3) (Int16_u.min a3 b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int16_u.min a_val b_val);
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
      Array.unsafe_set out_arr (out_offset + k) (Int16_u.min a_val b_val)
    done

let min_int32 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n16 = n - 15 in
    while !i < n16 do
      let idx = !i in
      let a0 = Int32x4.Array.unsafe_get a_arr ~idx:(a_base + idx) in
      let b0 = Int32x4.Array.unsafe_get b_arr ~idx:(b_base + idx) in
      let a1 = Int32x4.Array.unsafe_get a_arr ~idx:(a_base + idx + 4) in
      let b1 = Int32x4.Array.unsafe_get b_arr ~idx:(b_base + idx + 4) in
      let a2 = Int32x4.Array.unsafe_get a_arr ~idx:(a_base + idx + 8) in
      let b2 = Int32x4.Array.unsafe_get b_arr ~idx:(b_base + idx + 8) in
      let a3 = Int32x4.Array.unsafe_get a_arr ~idx:(a_base + idx + 12) in
      let b3 = Int32x4.Array.unsafe_get b_arr ~idx:(b_base + idx + 12) in
      Int32x4.Array.unsafe_set out_arr ~idx:(out_base + idx) (Int32x4.min a0 b0);
      Int32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 4) (Int32x4.min a1 b1);
      Int32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 8) (Int32x4.min a2 b2);
      Int32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 12) (Int32x4.min a3 b3);
      i := idx + 16
    done;
    let n4 = n - 3 in
    while !i < n4 do
      let idx = !i in
      let a_vec = Int32x4.Array.unsafe_get a_arr ~idx:(a_base + idx) in
      let b_vec = Int32x4.Array.unsafe_get b_arr ~idx:(b_base + idx) in
      Int32x4.Array.unsafe_set out_arr ~idx:(out_base + idx) (Int32x4.min a_vec b_vec);
      i := idx + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int32_u.min a_val b_val);
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
      Array.unsafe_set out_arr (out_offset + k) (Int32_u.min a_val b_val)
    done

let min_int64 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n8 = n - 7 in
    while !i < n8 do
      let idx = !i in
      let a0 = Int64x2.Array.unsafe_get a_arr ~idx:(a_base + idx) in
      let b0 = Int64x2.Array.unsafe_get b_arr ~idx:(b_base + idx) in
      let a1 = Int64x2.Array.unsafe_get a_arr ~idx:(a_base + idx + 2) in
      let b1 = Int64x2.Array.unsafe_get b_arr ~idx:(b_base + idx + 2) in
      let a2 = Int64x2.Array.unsafe_get a_arr ~idx:(a_base + idx + 4) in
      let b2 = Int64x2.Array.unsafe_get b_arr ~idx:(b_base + idx + 4) in
      let a3 = Int64x2.Array.unsafe_get a_arr ~idx:(a_base + idx + 6) in
      let b3 = Int64x2.Array.unsafe_get b_arr ~idx:(b_base + idx + 6) in
      Int64x2.Array.unsafe_set out_arr ~idx:(out_base + idx) (Int64x2.min a0 b0);
      Int64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 2) (Int64x2.min a1 b1);
      Int64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 4) (Int64x2.min a2 b2);
      Int64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 6) (Int64x2.min a3 b3);
      i := idx + 8
    done;
    let n2 = n - 1 in
    while !i < n2 do
      let idx = !i in
      let a_vec = Int64x2.Array.unsafe_get a_arr ~idx:(a_base + idx) in
      let b_vec = Int64x2.Array.unsafe_get b_arr ~idx:(b_base + idx) in
      Int64x2.Array.unsafe_set out_arr ~idx:(out_base + idx) (Int64x2.min a_vec b_vec);
      i := idx + 2
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int64_u.min a_val b_val);
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
      Array.unsafe_set out_arr (out_offset + k) (Int64_u.min a_val b_val)
    done
