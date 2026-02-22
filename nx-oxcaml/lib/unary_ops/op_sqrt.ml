(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let sqrt_float64 a_arr out_arr va vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n2 = n - 1 in
    while !i < n2 do
      let idx = !i in
      let a_vec = Float64x2.Array.unsafe_get a_arr ~idx:(a_base + idx) in
      let out_vec = Float64x2.sqrt a_vec in
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx) out_vec;
      i := idx + 2
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Float_u.sqrt a_val);
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
      Array.unsafe_set out_arr (out_offset + k) (Float_u.sqrt a_val)
    done

let sqrt_float32 a_arr out_arr va vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let idx = !i in
      let a_vec = Float32x4.Array.unsafe_get a_arr ~idx:(a_base + idx) in
      let out_vec = Float32x4.sqrt a_vec in
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx) out_vec;
      i := idx + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Float32_u.sqrt a_val);
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
      Array.unsafe_set out_arr (out_offset + k) (Float32_u.sqrt a_val)
    done
