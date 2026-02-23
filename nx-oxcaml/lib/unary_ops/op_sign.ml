(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let[@inline] sign_float64_scalar x =
  if Float_u.is_nan x then x
  else
    let c = Float_u.compare x #0.0 in
    if c > 0 then #1.0 else if c < 0 then -#1.0 else #0.0

let[@inline] sign_float32_scalar x =
  if Float32_u.is_nan x then x
  else
    let c = Float32_u.compare x #0.0s in
    if c > 0 then #1.0s else if c < 0 then -#1.0s else #0.0s

let[@inline] sign_int8_scalar x =
  let c = Int8_u.compare x #0s in
  if c > 0 then #1s else if c < 0 then -#1s else #0s

let[@inline] sign_int16_scalar x =
  let c = Int16_u.compare x #0S in
  if c > 0 then #1S else if c < 0 then -#1S else #0S

let[@inline] sign_int32_scalar x =
  let c = Int32_u.compare x #0l in
  if c > 0 then #1l else if c < 0 then -#1l else #0l

let[@inline] sign_int64_scalar x =
  let c = Int64_u.compare x #0L in
  if c > 0 then #1L else if c < 0 then -#1L else #0L

let sign_float64 a_arr out_arr va vout start_idx end_idx =
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
      Array.unsafe_set out_arr (out_base + i0) (sign_float64_scalar a0);
      Array.unsafe_set out_arr (out_base + i1) (sign_float64_scalar a1);
      Array.unsafe_set out_arr (out_base + i2) (sign_float64_scalar a2);
      Array.unsafe_set out_arr (out_base + i3) (sign_float64_scalar a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (sign_float64_scalar a_val);
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
      Array.unsafe_set out_arr (out_offset + k) (sign_float64_scalar a_val)
    done

let sign_float32 a_arr out_arr va vout start_idx end_idx =
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
      Array.unsafe_set out_arr (out_base + i0) (sign_float32_scalar a0);
      Array.unsafe_set out_arr (out_base + i1) (sign_float32_scalar a1);
      Array.unsafe_set out_arr (out_base + i2) (sign_float32_scalar a2);
      Array.unsafe_set out_arr (out_base + i3) (sign_float32_scalar a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (sign_float32_scalar a_val);
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
      Array.unsafe_set out_arr (out_offset + k) (sign_float32_scalar a_val)
    done

let sign_int8 a_arr out_arr va vout start_idx end_idx =
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
      Array.unsafe_set out_arr (out_base + i0) (sign_int8_scalar a0);
      Array.unsafe_set out_arr (out_base + i1) (sign_int8_scalar a1);
      Array.unsafe_set out_arr (out_base + i2) (sign_int8_scalar a2);
      Array.unsafe_set out_arr (out_base + i3) (sign_int8_scalar a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (sign_int8_scalar a_val);
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
      Array.unsafe_set out_arr (out_offset + k) (sign_int8_scalar a_val)
    done

let sign_int16 a_arr out_arr va vout start_idx end_idx =
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
      Array.unsafe_set out_arr (out_base + i0) (sign_int16_scalar a0);
      Array.unsafe_set out_arr (out_base + i1) (sign_int16_scalar a1);
      Array.unsafe_set out_arr (out_base + i2) (sign_int16_scalar a2);
      Array.unsafe_set out_arr (out_base + i3) (sign_int16_scalar a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (sign_int16_scalar a_val);
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
      Array.unsafe_set out_arr (out_offset + k) (sign_int16_scalar a_val)
    done

let sign_int32 a_arr out_arr va vout start_idx end_idx =
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
      Array.unsafe_set out_arr (out_base + i0) (sign_int32_scalar a0);
      Array.unsafe_set out_arr (out_base + i1) (sign_int32_scalar a1);
      Array.unsafe_set out_arr (out_base + i2) (sign_int32_scalar a2);
      Array.unsafe_set out_arr (out_base + i3) (sign_int32_scalar a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (sign_int32_scalar a_val);
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
      Array.unsafe_set out_arr (out_offset + k) (sign_int32_scalar a_val)
    done

let sign_int64 a_arr out_arr va vout start_idx end_idx =
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
      Array.unsafe_set out_arr (out_base + i0) (sign_int64_scalar a0);
      Array.unsafe_set out_arr (out_base + i1) (sign_int64_scalar a1);
      Array.unsafe_set out_arr (out_base + i2) (sign_int64_scalar a2);
      Array.unsafe_set out_arr (out_base + i3) (sign_int64_scalar a3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (sign_int64_scalar a_val);
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
      Array.unsafe_set out_arr (out_offset + k) (sign_int64_scalar a_val)
    done

let sign_bool a_arr out_arr va vout start_idx end_idx =
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
      Array.unsafe_set out_arr (out_base + i0) (Array.unsafe_get a_arr (a_base + i0));
      Array.unsafe_set out_arr (out_base + i1) (Array.unsafe_get a_arr (a_base + i1));
      Array.unsafe_set out_arr (out_base + i2) (Array.unsafe_get a_arr (a_base + i2));
      Array.unsafe_set out_arr (out_base + i3) (Array.unsafe_get a_arr (a_base + i3));
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      Array.unsafe_set out_arr (out_base + idx) (Array.unsafe_get a_arr (a_base + idx));
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
      Array.unsafe_set out_arr (out_offset + k) (Array.unsafe_get a_arr (a_offset + a_lin))
    done
