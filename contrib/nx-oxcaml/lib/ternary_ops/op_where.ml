(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let where_float64
    cond_arr true_arr false_arr out_arr
    vcond vtrue vfalse vout
    start_idx end_idx =
  let[@inline] select_f64x2 (mask : Int64x2.t) (a : Float64x2.t) (b : Float64x2.t) =
    let ai = Int64x2.of_float64x2 a in
    let bi = Int64x2.of_float64x2 b in
    let r =
      Int64x2.bitwise_or
        (Int64x2.bitwise_and mask ai)
        (Int64x2.bitwise_and (Int64x2.bitwise_not mask) bi)
    in
    Float64x2.of_int64x2 r
  in
  let[@inline] mask2 cond_arr base i =
    let m0 =
      if Array.unsafe_get cond_arr (base + i)
      then (-#1L)
      else #0L
    in
    let m1 =
      if Array.unsafe_get cond_arr (base + i + 1)
      then (-#1L)
      else #0L
    in
    Int64x2.set m0 m1
  in
  let cond_base = View.offset vcond + start_idx in
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset vtrue + start_idx in
  let b_base = View.offset vfalse + start_idx in
  if
    View.is_c_contiguous vout &&
    View.is_c_contiguous vtrue &&
    View.is_c_contiguous vfalse &&
    View.is_c_contiguous vcond
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n8 = n - 7 in
    while !i < n8 do
      let idx = !i in
      let a0 = Float64x2.Array.unsafe_get true_arr ~idx:(a_base + idx) in
      let b0 = Float64x2.Array.unsafe_get false_arr ~idx:(b_base + idx) in
      let m0 = mask2 cond_arr cond_base idx in
      let a1 = Float64x2.Array.unsafe_get true_arr ~idx:(a_base + idx + 2) in
      let b1 = Float64x2.Array.unsafe_get false_arr ~idx:(b_base + idx + 2) in
      let m1 = mask2 cond_arr cond_base (idx + 2) in
      let a2 = Float64x2.Array.unsafe_get true_arr ~idx:(a_base + idx + 4) in
      let b2 = Float64x2.Array.unsafe_get false_arr ~idx:(b_base + idx + 4) in
      let m2 = mask2 cond_arr cond_base (idx + 4) in
      let a3 = Float64x2.Array.unsafe_get true_arr ~idx:(a_base + idx + 6) in
      let b3 = Float64x2.Array.unsafe_get false_arr ~idx:(b_base + idx + 6) in
      let m3 = mask2 cond_arr cond_base (idx + 6) in
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx)
        (select_f64x2 m0 a0 b0);
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 2)
        (select_f64x2 m1 a1 b1);
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 4)
        (select_f64x2 m2 a2 b2);
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 6)
        (select_f64x2 m3 a3 b3);
      i := idx + 8
    done;
    let n2 = n - 1 in
    while !i < n2 do
      let idx = !i in
      let a = Float64x2.Array.unsafe_get true_arr ~idx:(a_base + idx) in
      let b = Float64x2.Array.unsafe_get false_arr ~idx:(b_base + idx) in
      let m = mask2 cond_arr cond_base idx in
      Float64x2.Array.unsafe_set out_arr ~idx:(out_base + idx)
        (select_f64x2 m a b);
      i := idx + 2
    done;
    while !i < n do
      let idx = !i in
      let a = Array.unsafe_get true_arr (a_base + idx) in
      let b = Array.unsafe_get false_arr (b_base + idx) in
      let c = Array.unsafe_get cond_arr (cond_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (if c then a else b);
      incr i
    done
  )
  else
    let out_shape = shape vout in
    let out_strides = View.strides vout in
    let a_shape = shape vtrue in
    let b_shape = shape vfalse in
    let c_shape = shape vcond in
    let a_strides = View.strides vtrue in
    let b_strides = View.strides vfalse in
    let c_strides = View.strides vcond in
    let a_offset = View.offset vtrue in
    let b_offset = View.offset vfalse in
    let c_offset = View.offset vcond in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    let c_idx = Array.make (Array.length c_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      Shape.broadcast_index_into md_idx c_shape c_idx;
      let c_lin = Shape.ravel_index c_idx c_strides in
      let out_lin = Shape.ravel_index md_idx out_strides in
      let a = Array.unsafe_get true_arr (a_offset + a_lin) in
      let b = Array.unsafe_get false_arr (b_offset + b_lin) in
      let c = Array.unsafe_get cond_arr (c_offset + c_lin) in
      Array.unsafe_set out_arr (out_offset + out_lin) (if c then a else b)
    done

let where_float32
    cond_arr true_arr false_arr out_arr
    vcond vtrue vfalse vout
    start_idx end_idx =
  let[@inline] select_f32x4
      (mask : Int32x4.t)
      (a : Float32x4.t)
      (b : Float32x4.t) =
    let ai = Int32x4.of_float32x4 a in
    let bi = Int32x4.of_float32x4 b in
    let r =
      Int32x4.bitwise_or
        (Int32x4.bitwise_and mask ai)
        (Int32x4.bitwise_and (Int32x4.bitwise_not mask) bi)
    in
    Float32x4.of_int32x4 r
  in
  let[@inline] mask4 cond_arr base i =
    let m0 =
      if Array.unsafe_get cond_arr (base + i)
      then (-#1l) else #0l
    in
    let m1 =
      if Array.unsafe_get cond_arr (base + i + 1)
      then (-#1l) else #0l
    in
    let m2 =
      if Array.unsafe_get cond_arr (base + i + 2)
      then (-#1l) else #0l
    in
    let m3 =
      if Array.unsafe_get cond_arr (base + i + 3)
      then (-#1l) else #0l
    in
    Int32x4.set m0 m1 m2 m3
  in
  let cond_base = View.offset vcond + start_idx in
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset vtrue + start_idx in
  let b_base = View.offset vfalse + start_idx in
  if
    View.is_c_contiguous vout &&
    View.is_c_contiguous vtrue &&
    View.is_c_contiguous vfalse &&
    View.is_c_contiguous vcond
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n16 = n - 15 in
    while !i < n16 do
      let idx = !i in
      let a0 = Float32x4.Array.unsafe_get true_arr ~idx:(a_base + idx) in
      let b0 = Float32x4.Array.unsafe_get false_arr ~idx:(b_base + idx) in
      let m0 = mask4 cond_arr cond_base idx in
      let a1 = Float32x4.Array.unsafe_get true_arr ~idx:(a_base + idx + 4) in
      let b1 = Float32x4.Array.unsafe_get false_arr ~idx:(b_base + idx + 4) in
      let m1 = mask4 cond_arr cond_base (idx + 4) in
      let a2 = Float32x4.Array.unsafe_get true_arr ~idx:(a_base + idx + 8) in
      let b2 = Float32x4.Array.unsafe_get false_arr ~idx:(b_base + idx + 8) in
      let m2 = mask4 cond_arr cond_base (idx + 8) in
      let a3 = Float32x4.Array.unsafe_get true_arr ~idx:(a_base + idx + 12) in
      let b3 = Float32x4.Array.unsafe_get false_arr ~idx:(b_base + idx + 12) in
      let m3 = mask4 cond_arr cond_base (idx + 12) in
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx)
        (select_f32x4 m0 a0 b0);
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 4)
        (select_f32x4 m1 a1 b1);
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 8)
        (select_f32x4 m2 a2 b2);
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 12)
        (select_f32x4 m3 a3 b3);
      i := idx + 16
    done;
    let n4 = n - 3 in
    while !i < n4 do
      let idx = !i in
      let a = Float32x4.Array.unsafe_get true_arr ~idx:(a_base + idx) in
      let b = Float32x4.Array.unsafe_get false_arr ~idx:(b_base + idx) in
      let m = mask4 cond_arr cond_base idx in
      Float32x4.Array.unsafe_set out_arr ~idx:(out_base + idx)
        (select_f32x4 m a b);
      i := idx + 4
    done;
    while !i < n do
      let idx = !i in
      let a = Array.unsafe_get true_arr (a_base + idx) in
      let b = Array.unsafe_get false_arr (b_base + idx) in
      let c = Array.unsafe_get cond_arr (cond_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (if c then a else b);
      incr i
    done
  )
  else
    let out_shape = shape vout in
    let out_strides = View.strides vout in
    let a_shape = shape vtrue in
    let b_shape = shape vfalse in
    let c_shape = shape vcond in
    let a_strides = View.strides vtrue in
    let b_strides = View.strides vfalse in
    let c_strides = View.strides vcond in
    let a_offset = View.offset vtrue in
    let b_offset = View.offset vfalse in
    let c_offset = View.offset vcond in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    let c_idx = Array.make (Array.length c_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      Shape.broadcast_index_into md_idx c_shape c_idx;
      let c_lin = Shape.ravel_index c_idx c_strides in
      let out_lin = Shape.ravel_index md_idx out_strides in
      let a = Array.unsafe_get true_arr (a_offset + a_lin) in
      let b = Array.unsafe_get false_arr (b_offset + b_lin) in
      let c = Array.unsafe_get cond_arr (c_offset + c_lin) in
      Array.unsafe_set out_arr (out_offset + out_lin) (if c then a else b)
    done

let where_int8
    (cond_arr : bool array)
    (true_arr : int8# array)
    (false_arr : int8# array)
    (out_arr : int8# array)
    vcond vtrue vfalse vout
    start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset vtrue + start_idx in
  let b_base = View.offset vfalse + start_idx in
  let c_base = View.offset vcond + start_idx in
  if
    View.is_c_contiguous vout && View.is_c_contiguous vtrue
    && View.is_c_contiguous vfalse && View.is_c_contiguous vcond
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get true_arr (a_base + i0) in
      let b0 = Array.unsafe_get false_arr (b_base + i0) in
      let c0 = Array.unsafe_get cond_arr (c_base + i0) in
      let a1 = Array.unsafe_get true_arr (a_base + i1) in
      let b1 = Array.unsafe_get false_arr (b_base + i1) in
      let c1 = Array.unsafe_get cond_arr (c_base + i1) in
      let a2 = Array.unsafe_get true_arr (a_base + i2) in
      let b2 = Array.unsafe_get false_arr (b_base + i2) in
      let c2 = Array.unsafe_get cond_arr (c_base + i2) in
      let a3 = Array.unsafe_get true_arr (a_base + i3) in
      let b3 = Array.unsafe_get false_arr (b_base + i3) in
      let c3 = Array.unsafe_get cond_arr (c_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (if c0 then a0 else b0);
      Array.unsafe_set out_arr (out_base + i1) (if c1 then a1 else b1);
      Array.unsafe_set out_arr (out_base + i2) (if c2 then a2 else b2);
      Array.unsafe_set out_arr (out_base + i3) (if c3 then a3 else b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get true_arr (a_base + idx) in
      let b_val = Array.unsafe_get false_arr (b_base + idx) in
      let c_val = Array.unsafe_get cond_arr (c_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (if c_val then a_val else b_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let out_strides = View.strides vout in
    let a_shape = shape vtrue in
    let b_shape = shape vfalse in
    let c_shape = shape vcond in
    let a_strides = View.strides vtrue in
    let b_strides = View.strides vfalse in
    let c_strides = View.strides vcond in
    let a_offset = View.offset vtrue in
    let b_offset = View.offset vfalse in
    let c_offset = View.offset vcond in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    let c_idx = Array.make (Array.length c_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      Shape.broadcast_index_into md_idx c_shape c_idx;
      let c_lin = Shape.ravel_index c_idx c_strides in
      let out_lin = Shape.ravel_index md_idx out_strides in
      let a_val = Array.unsafe_get true_arr (a_offset + a_lin) in
      let b_val = Array.unsafe_get false_arr (b_offset + b_lin) in
      let c_val = Array.unsafe_get cond_arr (c_offset + c_lin) in
      Array.unsafe_set out_arr (out_offset + out_lin) (if c_val then a_val else b_val)
    done

let where_int16
    (cond_arr : bool array)
    (true_arr : int16# array)
    (false_arr : int16# array)
    (out_arr : int16# array)
    vcond vtrue vfalse vout
    start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset vtrue + start_idx in
  let b_base = View.offset vfalse + start_idx in
  let c_base = View.offset vcond + start_idx in
  if
    View.is_c_contiguous vout && View.is_c_contiguous vtrue
    && View.is_c_contiguous vfalse && View.is_c_contiguous vcond
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get true_arr (a_base + i0) in
      let b0 = Array.unsafe_get false_arr (b_base + i0) in
      let c0 = Array.unsafe_get cond_arr (c_base + i0) in
      let a1 = Array.unsafe_get true_arr (a_base + i1) in
      let b1 = Array.unsafe_get false_arr (b_base + i1) in
      let c1 = Array.unsafe_get cond_arr (c_base + i1) in
      let a2 = Array.unsafe_get true_arr (a_base + i2) in
      let b2 = Array.unsafe_get false_arr (b_base + i2) in
      let c2 = Array.unsafe_get cond_arr (c_base + i2) in
      let a3 = Array.unsafe_get true_arr (a_base + i3) in
      let b3 = Array.unsafe_get false_arr (b_base + i3) in
      let c3 = Array.unsafe_get cond_arr (c_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (if c0 then a0 else b0);
      Array.unsafe_set out_arr (out_base + i1) (if c1 then a1 else b1);
      Array.unsafe_set out_arr (out_base + i2) (if c2 then a2 else b2);
      Array.unsafe_set out_arr (out_base + i3) (if c3 then a3 else b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get true_arr (a_base + idx) in
      let b_val = Array.unsafe_get false_arr (b_base + idx) in
      let c_val = Array.unsafe_get cond_arr (c_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (if c_val then a_val else b_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let out_strides = View.strides vout in
    let a_shape = shape vtrue in
    let b_shape = shape vfalse in
    let c_shape = shape vcond in
    let a_strides = View.strides vtrue in
    let b_strides = View.strides vfalse in
    let c_strides = View.strides vcond in
    let a_offset = View.offset vtrue in
    let b_offset = View.offset vfalse in
    let c_offset = View.offset vcond in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    let c_idx = Array.make (Array.length c_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      Shape.broadcast_index_into md_idx c_shape c_idx;
      let c_lin = Shape.ravel_index c_idx c_strides in
      let out_lin = Shape.ravel_index md_idx out_strides in
      let a_val = Array.unsafe_get true_arr (a_offset + a_lin) in
      let b_val = Array.unsafe_get false_arr (b_offset + b_lin) in
      let c_val = Array.unsafe_get cond_arr (c_offset + c_lin) in
      Array.unsafe_set out_arr (out_offset + out_lin) (if c_val then a_val else b_val)
    done

let where_int32
    cond_arr true_arr false_arr out_arr
    vcond vtrue vfalse vout
    start_idx end_idx =
  let[@inline] select_i32x4
      (mask : Int32x4.t)
      (a : Int32x4.t)
      (b : Int32x4.t) =
    Int32x4.bitwise_or
      (Int32x4.bitwise_and mask a)
      (Int32x4.bitwise_and (Int32x4.bitwise_not mask) b)
  in
  let[@inline] mask4 cond_arr base i =
    let m0 = if Array.unsafe_get cond_arr (base + i) then -#1l else #0l in
    let m1 = if Array.unsafe_get cond_arr (base + i + 1) then -#1l else #0l in
    let m2 = if Array.unsafe_get cond_arr (base + i + 2) then -#1l else #0l in
    let m3 = if Array.unsafe_get cond_arr (base + i + 3) then -#1l else #0l in
    Int32x4.set m0 m1 m2 m3
  in
  let cond_base = View.offset vcond + start_idx in
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset vtrue + start_idx in
  let b_base = View.offset vfalse + start_idx in
  if
    View.is_c_contiguous vout &&
    View.is_c_contiguous vtrue &&
    View.is_c_contiguous vfalse &&
    View.is_c_contiguous vcond
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n16 = n - 15 in
    while !i < n16 do
      let idx = !i in
      let a0 = Int32x4.Array.unsafe_get true_arr ~idx:(a_base + idx) in
      let b0 = Int32x4.Array.unsafe_get false_arr ~idx:(b_base + idx) in
      let m0 = mask4 cond_arr cond_base idx in
      let a1 = Int32x4.Array.unsafe_get true_arr ~idx:(a_base + idx + 4) in
      let b1 = Int32x4.Array.unsafe_get false_arr ~idx:(b_base + idx + 4) in
      let m1 = mask4 cond_arr cond_base (idx + 4) in
      let a2 = Int32x4.Array.unsafe_get true_arr ~idx:(a_base + idx + 8) in
      let b2 = Int32x4.Array.unsafe_get false_arr ~idx:(b_base + idx + 8) in
      let m2 = mask4 cond_arr cond_base (idx + 8) in
      let a3 = Int32x4.Array.unsafe_get true_arr ~idx:(a_base + idx + 12) in
      let b3 = Int32x4.Array.unsafe_get false_arr ~idx:(b_base + idx + 12) in
      let m3 = mask4 cond_arr cond_base (idx + 12) in
      Int32x4.Array.unsafe_set out_arr ~idx:(out_base + idx)
        (select_i32x4 m0 a0 b0);
      Int32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 4)
        (select_i32x4 m1 a1 b1);
      Int32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 8)
        (select_i32x4 m2 a2 b2);
      Int32x4.Array.unsafe_set out_arr ~idx:(out_base + idx + 12)
        (select_i32x4 m3 a3 b3);
      i := idx + 16
    done;
    let n4 = n - 3 in
    while !i < n4 do
      let idx = !i in
      let a = Int32x4.Array.unsafe_get true_arr ~idx:(a_base + idx) in
      let b = Int32x4.Array.unsafe_get false_arr ~idx:(b_base + idx) in
      let m = mask4 cond_arr cond_base idx in
      Int32x4.Array.unsafe_set out_arr ~idx:(out_base + idx)
        (select_i32x4 m a b);
      i := idx + 4
    done;
    while !i < n do
      let idx = !i in
      let a = Array.unsafe_get true_arr (a_base + idx) in
      let b = Array.unsafe_get false_arr (b_base + idx) in
      let c = Array.unsafe_get cond_arr (cond_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (if c then a else b);
      incr i
    done
  )
  else
    let out_shape = shape vout in
    let out_strides = View.strides vout in
    let a_shape = shape vtrue in
    let b_shape = shape vfalse in
    let c_shape = shape vcond in
    let a_strides = View.strides vtrue in
    let b_strides = View.strides vfalse in
    let c_strides = View.strides vcond in
    let a_offset = View.offset vtrue in
    let b_offset = View.offset vfalse in
    let c_offset = View.offset vcond in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    let c_idx = Array.make (Array.length c_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      Shape.broadcast_index_into md_idx c_shape c_idx;
      let c_lin = Shape.ravel_index c_idx c_strides in
      let out_lin = Shape.ravel_index md_idx out_strides in
      let a = Array.unsafe_get true_arr (a_offset + a_lin) in
      let b = Array.unsafe_get false_arr (b_offset + b_lin) in
      let c = Array.unsafe_get cond_arr (c_offset + c_lin) in
      Array.unsafe_set out_arr (out_offset + out_lin) (if c then a else b)
    done

let where_int64
    cond_arr true_arr false_arr out_arr
    vcond vtrue vfalse vout
    start_idx end_idx =
  let[@inline] select_i64x2
      (mask : Int64x2.t)
      (a : Int64x2.t)
      (b : Int64x2.t) =
    Int64x2.bitwise_or
      (Int64x2.bitwise_and mask a)
      (Int64x2.bitwise_and (Int64x2.bitwise_not mask) b)
  in
  let[@inline] mask2 cond_arr base i =
    let m0 = if Array.unsafe_get cond_arr (base + i) then -#1L else #0L in
    let m1 = if Array.unsafe_get cond_arr (base + i + 1) then -#1L else #0L in
    Int64x2.set m0 m1
  in
  let cond_base = View.offset vcond + start_idx in
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset vtrue + start_idx in
  let b_base = View.offset vfalse + start_idx in
  if
    View.is_c_contiguous vout &&
    View.is_c_contiguous vtrue &&
    View.is_c_contiguous vfalse &&
    View.is_c_contiguous vcond
  then (
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n8 = n - 7 in
    while !i < n8 do
      let idx = !i in
      let a0 = Int64x2.Array.unsafe_get true_arr ~idx:(a_base + idx) in
      let b0 = Int64x2.Array.unsafe_get false_arr ~idx:(b_base + idx) in
      let m0 = mask2 cond_arr cond_base idx in
      let a1 = Int64x2.Array.unsafe_get true_arr ~idx:(a_base + idx + 2) in
      let b1 = Int64x2.Array.unsafe_get false_arr ~idx:(b_base + idx + 2) in
      let m1 = mask2 cond_arr cond_base (idx + 2) in
      let a2 = Int64x2.Array.unsafe_get true_arr ~idx:(a_base + idx + 4) in
      let b2 = Int64x2.Array.unsafe_get false_arr ~idx:(b_base + idx + 4) in
      let m2 = mask2 cond_arr cond_base (idx + 4) in
      let a3 = Int64x2.Array.unsafe_get true_arr ~idx:(a_base + idx + 6) in
      let b3 = Int64x2.Array.unsafe_get false_arr ~idx:(b_base + idx + 6) in
      let m3 = mask2 cond_arr cond_base (idx + 6) in
      Int64x2.Array.unsafe_set out_arr ~idx:(out_base + idx)
        (select_i64x2 m0 a0 b0);
      Int64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 2)
        (select_i64x2 m1 a1 b1);
      Int64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 4)
        (select_i64x2 m2 a2 b2);
      Int64x2.Array.unsafe_set out_arr ~idx:(out_base + idx + 6)
        (select_i64x2 m3 a3 b3);
      i := idx + 8
    done;
    let n2 = n - 1 in
    while !i < n2 do
      let idx = !i in
      let a = Int64x2.Array.unsafe_get true_arr ~idx:(a_base + idx) in
      let b = Int64x2.Array.unsafe_get false_arr ~idx:(b_base + idx) in
      let m = mask2 cond_arr cond_base idx in
      Int64x2.Array.unsafe_set out_arr ~idx:(out_base + idx)
        (select_i64x2 m a b);
      i := idx + 2
    done;
    while !i < n do
      let idx = !i in
      let a = Array.unsafe_get true_arr (a_base + idx) in
      let b = Array.unsafe_get false_arr (b_base + idx) in
      let c = Array.unsafe_get cond_arr (cond_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (if c then a else b);
      incr i
    done
  )
  else
    let out_shape = shape vout in
    let out_strides = View.strides vout in
    let a_shape = shape vtrue in
    let b_shape = shape vfalse in
    let c_shape = shape vcond in
    let a_strides = View.strides vtrue in
    let b_strides = View.strides vfalse in
    let c_strides = View.strides vcond in
    let a_offset = View.offset vtrue in
    let b_offset = View.offset vfalse in
    let c_offset = View.offset vcond in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    let c_idx = Array.make (Array.length c_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      Shape.broadcast_index_into md_idx c_shape c_idx;
      let c_lin = Shape.ravel_index c_idx c_strides in
      let out_lin = Shape.ravel_index md_idx out_strides in
      let a = Array.unsafe_get true_arr (a_offset + a_lin) in
      let b = Array.unsafe_get false_arr (b_offset + b_lin) in
      let c = Array.unsafe_get cond_arr (c_offset + c_lin) in
      Array.unsafe_set out_arr (out_offset + out_lin) (if c then a else b)
    done
