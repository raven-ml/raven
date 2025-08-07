open Bigarray_ext
module Dtype = Nx_core.Dtype
module Shape = Nx_core.Shape
open Internal

let cast_f16_to_f32 (src : (float, float16_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_f16_to_f64 (src : (float, float16_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_f16_to_i8 (src : (float, float16_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f16_to_u8 (src : (float, float16_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f16_to_i16 (src : (float, float16_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f16_to_u16 (src : (float, float16_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f16_to_i32 (src : (float, float16_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_float src_val)
    done

let cast_f16_to_i64 (src : (float, float16_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_float src_val)
    done

let cast_f16_to_c32 (src : (float, float16_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = src_val; im = 0.0 }
    done

let cast_f16_to_c64 (src : (float, float16_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = src_val; im = 0.0 }
    done

let cast_f16_to_int (src : (float, float16_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f16_to_nativeint (src : (float, float16_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_float src_val)
    done

let cast_f32_to_f16 (src : (float, float32_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_f32_to_f64 (src : (float, float32_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_f32_to_i8 (src : (float, float32_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f32_to_u8 (src : (float, float32_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f32_to_i16 (src : (float, float32_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f32_to_u16 (src : (float, float32_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f32_to_i32 (src : (float, float32_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_float src_val)
    done

let cast_f32_to_i64 (src : (float, float32_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_float src_val)
    done

let cast_f32_to_c32 (src : (float, float32_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = src_val; im = 0.0 }
    done

let cast_f32_to_c64 (src : (float, float32_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = src_val; im = 0.0 }
    done

let cast_f32_to_int (src : (float, float32_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f32_to_nativeint (src : (float, float32_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_float src_val)
    done

let cast_f64_to_f16 (src : (float, float64_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_f64_to_f32 (src : (float, float64_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_f64_to_i8 (src : (float, float64_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f64_to_u8 (src : (float, float64_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f64_to_i16 (src : (float, float64_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f64_to_u16 (src : (float, float64_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f64_to_i32 (src : (float, float64_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_float src_val)
    done

let cast_f64_to_i64 (src : (float, float64_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_float src_val)
    done

let cast_f64_to_c32 (src : (float, float64_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = src_val; im = 0.0 }
    done

let cast_f64_to_c64 (src : (float, float64_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = src_val; im = 0.0 }
    done

let cast_f64_to_int (src : (float, float64_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (int_of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (int_of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (int_of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (int_of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
    done

let cast_f64_to_nativeint (src : (float, float64_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.of_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.of_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.of_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.of_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.of_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_float src_val)
    done

let cast_i8_to_f16 (src : (int, int8_signed_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_i8_to_f32 (src : (int, int8_signed_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_i8_to_f64 (src : (int, int8_signed_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_i8_to_u8 (src : (int, int8_signed_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_i8_to_i16 (src : (int, int8_signed_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_i8_to_u16 (src : (int, int8_signed_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_i8_to_i32 (src : (int, int8_signed_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_int src_val)
    done

let cast_i8_to_i64 (src : (int, int8_signed_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_int src_val)
    done

let cast_i8_to_c32 (src : (int, int8_signed_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_i8_to_c64 (src : (int, int8_signed_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_i8_to_int (src : (int, int8_signed_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_i8_to_nativeint (src : (int, int8_signed_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_int src_val)
    done

let cast_u8_to_f16 (src : (int, int8_unsigned_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_u8_to_f32 (src : (int, int8_unsigned_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_u8_to_f64 (src : (int, int8_unsigned_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_u8_to_i8 (src : (int, int8_unsigned_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_u8_to_i16 (src : (int, int8_unsigned_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_u8_to_u16 (src : (int, int8_unsigned_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_u8_to_i32 (src : (int, int8_unsigned_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_int src_val)
    done

let cast_u8_to_i64 (src : (int, int8_unsigned_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_int src_val)
    done

let cast_u8_to_c32 (src : (int, int8_unsigned_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_u8_to_c64 (src : (int, int8_unsigned_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_u8_to_int (src : (int, int8_unsigned_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_u8_to_nativeint (src : (int, int8_unsigned_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_int src_val)
    done

let cast_i16_to_f16 (src : (int, int16_signed_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_i16_to_f32 (src : (int, int16_signed_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_i16_to_f64 (src : (int, int16_signed_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_i16_to_i8 (src : (int, int16_signed_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_i16_to_u8 (src : (int, int16_signed_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_i16_to_u16 (src : (int, int16_signed_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_i16_to_i32 (src : (int, int16_signed_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_int src_val)
    done

let cast_i16_to_i64 (src : (int, int16_signed_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_int src_val)
    done

let cast_i16_to_c32 (src : (int, int16_signed_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_i16_to_c64 (src : (int, int16_signed_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_i16_to_int (src : (int, int16_signed_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_i16_to_nativeint (src : (int, int16_signed_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_int src_val)
    done

let cast_u16_to_f16 (src : (int, int16_unsigned_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_u16_to_f32 (src : (int, int16_unsigned_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_u16_to_f64 (src : (int, int16_unsigned_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_u16_to_i8 (src : (int, int16_unsigned_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_u16_to_u8 (src : (int, int16_unsigned_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_u16_to_i16 (src : (int, int16_unsigned_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_u16_to_i32 (src : (int, int16_unsigned_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_int src_val)
    done

let cast_u16_to_i64 (src : (int, int16_unsigned_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_int src_val)
    done

let cast_u16_to_c32 (src : (int, int16_unsigned_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_u16_to_c64 (src : (int, int16_unsigned_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_u16_to_int (src : (int, int16_unsigned_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_u16_to_nativeint (src : (int, int16_unsigned_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_int src_val)
    done

let cast_i32_to_f16 (src : (int32, int32_elt) t) (dst : (float, float16_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.to_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.to_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.to_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.to_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.to_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_float src_val)
    done

let cast_i32_to_f32 (src : (int32, int32_elt) t) (dst : (float, float32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.to_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.to_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.to_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.to_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.to_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_float src_val)
    done

let cast_i32_to_f64 (src : (int32, int32_elt) t) (dst : (float, float64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.to_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.to_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.to_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.to_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.to_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_float src_val)
    done

let cast_i32_to_i8 (src : (int32, int32_elt) t) (dst : (int, int8_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_int src_val)
    done

let cast_i32_to_u8 (src : (int32, int32_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_int src_val)
    done

let cast_i32_to_i16 (src : (int32, int32_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_int src_val)
    done

let cast_i32_to_u16 (src : (int32, int32_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_int src_val)
    done

let cast_i32_to_i64 (src : (int32, int32_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.of_int32 src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.of_int32 src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.of_int32 src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.of_int32 src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.of_int32 src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_int32 src_val)
    done

let cast_i32_to_c32 (src : (int32, int32_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = Int32.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = Int32.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = Int32.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = Int32.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = Int32.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = Int32.to_float src_val; im = 0.0 }
    done

let cast_i32_to_c64 (src : (int32, int32_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = Int32.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = Int32.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = Int32.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = Int32.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = Int32.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = Int32.to_float src_val; im = 0.0 }
    done

let cast_i32_to_int (src : (int32, int32_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_int src_val)
    done

let cast_i32_to_nativeint (src : (int32, int32_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.of_int32 src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.of_int32 src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.of_int32 src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.of_int32 src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.of_int32 src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_int32 src_val)
    done

let cast_i64_to_f16 (src : (int64, int64_elt) t) (dst : (float, float16_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.to_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.to_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.to_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.to_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.to_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_float src_val)
    done

let cast_i64_to_f32 (src : (int64, int64_elt) t) (dst : (float, float32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.to_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.to_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.to_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.to_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.to_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_float src_val)
    done

let cast_i64_to_f64 (src : (int64, int64_elt) t) (dst : (float, float64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.to_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.to_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.to_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.to_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.to_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_float src_val)
    done

let cast_i64_to_i8 (src : (int64, int64_elt) t) (dst : (int, int8_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_int src_val)
    done

let cast_i64_to_u8 (src : (int64, int64_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_int src_val)
    done

let cast_i64_to_i16 (src : (int64, int64_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_int src_val)
    done

let cast_i64_to_u16 (src : (int64, int64_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_int src_val)
    done

let cast_i64_to_i32 (src : (int64, int64_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.to_int32 src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.to_int32 src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.to_int32 src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.to_int32 src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.to_int32 src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_int32 src_val)
    done

let cast_i64_to_c32 (src : (int64, int64_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = Int64.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = Int64.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = Int64.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = Int64.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = Int64.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = Int64.to_float src_val; im = 0.0 }
    done

let cast_i64_to_c64 (src : (int64, int64_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = Int64.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = Int64.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = Int64.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = Int64.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = Int64.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = Int64.to_float src_val; im = 0.0 }
    done

let cast_i64_to_int (src : (int64, int64_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_int src_val)
    done

let cast_i64_to_nativeint (src : (int64, int64_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.to_nativeint src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.to_nativeint src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.to_nativeint src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.to_nativeint src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.to_nativeint src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_nativeint src_val)
    done

let cast_c32_to_f16 (src : (Complex.t, complex32_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val.Complex.re;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
    done

let cast_c32_to_f32 (src : (Complex.t, complex32_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val.Complex.re;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
    done

let cast_c32_to_f64 (src : (Complex.t, complex32_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val.Complex.re;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
    done

let cast_c32_to_i8 (src : (Complex.t, complex32_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (int_of_float src_val.Complex.re)
    done

let cast_c32_to_u8 (src : (Complex.t, complex32_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (int_of_float src_val.Complex.re)
    done

let cast_c32_to_i16 (src : (Complex.t, complex32_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (int_of_float src_val.Complex.re)
    done

let cast_c32_to_u16 (src : (Complex.t, complex32_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (int_of_float src_val.Complex.re)
    done

let cast_c32_to_i32 (src : (Complex.t, complex32_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (Int32.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (Int32.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (Int32.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (Int32.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (Int32.of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (Int32.of_float src_val.Complex.re)
    done

let cast_c32_to_i64 (src : (Complex.t, complex32_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (Int64.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (Int64.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (Int64.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (Int64.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (Int64.of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (Int64.of_float src_val.Complex.re)
    done

let cast_c32_to_c64 (src : (Complex.t, complex32_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_c32_to_int (src : (Complex.t, complex32_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (int_of_float src_val.Complex.re)
    done

let cast_c32_to_nativeint (src : (Complex.t, complex32_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (Nativeint.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (Nativeint.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (Nativeint.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (Nativeint.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (Nativeint.of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (Nativeint.of_float src_val.Complex.re)
    done

let cast_c64_to_f16 (src : (Complex.t, complex64_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val.Complex.re;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
    done

let cast_c64_to_f32 (src : (Complex.t, complex64_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val.Complex.re;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
    done

let cast_c64_to_f64 (src : (Complex.t, complex64_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2.Complex.re;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val.Complex.re;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
    done

let cast_c64_to_i8 (src : (Complex.t, complex64_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (int_of_float src_val.Complex.re)
    done

let cast_c64_to_u8 (src : (Complex.t, complex64_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (int_of_float src_val.Complex.re)
    done

let cast_c64_to_i16 (src : (Complex.t, complex64_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (int_of_float src_val.Complex.re)
    done

let cast_c64_to_u16 (src : (Complex.t, complex64_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (int_of_float src_val.Complex.re)
    done

let cast_c64_to_i32 (src : (Complex.t, complex64_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (Int32.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (Int32.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (Int32.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (Int32.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (Int32.of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (Int32.of_float src_val.Complex.re)
    done

let cast_c64_to_i64 (src : (Complex.t, complex64_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (Int64.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (Int64.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (Int64.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (Int64.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (Int64.of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (Int64.of_float src_val.Complex.re)
    done

let cast_c64_to_c32 (src : (Complex.t, complex64_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_c64_to_int (src : (Complex.t, complex64_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (int_of_float src_val.Complex.re)
    done

let cast_c64_to_nativeint (src : (Complex.t, complex64_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        (Nativeint.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i1)
        (Nativeint.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i2)
        (Nativeint.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf (dst_base + i3)
        (Nativeint.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        (Nativeint.of_float src_val.Complex.re);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        (Nativeint.of_float src_val.Complex.re)
    done

let cast_int_to_f16 (src : (int, int_elt) t) (dst : (float, float16_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_int_to_f32 (src : (int, int_elt) t) (dst : (float, float32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_int_to_f64 (src : (int, int_elt) t) (dst : (float, float64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (float_of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (float_of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (float_of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (float_of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
    done

let cast_int_to_i8 (src : (int, int_elt) t) (dst : (int, int8_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_int_to_u8 (src : (int, int_elt) t) (dst : (int, int8_unsigned_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_int_to_i16 (src : (int, int_elt) t) (dst : (int, int16_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_int_to_u16 (src : (int, int_elt) t) (dst : (int, int16_unsigned_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) src_val0;
      Array1.unsafe_set dst_buf (dst_base + i1) src_val1;
      Array1.unsafe_set dst_buf (dst_base + i2) src_val2;
      Array1.unsafe_set dst_buf (dst_base + i3) src_val3;
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) src_val;
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) src_val
    done

let cast_int_to_i32 (src : (int, int_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int32.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int32.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int32.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int32.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int32.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_int src_val)
    done

let cast_int_to_i64 (src : (int, int_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_int src_val)
    done

let cast_int_to_c32 (src : (int, int_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_int_to_c64 (src : (int, int_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_int_to_nativeint (src : (int, int_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.of_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.of_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.of_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.of_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.of_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_int src_val)
    done

let cast_nativeint_to_f16 (src : (nativeint, nativeint_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.to_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.to_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.to_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.to_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.to_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_float src_val)
    done

let cast_nativeint_to_f32 (src : (nativeint, nativeint_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.to_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.to_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.to_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.to_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.to_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_float src_val)
    done

let cast_nativeint_to_f64 (src : (nativeint, nativeint_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.to_float src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.to_float src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.to_float src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.to_float src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.to_float src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_float src_val)
    done

let cast_nativeint_to_i8 (src : (nativeint, nativeint_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_int src_val)
    done

let cast_nativeint_to_u8 (src : (nativeint, nativeint_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_int src_val)
    done

let cast_nativeint_to_i16 (src : (nativeint, nativeint_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_int src_val)
    done

let cast_nativeint_to_u16 (src : (nativeint, nativeint_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_int src_val)
    done

let cast_nativeint_to_i32 (src : (nativeint, nativeint_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.to_int32 src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.to_int32 src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.to_int32 src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.to_int32 src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.to_int32 src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_int32 src_val)
    done

let cast_nativeint_to_i64 (src : (nativeint, nativeint_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Int64.of_nativeint src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Int64.of_nativeint src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Int64.of_nativeint src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Int64.of_nativeint src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Int64.of_nativeint src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_nativeint src_val)
    done

let cast_nativeint_to_c32 (src : (nativeint, nativeint_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = Nativeint.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = Nativeint.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = Nativeint.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = Nativeint.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = Nativeint.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = Nativeint.to_float src_val; im = 0.0 }
    done

let cast_nativeint_to_c64 (src : (nativeint, nativeint_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0)
        { Complex.re = Nativeint.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i1)
        { Complex.re = Nativeint.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i2)
        { Complex.re = Nativeint.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf (dst_base + i3)
        { Complex.re = Nativeint.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx)
        { Complex.re = Nativeint.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf
        (offset dst + k)
        { Complex.re = Nativeint.to_float src_val; im = 0.0 }
    done

let cast_nativeint_to_int (src : (nativeint, nativeint_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_c_contiguous src then (
    let src_base = offset src + start_idx in
    let dst_base = offset dst + start_idx in
    let n = end_idx - start_idx in
    let i = ref 0 in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (src_base + i0) in
      let src_val1 = Array1.unsafe_get src_buf (src_base + i1) in
      let src_val2 = Array1.unsafe_get src_buf (src_base + i2) in
      let src_val3 = Array1.unsafe_get src_buf (src_base + i3) in
      Array1.unsafe_set dst_buf (dst_base + i0) (Nativeint.to_int src_val0);
      Array1.unsafe_set dst_buf (dst_base + i1) (Nativeint.to_int src_val1);
      Array1.unsafe_set dst_buf (dst_base + i2) (Nativeint.to_int src_val2);
      Array1.unsafe_set dst_buf (dst_base + i3) (Nativeint.to_int src_val3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (src_base + idx) in
      Array1.unsafe_set dst_buf (dst_base + idx) (Nativeint.to_int src_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape dst)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape dst) md_index;
      let src_lin = Shape.ravel_index md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_int src_val)
    done

(* ===== Extended Type Cast Functions ===== *)

(* Float16 to Extended Types *)
let cast_f16_to_bfloat16 (src : (float, float16_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  (* Direct copy - both are 16-bit floats *)
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_f16_to_bool (src : (float, float16_elt) t) (dst : (bool, bool_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0.0)
  done

let cast_f16_to_int4 (src : (float, float16_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_f16_to_uint4 (src : (float, float16_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_f16_to_float8_e4m3 (src : (float, float16_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_f16_to_float8_e5m2 (src : (float, float16_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_f16_to_complex16 (src : (float, float16_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = src_val; im = 0.0 }
  done

let cast_f16_to_qint8 (src : (float, float16_elt) t) (dst : (int, qint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_f16_to_quint8 (src : (float, float16_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* Float32 to Extended Types *)
let cast_f32_to_bfloat16 (src : (float, float32_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_f32_to_bool (src : (float, float32_elt) t) (dst : (bool, bool_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0.0)
  done

let cast_f32_to_int4 (src : (float, float32_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_f32_to_uint4 (src : (float, float32_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_f32_to_float8_e4m3 (src : (float, float32_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_f32_to_float8_e5m2 (src : (float, float32_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_f32_to_complex16 (src : (float, float32_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = src_val; im = 0.0 }
  done

let cast_f32_to_qint8 (src : (float, float32_elt) t) (dst : (int, qint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_f32_to_quint8 (src : (float, float32_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* BFloat16 as source - to all standard types *)
let cast_bfloat16_to_f16 (src : (float, bfloat16_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_bfloat16_to_f32 (src : (float, bfloat16_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_bfloat16_to_f64 (src : (float, bfloat16_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_bfloat16_to_i8 (src : (float, bfloat16_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_bfloat16_to_u8 (src : (float, bfloat16_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_bfloat16_to_i16 (src : (float, bfloat16_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-32768) (min 32767 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_bfloat16_to_u16 (src : (float, bfloat16_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 65535 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_bfloat16_to_i32 (src : (float, bfloat16_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_float src_val)
  done

let cast_bfloat16_to_i64 (src : (float, bfloat16_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_float src_val)
  done

let cast_bfloat16_to_c32 (src : (float, bfloat16_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = src_val; im = 0.0 }
  done

let cast_bfloat16_to_c64 (src : (float, bfloat16_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = src_val; im = 0.0 }
  done

let cast_bfloat16_to_int (src : (float, bfloat16_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
  done

let cast_bfloat16_to_nativeint (src : (float, bfloat16_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_float src_val)
  done

(* BFloat16 to extended types *)
let cast_bfloat16_to_bool (src : (float, bfloat16_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0.0)
  done

let cast_bfloat16_to_int4 (src : (float, bfloat16_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_bfloat16_to_uint4 (src : (float, bfloat16_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_bfloat16_to_float8_e4m3 (src : (float, bfloat16_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_bfloat16_to_float8_e5m2 (src : (float, bfloat16_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_bfloat16_to_complex16 (src : (float, bfloat16_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = src_val; im = 0.0 }
  done

let cast_bfloat16_to_qint8 (src : (float, bfloat16_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_bfloat16_to_quint8 (src : (float, bfloat16_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* Float64 to Extended Types *)
let cast_f64_to_bfloat16 (src : (float, float64_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_f64_to_bool (src : (float, float64_elt) t) (dst : (bool, bool_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0.0)
  done

let cast_f64_to_int4 (src : (float, float64_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_f64_to_uint4 (src : (float, float64_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_f64_to_float8_e4m3 (src : (float, float64_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_f64_to_float8_e5m2 (src : (float, float64_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_f64_to_complex16 (src : (float, float64_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = src_val; im = 0.0 }
  done

let cast_f64_to_qint8 (src : (float, float64_elt) t) (dst : (int, qint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_f64_to_quint8 (src : (float, float64_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* Int8 to Extended Types *)
let cast_i8_to_bfloat16 (src : (int, int8_signed_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_i8_to_bool (src : (int, int8_signed_elt) t) (dst : (bool, bool_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0)
  done

let cast_i8_to_int4 (src : (int, int8_signed_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max (-8) (min 7 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i8_to_uint4 (src : (int, int8_signed_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 15 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i8_to_float8_e4m3 (src : (int, int8_signed_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_i8_to_float8_e5m2 (src : (int, int8_signed_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_i8_to_complex16 (src : (int, int8_signed_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = float_of_int src_val; im = 0.0 }
  done

let cast_i8_to_qint8 (src : (int, int8_signed_elt) t) (dst : (int, qint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_i8_to_quint8 (src : (int, int8_signed_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 255 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* UInt8 to Extended Types *)
let cast_u8_to_bfloat16 (src : (int, int8_unsigned_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_u8_to_bool (src : (int, int8_unsigned_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0)
  done

let cast_u8_to_int4 (src : (int, int8_unsigned_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max (-8) (min 7 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_u8_to_uint4 (src : (int, int8_unsigned_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 15 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_u8_to_float8_e4m3 (src : (int, int8_unsigned_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_u8_to_float8_e5m2 (src : (int, int8_unsigned_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_u8_to_complex16 (src : (int, int8_unsigned_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = float_of_int src_val; im = 0.0 }
  done

let cast_u8_to_qint8 (src : (int, int8_unsigned_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 127 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_u8_to_quint8 (src : (int, int8_unsigned_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

(* Int16, UInt16, Int32, Int64 to Extended Types - simplified implementations *)
let cast_i16_to_bfloat16 (src : (int, int16_signed_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_u16_to_bfloat16 (src : (int, int16_unsigned_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_i32_to_bfloat16 (src : (int32, int32_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_float src_val)
  done

let cast_i64_to_bfloat16 (src : (int64, int64_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_float src_val)
  done

let cast_i16_to_bool (src : (int, int16_signed_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0)
  done

let cast_u16_to_bool (src : (int, int16_unsigned_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0)
  done

let cast_i32_to_bool (src : (int32, int32_elt) t) (dst : (bool, bool_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0l)
  done

let cast_i64_to_bool (src : (int64, int64_elt) t) (dst : (bool, bool_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0L)
  done

let cast_i16_to_int4 (src : (int, int16_signed_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max (-8) (min 7 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_u16_to_int4 (src : (int, int16_unsigned_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max (-8) (min 7 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i32_to_int4 (src : (int32, int32_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Int32.to_int src_val in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i64_to_int4 (src : (int64, int64_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Int64.to_int src_val in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i16_to_uint4 (src : (int, int16_signed_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 15 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_u16_to_uint4 (src : (int, int16_unsigned_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 15 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i32_to_uint4 (src : (int32, int32_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Int32.to_int src_val in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i64_to_uint4 (src : (int64, int64_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Int64.to_int src_val in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i16_to_float8_e4m3 (src : (int, int16_signed_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_u16_to_float8_e4m3 (src : (int, int16_unsigned_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_i32_to_float8_e4m3 (src : (int32, int32_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_float src_val)
  done

let cast_i64_to_float8_e4m3 (src : (int64, int64_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_float src_val)
  done

let cast_i16_to_float8_e5m2 (src : (int, int16_signed_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_u16_to_float8_e5m2 (src : (int, int16_unsigned_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_i32_to_float8_e5m2 (src : (int32, int32_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int32.to_float src_val)
  done

let cast_i64_to_float8_e5m2 (src : (int64, int64_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int64.to_float src_val)
  done

let cast_i16_to_complex16 (src : (int, int16_signed_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = float_of_int src_val; im = 0.0 }
  done

let cast_u16_to_complex16 (src : (int, int16_unsigned_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = float_of_int src_val; im = 0.0 }
  done

let cast_i32_to_complex16 (src : (int32, int32_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = Int32.to_float src_val; im = 0.0 }
  done

let cast_i64_to_complex16 (src : (int64, int64_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = Int64.to_float src_val; im = 0.0 }
  done

let cast_i16_to_qint8 (src : (int, int16_signed_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max (-128) (min 127 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_u16_to_qint8 (src : (int, int16_unsigned_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 127 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i32_to_qint8 (src : (int32, int32_elt) t) (dst : (int, qint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Int32.to_int src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i64_to_qint8 (src : (int64, int64_elt) t) (dst : (int, qint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Int64.to_int src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i16_to_quint8 (src : (int, int16_signed_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 255 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_u16_to_quint8 (src : (int, int16_unsigned_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 255 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i32_to_quint8 (src : (int32, int32_elt) t) (dst : (int, quint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Int32.to_int src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_i64_to_quint8 (src : (int64, int64_elt) t) (dst : (int, quint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Int64.to_int src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* Complex32 and Complex64 to Extended Types *)
let cast_c32_to_bfloat16 (src : (Complex.t, complex32_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_c64_to_bfloat16 (src : (Complex.t, complex64_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_c32_to_bool (src : (Complex.t, complex32_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      (src_val.Complex.re <> 0.0 || src_val.Complex.im <> 0.0)
  done

let cast_c64_to_bool (src : (Complex.t, complex64_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      (src_val.Complex.re <> 0.0 || src_val.Complex.im <> 0.0)
  done

let cast_c32_to_int4 (src : (Complex.t, complex32_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_c64_to_int4 (src : (Complex.t, complex64_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_c32_to_uint4 (src : (Complex.t, complex32_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_c64_to_uint4 (src : (Complex.t, complex64_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_c32_to_float8_e4m3 (src : (Complex.t, complex32_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_c64_to_float8_e4m3 (src : (Complex.t, complex64_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_c32_to_float8_e5m2 (src : (Complex.t, complex32_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_c64_to_float8_e5m2 (src : (Complex.t, complex64_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_c32_to_complex16 (src : (Complex.t, complex32_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_c64_to_complex16 (src : (Complex.t, complex64_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_c32_to_qint8 (src : (Complex.t, complex32_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_c64_to_qint8 (src : (Complex.t, complex64_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_c32_to_quint8 (src : (Complex.t, complex32_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_c64_to_quint8 (src : (Complex.t, complex64_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* Int and NativeInt to Extended Types *)
let cast_int_to_bfloat16 (src : (int, int_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_nativeint_to_bfloat16 (src : (nativeint, nativeint_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_float src_val)
  done

let cast_int_to_bool (src : (int, int_elt) t) (dst : (bool, bool_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0)
  done

let cast_nativeint_to_bool (src : (nativeint, nativeint_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0n)
  done

let cast_int_to_int4 (src : (int, int_elt) t) (dst : (int, int4_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max (-8) (min 7 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_nativeint_to_int4 (src : (nativeint, nativeint_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Nativeint.to_int src_val in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_int_to_uint4 (src : (int, int_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 15 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_nativeint_to_uint4 (src : (nativeint, nativeint_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Nativeint.to_int src_val in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_int_to_float8_e4m3 (src : (int, int_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_nativeint_to_float8_e4m3 (src : (nativeint, nativeint_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_float src_val)
  done

let cast_int_to_float8_e5m2 (src : (int, int_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_nativeint_to_float8_e5m2 (src : (nativeint, nativeint_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.to_float src_val)
  done

let cast_int_to_complex16 (src : (int, int_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = float_of_int src_val; im = 0.0 }
  done

let cast_nativeint_to_complex16 (src : (nativeint, nativeint_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = Nativeint.to_float src_val; im = 0.0 }
  done

let cast_int_to_qint8 (src : (int, int_elt) t) (dst : (int, qint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max (-128) (min 127 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_nativeint_to_qint8 (src : (nativeint, nativeint_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Nativeint.to_int src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_int_to_quint8 (src : (int, int_elt) t) (dst : (int, quint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 255 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_nativeint_to_quint8 (src : (nativeint, nativeint_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = Nativeint.to_int src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* Bool as source *)
let cast_bool_to_f16 (src : (bool, bool_elt) t) (dst : (float, float16_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1.0 else 0.0)
  done

let cast_bool_to_f32 (src : (bool, bool_elt) t) (dst : (float, float32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1.0 else 0.0)
  done

let cast_bool_to_f64 (src : (bool, bool_elt) t) (dst : (float, float64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1.0 else 0.0)
  done

let cast_bool_to_i8 (src : (bool, bool_elt) t) (dst : (int, int8_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1 else 0)
  done

let cast_bool_to_u8 (src : (bool, bool_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1 else 0)
  done

let cast_bool_to_i16 (src : (bool, bool_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1 else 0)
  done

let cast_bool_to_u16 (src : (bool, bool_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1 else 0)
  done

let cast_bool_to_i32 (src : (bool, bool_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1l else 0l)
  done

let cast_bool_to_i64 (src : (bool, bool_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1L else 0L)
  done

let cast_bool_to_c32 (src : (bool, bool_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = (if src_val then 1.0 else 0.0); im = 0.0 }
  done

let cast_bool_to_c64 (src : (bool, bool_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = (if src_val then 1.0 else 0.0); im = 0.0 }
  done

let cast_bool_to_int (src : (bool, bool_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1 else 0)
  done

let cast_bool_to_nativeint (src : (bool, bool_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1n else 0n)
  done

let cast_bool_to_bfloat16 (src : (bool, bool_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1.0 else 0.0)
  done

let cast_bool_to_int4 (src : (bool, bool_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1 else 0)
  done

let cast_bool_to_uint4 (src : (bool, bool_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1 else 0)
  done

let cast_bool_to_float8_e4m3 (src : (bool, bool_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1.0 else 0.0)
  done

let cast_bool_to_float8_e5m2 (src : (bool, bool_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1.0 else 0.0)
  done

let cast_bool_to_complex16 (src : (bool, bool_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      { Complex.re = (if src_val then 1.0 else 0.0); im = 0.0 }
  done

let cast_bool_to_qint8 (src : (bool, bool_elt) t) (dst : (int, qint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1 else 0)
  done

let cast_bool_to_quint8 (src : (bool, bool_elt) t) (dst : (int, quint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (if src_val then 1 else 0)
  done

(* Int4 as source *)
let cast_int4_to_f16 (src : (int, int4_signed_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_int4_to_f32 (src : (int, int4_signed_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_int4_to_f64 (src : (int, int4_signed_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_int4_to_i8 (src : (int, int4_signed_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_int4_to_u8 (src : (int, int4_signed_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 255 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_int4_to_i16 (src : (int, int4_signed_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_int4_to_u16 (src : (int, int4_signed_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 65535 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_int4_to_i32 (src : (int, int4_signed_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_int src_val)
  done

let cast_int4_to_i64 (src : (int, int4_signed_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_int src_val)
  done

let cast_int4_to_c32 (src : (int, int4_signed_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_int4_to_c64 (src : (int, int4_signed_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_int4_to_int (src : (int, int4_signed_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_int4_to_nativeint (src : (int, int4_signed_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_int src_val)
  done

let cast_int4_to_bfloat16 (src : (int, int4_signed_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_int4_to_bool (src : (int, int4_signed_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0)
  done

let cast_int4_to_uint4 (src : (int, int4_signed_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 15 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_int4_to_float8_e4m3 (src : (int, int4_signed_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_int4_to_float8_e5m2 (src : (int, int4_signed_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_int4_to_complex16 (src : (int, int4_signed_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_int4_to_qint8 (src : (int, int4_signed_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_int4_to_quint8 (src : (int, int4_signed_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 255 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* UInt4 as source *)
let cast_uint4_to_f16 (src : (int, int4_unsigned_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_uint4_to_f32 (src : (int, int4_unsigned_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_uint4_to_f64 (src : (int, int4_unsigned_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_uint4_to_i8 (src : (int, int4_unsigned_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_uint4_to_u8 (src : (int, int4_unsigned_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_uint4_to_i16 (src : (int, int4_unsigned_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_uint4_to_u16 (src : (int, int4_unsigned_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_uint4_to_i32 (src : (int, int4_unsigned_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_int src_val)
  done

let cast_uint4_to_i64 (src : (int, int4_unsigned_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_int src_val)
  done

let cast_uint4_to_c32 (src : (int, int4_unsigned_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_uint4_to_c64 (src : (int, int4_unsigned_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_uint4_to_int (src : (int, int4_unsigned_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_uint4_to_nativeint (src : (int, int4_unsigned_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_int src_val)
  done

let cast_uint4_to_bfloat16 (src : (int, int4_unsigned_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_uint4_to_bool (src : (int, int4_unsigned_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0)
  done

let cast_uint4_to_int4 (src : (int, int4_unsigned_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 7 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_uint4_to_float8_e4m3 (src : (int, int4_unsigned_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_uint4_to_float8_e5m2 (src : (int, int4_unsigned_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_uint4_to_complex16 (src : (int, int4_unsigned_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_uint4_to_qint8 (src : (int, int4_unsigned_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 127 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_uint4_to_quint8 (src : (int, int4_unsigned_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

(* Float8_e4m3 as source *)
let cast_float8_e4m3_to_f16 (src : (float, float8_e4m3_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_float8_e4m3_to_f32 (src : (float, float8_e4m3_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_float8_e4m3_to_f64 (src : (float, float8_e4m3_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_float8_e4m3_to_i8 (src : (float, float8_e4m3_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e4m3_to_u8 (src : (float, float8_e4m3_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e4m3_to_i16 (src : (float, float8_e4m3_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-32768) (min 32767 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e4m3_to_u16 (src : (float, float8_e4m3_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 65535 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e4m3_to_i32 (src : (float, float8_e4m3_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_float src_val)
  done

let cast_float8_e4m3_to_i64 (src : (float, float8_e4m3_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_float src_val)
  done

let cast_float8_e4m3_to_c32 (src : (float, float8_e4m3_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = src_val; im = 0.0 }
  done

let cast_float8_e4m3_to_c64 (src : (float, float8_e4m3_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = src_val; im = 0.0 }
  done

let cast_float8_e4m3_to_int (src : (float, float8_e4m3_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
  done

let cast_float8_e4m3_to_nativeint (src : (float, float8_e4m3_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_float src_val)
  done

let cast_float8_e4m3_to_bfloat16 (src : (float, float8_e4m3_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_float8_e4m3_to_bool (src : (float, float8_e4m3_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0.0)
  done

let cast_float8_e4m3_to_int4 (src : (float, float8_e4m3_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e4m3_to_uint4 (src : (float, float8_e4m3_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e4m3_to_float8_e5m2 (src : (float, float8_e4m3_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_float8_e4m3_to_complex16 (src : (float, float8_e4m3_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = src_val; im = 0.0 }
  done

let cast_float8_e4m3_to_qint8 (src : (float, float8_e4m3_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e4m3_to_quint8 (src : (float, float8_e4m3_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* Float8_e5m2 as source *)
let cast_float8_e5m2_to_f16 (src : (float, float8_e5m2_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_float8_e5m2_to_f32 (src : (float, float8_e5m2_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_float8_e5m2_to_f64 (src : (float, float8_e5m2_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_float8_e5m2_to_i8 (src : (float, float8_e5m2_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e5m2_to_u8 (src : (float, float8_e5m2_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e5m2_to_i16 (src : (float, float8_e5m2_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-32768) (min 32767 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e5m2_to_u16 (src : (float, float8_e5m2_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 65535 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e5m2_to_i32 (src : (float, float8_e5m2_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_float src_val)
  done

let cast_float8_e5m2_to_i64 (src : (float, float8_e5m2_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_float src_val)
  done

let cast_float8_e5m2_to_c32 (src : (float, float8_e5m2_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = src_val; im = 0.0 }
  done

let cast_float8_e5m2_to_c64 (src : (float, float8_e5m2_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = src_val; im = 0.0 }
  done

let cast_float8_e5m2_to_int (src : (float, float8_e5m2_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val)
  done

let cast_float8_e5m2_to_nativeint (src : (float, float8_e5m2_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_float src_val)
  done

let cast_float8_e5m2_to_bfloat16 (src : (float, float8_e5m2_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_float8_e5m2_to_bool (src : (float, float8_e5m2_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0.0)
  done

let cast_float8_e5m2_to_int4 (src : (float, float8_e5m2_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e5m2_to_uint4 (src : (float, float8_e5m2_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e5m2_to_float8_e4m3 (src : (float, float8_e5m2_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_float8_e5m2_to_complex16 (src : (float, float8_e5m2_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = src_val; im = 0.0 }
  done

let cast_float8_e5m2_to_qint8 (src : (float, float8_e5m2_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_float8_e5m2_to_quint8 (src : (float, float8_e5m2_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* Complex16 as source *)
let cast_complex16_to_f16 (src : (Complex.t, complex16_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_complex16_to_f32 (src : (Complex.t, complex16_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_complex16_to_f64 (src : (Complex.t, complex16_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_complex16_to_i8 (src : (Complex.t, complex16_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_complex16_to_u8 (src : (Complex.t, complex16_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_complex16_to_i16 (src : (Complex.t, complex16_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max (-32768) (min 32767 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_complex16_to_u16 (src : (Complex.t, complex16_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max 0 (min 65535 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_complex16_to_i32 (src : (Complex.t, complex16_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      (Int32.of_float src_val.Complex.re)
  done

let cast_complex16_to_i64 (src : (Complex.t, complex16_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      (Int64.of_float src_val.Complex.re)
  done

let cast_complex16_to_c32 (src : (Complex.t, complex16_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_complex16_to_c64 (src : (Complex.t, complex16_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_complex16_to_int (src : (Complex.t, complex16_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (int_of_float src_val.Complex.re)
  done

let cast_complex16_to_nativeint (src : (Complex.t, complex16_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      (Nativeint.of_float src_val.Complex.re)
  done

let cast_complex16_to_bfloat16 (src : (Complex.t, complex16_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_complex16_to_bool (src : (Complex.t, complex16_elt) t)
    (dst : (bool, bool_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      (src_val.Complex.re <> 0.0 || src_val.Complex.im <> 0.0)
  done

let cast_complex16_to_int4 (src : (Complex.t, complex16_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max (-8) (min 7 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_complex16_to_uint4 (src : (Complex.t, complex16_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max 0 (min 15 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_complex16_to_float8_e4m3 (src : (Complex.t, complex16_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_complex16_to_float8_e5m2 (src : (Complex.t, complex16_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val.Complex.re
  done

let cast_complex16_to_qint8 (src : (Complex.t, complex16_elt) t)
    (dst : (int, qint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max (-128) (min 127 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_complex16_to_quint8 (src : (Complex.t, complex16_elt) t)
    (dst : (int, quint8_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let int_val = int_of_float src_val.Complex.re in
    let clamped = max 0 (min 255 int_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* QInt8 as source *)
let cast_qint8_to_f16 (src : (int, qint8_elt) t) (dst : (float, float16_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_qint8_to_f32 (src : (int, qint8_elt) t) (dst : (float, float32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_qint8_to_f64 (src : (int, qint8_elt) t) (dst : (float, float64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_qint8_to_i8 (src : (int, qint8_elt) t) (dst : (int, int8_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_qint8_to_u8 (src : (int, qint8_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 255 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_qint8_to_i16 (src : (int, qint8_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_qint8_to_u16 (src : (int, qint8_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 65535 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_qint8_to_i32 (src : (int, qint8_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_int src_val)
  done

let cast_qint8_to_i64 (src : (int, qint8_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_int src_val)
  done

let cast_qint8_to_c32 (src : (int, qint8_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_qint8_to_c64 (src : (int, qint8_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_qint8_to_int (src : (int, qint8_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_qint8_to_nativeint (src : (int, qint8_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_int src_val)
  done

let cast_qint8_to_bfloat16 (src : (int, qint8_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_qint8_to_bool (src : (int, qint8_elt) t) (dst : (bool, bool_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0)
  done

let cast_qint8_to_int4 (src : (int, qint8_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max (-8) (min 7 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_qint8_to_uint4 (src : (int, qint8_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 15 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_qint8_to_float8_e4m3 (src : (int, qint8_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_qint8_to_float8_e5m2 (src : (int, qint8_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_qint8_to_complex16 (src : (int, qint8_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_qint8_to_quint8 (src : (int, qint8_elt) t) (dst : (int, quint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = max 0 (min 255 src_val) in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

(* QUInt8 as source *)
let cast_quint8_to_f16 (src : (int, quint8_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_quint8_to_f32 (src : (int, quint8_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_quint8_to_f64 (src : (int, quint8_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_quint8_to_i8 (src : (int, quint8_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 127 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_quint8_to_u8 (src : (int, quint8_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_quint8_to_i16 (src : (int, quint8_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_quint8_to_u16 (src : (int, quint8_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_quint8_to_i32 (src : (int, quint8_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int32.of_int src_val)
  done

let cast_quint8_to_i64 (src : (int, quint8_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Int64.of_int src_val)
  done

let cast_quint8_to_c32 (src : (int, quint8_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_quint8_to_c64 (src : (int, quint8_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_quint8_to_int (src : (int, quint8_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) src_val
  done

let cast_quint8_to_nativeint (src : (int, quint8_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (Nativeint.of_int src_val)
  done

let cast_quint8_to_bfloat16 (src : (int, quint8_elt) t)
    (dst : (float, bfloat16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_quint8_to_bool (src : (int, quint8_elt) t) (dst : (bool, bool_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (src_val <> 0)
  done

let cast_quint8_to_int4 (src : (int, quint8_elt) t)
    (dst : (int, int4_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 7 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_quint8_to_uint4 (src : (int, quint8_elt) t)
    (dst : (int, int4_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 15 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_quint8_to_float8_e4m3 (src : (int, quint8_elt) t)
    (dst : (float, float8_e4m3_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_quint8_to_float8_e5m2 (src : (int, quint8_elt) t)
    (dst : (float, float8_e5m2_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf (offset dst + k) (float_of_int src_val)
  done

let cast_quint8_to_complex16 (src : (int, quint8_elt) t)
    (dst : (Complex.t, complex16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    Array1.unsafe_set dst_buf
      (offset dst + k)
      Complex.{ re = float_of_int src_val; im = 0.0 }
  done

let cast_quint8_to_qint8 (src : (int, quint8_elt) t) (dst : (int, qint8_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  for k = start_idx to end_idx - 1 do
    let src_val = Array1.unsafe_get src_buf (offset src + k) in
    let clamped = min 127 src_val in
    Array1.unsafe_set dst_buf (offset dst + k) clamped
  done

let cast_kernel (type a b c d) (src : (a, b) t) (dst : (c, d) t) start_idx
    end_idx =
  match (dtype src, dtype dst) with
  (* Float16 Source *)
  | Float16, Float16 -> ()
  | Float16, Float32 -> cast_f16_to_f32 src dst start_idx end_idx
  | Float16, Float64 -> cast_f16_to_f64 src dst start_idx end_idx
  | Float16, Int8 -> cast_f16_to_i8 src dst start_idx end_idx
  | Float16, UInt8 -> cast_f16_to_u8 src dst start_idx end_idx
  | Float16, Int16 -> cast_f16_to_i16 src dst start_idx end_idx
  | Float16, UInt16 -> cast_f16_to_u16 src dst start_idx end_idx
  | Float16, Int32 -> cast_f16_to_i32 src dst start_idx end_idx
  | Float16, Int64 -> cast_f16_to_i64 src dst start_idx end_idx
  | Float16, Complex32 -> cast_f16_to_c32 src dst start_idx end_idx
  | Float16, Complex64 -> cast_f16_to_c64 src dst start_idx end_idx
  | Float16, Int -> cast_f16_to_int src dst start_idx end_idx
  | Float16, NativeInt -> cast_f16_to_nativeint src dst start_idx end_idx
  | Float16, BFloat16 -> cast_f16_to_bfloat16 src dst start_idx end_idx
  | Float16, Bool -> cast_f16_to_bool src dst start_idx end_idx
  | Float16, Int4 -> cast_f16_to_int4 src dst start_idx end_idx
  | Float16, UInt4 -> cast_f16_to_uint4 src dst start_idx end_idx
  | Float16, Float8_e4m3 -> cast_f16_to_float8_e4m3 src dst start_idx end_idx
  | Float16, Float8_e5m2 -> cast_f16_to_float8_e5m2 src dst start_idx end_idx
  | Float16, Complex16 -> cast_f16_to_complex16 src dst start_idx end_idx
  | Float16, QInt8 -> cast_f16_to_qint8 src dst start_idx end_idx
  | Float16, QUInt8 -> cast_f16_to_quint8 src dst start_idx end_idx
  (* Float32 Source *)
  | Float32, Float16 -> cast_f32_to_f16 src dst start_idx end_idx
  | Float32, Float32 -> ()
  | Float32, Float64 -> cast_f32_to_f64 src dst start_idx end_idx
  | Float32, Int8 -> cast_f32_to_i8 src dst start_idx end_idx
  | Float32, UInt8 -> cast_f32_to_u8 src dst start_idx end_idx
  | Float32, Int16 -> cast_f32_to_i16 src dst start_idx end_idx
  | Float32, UInt16 -> cast_f32_to_u16 src dst start_idx end_idx
  | Float32, Int32 -> cast_f32_to_i32 src dst start_idx end_idx
  | Float32, Int64 -> cast_f32_to_i64 src dst start_idx end_idx
  | Float32, Complex32 -> cast_f32_to_c32 src dst start_idx end_idx
  | Float32, Complex64 -> cast_f32_to_c64 src dst start_idx end_idx
  | Float32, Int -> cast_f32_to_int src dst start_idx end_idx
  | Float32, NativeInt -> cast_f32_to_nativeint src dst start_idx end_idx
  | Float32, BFloat16 -> cast_f32_to_bfloat16 src dst start_idx end_idx
  | Float32, Bool -> cast_f32_to_bool src dst start_idx end_idx
  | Float32, Int4 -> cast_f32_to_int4 src dst start_idx end_idx
  | Float32, UInt4 -> cast_f32_to_uint4 src dst start_idx end_idx
  | Float32, Float8_e4m3 -> cast_f32_to_float8_e4m3 src dst start_idx end_idx
  | Float32, Float8_e5m2 -> cast_f32_to_float8_e5m2 src dst start_idx end_idx
  | Float32, Complex16 -> cast_f32_to_complex16 src dst start_idx end_idx
  | Float32, QInt8 -> cast_f32_to_qint8 src dst start_idx end_idx
  | Float32, QUInt8 -> cast_f32_to_quint8 src dst start_idx end_idx
  (* Float64 Source *)
  | Float64, Float16 -> cast_f64_to_f16 src dst start_idx end_idx
  | Float64, Float32 -> cast_f64_to_f32 src dst start_idx end_idx
  | Float64, Float64 -> ()
  | Float64, Int8 -> cast_f64_to_i8 src dst start_idx end_idx
  | Float64, UInt8 -> cast_f64_to_u8 src dst start_idx end_idx
  | Float64, Int16 -> cast_f64_to_i16 src dst start_idx end_idx
  | Float64, UInt16 -> cast_f64_to_u16 src dst start_idx end_idx
  | Float64, Int32 -> cast_f64_to_i32 src dst start_idx end_idx
  | Float64, Int64 -> cast_f64_to_i64 src dst start_idx end_idx
  | Float64, Complex32 -> cast_f64_to_c32 src dst start_idx end_idx
  | Float64, Complex64 -> cast_f64_to_c64 src dst start_idx end_idx
  | Float64, Int -> cast_f64_to_int src dst start_idx end_idx
  | Float64, NativeInt -> cast_f64_to_nativeint src dst start_idx end_idx
  | Float64, BFloat16 -> cast_f64_to_bfloat16 src dst start_idx end_idx
  | Float64, Bool -> cast_f64_to_bool src dst start_idx end_idx
  | Float64, Int4 -> cast_f64_to_int4 src dst start_idx end_idx
  | Float64, UInt4 -> cast_f64_to_uint4 src dst start_idx end_idx
  | Float64, Float8_e4m3 -> cast_f64_to_float8_e4m3 src dst start_idx end_idx
  | Float64, Float8_e5m2 -> cast_f64_to_float8_e5m2 src dst start_idx end_idx
  | Float64, Complex16 -> cast_f64_to_complex16 src dst start_idx end_idx
  | Float64, QInt8 -> cast_f64_to_qint8 src dst start_idx end_idx
  | Float64, QUInt8 -> cast_f64_to_quint8 src dst start_idx end_idx
  (* Int8 Source *)
  | Int8, Float16 -> cast_i8_to_f16 src dst start_idx end_idx
  | Int8, Float32 -> cast_i8_to_f32 src dst start_idx end_idx
  | Int8, Float64 -> cast_i8_to_f64 src dst start_idx end_idx
  | Int8, Int8 -> ()
  | Int8, UInt8 -> cast_i8_to_u8 src dst start_idx end_idx
  | Int8, Int16 -> cast_i8_to_i16 src dst start_idx end_idx
  | Int8, UInt16 -> cast_i8_to_u16 src dst start_idx end_idx
  | Int8, Int32 -> cast_i8_to_i32 src dst start_idx end_idx
  | Int8, Int64 -> cast_i8_to_i64 src dst start_idx end_idx
  | Int8, Complex32 -> cast_i8_to_c32 src dst start_idx end_idx
  | Int8, Complex64 -> cast_i8_to_c64 src dst start_idx end_idx
  | Int8, Int -> cast_i8_to_int src dst start_idx end_idx
  | Int8, NativeInt -> cast_i8_to_nativeint src dst start_idx end_idx
  | Int8, BFloat16 -> cast_i8_to_bfloat16 src dst start_idx end_idx
  | Int8, Bool -> cast_i8_to_bool src dst start_idx end_idx
  | Int8, Int4 -> cast_i8_to_int4 src dst start_idx end_idx
  | Int8, UInt4 -> cast_i8_to_uint4 src dst start_idx end_idx
  | Int8, Float8_e4m3 -> cast_i8_to_float8_e4m3 src dst start_idx end_idx
  | Int8, Float8_e5m2 -> cast_i8_to_float8_e5m2 src dst start_idx end_idx
  | Int8, Complex16 -> cast_i8_to_complex16 src dst start_idx end_idx
  | Int8, QInt8 -> cast_i8_to_qint8 src dst start_idx end_idx
  | Int8, QUInt8 -> cast_i8_to_quint8 src dst start_idx end_idx
  (* UInt8 Source *)
  | UInt8, Float16 -> cast_u8_to_f16 src dst start_idx end_idx
  | UInt8, Float32 -> cast_u8_to_f32 src dst start_idx end_idx
  | UInt8, Float64 -> cast_u8_to_f64 src dst start_idx end_idx
  | UInt8, Int8 -> cast_u8_to_i8 src dst start_idx end_idx
  | UInt8, UInt8 -> ()
  | UInt8, Int16 -> cast_u8_to_i16 src dst start_idx end_idx
  | UInt8, UInt16 -> cast_u8_to_u16 src dst start_idx end_idx
  | UInt8, Int32 -> cast_u8_to_i32 src dst start_idx end_idx
  | UInt8, Int64 -> cast_u8_to_i64 src dst start_idx end_idx
  | UInt8, Complex32 -> cast_u8_to_c32 src dst start_idx end_idx
  | UInt8, Complex64 -> cast_u8_to_c64 src dst start_idx end_idx
  | UInt8, Int -> cast_u8_to_int src dst start_idx end_idx
  | UInt8, NativeInt -> cast_u8_to_nativeint src dst start_idx end_idx
  | UInt8, BFloat16 -> cast_u8_to_bfloat16 src dst start_idx end_idx
  | UInt8, Bool -> cast_u8_to_bool src dst start_idx end_idx
  | UInt8, Int4 -> cast_u8_to_int4 src dst start_idx end_idx
  | UInt8, UInt4 -> cast_u8_to_uint4 src dst start_idx end_idx
  | UInt8, Float8_e4m3 -> cast_u8_to_float8_e4m3 src dst start_idx end_idx
  | UInt8, Float8_e5m2 -> cast_u8_to_float8_e5m2 src dst start_idx end_idx
  | UInt8, Complex16 -> cast_u8_to_complex16 src dst start_idx end_idx
  | UInt8, QInt8 -> cast_u8_to_qint8 src dst start_idx end_idx
  | UInt8, QUInt8 -> cast_u8_to_quint8 src dst start_idx end_idx
  (* Int16 Source *)
  | Int16, Float16 -> cast_i16_to_f16 src dst start_idx end_idx
  | Int16, Float32 -> cast_i16_to_f32 src dst start_idx end_idx
  | Int16, Float64 -> cast_i16_to_f64 src dst start_idx end_idx
  | Int16, Int8 -> cast_i16_to_i8 src dst start_idx end_idx
  | Int16, UInt8 -> cast_i16_to_u8 src dst start_idx end_idx
  | Int16, Int16 -> ()
  | Int16, UInt16 -> cast_i16_to_u16 src dst start_idx end_idx
  | Int16, Int32 -> cast_i16_to_i32 src dst start_idx end_idx
  | Int16, Int64 -> cast_i16_to_i64 src dst start_idx end_idx
  | Int16, Complex32 -> cast_i16_to_c32 src dst start_idx end_idx
  | Int16, Complex64 -> cast_i16_to_c64 src dst start_idx end_idx
  | Int16, Int -> cast_i16_to_int src dst start_idx end_idx
  | Int16, NativeInt -> cast_i16_to_nativeint src dst start_idx end_idx
  | Int16, BFloat16 -> cast_i16_to_bfloat16 src dst start_idx end_idx
  | Int16, Bool -> cast_i16_to_bool src dst start_idx end_idx
  | Int16, Int4 -> cast_i16_to_int4 src dst start_idx end_idx
  | Int16, UInt4 -> cast_i16_to_uint4 src dst start_idx end_idx
  | Int16, Float8_e4m3 -> cast_i16_to_float8_e4m3 src dst start_idx end_idx
  | Int16, Float8_e5m2 -> cast_i16_to_float8_e5m2 src dst start_idx end_idx
  | Int16, Complex16 -> cast_i16_to_complex16 src dst start_idx end_idx
  | Int16, QInt8 -> cast_i16_to_qint8 src dst start_idx end_idx
  | Int16, QUInt8 -> cast_i16_to_quint8 src dst start_idx end_idx
  (* UInt16 Source *)
  | UInt16, Float16 -> cast_u16_to_f16 src dst start_idx end_idx
  | UInt16, Float32 -> cast_u16_to_f32 src dst start_idx end_idx
  | UInt16, Float64 -> cast_u16_to_f64 src dst start_idx end_idx
  | UInt16, Int8 -> cast_u16_to_i8 src dst start_idx end_idx
  | UInt16, UInt8 -> cast_u16_to_u8 src dst start_idx end_idx
  | UInt16, Int16 -> cast_u16_to_i16 src dst start_idx end_idx
  | UInt16, UInt16 -> ()
  | UInt16, Int32 -> cast_u16_to_i32 src dst start_idx end_idx
  | UInt16, Int64 -> cast_u16_to_i64 src dst start_idx end_idx
  | UInt16, Complex32 -> cast_u16_to_c32 src dst start_idx end_idx
  | UInt16, Complex64 -> cast_u16_to_c64 src dst start_idx end_idx
  | UInt16, Int -> cast_u16_to_int src dst start_idx end_idx
  | UInt16, NativeInt -> cast_u16_to_nativeint src dst start_idx end_idx
  | UInt16, BFloat16 -> cast_u16_to_bfloat16 src dst start_idx end_idx
  | UInt16, Bool -> cast_u16_to_bool src dst start_idx end_idx
  | UInt16, Int4 -> cast_u16_to_int4 src dst start_idx end_idx
  | UInt16, UInt4 -> cast_u16_to_uint4 src dst start_idx end_idx
  | UInt16, Float8_e4m3 -> cast_u16_to_float8_e4m3 src dst start_idx end_idx
  | UInt16, Float8_e5m2 -> cast_u16_to_float8_e5m2 src dst start_idx end_idx
  | UInt16, Complex16 -> cast_u16_to_complex16 src dst start_idx end_idx
  | UInt16, QInt8 -> cast_u16_to_qint8 src dst start_idx end_idx
  | UInt16, QUInt8 -> cast_u16_to_quint8 src dst start_idx end_idx
  (* Int32 Source *)
  | Int32, Float16 -> cast_i32_to_f16 src dst start_idx end_idx
  | Int32, Float32 -> cast_i32_to_f32 src dst start_idx end_idx
  | Int32, Float64 -> cast_i32_to_f64 src dst start_idx end_idx
  | Int32, Int8 -> cast_i32_to_i8 src dst start_idx end_idx
  | Int32, UInt8 -> cast_i32_to_u8 src dst start_idx end_idx
  | Int32, Int16 -> cast_i32_to_i16 src dst start_idx end_idx
  | Int32, UInt16 -> cast_i32_to_u16 src dst start_idx end_idx
  | Int32, Int32 -> ()
  | Int32, Int64 -> cast_i32_to_i64 src dst start_idx end_idx
  | Int32, Complex32 -> cast_i32_to_c32 src dst start_idx end_idx
  | Int32, Complex64 -> cast_i32_to_c64 src dst start_idx end_idx
  | Int32, Int -> cast_i32_to_int src dst start_idx end_idx
  | Int32, NativeInt -> cast_i32_to_nativeint src dst start_idx end_idx
  | Int32, BFloat16 -> cast_i32_to_bfloat16 src dst start_idx end_idx
  | Int32, Bool -> cast_i32_to_bool src dst start_idx end_idx
  | Int32, Int4 -> cast_i32_to_int4 src dst start_idx end_idx
  | Int32, UInt4 -> cast_i32_to_uint4 src dst start_idx end_idx
  | Int32, Float8_e4m3 -> cast_i32_to_float8_e4m3 src dst start_idx end_idx
  | Int32, Float8_e5m2 -> cast_i32_to_float8_e5m2 src dst start_idx end_idx
  | Int32, Complex16 -> cast_i32_to_complex16 src dst start_idx end_idx
  | Int32, QInt8 -> cast_i32_to_qint8 src dst start_idx end_idx
  | Int32, QUInt8 -> cast_i32_to_quint8 src dst start_idx end_idx
  (* Int64 Source *)
  | Int64, Float16 -> cast_i64_to_f16 src dst start_idx end_idx
  | Int64, Float32 -> cast_i64_to_f32 src dst start_idx end_idx
  | Int64, Float64 -> cast_i64_to_f64 src dst start_idx end_idx
  | Int64, Int8 -> cast_i64_to_i8 src dst start_idx end_idx
  | Int64, UInt8 -> cast_i64_to_u8 src dst start_idx end_idx
  | Int64, Int16 -> cast_i64_to_i16 src dst start_idx end_idx
  | Int64, UInt16 -> cast_i64_to_u16 src dst start_idx end_idx
  | Int64, Int32 -> cast_i64_to_i32 src dst start_idx end_idx
  | Int64, Int64 -> ()
  | Int64, Complex32 -> cast_i64_to_c32 src dst start_idx end_idx
  | Int64, Complex64 -> cast_i64_to_c64 src dst start_idx end_idx
  | Int64, Int -> cast_i64_to_int src dst start_idx end_idx
  | Int64, NativeInt -> cast_i64_to_nativeint src dst start_idx end_idx
  | Int64, BFloat16 -> cast_i64_to_bfloat16 src dst start_idx end_idx
  | Int64, Bool -> cast_i64_to_bool src dst start_idx end_idx
  | Int64, Int4 -> cast_i64_to_int4 src dst start_idx end_idx
  | Int64, UInt4 -> cast_i64_to_uint4 src dst start_idx end_idx
  | Int64, Float8_e4m3 -> cast_i64_to_float8_e4m3 src dst start_idx end_idx
  | Int64, Float8_e5m2 -> cast_i64_to_float8_e5m2 src dst start_idx end_idx
  | Int64, Complex16 -> cast_i64_to_complex16 src dst start_idx end_idx
  | Int64, QInt8 -> cast_i64_to_qint8 src dst start_idx end_idx
  | Int64, QUInt8 -> cast_i64_to_quint8 src dst start_idx end_idx
  (* Complex32 Source *)
  | Complex32, Float16 -> cast_c32_to_f16 src dst start_idx end_idx
  | Complex32, Float32 -> cast_c32_to_f32 src dst start_idx end_idx
  | Complex32, Float64 -> cast_c32_to_f64 src dst start_idx end_idx
  | Complex32, Int8 -> cast_c32_to_i8 src dst start_idx end_idx
  | Complex32, UInt8 -> cast_c32_to_u8 src dst start_idx end_idx
  | Complex32, Int16 -> cast_c32_to_i16 src dst start_idx end_idx
  | Complex32, UInt16 -> cast_c32_to_u16 src dst start_idx end_idx
  | Complex32, Int32 -> cast_c32_to_i32 src dst start_idx end_idx
  | Complex32, Int64 -> cast_c32_to_i64 src dst start_idx end_idx
  | Complex32, Complex32 -> ()
  | Complex32, Complex64 -> cast_c32_to_c64 src dst start_idx end_idx
  | Complex32, Int -> cast_c32_to_int src dst start_idx end_idx
  | Complex32, NativeInt -> cast_c32_to_nativeint src dst start_idx end_idx
  | Complex32, BFloat16 -> cast_c32_to_bfloat16 src dst start_idx end_idx
  | Complex32, Bool -> cast_c32_to_bool src dst start_idx end_idx
  | Complex32, Int4 -> cast_c32_to_int4 src dst start_idx end_idx
  | Complex32, UInt4 -> cast_c32_to_uint4 src dst start_idx end_idx
  | Complex32, Float8_e4m3 -> cast_c32_to_float8_e4m3 src dst start_idx end_idx
  | Complex32, Float8_e5m2 -> cast_c32_to_float8_e5m2 src dst start_idx end_idx
  | Complex32, Complex16 -> cast_c32_to_complex16 src dst start_idx end_idx
  | Complex32, QInt8 -> cast_c32_to_qint8 src dst start_idx end_idx
  | Complex32, QUInt8 -> cast_c32_to_quint8 src dst start_idx end_idx
  (* Complex64 Source *)
  | Complex64, Float16 -> cast_c64_to_f16 src dst start_idx end_idx
  | Complex64, Float32 -> cast_c64_to_f32 src dst start_idx end_idx
  | Complex64, Float64 -> cast_c64_to_f64 src dst start_idx end_idx
  | Complex64, Int8 -> cast_c64_to_i8 src dst start_idx end_idx
  | Complex64, UInt8 -> cast_c64_to_u8 src dst start_idx end_idx
  | Complex64, Int16 -> cast_c64_to_i16 src dst start_idx end_idx
  | Complex64, UInt16 -> cast_c64_to_u16 src dst start_idx end_idx
  | Complex64, Int32 -> cast_c64_to_i32 src dst start_idx end_idx
  | Complex64, Int64 -> cast_c64_to_i64 src dst start_idx end_idx
  | Complex64, Complex32 -> cast_c64_to_c32 src dst start_idx end_idx
  | Complex64, Complex64 -> ()
  | Complex64, Int -> cast_c64_to_int src dst start_idx end_idx
  | Complex64, NativeInt -> cast_c64_to_nativeint src dst start_idx end_idx
  | Complex64, BFloat16 -> cast_c64_to_bfloat16 src dst start_idx end_idx
  | Complex64, Bool -> cast_c64_to_bool src dst start_idx end_idx
  | Complex64, Int4 -> cast_c64_to_int4 src dst start_idx end_idx
  | Complex64, UInt4 -> cast_c64_to_uint4 src dst start_idx end_idx
  | Complex64, Float8_e4m3 -> cast_c64_to_float8_e4m3 src dst start_idx end_idx
  | Complex64, Float8_e5m2 -> cast_c64_to_float8_e5m2 src dst start_idx end_idx
  | Complex64, Complex16 -> cast_c64_to_complex16 src dst start_idx end_idx
  | Complex64, QInt8 -> cast_c64_to_qint8 src dst start_idx end_idx
  | Complex64, QUInt8 -> cast_c64_to_quint8 src dst start_idx end_idx
  (* Int Source *)
  | Int, Float16 -> cast_int_to_f16 src dst start_idx end_idx
  | Int, Float32 -> cast_int_to_f32 src dst start_idx end_idx
  | Int, Float64 -> cast_int_to_f64 src dst start_idx end_idx
  | Int, Int8 -> cast_int_to_i8 src dst start_idx end_idx
  | Int, UInt8 -> cast_int_to_u8 src dst start_idx end_idx
  | Int, Int16 -> cast_int_to_i16 src dst start_idx end_idx
  | Int, UInt16 -> cast_int_to_u16 src dst start_idx end_idx
  | Int, Int32 -> cast_int_to_i32 src dst start_idx end_idx
  | Int, Int64 -> cast_int_to_i64 src dst start_idx end_idx
  | Int, Complex32 -> cast_int_to_c32 src dst start_idx end_idx
  | Int, Complex64 -> cast_int_to_c64 src dst start_idx end_idx
  | Int, Int -> ()
  | Int, NativeInt -> cast_int_to_nativeint src dst start_idx end_idx
  | Int, BFloat16 -> cast_int_to_bfloat16 src dst start_idx end_idx
  | Int, Bool -> cast_int_to_bool src dst start_idx end_idx
  | Int, Int4 -> cast_int_to_int4 src dst start_idx end_idx
  | Int, UInt4 -> cast_int_to_uint4 src dst start_idx end_idx
  | Int, Float8_e4m3 -> cast_int_to_float8_e4m3 src dst start_idx end_idx
  | Int, Float8_e5m2 -> cast_int_to_float8_e5m2 src dst start_idx end_idx
  | Int, Complex16 -> cast_int_to_complex16 src dst start_idx end_idx
  | Int, QInt8 -> cast_int_to_qint8 src dst start_idx end_idx
  | Int, QUInt8 -> cast_int_to_quint8 src dst start_idx end_idx
  (* NativeInt Source *)
  | NativeInt, Float16 -> cast_nativeint_to_f16 src dst start_idx end_idx
  | NativeInt, Float32 -> cast_nativeint_to_f32 src dst start_idx end_idx
  | NativeInt, Float64 -> cast_nativeint_to_f64 src dst start_idx end_idx
  | NativeInt, Int8 -> cast_nativeint_to_i8 src dst start_idx end_idx
  | NativeInt, UInt8 -> cast_nativeint_to_u8 src dst start_idx end_idx
  | NativeInt, Int16 -> cast_nativeint_to_i16 src dst start_idx end_idx
  | NativeInt, UInt16 -> cast_nativeint_to_u16 src dst start_idx end_idx
  | NativeInt, Int32 -> cast_nativeint_to_i32 src dst start_idx end_idx
  | NativeInt, Int64 -> cast_nativeint_to_i64 src dst start_idx end_idx
  | NativeInt, Complex32 -> cast_nativeint_to_c32 src dst start_idx end_idx
  | NativeInt, Complex64 -> cast_nativeint_to_c64 src dst start_idx end_idx
  | NativeInt, Int -> cast_nativeint_to_int src dst start_idx end_idx
  | NativeInt, NativeInt -> ()
  | NativeInt, BFloat16 -> cast_nativeint_to_bfloat16 src dst start_idx end_idx
  | NativeInt, Bool -> cast_nativeint_to_bool src dst start_idx end_idx
  | NativeInt, Int4 -> cast_nativeint_to_int4 src dst start_idx end_idx
  | NativeInt, UInt4 -> cast_nativeint_to_uint4 src dst start_idx end_idx
  | NativeInt, Float8_e4m3 ->
      cast_nativeint_to_float8_e4m3 src dst start_idx end_idx
  | NativeInt, Float8_e5m2 ->
      cast_nativeint_to_float8_e5m2 src dst start_idx end_idx
  | NativeInt, Complex16 ->
      cast_nativeint_to_complex16 src dst start_idx end_idx
  | NativeInt, QInt8 -> cast_nativeint_to_qint8 src dst start_idx end_idx
  | NativeInt, QUInt8 -> cast_nativeint_to_quint8 src dst start_idx end_idx
  (* BFloat16 Source *)
  | BFloat16, Float16 -> cast_bfloat16_to_f16 src dst start_idx end_idx
  | BFloat16, Float32 -> cast_bfloat16_to_f32 src dst start_idx end_idx
  | BFloat16, Float64 -> cast_bfloat16_to_f64 src dst start_idx end_idx
  | BFloat16, Int8 -> cast_bfloat16_to_i8 src dst start_idx end_idx
  | BFloat16, UInt8 -> cast_bfloat16_to_u8 src dst start_idx end_idx
  | BFloat16, Int16 -> cast_bfloat16_to_i16 src dst start_idx end_idx
  | BFloat16, UInt16 -> cast_bfloat16_to_u16 src dst start_idx end_idx
  | BFloat16, Int32 -> cast_bfloat16_to_i32 src dst start_idx end_idx
  | BFloat16, Int64 -> cast_bfloat16_to_i64 src dst start_idx end_idx
  | BFloat16, Complex32 -> cast_bfloat16_to_c32 src dst start_idx end_idx
  | BFloat16, Complex64 -> cast_bfloat16_to_c64 src dst start_idx end_idx
  | BFloat16, Int -> cast_bfloat16_to_int src dst start_idx end_idx
  | BFloat16, NativeInt -> cast_bfloat16_to_nativeint src dst start_idx end_idx
  | BFloat16, BFloat16 -> ()
  | BFloat16, Bool -> cast_bfloat16_to_bool src dst start_idx end_idx
  | BFloat16, Int4 -> cast_bfloat16_to_int4 src dst start_idx end_idx
  | BFloat16, UInt4 -> cast_bfloat16_to_uint4 src dst start_idx end_idx
  | BFloat16, Float8_e4m3 ->
      cast_bfloat16_to_float8_e4m3 src dst start_idx end_idx
  | BFloat16, Float8_e5m2 ->
      cast_bfloat16_to_float8_e5m2 src dst start_idx end_idx
  | BFloat16, Complex16 -> cast_bfloat16_to_complex16 src dst start_idx end_idx
  | BFloat16, QInt8 -> cast_bfloat16_to_qint8 src dst start_idx end_idx
  | BFloat16, QUInt8 -> cast_bfloat16_to_quint8 src dst start_idx end_idx
  (* Bool Source *)
  | Bool, Float16 -> cast_bool_to_f16 src dst start_idx end_idx
  | Bool, Float32 -> cast_bool_to_f32 src dst start_idx end_idx
  | Bool, Float64 -> cast_bool_to_f64 src dst start_idx end_idx
  | Bool, Int8 -> cast_bool_to_i8 src dst start_idx end_idx
  | Bool, UInt8 -> cast_bool_to_u8 src dst start_idx end_idx
  | Bool, Int16 -> cast_bool_to_i16 src dst start_idx end_idx
  | Bool, UInt16 -> cast_bool_to_u16 src dst start_idx end_idx
  | Bool, Int32 -> cast_bool_to_i32 src dst start_idx end_idx
  | Bool, Int64 -> cast_bool_to_i64 src dst start_idx end_idx
  | Bool, Complex32 -> cast_bool_to_c32 src dst start_idx end_idx
  | Bool, Complex64 -> cast_bool_to_c64 src dst start_idx end_idx
  | Bool, Int -> cast_bool_to_int src dst start_idx end_idx
  | Bool, NativeInt -> cast_bool_to_nativeint src dst start_idx end_idx
  | Bool, BFloat16 -> cast_bool_to_bfloat16 src dst start_idx end_idx
  | Bool, Bool -> ()
  | Bool, Int4 -> cast_bool_to_int4 src dst start_idx end_idx
  | Bool, UInt4 -> cast_bool_to_uint4 src dst start_idx end_idx
  | Bool, Float8_e4m3 -> cast_bool_to_float8_e4m3 src dst start_idx end_idx
  | Bool, Float8_e5m2 -> cast_bool_to_float8_e5m2 src dst start_idx end_idx
  | Bool, Complex16 -> cast_bool_to_complex16 src dst start_idx end_idx
  | Bool, QInt8 -> cast_bool_to_qint8 src dst start_idx end_idx
  | Bool, QUInt8 -> cast_bool_to_quint8 src dst start_idx end_idx
  (* Int4 Source *)
  | Int4, Float16 -> cast_int4_to_f16 src dst start_idx end_idx
  | Int4, Float32 -> cast_int4_to_f32 src dst start_idx end_idx
  | Int4, Float64 -> cast_int4_to_f64 src dst start_idx end_idx
  | Int4, Int8 -> cast_int4_to_i8 src dst start_idx end_idx
  | Int4, UInt8 -> cast_int4_to_u8 src dst start_idx end_idx
  | Int4, Int16 -> cast_int4_to_i16 src dst start_idx end_idx
  | Int4, UInt16 -> cast_int4_to_u16 src dst start_idx end_idx
  | Int4, Int32 -> cast_int4_to_i32 src dst start_idx end_idx
  | Int4, Int64 -> cast_int4_to_i64 src dst start_idx end_idx
  | Int4, Complex32 -> cast_int4_to_c32 src dst start_idx end_idx
  | Int4, Complex64 -> cast_int4_to_c64 src dst start_idx end_idx
  | Int4, Int -> cast_int4_to_int src dst start_idx end_idx
  | Int4, NativeInt -> cast_int4_to_nativeint src dst start_idx end_idx
  | Int4, BFloat16 -> cast_int4_to_bfloat16 src dst start_idx end_idx
  | Int4, Bool -> cast_int4_to_bool src dst start_idx end_idx
  | Int4, Int4 -> ()
  | Int4, UInt4 -> cast_int4_to_uint4 src dst start_idx end_idx
  | Int4, Float8_e4m3 -> cast_int4_to_float8_e4m3 src dst start_idx end_idx
  | Int4, Float8_e5m2 -> cast_int4_to_float8_e5m2 src dst start_idx end_idx
  | Int4, Complex16 -> cast_int4_to_complex16 src dst start_idx end_idx
  | Int4, QInt8 -> cast_int4_to_qint8 src dst start_idx end_idx
  | Int4, QUInt8 -> cast_int4_to_quint8 src dst start_idx end_idx
  (* UInt4 Source *)
  | UInt4, Float16 -> cast_uint4_to_f16 src dst start_idx end_idx
  | UInt4, Float32 -> cast_uint4_to_f32 src dst start_idx end_idx
  | UInt4, Float64 -> cast_uint4_to_f64 src dst start_idx end_idx
  | UInt4, Int8 -> cast_uint4_to_i8 src dst start_idx end_idx
  | UInt4, UInt8 -> cast_uint4_to_u8 src dst start_idx end_idx
  | UInt4, Int16 -> cast_uint4_to_i16 src dst start_idx end_idx
  | UInt4, UInt16 -> cast_uint4_to_u16 src dst start_idx end_idx
  | UInt4, Int32 -> cast_uint4_to_i32 src dst start_idx end_idx
  | UInt4, Int64 -> cast_uint4_to_i64 src dst start_idx end_idx
  | UInt4, Complex32 -> cast_uint4_to_c32 src dst start_idx end_idx
  | UInt4, Complex64 -> cast_uint4_to_c64 src dst start_idx end_idx
  | UInt4, Int -> cast_uint4_to_int src dst start_idx end_idx
  | UInt4, NativeInt -> cast_uint4_to_nativeint src dst start_idx end_idx
  | UInt4, BFloat16 -> cast_uint4_to_bfloat16 src dst start_idx end_idx
  | UInt4, Bool -> cast_uint4_to_bool src dst start_idx end_idx
  | UInt4, Int4 -> cast_uint4_to_int4 src dst start_idx end_idx
  | UInt4, UInt4 -> ()
  | UInt4, Float8_e4m3 -> cast_uint4_to_float8_e4m3 src dst start_idx end_idx
  | UInt4, Float8_e5m2 -> cast_uint4_to_float8_e5m2 src dst start_idx end_idx
  | UInt4, Complex16 -> cast_uint4_to_complex16 src dst start_idx end_idx
  | UInt4, QInt8 -> cast_uint4_to_qint8 src dst start_idx end_idx
  | UInt4, QUInt8 -> cast_uint4_to_quint8 src dst start_idx end_idx
  (* Float8_e4m3 Source *)
  | Float8_e4m3, Float16 -> cast_float8_e4m3_to_f16 src dst start_idx end_idx
  | Float8_e4m3, Float32 -> cast_float8_e4m3_to_f32 src dst start_idx end_idx
  | Float8_e4m3, Float64 -> cast_float8_e4m3_to_f64 src dst start_idx end_idx
  | Float8_e4m3, Int8 -> cast_float8_e4m3_to_i8 src dst start_idx end_idx
  | Float8_e4m3, UInt8 -> cast_float8_e4m3_to_u8 src dst start_idx end_idx
  | Float8_e4m3, Int16 -> cast_float8_e4m3_to_i16 src dst start_idx end_idx
  | Float8_e4m3, UInt16 -> cast_float8_e4m3_to_u16 src dst start_idx end_idx
  | Float8_e4m3, Int32 -> cast_float8_e4m3_to_i32 src dst start_idx end_idx
  | Float8_e4m3, Int64 -> cast_float8_e4m3_to_i64 src dst start_idx end_idx
  | Float8_e4m3, Complex32 -> cast_float8_e4m3_to_c32 src dst start_idx end_idx
  | Float8_e4m3, Complex64 -> cast_float8_e4m3_to_c64 src dst start_idx end_idx
  | Float8_e4m3, Int -> cast_float8_e4m3_to_int src dst start_idx end_idx
  | Float8_e4m3, NativeInt ->
      cast_float8_e4m3_to_nativeint src dst start_idx end_idx
  | Float8_e4m3, BFloat16 ->
      cast_float8_e4m3_to_bfloat16 src dst start_idx end_idx
  | Float8_e4m3, Bool -> cast_float8_e4m3_to_bool src dst start_idx end_idx
  | Float8_e4m3, Int4 -> cast_float8_e4m3_to_int4 src dst start_idx end_idx
  | Float8_e4m3, UInt4 -> cast_float8_e4m3_to_uint4 src dst start_idx end_idx
  | Float8_e4m3, Float8_e4m3 -> ()
  | Float8_e4m3, Float8_e5m2 ->
      cast_float8_e4m3_to_float8_e5m2 src dst start_idx end_idx
  | Float8_e4m3, Complex16 ->
      cast_float8_e4m3_to_complex16 src dst start_idx end_idx
  | Float8_e4m3, QInt8 -> cast_float8_e4m3_to_qint8 src dst start_idx end_idx
  | Float8_e4m3, QUInt8 -> cast_float8_e4m3_to_quint8 src dst start_idx end_idx
  (* Float8_e5m2 Source *)
  | Float8_e5m2, Float16 -> cast_float8_e5m2_to_f16 src dst start_idx end_idx
  | Float8_e5m2, Float32 -> cast_float8_e5m2_to_f32 src dst start_idx end_idx
  | Float8_e5m2, Float64 -> cast_float8_e5m2_to_f64 src dst start_idx end_idx
  | Float8_e5m2, Int8 -> cast_float8_e5m2_to_i8 src dst start_idx end_idx
  | Float8_e5m2, UInt8 -> cast_float8_e5m2_to_u8 src dst start_idx end_idx
  | Float8_e5m2, Int16 -> cast_float8_e5m2_to_i16 src dst start_idx end_idx
  | Float8_e5m2, UInt16 -> cast_float8_e5m2_to_u16 src dst start_idx end_idx
  | Float8_e5m2, Int32 -> cast_float8_e5m2_to_i32 src dst start_idx end_idx
  | Float8_e5m2, Int64 -> cast_float8_e5m2_to_i64 src dst start_idx end_idx
  | Float8_e5m2, Complex32 -> cast_float8_e5m2_to_c32 src dst start_idx end_idx
  | Float8_e5m2, Complex64 -> cast_float8_e5m2_to_c64 src dst start_idx end_idx
  | Float8_e5m2, Int -> cast_float8_e5m2_to_int src dst start_idx end_idx
  | Float8_e5m2, NativeInt ->
      cast_float8_e5m2_to_nativeint src dst start_idx end_idx
  | Float8_e5m2, BFloat16 ->
      cast_float8_e5m2_to_bfloat16 src dst start_idx end_idx
  | Float8_e5m2, Bool -> cast_float8_e5m2_to_bool src dst start_idx end_idx
  | Float8_e5m2, Int4 -> cast_float8_e5m2_to_int4 src dst start_idx end_idx
  | Float8_e5m2, UInt4 -> cast_float8_e5m2_to_uint4 src dst start_idx end_idx
  | Float8_e5m2, Float8_e4m3 ->
      cast_float8_e5m2_to_float8_e4m3 src dst start_idx end_idx
  | Float8_e5m2, Float8_e5m2 -> ()
  | Float8_e5m2, Complex16 ->
      cast_float8_e5m2_to_complex16 src dst start_idx end_idx
  | Float8_e5m2, QInt8 -> cast_float8_e5m2_to_qint8 src dst start_idx end_idx
  | Float8_e5m2, QUInt8 -> cast_float8_e5m2_to_quint8 src dst start_idx end_idx
  (* Complex16 Source *)
  | Complex16, Float16 -> cast_complex16_to_f16 src dst start_idx end_idx
  | Complex16, Float32 -> cast_complex16_to_f32 src dst start_idx end_idx
  | Complex16, Float64 -> cast_complex16_to_f64 src dst start_idx end_idx
  | Complex16, Int8 -> cast_complex16_to_i8 src dst start_idx end_idx
  | Complex16, UInt8 -> cast_complex16_to_u8 src dst start_idx end_idx
  | Complex16, Int16 -> cast_complex16_to_i16 src dst start_idx end_idx
  | Complex16, UInt16 -> cast_complex16_to_u16 src dst start_idx end_idx
  | Complex16, Int32 -> cast_complex16_to_i32 src dst start_idx end_idx
  | Complex16, Int64 -> cast_complex16_to_i64 src dst start_idx end_idx
  | Complex16, Complex32 -> cast_complex16_to_c32 src dst start_idx end_idx
  | Complex16, Complex64 -> cast_complex16_to_c64 src dst start_idx end_idx
  | Complex16, Int -> cast_complex16_to_int src dst start_idx end_idx
  | Complex16, NativeInt ->
      cast_complex16_to_nativeint src dst start_idx end_idx
  | Complex16, BFloat16 -> cast_complex16_to_bfloat16 src dst start_idx end_idx
  | Complex16, Bool -> cast_complex16_to_bool src dst start_idx end_idx
  | Complex16, Int4 -> cast_complex16_to_int4 src dst start_idx end_idx
  | Complex16, UInt4 -> cast_complex16_to_uint4 src dst start_idx end_idx
  | Complex16, Float8_e4m3 ->
      cast_complex16_to_float8_e4m3 src dst start_idx end_idx
  | Complex16, Float8_e5m2 ->
      cast_complex16_to_float8_e5m2 src dst start_idx end_idx
  | Complex16, Complex16 -> ()
  | Complex16, QInt8 -> cast_complex16_to_qint8 src dst start_idx end_idx
  | Complex16, QUInt8 -> cast_complex16_to_quint8 src dst start_idx end_idx
  (* QInt8 Source *)
  | QInt8, Float16 -> cast_qint8_to_f16 src dst start_idx end_idx
  | QInt8, Float32 -> cast_qint8_to_f32 src dst start_idx end_idx
  | QInt8, Float64 -> cast_qint8_to_f64 src dst start_idx end_idx
  | QInt8, Int8 -> cast_qint8_to_i8 src dst start_idx end_idx
  | QInt8, UInt8 -> cast_qint8_to_u8 src dst start_idx end_idx
  | QInt8, Int16 -> cast_qint8_to_i16 src dst start_idx end_idx
  | QInt8, UInt16 -> cast_qint8_to_u16 src dst start_idx end_idx
  | QInt8, Int32 -> cast_qint8_to_i32 src dst start_idx end_idx
  | QInt8, Int64 -> cast_qint8_to_i64 src dst start_idx end_idx
  | QInt8, Complex32 -> cast_qint8_to_c32 src dst start_idx end_idx
  | QInt8, Complex64 -> cast_qint8_to_c64 src dst start_idx end_idx
  | QInt8, Int -> cast_qint8_to_int src dst start_idx end_idx
  | QInt8, NativeInt -> cast_qint8_to_nativeint src dst start_idx end_idx
  | QInt8, BFloat16 -> cast_qint8_to_bfloat16 src dst start_idx end_idx
  | QInt8, Bool -> cast_qint8_to_bool src dst start_idx end_idx
  | QInt8, Int4 -> cast_qint8_to_int4 src dst start_idx end_idx
  | QInt8, UInt4 -> cast_qint8_to_uint4 src dst start_idx end_idx
  | QInt8, Float8_e4m3 -> cast_qint8_to_float8_e4m3 src dst start_idx end_idx
  | QInt8, Float8_e5m2 -> cast_qint8_to_float8_e5m2 src dst start_idx end_idx
  | QInt8, Complex16 -> cast_qint8_to_complex16 src dst start_idx end_idx
  | QInt8, QInt8 -> ()
  | QInt8, QUInt8 -> cast_qint8_to_quint8 src dst start_idx end_idx
  (* QUInt8 Source *)
  | QUInt8, Float16 -> cast_quint8_to_f16 src dst start_idx end_idx
  | QUInt8, Float32 -> cast_quint8_to_f32 src dst start_idx end_idx
  | QUInt8, Float64 -> cast_quint8_to_f64 src dst start_idx end_idx
  | QUInt8, Int8 -> cast_quint8_to_i8 src dst start_idx end_idx
  | QUInt8, UInt8 -> cast_quint8_to_u8 src dst start_idx end_idx
  | QUInt8, Int16 -> cast_quint8_to_i16 src dst start_idx end_idx
  | QUInt8, UInt16 -> cast_quint8_to_u16 src dst start_idx end_idx
  | QUInt8, Int32 -> cast_quint8_to_i32 src dst start_idx end_idx
  | QUInt8, Int64 -> cast_quint8_to_i64 src dst start_idx end_idx
  | QUInt8, Complex32 -> cast_quint8_to_c32 src dst start_idx end_idx
  | QUInt8, Complex64 -> cast_quint8_to_c64 src dst start_idx end_idx
  | QUInt8, Int -> cast_quint8_to_int src dst start_idx end_idx
  | QUInt8, NativeInt -> cast_quint8_to_nativeint src dst start_idx end_idx
  | QUInt8, BFloat16 -> cast_quint8_to_bfloat16 src dst start_idx end_idx
  | QUInt8, Bool -> cast_quint8_to_bool src dst start_idx end_idx
  | QUInt8, Int4 -> cast_quint8_to_int4 src dst start_idx end_idx
  | QUInt8, UInt4 -> cast_quint8_to_uint4 src dst start_idx end_idx
  | QUInt8, Float8_e4m3 -> cast_quint8_to_float8_e4m3 src dst start_idx end_idx
  | QUInt8, Float8_e5m2 -> cast_quint8_to_float8_e5m2 src dst start_idx end_idx
  | QUInt8, Complex16 -> cast_quint8_to_complex16 src dst start_idx end_idx
  | QUInt8, QInt8 -> cast_quint8_to_qint8 src dst start_idx end_idx
  | QUInt8, QUInt8 -> ()

let cast (type a b c d) ctx (src_tensor : (a, b) t) (dst_tensor : (c, d) t) =
  match Dtype.equal_witness (dtype src_tensor) (dtype dst_tensor) with
  | Some Equal -> () (* No casting needed *)
  | None ->
      let src_size = size src_tensor in
      if src_size = 0 then () (* Nothing to cast for empty tensors *)
      else
        let total_size = size src_tensor in
        if total_size > 1000 then
          let pool = ctx.pool in
          Parallel.parallel_for pool 0 (total_size - 1)
            (fun start_idx end_idx ->
              cast_kernel src_tensor dst_tensor start_idx end_idx)
        else cast_kernel src_tensor dst_tensor 0 total_size
