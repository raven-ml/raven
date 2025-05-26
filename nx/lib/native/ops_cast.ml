open Bigarray
module Dtype = Nx_core.Dtype
open Nx_core.View
open Internal

let cast_f16_to_f32 (src : (float, float16_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_f16_to_f64 (src : (float, float16_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_f16_to_i8 (src : (float, float16_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f16_to_u8 (src : (float, float16_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f16_to_i16 (src : (float, float16_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f16_to_u16 (src : (float, float16_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f16_to_i32 (src : (float, float16_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.of_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.of_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.of_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.of_float src_val)
    done

let cast_f16_to_i64 (src : (float, float16_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.of_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.of_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_float src_val)
    done

let cast_f16_to_c32 (src : (float, float16_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = src_val; im = 0.0 }
    done

let cast_f16_to_c64 (src : (float, float16_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = src_val; im = 0.0 }
    done

let cast_f16_to_int (src : (float, float16_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f16_to_nativeint (src : (float, float16_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_float src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_float src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_float src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_float src_val)
    done

let cast_f32_to_f16 (src : (float, float32_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_f32_to_f64 (src : (float, float32_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_f32_to_i8 (src : (float, float32_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f32_to_u8 (src : (float, float32_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f32_to_i16 (src : (float, float32_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f32_to_u16 (src : (float, float32_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f32_to_i32 (src : (float, float32_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.of_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.of_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.of_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.of_float src_val)
    done

let cast_f32_to_i64 (src : (float, float32_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.of_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.of_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_float src_val)
    done

let cast_f32_to_c32 (src : (float, float32_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = src_val; im = 0.0 }
    done

let cast_f32_to_c64 (src : (float, float32_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = src_val; im = 0.0 }
    done

let cast_f32_to_int (src : (float, float32_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f32_to_nativeint (src : (float, float32_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_float src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_float src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_float src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_float src_val)
    done

let cast_f64_to_f16 (src : (float, float64_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_f64_to_f32 (src : (float, float64_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_f64_to_i8 (src : (float, float64_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f64_to_u8 (src : (float, float64_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f64_to_i16 (src : (float, float64_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f64_to_u16 (src : (float, float64_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f64_to_i32 (src : (float, float64_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.of_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.of_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.of_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.of_float src_val)
    done

let cast_f64_to_i64 (src : (float, float64_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.of_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.of_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_float src_val)
    done

let cast_f64_to_c32 (src : (float, float64_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = src_val; im = 0.0 }
    done

let cast_f64_to_c64 (src : (float, float64_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 { Complex.re = src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = src_val; im = 0.0 }
    done

let cast_f64_to_int (src : (float, float64_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val)
    done

let cast_f64_to_nativeint (src : (float, float64_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_float src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_float src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_float src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_float src_val)
    done

let cast_i8_to_f16 (src : (int, int8_signed_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_i8_to_f32 (src : (int, int8_signed_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_i8_to_f64 (src : (int, int8_signed_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_i8_to_u8 (src : (int, int8_signed_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_i8_to_i16 (src : (int, int8_signed_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_i8_to_u16 (src : (int, int8_signed_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_i8_to_i32 (src : (int, int8_signed_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.of_int src_val)
    done

let cast_i8_to_i64 (src : (int, int8_signed_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_int src_val)
    done

let cast_i8_to_c32 (src : (int, int8_signed_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_i8_to_c64 (src : (int, int8_signed_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_i8_to_int (src : (int, int8_signed_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_i8_to_nativeint (src : (int, int8_signed_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_int src_val)
    done

let cast_u8_to_f16 (src : (int, int8_unsigned_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_u8_to_f32 (src : (int, int8_unsigned_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_u8_to_f64 (src : (int, int8_unsigned_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_u8_to_i8 (src : (int, int8_unsigned_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_u8_to_i16 (src : (int, int8_unsigned_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_u8_to_u16 (src : (int, int8_unsigned_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_u8_to_i32 (src : (int, int8_unsigned_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.of_int src_val)
    done

let cast_u8_to_i64 (src : (int, int8_unsigned_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_int src_val)
    done

let cast_u8_to_c32 (src : (int, int8_unsigned_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_u8_to_c64 (src : (int, int8_unsigned_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_u8_to_int (src : (int, int8_unsigned_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_u8_to_nativeint (src : (int, int8_unsigned_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_int src_val)
    done

let cast_i16_to_f16 (src : (int, int16_signed_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_i16_to_f32 (src : (int, int16_signed_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_i16_to_f64 (src : (int, int16_signed_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_i16_to_i8 (src : (int, int16_signed_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_i16_to_u8 (src : (int, int16_signed_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_i16_to_u16 (src : (int, int16_signed_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_i16_to_i32 (src : (int, int16_signed_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.of_int src_val)
    done

let cast_i16_to_i64 (src : (int, int16_signed_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_int src_val)
    done

let cast_i16_to_c32 (src : (int, int16_signed_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_i16_to_c64 (src : (int, int16_signed_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_i16_to_int (src : (int, int16_signed_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_i16_to_nativeint (src : (int, int16_signed_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_int src_val)
    done

let cast_u16_to_f16 (src : (int, int16_unsigned_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_u16_to_f32 (src : (int, int16_unsigned_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_u16_to_f64 (src : (int, int16_unsigned_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_u16_to_i8 (src : (int, int16_unsigned_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_u16_to_u8 (src : (int, int16_unsigned_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_u16_to_i16 (src : (int, int16_unsigned_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_u16_to_i32 (src : (int, int16_unsigned_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.of_int src_val)
    done

let cast_u16_to_i64 (src : (int, int16_unsigned_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_int src_val)
    done

let cast_u16_to_c32 (src : (int, int16_unsigned_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_u16_to_c64 (src : (int, int16_unsigned_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_u16_to_int (src : (int, int16_unsigned_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_u16_to_nativeint (src : (int, int16_unsigned_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_int src_val)
    done

let cast_i32_to_f16 (src : (int32, int32_elt) t) (dst : (float, float16_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.to_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.to_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.to_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.to_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.to_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.to_float src_val)
    done

let cast_i32_to_f32 (src : (int32, int32_elt) t) (dst : (float, float32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.to_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.to_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.to_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.to_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.to_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.to_float src_val)
    done

let cast_i32_to_f64 (src : (int32, int32_elt) t) (dst : (float, float64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.to_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.to_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.to_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.to_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.to_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.to_float src_val)
    done

let cast_i32_to_i8 (src : (int32, int32_elt) t) (dst : (int, int8_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.to_int src_val)
    done

let cast_i32_to_u8 (src : (int32, int32_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.to_int src_val)
    done

let cast_i32_to_i16 (src : (int32, int32_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.to_int src_val)
    done

let cast_i32_to_u16 (src : (int32, int32_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.to_int src_val)
    done

let cast_i32_to_i64 (src : (int32, int32_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_int32 src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.of_int32 src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.of_int32 src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.of_int32 src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_int32 src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_int32 src_val)
    done

let cast_i32_to_c32 (src : (int32, int32_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = Int32.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = Int32.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = Int32.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = Int32.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = Int32.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = Int32.to_float src_val; im = 0.0 }
    done

let cast_i32_to_c64 (src : (int32, int32_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = Int32.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = Int32.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = Int32.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = Int32.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = Int32.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = Int32.to_float src_val; im = 0.0 }
    done

let cast_i32_to_int (src : (int32, int32_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.to_int src_val)
    done

let cast_i32_to_nativeint (src : (int32, int32_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_int32 src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_int32 src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_int32 src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_int32 src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_int32 src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_int32 src_val)
    done

let cast_i64_to_f16 (src : (int64, int64_elt) t) (dst : (float, float16_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.to_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.to_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.to_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.to_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.to_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.to_float src_val)
    done

let cast_i64_to_f32 (src : (int64, int64_elt) t) (dst : (float, float32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.to_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.to_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.to_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.to_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.to_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.to_float src_val)
    done

let cast_i64_to_f64 (src : (int64, int64_elt) t) (dst : (float, float64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.to_float src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.to_float src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.to_float src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.to_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.to_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.to_float src_val)
    done

let cast_i64_to_i8 (src : (int64, int64_elt) t) (dst : (int, int8_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.to_int src_val)
    done

let cast_i64_to_u8 (src : (int64, int64_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.to_int src_val)
    done

let cast_i64_to_i16 (src : (int64, int64_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.to_int src_val)
    done

let cast_i64_to_u16 (src : (int64, int64_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.to_int src_val)
    done

let cast_i64_to_i32 (src : (int64, int64_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.to_int32 src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.to_int32 src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.to_int32 src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.to_int32 src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.to_int32 src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.to_int32 src_val)
    done

let cast_i64_to_c32 (src : (int64, int64_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = Int64.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = Int64.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = Int64.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = Int64.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = Int64.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = Int64.to_float src_val; im = 0.0 }
    done

let cast_i64_to_c64 (src : (int64, int64_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = Int64.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = Int64.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = Int64.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = Int64.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = Int64.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = Int64.to_float src_val; im = 0.0 }
    done

let cast_i64_to_int (src : (int64, int64_elt) t) (dst : (int, int_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.to_int src_val)
    done

let cast_i64_to_nativeint (src : (int64, int64_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.to_nativeint src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.to_nativeint src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.to_nativeint src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.to_nativeint src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.to_nativeint src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.to_nativeint src_val)
    done

let cast_c32_to_f16 (src : (Complex.t, complex32_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val.Complex.re
    done

let cast_c32_to_f32 (src : (Complex.t, complex32_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val.Complex.re
    done

let cast_c32_to_f64 (src : (Complex.t, complex32_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val.Complex.re
    done

let cast_c32_to_i8 (src : (Complex.t, complex32_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val.Complex.re)
    done

let cast_c32_to_u8 (src : (Complex.t, complex32_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val.Complex.re)
    done

let cast_c32_to_i16 (src : (Complex.t, complex32_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val.Complex.re)
    done

let cast_c32_to_u16 (src : (Complex.t, complex32_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val.Complex.re)
    done

let cast_c32_to_i32 (src : (Complex.t, complex32_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (Int32.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (Int32.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (Int32.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.of_float src_val.Complex.re)
    done

let cast_c32_to_i64 (src : (Complex.t, complex32_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (Int64.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (Int64.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (Int64.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_float src_val.Complex.re)
    done

let cast_c32_to_c64 (src : (Complex.t, complex32_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_c32_to_int (src : (Complex.t, complex32_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val.Complex.re)
    done

let cast_c32_to_nativeint (src : (Complex.t, complex32_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_float src_val.Complex.re)
    done

let cast_c64_to_f16 (src : (Complex.t, complex64_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val.Complex.re
    done

let cast_c64_to_f32 (src : (Complex.t, complex64_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val.Complex.re
    done

let cast_c64_to_f64 (src : (Complex.t, complex64_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val.Complex.re
    done

let cast_c64_to_i8 (src : (Complex.t, complex64_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val.Complex.re)
    done

let cast_c64_to_u8 (src : (Complex.t, complex64_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val.Complex.re)
    done

let cast_c64_to_i16 (src : (Complex.t, complex64_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val.Complex.re)
    done

let cast_c64_to_u16 (src : (Complex.t, complex64_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val.Complex.re)
    done

let cast_c64_to_i32 (src : (Complex.t, complex64_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (Int32.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (Int32.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (Int32.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.of_float src_val.Complex.re)
    done

let cast_c64_to_i64 (src : (Complex.t, complex64_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (Int64.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (Int64.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (Int64.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_float src_val.Complex.re)
    done

let cast_c64_to_c32 (src : (Complex.t, complex64_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_c64_to_int (src : (Complex.t, complex64_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (int_of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (int_of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (int_of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (int_of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (int_of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (int_of_float src_val.Complex.re)
    done

let cast_c64_to_nativeint (src : (Complex.t, complex64_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_float src_val0.Complex.re);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_float src_val1.Complex.re);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_float src_val2.Complex.re);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_float src_val3.Complex.re);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_float src_val.Complex.re);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_float src_val.Complex.re)
    done

let cast_int_to_f16 (src : (int, int_elt) t) (dst : (float, float16_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_int_to_f32 (src : (int, int_elt) t) (dst : (float, float32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_int_to_f64 (src : (int, int_elt) t) (dst : (float, float64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (float_of_int src_val0);
      Array1.unsafe_set dst_buf i1 (float_of_int src_val1);
      Array1.unsafe_set dst_buf i2 (float_of_int src_val2);
      Array1.unsafe_set dst_buf i3 (float_of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (float_of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (float_of_int src_val)
    done

let cast_int_to_i8 (src : (int, int_elt) t) (dst : (int, int8_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_int_to_u8 (src : (int, int_elt) t) (dst : (int, int8_unsigned_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_int_to_i16 (src : (int, int_elt) t) (dst : (int, int16_signed_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_int_to_u16 (src : (int, int_elt) t) (dst : (int, int16_unsigned_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 src_val0;
      Array1.unsafe_set dst_buf i1 src_val1;
      Array1.unsafe_set dst_buf i2 src_val2;
      Array1.unsafe_set dst_buf i3 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k src_val
    done

let cast_int_to_i32 (src : (int, int_elt) t) (dst : (int32, int32_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int32.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int32.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int32.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int32.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int32.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int32.of_int src_val)
    done

let cast_int_to_i64 (src : (int, int_elt) t) (dst : (int64, int64_elt) t)
    start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_int src_val)
    done

let cast_int_to_c32 (src : (int, int_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_int_to_c64 (src : (int, int_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = float_of_int src_val; im = 0.0 }
    done

let cast_int_to_nativeint (src : (int, int_elt) t)
    (dst : (nativeint, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.of_int src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.of_int src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.of_int src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.of_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.of_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.of_int src_val)
    done

let cast_nativeint_to_f16 (src : (nativeint, nativeint_elt) t)
    (dst : (float, float16_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.to_float src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.to_float src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.to_float src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.to_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.to_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.to_float src_val)
    done

let cast_nativeint_to_f32 (src : (nativeint, nativeint_elt) t)
    (dst : (float, float32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.to_float src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.to_float src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.to_float src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.to_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.to_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.to_float src_val)
    done

let cast_nativeint_to_f64 (src : (nativeint, nativeint_elt) t)
    (dst : (float, float64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.to_float src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.to_float src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.to_float src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.to_float src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.to_float src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.to_float src_val)
    done

let cast_nativeint_to_i8 (src : (nativeint, nativeint_elt) t)
    (dst : (int, int8_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.to_int src_val)
    done

let cast_nativeint_to_u8 (src : (nativeint, nativeint_elt) t)
    (dst : (int, int8_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.to_int src_val)
    done

let cast_nativeint_to_i16 (src : (nativeint, nativeint_elt) t)
    (dst : (int, int16_signed_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.to_int src_val)
    done

let cast_nativeint_to_u16 (src : (nativeint, nativeint_elt) t)
    (dst : (int, int16_unsigned_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.to_int src_val)
    done

let cast_nativeint_to_i32 (src : (nativeint, nativeint_elt) t)
    (dst : (int32, int32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.to_int32 src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.to_int32 src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.to_int32 src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.to_int32 src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.to_int32 src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.to_int32 src_val)
    done

let cast_nativeint_to_i64 (src : (nativeint, nativeint_elt) t)
    (dst : (int64, int64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Int64.of_nativeint src_val0);
      Array1.unsafe_set dst_buf i1 (Int64.of_nativeint src_val1);
      Array1.unsafe_set dst_buf i2 (Int64.of_nativeint src_val2);
      Array1.unsafe_set dst_buf i3 (Int64.of_nativeint src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Int64.of_nativeint src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Int64.of_nativeint src_val)
    done

let cast_nativeint_to_c32 (src : (nativeint, nativeint_elt) t)
    (dst : (Complex.t, complex32_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = Nativeint.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = Nativeint.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = Nativeint.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = Nativeint.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = Nativeint.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = Nativeint.to_float src_val; im = 0.0 }
    done

let cast_nativeint_to_c64 (src : (nativeint, nativeint_elt) t)
    (dst : (Complex.t, complex64_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0
        { Complex.re = Nativeint.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1
        { Complex.re = Nativeint.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2
        { Complex.re = Nativeint.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3
        { Complex.re = Nativeint.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx
        { Complex.re = Nativeint.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k
        { Complex.re = Nativeint.to_float src_val; im = 0.0 }
    done

let cast_nativeint_to_int (src : (nativeint, nativeint_elt) t)
    (dst : (int, int_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 (Nativeint.to_int src_val0);
      Array1.unsafe_set dst_buf i1 (Nativeint.to_int src_val1);
      Array1.unsafe_set dst_buf i2 (Nativeint.to_int src_val2);
      Array1.unsafe_set dst_buf i3 (Nativeint.to_int src_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx (Nativeint.to_int src_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k (Nativeint.to_int src_val)
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

let cast (type a b c d) ctx (src_tensor : (a, b) t) (dst_tensor : (c, d) t) =
  match Dtype.eq_gadt (dtype src_tensor) (dtype dst_tensor) with
  | Some Refl -> () (* No casting needed *)
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
