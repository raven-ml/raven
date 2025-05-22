(* BEGIN GENERATED OCAML CODE *)
(* Assumed open statements: *)
(* open Bigarray *)
(* module Dtype = Nx_core.Dtype *)
(* open Nx_core.View *)
(* open Internal *)
(* (* Complex module may also be needed *) *)

(* Specific (non-identity) Casting Functions *)
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
    done


let cast_f16_to_i32 (src : (float, float16_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int32.of_float src_val0;
      Array1.unsafe_set dst_buf i1 Int32.of_float src_val1;
      Array1.unsafe_set dst_buf i2 Int32.of_float src_val2;
      Array1.unsafe_set dst_buf i3 Int32.of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.of_float src_val
    done


let cast_f16_to_i64 (src : (float, float16_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.of_float src_val0;
      Array1.unsafe_set dst_buf i1 Int64.of_float src_val1;
      Array1.unsafe_set dst_buf i2 Int64.of_float src_val2;
      Array1.unsafe_set dst_buf i3 Int64.of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_float src_val
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


let cast_f16_to_int (src : (float, float16_elt) t)
    (dst : (int, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_float src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.of_float src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.of_float src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
    done


let cast_f32_to_i32 (src : (float, float32_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int32.of_float src_val0;
      Array1.unsafe_set dst_buf i1 Int32.of_float src_val1;
      Array1.unsafe_set dst_buf i2 Int32.of_float src_val2;
      Array1.unsafe_set dst_buf i3 Int32.of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.of_float src_val
    done


let cast_f32_to_i64 (src : (float, float32_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.of_float src_val0;
      Array1.unsafe_set dst_buf i1 Int64.of_float src_val1;
      Array1.unsafe_set dst_buf i2 Int64.of_float src_val2;
      Array1.unsafe_set dst_buf i3 Int64.of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_float src_val
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


let cast_f32_to_int (src : (float, float32_elt) t)
    (dst : (int, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_float src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.of_float src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.of_float src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
    done


let cast_f64_to_i32 (src : (float, float64_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int32.of_float src_val0;
      Array1.unsafe_set dst_buf i1 Int32.of_float src_val1;
      Array1.unsafe_set dst_buf i2 Int32.of_float src_val2;
      Array1.unsafe_set dst_buf i3 Int32.of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.of_float src_val
    done


let cast_f64_to_i64 (src : (float, float64_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.of_float src_val0;
      Array1.unsafe_set dst_buf i1 Int64.of_float src_val1;
      Array1.unsafe_set dst_buf i2 Int64.of_float src_val2;
      Array1.unsafe_set dst_buf i3 Int64.of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_float src_val
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


let cast_f64_to_int (src : (float, float64_elt) t)
    (dst : (int, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 int_of_float src_val0;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_float src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.of_float src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.of_float src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.of_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_float src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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


let cast_i8_to_i32 (src : (int, int8_signed_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int32.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Int32.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Int32.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Int32.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.of_int src_val
    done


let cast_i8_to_i64 (src : (int, int8_signed_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Int64.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Int64.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Int64.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_int src_val
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
      Array1.unsafe_set dst_buf i0 { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = float_of_int src_val; im = 0.0 }
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
      Array1.unsafe_set dst_buf i0 { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = float_of_int src_val; im = 0.0 }
    done


let cast_i8_to_int (src : (int, int8_signed_elt) t)
    (dst : (int, nativeint_elt) t) start_idx end_idx =
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 Int32.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Int32.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Int32.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Int32.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.of_int src_val
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
      Array1.unsafe_set dst_buf i0 Int64.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Int64.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Int64.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Int64.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_int src_val
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
      Array1.unsafe_set dst_buf i0 { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = float_of_int src_val; im = 0.0 }
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
      Array1.unsafe_set dst_buf i0 { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = float_of_int src_val; im = 0.0 }
    done


let cast_u8_to_int (src : (int, int8_unsigned_elt) t)
    (dst : (int, nativeint_elt) t) start_idx end_idx =
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 Int32.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Int32.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Int32.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Int32.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.of_int src_val
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
      Array1.unsafe_set dst_buf i0 Int64.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Int64.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Int64.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Int64.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_int src_val
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
      Array1.unsafe_set dst_buf i0 { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = float_of_int src_val; im = 0.0 }
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
      Array1.unsafe_set dst_buf i0 { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = float_of_int src_val; im = 0.0 }
    done


let cast_i16_to_int (src : (int, int16_signed_elt) t)
    (dst : (int, nativeint_elt) t) start_idx end_idx =
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
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
      Array1.unsafe_set dst_buf i0 Int32.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Int32.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Int32.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Int32.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.of_int src_val
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
      Array1.unsafe_set dst_buf i0 Int64.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Int64.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Int64.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Int64.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_int src_val
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
      Array1.unsafe_set dst_buf i0 { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = float_of_int src_val; im = 0.0 }
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
      Array1.unsafe_set dst_buf i0 { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = float_of_int src_val; im = 0.0 }
    done


let cast_u16_to_int (src : (int, int16_unsigned_elt) t)
    (dst : (int, nativeint_elt) t) start_idx end_idx =
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_int src_val
    done


let cast_i32_to_f16 (src : (int32, int32_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int32.to_float src_val0;
      Array1.unsafe_set dst_buf i1 Int32.to_float src_val1;
      Array1.unsafe_set dst_buf i2 Int32.to_float src_val2;
      Array1.unsafe_set dst_buf i3 Int32.to_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.to_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.to_float src_val
    done


let cast_i32_to_f32 (src : (int32, int32_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int32.to_float src_val0;
      Array1.unsafe_set dst_buf i1 Int32.to_float src_val1;
      Array1.unsafe_set dst_buf i2 Int32.to_float src_val2;
      Array1.unsafe_set dst_buf i3 Int32.to_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.to_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.to_float src_val
    done


let cast_i32_to_f64 (src : (int32, int32_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int32.to_float src_val0;
      Array1.unsafe_set dst_buf i1 Int32.to_float src_val1;
      Array1.unsafe_set dst_buf i2 Int32.to_float src_val2;
      Array1.unsafe_set dst_buf i3 Int32.to_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.to_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.to_float src_val
    done


let cast_i32_to_i8 (src : (int32, int32_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int32.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Int32.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Int32.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Int32.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Int32.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Int32.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Int32.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Int32.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Int32.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Int32.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Int32.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Int32.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Int32.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Int32.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Int32.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Int32.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.to_int src_val
    done


let cast_i32_to_i64 (src : (int32, int32_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.of_int32 src_val0;
      Array1.unsafe_set dst_buf i1 Int64.of_int32 src_val1;
      Array1.unsafe_set dst_buf i2 Int64.of_int32 src_val2;
      Array1.unsafe_set dst_buf i3 Int64.of_int32 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_int32 src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_int32 src_val
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
      Array1.unsafe_set dst_buf i0 { Complex.re = Int32.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = Int32.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = Int32.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = Int32.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = Int32.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = Int32.to_float src_val; im = 0.0 }
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
      Array1.unsafe_set dst_buf i0 { Complex.re = Int32.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = Int32.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = Int32.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = Int32.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = Int32.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = Int32.to_float src_val; im = 0.0 }
    done


let cast_i32_to_int (src : (int32, int32_elt) t)
    (dst : (int, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 Int32.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Int32.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Int32.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Int32.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_int32 src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.of_int32 src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.of_int32 src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.of_int32 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_int32 src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_int32 src_val
    done


let cast_i64_to_f16 (src : (int64, int64_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.to_float src_val0;
      Array1.unsafe_set dst_buf i1 Int64.to_float src_val1;
      Array1.unsafe_set dst_buf i2 Int64.to_float src_val2;
      Array1.unsafe_set dst_buf i3 Int64.to_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.to_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.to_float src_val
    done


let cast_i64_to_f32 (src : (int64, int64_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.to_float src_val0;
      Array1.unsafe_set dst_buf i1 Int64.to_float src_val1;
      Array1.unsafe_set dst_buf i2 Int64.to_float src_val2;
      Array1.unsafe_set dst_buf i3 Int64.to_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.to_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.to_float src_val
    done


let cast_i64_to_f64 (src : (int64, int64_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.to_float src_val0;
      Array1.unsafe_set dst_buf i1 Int64.to_float src_val1;
      Array1.unsafe_set dst_buf i2 Int64.to_float src_val2;
      Array1.unsafe_set dst_buf i3 Int64.to_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.to_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.to_float src_val
    done


let cast_i64_to_i8 (src : (int64, int64_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Int64.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Int64.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Int64.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Int64.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Int64.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Int64.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Int64.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Int64.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Int64.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Int64.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Int64.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Int64.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Int64.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Int64.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Int64.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.to_int src_val
    done


let cast_i64_to_i32 (src : (int64, int64_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.to_int32 src_val0;
      Array1.unsafe_set dst_buf i1 Int64.to_int32 src_val1;
      Array1.unsafe_set dst_buf i2 Int64.to_int32 src_val2;
      Array1.unsafe_set dst_buf i3 Int64.to_int32 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.to_int32 src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.to_int32 src_val
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
      Array1.unsafe_set dst_buf i0 { Complex.re = Int64.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = Int64.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = Int64.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = Int64.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = Int64.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = Int64.to_float src_val; im = 0.0 }
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
      Array1.unsafe_set dst_buf i0 { Complex.re = Int64.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = Int64.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = Int64.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = Int64.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = Int64.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = Int64.to_float src_val; im = 0.0 }
    done


let cast_i64_to_int (src : (int64, int64_elt) t)
    (dst : (int, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 Int64.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Int64.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Int64.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Int64.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_int64 src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.of_int64 src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.of_int64 src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.of_int64 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_int64 src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_int64 src_val
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 Int32.of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 Int32.of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 Int32.of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 Int32.of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 Int64.of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 Int64.of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 Int64.of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 Int64.of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_float src_val.Complex.re
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
    (dst : (int, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 int_of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 Nativeint.of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 Nativeint.of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 Nativeint.of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 int_of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 Int32.of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 Int32.of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 Int32.of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 Int32.of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 Int64.of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 Int64.of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 Int64.of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 Int64.of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_float src_val.Complex.re
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
    (dst : (int, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 int_of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 int_of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 int_of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 int_of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx int_of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k int_of_float src_val.Complex.re
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_float src_val0.Complex.re;
      Array1.unsafe_set dst_buf i1 Nativeint.of_float src_val1.Complex.re;
      Array1.unsafe_set dst_buf i2 Nativeint.of_float src_val2.Complex.re;
      Array1.unsafe_set dst_buf i3 Nativeint.of_float src_val3.Complex.re;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_float src_val.Complex.re;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_float src_val.Complex.re
    done


let cast_int_to_f16 (src : (int, nativeint_elt) t)
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
    done


let cast_int_to_f32 (src : (int, nativeint_elt) t)
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
    done


let cast_int_to_f64 (src : (int, nativeint_elt) t)
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
      Array1.unsafe_set dst_buf i0 float_of_int src_val0;
      Array1.unsafe_set dst_buf i1 float_of_int src_val1;
      Array1.unsafe_set dst_buf i2 float_of_int src_val2;
      Array1.unsafe_set dst_buf i3 float_of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx float_of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k float_of_int src_val
    done


let cast_int_to_i8 (src : (int, nativeint_elt) t)
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


let cast_int_to_u8 (src : (int, nativeint_elt) t)
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


let cast_int_to_i16 (src : (int, nativeint_elt) t)
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


let cast_int_to_u16 (src : (int, nativeint_elt) t)
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


let cast_int_to_i32 (src : (int, nativeint_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int32.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Int32.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Int32.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Int32.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int32.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int32.of_int src_val
    done


let cast_int_to_i64 (src : (int, nativeint_elt) t)
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
      Array1.unsafe_set dst_buf i0 Int64.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Int64.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Int64.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Int64.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Int64.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Int64.of_int src_val
    done


let cast_int_to_c32 (src : (int, nativeint_elt) t)
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
      Array1.unsafe_set dst_buf i0 { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = float_of_int src_val; im = 0.0 }
    done


let cast_int_to_c64 (src : (int, nativeint_elt) t)
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
      Array1.unsafe_set dst_buf i0 { Complex.re = float_of_int src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = float_of_int src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = float_of_int src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = float_of_int src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = float_of_int src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = float_of_int src_val; im = 0.0 }
    done


let cast_int_to_nativeint (src : (int, nativeint_elt) t)
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
      Array1.unsafe_set dst_buf i0 Nativeint.of_int src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.of_int src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.of_int src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.of_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.of_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.of_int src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.to_float src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.to_float src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.to_float src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.to_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.to_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.to_float src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.to_float src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.to_float src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.to_float src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.to_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.to_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.to_float src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.to_float src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.to_float src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.to_float src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.to_float src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.to_float src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.to_float src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.to_int src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.to_int32 src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.to_int32 src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.to_int32 src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.to_int32 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.to_int32 src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.to_int32 src_val
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
      Array1.unsafe_set dst_buf i0 Nativeint.to_int64 src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.to_int64 src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.to_int64 src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.to_int64 src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.to_int64 src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.to_int64 src_val
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
      Array1.unsafe_set dst_buf i0 { Complex.re = Nativeint.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = Nativeint.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = Nativeint.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = Nativeint.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = Nativeint.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = Nativeint.to_float src_val; im = 0.0 }
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
      Array1.unsafe_set dst_buf i0 { Complex.re = Nativeint.to_float src_val0; im = 0.0 };
      Array1.unsafe_set dst_buf i1 { Complex.re = Nativeint.to_float src_val1; im = 0.0 };
      Array1.unsafe_set dst_buf i2 { Complex.re = Nativeint.to_float src_val2; im = 0.0 };
      Array1.unsafe_set dst_buf i3 { Complex.re = Nativeint.to_float src_val3; im = 0.0 };
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx { Complex.re = Nativeint.to_float src_val; im = 0.0 };
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k { Complex.re = Nativeint.to_float src_val; im = 0.0 }
    done


let cast_nativeint_to_int (src : (nativeint, nativeint_elt) t)
    (dst : (int, nativeint_elt) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in
  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 Nativeint.to_int src_val0;
      Array1.unsafe_set dst_buf i1 Nativeint.to_int src_val1;
      Array1.unsafe_set dst_buf i2 Nativeint.to_int src_val2;
      Array1.unsafe_set dst_buf i3 Nativeint.to_int src_val3;
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx Nativeint.to_int src_val;
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k Nativeint.to_int src_val
    done


(* Total specific (non-identity) cast functions generated: 156 *)


(* Generated cast_kernel Dispatch Function *)
let cast_kernel (type a b c d) (src_dtype : (a, b) Dtype.t)
    (dst_dtype : (c, d) Dtype.t)
    : ((a,b) t -> (c,d) t -> int -> int -> unit) =
  match (src_dtype, dst_dtype) with
  (* Float16 Source *)
  | Float16, Float16 ->
      (* Identity DType: Float16 to Float16. *)
      (* Specific kernel cast_f16_to_f16 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | Float16, Float32 -> Obj.magic cast_f16_to_f32
  | Float16, Float64 -> Obj.magic cast_f16_to_f64
  | Float16, Int8 -> Obj.magic cast_f16_to_i8
  | Float16, UInt8 -> Obj.magic cast_f16_to_u8
  | Float16, Int16 -> Obj.magic cast_f16_to_i16
  | Float16, UInt16 -> Obj.magic cast_f16_to_u16
  | Float16, Int32 -> Obj.magic cast_f16_to_i32
  | Float16, Int64 -> Obj.magic cast_f16_to_i64
  | Float16, Complex32 -> Obj.magic cast_f16_to_c32
  | Float16, Complex64 -> Obj.magic cast_f16_to_c64
  | Float16, Int -> Obj.magic cast_f16_to_int
  | Float16, NativeInt -> Obj.magic cast_f16_to_nativeint
  (* Float32 Source *)
  | Float32, Float16 -> Obj.magic cast_f32_to_f16
  | Float32, Float32 ->
      (* Identity DType: Float32 to Float32. *)
      (* Specific kernel cast_f32_to_f32 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | Float32, Float64 -> Obj.magic cast_f32_to_f64
  | Float32, Int8 -> Obj.magic cast_f32_to_i8
  | Float32, UInt8 -> Obj.magic cast_f32_to_u8
  | Float32, Int16 -> Obj.magic cast_f32_to_i16
  | Float32, UInt16 -> Obj.magic cast_f32_to_u16
  | Float32, Int32 -> Obj.magic cast_f32_to_i32
  | Float32, Int64 -> Obj.magic cast_f32_to_i64
  | Float32, Complex32 -> Obj.magic cast_f32_to_c32
  | Float32, Complex64 -> Obj.magic cast_f32_to_c64
  | Float32, Int -> Obj.magic cast_f32_to_int
  | Float32, NativeInt -> Obj.magic cast_f32_to_nativeint
  (* Float64 Source *)
  | Float64, Float16 -> Obj.magic cast_f64_to_f16
  | Float64, Float32 -> Obj.magic cast_f64_to_f32
  | Float64, Float64 ->
      (* Identity DType: Float64 to Float64. *)
      (* Specific kernel cast_f64_to_f64 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | Float64, Int8 -> Obj.magic cast_f64_to_i8
  | Float64, UInt8 -> Obj.magic cast_f64_to_u8
  | Float64, Int16 -> Obj.magic cast_f64_to_i16
  | Float64, UInt16 -> Obj.magic cast_f64_to_u16
  | Float64, Int32 -> Obj.magic cast_f64_to_i32
  | Float64, Int64 -> Obj.magic cast_f64_to_i64
  | Float64, Complex32 -> Obj.magic cast_f64_to_c32
  | Float64, Complex64 -> Obj.magic cast_f64_to_c64
  | Float64, Int -> Obj.magic cast_f64_to_int
  | Float64, NativeInt -> Obj.magic cast_f64_to_nativeint
  (* Int8 Source *)
  | Int8, Float16 -> Obj.magic cast_i8_to_f16
  | Int8, Float32 -> Obj.magic cast_i8_to_f32
  | Int8, Float64 -> Obj.magic cast_i8_to_f64
  | Int8, Int8 ->
      (* Identity DType: Int8 to Int8. *)
      (* Specific kernel cast_i8_to_i8 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | Int8, UInt8 -> Obj.magic cast_i8_to_u8
  | Int8, Int16 -> Obj.magic cast_i8_to_i16
  | Int8, UInt16 -> Obj.magic cast_i8_to_u16
  | Int8, Int32 -> Obj.magic cast_i8_to_i32
  | Int8, Int64 -> Obj.magic cast_i8_to_i64
  | Int8, Complex32 -> Obj.magic cast_i8_to_c32
  | Int8, Complex64 -> Obj.magic cast_i8_to_c64
  | Int8, Int -> Obj.magic cast_i8_to_int
  | Int8, NativeInt -> Obj.magic cast_i8_to_nativeint
  (* UInt8 Source *)
  | UInt8, Float16 -> Obj.magic cast_u8_to_f16
  | UInt8, Float32 -> Obj.magic cast_u8_to_f32
  | UInt8, Float64 -> Obj.magic cast_u8_to_f64
  | UInt8, Int8 -> Obj.magic cast_u8_to_i8
  | UInt8, UInt8 ->
      (* Identity DType: UInt8 to UInt8. *)
      (* Specific kernel cast_u8_to_u8 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | UInt8, Int16 -> Obj.magic cast_u8_to_i16
  | UInt8, UInt16 -> Obj.magic cast_u8_to_u16
  | UInt8, Int32 -> Obj.magic cast_u8_to_i32
  | UInt8, Int64 -> Obj.magic cast_u8_to_i64
  | UInt8, Complex32 -> Obj.magic cast_u8_to_c32
  | UInt8, Complex64 -> Obj.magic cast_u8_to_c64
  | UInt8, Int -> Obj.magic cast_u8_to_int
  | UInt8, NativeInt -> Obj.magic cast_u8_to_nativeint
  (* Int16 Source *)
  | Int16, Float16 -> Obj.magic cast_i16_to_f16
  | Int16, Float32 -> Obj.magic cast_i16_to_f32
  | Int16, Float64 -> Obj.magic cast_i16_to_f64
  | Int16, Int8 -> Obj.magic cast_i16_to_i8
  | Int16, UInt8 -> Obj.magic cast_i16_to_u8
  | Int16, Int16 ->
      (* Identity DType: Int16 to Int16. *)
      (* Specific kernel cast_i16_to_i16 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | Int16, UInt16 -> Obj.magic cast_i16_to_u16
  | Int16, Int32 -> Obj.magic cast_i16_to_i32
  | Int16, Int64 -> Obj.magic cast_i16_to_i64
  | Int16, Complex32 -> Obj.magic cast_i16_to_c32
  | Int16, Complex64 -> Obj.magic cast_i16_to_c64
  | Int16, Int -> Obj.magic cast_i16_to_int
  | Int16, NativeInt -> Obj.magic cast_i16_to_nativeint
  (* UInt16 Source *)
  | UInt16, Float16 -> Obj.magic cast_u16_to_f16
  | UInt16, Float32 -> Obj.magic cast_u16_to_f32
  | UInt16, Float64 -> Obj.magic cast_u16_to_f64
  | UInt16, Int8 -> Obj.magic cast_u16_to_i8
  | UInt16, UInt8 -> Obj.magic cast_u16_to_u8
  | UInt16, Int16 -> Obj.magic cast_u16_to_i16
  | UInt16, UInt16 ->
      (* Identity DType: UInt16 to UInt16. *)
      (* Specific kernel cast_u16_to_u16 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | UInt16, Int32 -> Obj.magic cast_u16_to_i32
  | UInt16, Int64 -> Obj.magic cast_u16_to_i64
  | UInt16, Complex32 -> Obj.magic cast_u16_to_c32
  | UInt16, Complex64 -> Obj.magic cast_u16_to_c64
  | UInt16, Int -> Obj.magic cast_u16_to_int
  | UInt16, NativeInt -> Obj.magic cast_u16_to_nativeint
  (* Int32 Source *)
  | Int32, Float16 -> Obj.magic cast_i32_to_f16
  | Int32, Float32 -> Obj.magic cast_i32_to_f32
  | Int32, Float64 -> Obj.magic cast_i32_to_f64
  | Int32, Int8 -> Obj.magic cast_i32_to_i8
  | Int32, UInt8 -> Obj.magic cast_i32_to_u8
  | Int32, Int16 -> Obj.magic cast_i32_to_i16
  | Int32, UInt16 -> Obj.magic cast_i32_to_u16
  | Int32, Int32 ->
      (* Identity DType: Int32 to Int32. *)
      (* Specific kernel cast_i32_to_i32 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | Int32, Int64 -> Obj.magic cast_i32_to_i64
  | Int32, Complex32 -> Obj.magic cast_i32_to_c32
  | Int32, Complex64 -> Obj.magic cast_i32_to_c64
  | Int32, Int -> Obj.magic cast_i32_to_int
  | Int32, NativeInt -> Obj.magic cast_i32_to_nativeint
  (* Int64 Source *)
  | Int64, Float16 -> Obj.magic cast_i64_to_f16
  | Int64, Float32 -> Obj.magic cast_i64_to_f32
  | Int64, Float64 -> Obj.magic cast_i64_to_f64
  | Int64, Int8 -> Obj.magic cast_i64_to_i8
  | Int64, UInt8 -> Obj.magic cast_i64_to_u8
  | Int64, Int16 -> Obj.magic cast_i64_to_i16
  | Int64, UInt16 -> Obj.magic cast_i64_to_u16
  | Int64, Int32 -> Obj.magic cast_i64_to_i32
  | Int64, Int64 ->
      (* Identity DType: Int64 to Int64. *)
      (* Specific kernel cast_i64_to_i64 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | Int64, Complex32 -> Obj.magic cast_i64_to_c32
  | Int64, Complex64 -> Obj.magic cast_i64_to_c64
  | Int64, Int -> Obj.magic cast_i64_to_int
  | Int64, NativeInt -> Obj.magic cast_i64_to_nativeint
  (* Complex32 Source *)
  | Complex32, Float16 -> Obj.magic cast_c32_to_f16
  | Complex32, Float32 -> Obj.magic cast_c32_to_f32
  | Complex32, Float64 -> Obj.magic cast_c32_to_f64
  | Complex32, Int8 -> Obj.magic cast_c32_to_i8
  | Complex32, UInt8 -> Obj.magic cast_c32_to_u8
  | Complex32, Int16 -> Obj.magic cast_c32_to_i16
  | Complex32, UInt16 -> Obj.magic cast_c32_to_u16
  | Complex32, Int32 -> Obj.magic cast_c32_to_i32
  | Complex32, Int64 -> Obj.magic cast_c32_to_i64
  | Complex32, Complex32 ->
      (* Identity DType: Complex32 to Complex32. *)
      (* Specific kernel cast_c32_to_c32 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | Complex32, Complex64 -> Obj.magic cast_c32_to_c64
  | Complex32, Int -> Obj.magic cast_c32_to_int
  | Complex32, NativeInt -> Obj.magic cast_c32_to_nativeint
  (* Complex64 Source *)
  | Complex64, Float16 -> Obj.magic cast_c64_to_f16
  | Complex64, Float32 -> Obj.magic cast_c64_to_f32
  | Complex64, Float64 -> Obj.magic cast_c64_to_f64
  | Complex64, Int8 -> Obj.magic cast_c64_to_i8
  | Complex64, UInt8 -> Obj.magic cast_c64_to_u8
  | Complex64, Int16 -> Obj.magic cast_c64_to_i16
  | Complex64, UInt16 -> Obj.magic cast_c64_to_u16
  | Complex64, Int32 -> Obj.magic cast_c64_to_i32
  | Complex64, Int64 -> Obj.magic cast_c64_to_i64
  | Complex64, Complex32 -> Obj.magic cast_c64_to_c32
  | Complex64, Complex64 ->
      (* Identity DType: Complex64 to Complex64. *)
      (* Specific kernel cast_c64_to_c64 was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | Complex64, Int -> Obj.magic cast_c64_to_int
  | Complex64, NativeInt -> Obj.magic cast_c64_to_nativeint
  (* Int Source *)
  | Int, Float16 -> Obj.magic cast_int_to_f16
  | Int, Float32 -> Obj.magic cast_int_to_f32
  | Int, Float64 -> Obj.magic cast_int_to_f64
  | Int, Int8 -> Obj.magic cast_int_to_i8
  | Int, UInt8 -> Obj.magic cast_int_to_u8
  | Int, Int16 -> Obj.magic cast_int_to_i16
  | Int, UInt16 -> Obj.magic cast_int_to_u16
  | Int, Int32 -> Obj.magic cast_int_to_i32
  | Int, Int64 -> Obj.magic cast_int_to_i64
  | Int, Complex32 -> Obj.magic cast_int_to_c32
  | Int, Complex64 -> Obj.magic cast_int_to_c64
  | Int, Int ->
      (* Identity DType: Int to Int. *)
      (* Specific kernel cast_int_to_int was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | Int, NativeInt -> Obj.magic cast_int_to_nativeint
  (* NativeInt Source *)
  | NativeInt, Float16 -> Obj.magic cast_nativeint_to_f16
  | NativeInt, Float32 -> Obj.magic cast_nativeint_to_f32
  | NativeInt, Float64 -> Obj.magic cast_nativeint_to_f64
  | NativeInt, Int8 -> Obj.magic cast_nativeint_to_i8
  | NativeInt, UInt8 -> Obj.magic cast_nativeint_to_u8
  | NativeInt, Int16 -> Obj.magic cast_nativeint_to_i16
  | NativeInt, UInt16 -> Obj.magic cast_nativeint_to_u16
  | NativeInt, Int32 -> Obj.magic cast_nativeint_to_i32
  | NativeInt, Int64 -> Obj.magic cast_nativeint_to_i64
  | NativeInt, Complex32 -> Obj.magic cast_nativeint_to_c32
  | NativeInt, Complex64 -> Obj.magic cast_nativeint_to_c64
  | NativeInt, Int -> Obj.magic cast_nativeint_to_int
  | NativeInt, NativeInt ->
      (* Identity DType: NativeInt to NativeInt. *)
      (* Specific kernel cast_nativeint_to_nativeint was not generated. *)
      (* This should be handled by a copy operation or a generic identity kernel. *)
      fun _ _ _ _ -> failwith ("Internal: Identity cast kernel for " ^ Dtype.to_string src_dtype ^ " should be pre-empted or use a generic copy.")
  | _s, _d ->
      failwith
        ("cast_kernel: BUG or Incomplete - unsupported dtype combination from " ^ Dtype.to_string src_dtype
       ^ " to " ^ Dtype.to_string dst_dtype)


(* END GENERATED OCAML CODE *)
