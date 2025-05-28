open Bigarray
open Nx_core.Dtype
open Nx_core.View
open Internal

let complex_sin (z : Complex.t) =
  let a, b = (z.re, z.im) in
  { Complex.re = sin a *. cosh b; im = cos a *. sinh b }

let ln2 = Stdlib.log 2.0

let complex_log2 (z : Complex.t) =
  let a, b = (z.re, z.im) in
  {
    Complex.re = 0.5 *. Stdlib.log ((a *. a) +. (b *. b)) /. ln2;
    im = Stdlib.atan2 b a /. ln2;
  }

let complex_exp2 (z : Complex.t) =
  let a, b = (z.re, z.im) in
  let e = Stdlib.exp (a *. ln2) in
  { Complex.re = e *. Stdlib.cos (b *. ln2); im = e *. Stdlib.sin (b *. ln2) }

let kernel_neg_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.neg v0);
      Array1.unsafe_set out_buf i1 (Float.neg v1);
      Array1.unsafe_set out_buf i2 (Float.neg v2);
      Array1.unsafe_set out_buf i3 (Float.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.neg v)
    done

let kernel_neg_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.neg v0);
      Array1.unsafe_set out_buf i1 (Float.neg v1);
      Array1.unsafe_set out_buf i2 (Float.neg v2);
      Array1.unsafe_set out_buf i3 (Float.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.neg v)
    done

let kernel_neg_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.neg v0);
      Array1.unsafe_set out_buf i1 (Float.neg v1);
      Array1.unsafe_set out_buf i2 (Float.neg v2);
      Array1.unsafe_set out_buf i3 (Float.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.neg v)
    done

let kernel_neg_int8 (a : (int, int8_elt) t) (out : (int, int8_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.neg v0);
      Array1.unsafe_set out_buf i1 (Int.neg v1);
      Array1.unsafe_set out_buf i2 (Int.neg v2);
      Array1.unsafe_set out_buf i3 (Int.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.neg v)
    done

let kernel_neg_uint8 (a : (int, uint8_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.neg v0);
      Array1.unsafe_set out_buf i1 (Int.neg v1);
      Array1.unsafe_set out_buf i2 (Int.neg v2);
      Array1.unsafe_set out_buf i3 (Int.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.neg v)
    done

let kernel_neg_int16 (a : (int, int16_elt) t) (out : (int, int16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.neg v0);
      Array1.unsafe_set out_buf i1 (Int.neg v1);
      Array1.unsafe_set out_buf i2 (Int.neg v2);
      Array1.unsafe_set out_buf i3 (Int.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.neg v)
    done

let kernel_neg_uint16 (a : (int, uint16_elt) t) (out : (int, uint16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.neg v0);
      Array1.unsafe_set out_buf i1 (Int.neg v1);
      Array1.unsafe_set out_buf i2 (Int.neg v2);
      Array1.unsafe_set out_buf i3 (Int.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.neg v)
    done

let kernel_neg_int32 (a : (int32, int32_elt) t) (out : (int32, int32_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int32.neg v0);
      Array1.unsafe_set out_buf i1 (Int32.neg v1);
      Array1.unsafe_set out_buf i2 (Int32.neg v2);
      Array1.unsafe_set out_buf i3 (Int32.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int32.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int32.neg v)
    done

let kernel_neg_int64 (a : (int64, int64_elt) t) (out : (int64, int64_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int64.neg v0);
      Array1.unsafe_set out_buf i1 (Int64.neg v1);
      Array1.unsafe_set out_buf i2 (Int64.neg v2);
      Array1.unsafe_set out_buf i3 (Int64.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int64.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int64.neg v)
    done

let kernel_neg_int (a : (int, int_elt) t) (out : (int, int_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.neg v0);
      Array1.unsafe_set out_buf i1 (Int.neg v1);
      Array1.unsafe_set out_buf i2 (Int.neg v2);
      Array1.unsafe_set out_buf i3 (Int.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.neg v)
    done

let kernel_neg_nativeint (a : (nativeint, nativeint_elt) t)
    (out : (nativeint, nativeint_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.neg v0);
      Array1.unsafe_set out_buf i1 (Nativeint.neg v1);
      Array1.unsafe_set out_buf i2 (Nativeint.neg v2);
      Array1.unsafe_set out_buf i3 (Nativeint.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Nativeint.neg v)
    done

let kernel_neg_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex.neg v0);
      Array1.unsafe_set out_buf i1 (Complex.neg v1);
      Array1.unsafe_set out_buf i2 (Complex.neg v2);
      Array1.unsafe_set out_buf i3 (Complex.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.neg v)
    done

let kernel_neg_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex.neg v0);
      Array1.unsafe_set out_buf i1 (Complex.neg v1);
      Array1.unsafe_set out_buf i2 (Complex.neg v2);
      Array1.unsafe_set out_buf i3 (Complex.neg v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex.neg v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.neg v)
    done

let kernel_sqrt_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.sqrt v0);
      Array1.unsafe_set out_buf i1 (Float.sqrt v1);
      Array1.unsafe_set out_buf i2 (Float.sqrt v2);
      Array1.unsafe_set out_buf i3 (Float.sqrt v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.sqrt v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sqrt v)
    done

let kernel_sqrt_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.sqrt v0);
      Array1.unsafe_set out_buf i1 (Float.sqrt v1);
      Array1.unsafe_set out_buf i2 (Float.sqrt v2);
      Array1.unsafe_set out_buf i3 (Float.sqrt v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.sqrt v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sqrt v)
    done

let kernel_sqrt_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.sqrt v0);
      Array1.unsafe_set out_buf i1 (Float.sqrt v1);
      Array1.unsafe_set out_buf i2 (Float.sqrt v2);
      Array1.unsafe_set out_buf i3 (Float.sqrt v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.sqrt v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sqrt v)
    done

let kernel_sqrt_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex.sqrt v0);
      Array1.unsafe_set out_buf i1 (Complex.sqrt v1);
      Array1.unsafe_set out_buf i2 (Complex.sqrt v2);
      Array1.unsafe_set out_buf i3 (Complex.sqrt v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex.sqrt v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.sqrt v)
    done

let kernel_sqrt_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex.sqrt v0);
      Array1.unsafe_set out_buf i1 (Complex.sqrt v1);
      Array1.unsafe_set out_buf i2 (Complex.sqrt v2);
      Array1.unsafe_set out_buf i3 (Complex.sqrt v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex.sqrt v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.sqrt v)
    done

let kernel_recip_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Bigarray.Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Bigarray.Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Bigarray.Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Bigarray.Array1.unsafe_get a_buf (offset a + i3) in
      Bigarray.Array1.unsafe_set out_buf i0 (1.0 /. v0);
      Bigarray.Array1.unsafe_set out_buf i1 (1.0 /. v1);
      Bigarray.Array1.unsafe_set out_buf i2 (1.0 /. v2);
      Bigarray.Array1.unsafe_set out_buf i3 (1.0 /. v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Bigarray.Array1.unsafe_get a_buf (offset a + idx) in
      Bigarray.Array1.unsafe_set out_buf idx (1.0 /. v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin_offset_in_a_data = index_to_offset md_index (strides a) in
      let v =
        Bigarray.Array1.unsafe_get a_buf (offset a + a_lin_offset_in_a_data)
      in
      Bigarray.Array1.unsafe_set out_buf k (1.0 /. v)
    done

let kernel_recip_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Bigarray.Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Bigarray.Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Bigarray.Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Bigarray.Array1.unsafe_get a_buf (offset a + i3) in
      Bigarray.Array1.unsafe_set out_buf i0 (1.0 /. v0);
      Bigarray.Array1.unsafe_set out_buf i1 (1.0 /. v1);
      Bigarray.Array1.unsafe_set out_buf i2 (1.0 /. v2);
      Bigarray.Array1.unsafe_set out_buf i3 (1.0 /. v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Bigarray.Array1.unsafe_get a_buf (offset a + idx) in
      Bigarray.Array1.unsafe_set out_buf idx (1.0 /. v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin_offset_in_a_data = index_to_offset md_index (strides a) in
      let v =
        Bigarray.Array1.unsafe_get a_buf (offset a + a_lin_offset_in_a_data)
      in
      Bigarray.Array1.unsafe_set out_buf k (1.0 /. v)
    done

let kernel_recip_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Bigarray.Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Bigarray.Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Bigarray.Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Bigarray.Array1.unsafe_get a_buf (offset a + i3) in
      Bigarray.Array1.unsafe_set out_buf i0 (1.0 /. v0);
      Bigarray.Array1.unsafe_set out_buf i1 (1.0 /. v1);
      Bigarray.Array1.unsafe_set out_buf i2 (1.0 /. v2);
      Bigarray.Array1.unsafe_set out_buf i3 (1.0 /. v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Bigarray.Array1.unsafe_get a_buf (offset a + idx) in
      Bigarray.Array1.unsafe_set out_buf idx (1.0 /. v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin_offset_in_a_data = index_to_offset md_index (strides a) in
      let v =
        Bigarray.Array1.unsafe_get a_buf (offset a + a_lin_offset_in_a_data)
      in
      Bigarray.Array1.unsafe_set out_buf k (1.0 /. v)
    done

let kernel_recip_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Bigarray.Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Bigarray.Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Bigarray.Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Bigarray.Array1.unsafe_get a_buf (offset a + i3) in
      Bigarray.Array1.unsafe_set out_buf i0 (Complex.inv v0);
      Bigarray.Array1.unsafe_set out_buf i1 (Complex.inv v1);
      Bigarray.Array1.unsafe_set out_buf i2 (Complex.inv v2);
      Bigarray.Array1.unsafe_set out_buf i3 (Complex.inv v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Bigarray.Array1.unsafe_get a_buf (offset a + idx) in
      Bigarray.Array1.unsafe_set out_buf idx (Complex.inv v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Bigarray.Array1.unsafe_get a_buf (offset a + a_lin) in
      Bigarray.Array1.unsafe_set out_buf k (Complex.inv v)
    done

let kernel_recip_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Bigarray.Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Bigarray.Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Bigarray.Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Bigarray.Array1.unsafe_get a_buf (offset a + i3) in
      Bigarray.Array1.unsafe_set out_buf i0 (Complex.inv v0);
      Bigarray.Array1.unsafe_set out_buf i1 (Complex.inv v1);
      Bigarray.Array1.unsafe_set out_buf i2 (Complex.inv v2);
      Bigarray.Array1.unsafe_set out_buf i3 (Complex.inv v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Bigarray.Array1.unsafe_get a_buf (offset a + idx) in
      Bigarray.Array1.unsafe_set out_buf idx (Complex.inv v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Bigarray.Array1.unsafe_get a_buf (offset a + a_lin) in
      Bigarray.Array1.unsafe_set out_buf k (Complex.inv v)
    done

let kernel_exp2_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.exp2 v0);
      Array1.unsafe_set out_buf i1 (Float.exp2 v1);
      Array1.unsafe_set out_buf i2 (Float.exp2 v2);
      Array1.unsafe_set out_buf i3 (Float.exp2 v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.exp2 v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.exp2 v)
    done

let kernel_exp2_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.exp2 v0);
      Array1.unsafe_set out_buf i1 (Float.exp2 v1);
      Array1.unsafe_set out_buf i2 (Float.exp2 v2);
      Array1.unsafe_set out_buf i3 (Float.exp2 v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.exp2 v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.exp2 v)
    done

let kernel_exp2_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.exp2 v0);
      Array1.unsafe_set out_buf i1 (Float.exp2 v1);
      Array1.unsafe_set out_buf i2 (Float.exp2 v2);
      Array1.unsafe_set out_buf i3 (Float.exp2 v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.exp2 v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.exp2 v)
    done

let kernel_exp2_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (complex_exp2 v0);
      Array1.unsafe_set out_buf i1 (complex_exp2 v1);
      Array1.unsafe_set out_buf i2 (complex_exp2 v2);
      Array1.unsafe_set out_buf i3 (complex_exp2 v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (complex_exp2 v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (complex_exp2 v)
    done

let kernel_exp2_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (complex_exp2 v0);
      Array1.unsafe_set out_buf i1 (complex_exp2 v1);
      Array1.unsafe_set out_buf i2 (complex_exp2 v2);
      Array1.unsafe_set out_buf i3 (complex_exp2 v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (complex_exp2 v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (complex_exp2 v)
    done

let kernel_log2_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.log2 v0);
      Array1.unsafe_set out_buf i1 (Float.log2 v1);
      Array1.unsafe_set out_buf i2 (Float.log2 v2);
      Array1.unsafe_set out_buf i3 (Float.log2 v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.log2 v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.log2 v)
    done

let kernel_log2_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.log2 v0);
      Array1.unsafe_set out_buf i1 (Float.log2 v1);
      Array1.unsafe_set out_buf i2 (Float.log2 v2);
      Array1.unsafe_set out_buf i3 (Float.log2 v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.log2 v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.log2 v)
    done

let kernel_log2_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.log2 v0);
      Array1.unsafe_set out_buf i1 (Float.log2 v1);
      Array1.unsafe_set out_buf i2 (Float.log2 v2);
      Array1.unsafe_set out_buf i3 (Float.log2 v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.log2 v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.log2 v)
    done

let kernel_log2_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (complex_log2 v0);
      Array1.unsafe_set out_buf i1 (complex_log2 v1);
      Array1.unsafe_set out_buf i2 (complex_log2 v2);
      Array1.unsafe_set out_buf i3 (complex_log2 v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (complex_log2 v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (complex_log2 v)
    done

let kernel_log2_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (complex_log2 v0);
      Array1.unsafe_set out_buf i1 (complex_log2 v1);
      Array1.unsafe_set out_buf i2 (complex_log2 v2);
      Array1.unsafe_set out_buf i3 (complex_log2 v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (complex_log2 v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (complex_log2 v)
    done

let kernel_sin_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.sin v0);
      Array1.unsafe_set out_buf i1 (Float.sin v1);
      Array1.unsafe_set out_buf i2 (Float.sin v2);
      Array1.unsafe_set out_buf i3 (Float.sin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.sin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sin v)
    done

let kernel_sin_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.sin v0);
      Array1.unsafe_set out_buf i1 (Float.sin v1);
      Array1.unsafe_set out_buf i2 (Float.sin v2);
      Array1.unsafe_set out_buf i3 (Float.sin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.sin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sin v)
    done

let kernel_sin_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.sin v0);
      Array1.unsafe_set out_buf i1 (Float.sin v1);
      Array1.unsafe_set out_buf i2 (Float.sin v2);
      Array1.unsafe_set out_buf i3 (Float.sin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.sin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sin v)
    done

let kernel_sin_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (complex_sin v0);
      Array1.unsafe_set out_buf i1 (complex_sin v1);
      Array1.unsafe_set out_buf i2 (complex_sin v2);
      Array1.unsafe_set out_buf i3 (complex_sin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (complex_sin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (complex_sin v)
    done

let kernel_sin_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_contiguous a && is_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (complex_sin v0);
      Array1.unsafe_set out_buf i1 (complex_sin v1);
      Array1.unsafe_set out_buf i2 (complex_sin v2);
      Array1.unsafe_set out_buf i3 (complex_sin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (complex_sin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (complex_sin v)
    done

let kernel_neg (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_neg_float16 a c start_idx end_idx
  | Float32 -> kernel_neg_float32 a c start_idx end_idx
  | Float64 -> kernel_neg_float64 a c start_idx end_idx
  | Complex32 -> kernel_neg_complex32 a c start_idx end_idx
  | Complex64 -> kernel_neg_complex64 a c start_idx end_idx
  | Int8_signed -> kernel_neg_int8 a c start_idx end_idx
  | Int8_unsigned -> kernel_neg_uint8 a c start_idx end_idx
  | Int16_signed -> kernel_neg_int16 a c start_idx end_idx
  | Int16_unsigned -> kernel_neg_uint16 a c start_idx end_idx
  | Int32 -> kernel_neg_int32 a c start_idx end_idx
  | Int64 -> kernel_neg_int64 a c start_idx end_idx
  | Int -> kernel_neg_int a c start_idx end_idx
  | Nativeint -> kernel_neg_nativeint a c start_idx end_idx
  | _ -> invalid_arg "kernel_neg: unsupported type"

let kernel_sqrt (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_sqrt_float16 a c start_idx end_idx
  | Float32 -> kernel_sqrt_float32 a c start_idx end_idx
  | Float64 -> kernel_sqrt_float64 a c start_idx end_idx
  | Complex32 -> kernel_sqrt_complex32 a c start_idx end_idx
  | Complex64 -> kernel_sqrt_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_sqrt: unsupported type"

let kernel_recip (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_recip_float16 a c start_idx end_idx
  | Float32 -> kernel_recip_float32 a c start_idx end_idx
  | Float64 -> kernel_recip_float64 a c start_idx end_idx
  | Complex32 -> kernel_recip_complex32 a c start_idx end_idx
  | Complex64 -> kernel_recip_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_recip: unsupported type"

let kernel_exp2 (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_exp2_float16 a c start_idx end_idx
  | Float32 -> kernel_exp2_float32 a c start_idx end_idx
  | Float64 -> kernel_exp2_float64 a c start_idx end_idx
  | Complex32 -> kernel_exp2_complex32 a c start_idx end_idx
  | Complex64 -> kernel_exp2_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_exp2: unsupported type"

let kernel_log2 (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_log2_float16 a c start_idx end_idx
  | Float32 -> kernel_log2_float32 a c start_idx end_idx
  | Float64 -> kernel_log2_float64 a c start_idx end_idx
  | Complex32 -> kernel_log2_complex32 a c start_idx end_idx
  | Complex64 -> kernel_log2_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_log: unsupported type"

let kernel_sin (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_sin_float16 a c start_idx end_idx
  | Float32 -> kernel_sin_float32 a c start_idx end_idx
  | Float64 -> kernel_sin_float64 a c start_idx end_idx
  | Complex32 -> kernel_sin_complex32 a c start_idx end_idx
  | Complex64 -> kernel_sin_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_sin: unsupported type"

let neg context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_neg a out start_idx end_idx)

let sqrt context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_sqrt a out start_idx end_idx)

let recip context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_recip a out start_idx end_idx)

let exp2 context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_exp2 a out start_idx end_idx)

let log2 context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_log2 a out start_idx end_idx)

let sin context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_sin a out start_idx end_idx)
