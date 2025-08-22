open Bigarray_ext
open Nx_core.Dtype
module Shape = Nx_core.Shape
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
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.neg v)
    done

let kernel_neg_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.neg v)
    done

let kernel_neg_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.neg v)
    done

let kernel_neg_int8 (a : (int, int8_elt) t) (out : (int, int8_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Int.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Int.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Int.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Int.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Int.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.neg v)
    done

let kernel_neg_uint8 (a : (int, uint8_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Int.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Int.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Int.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Int.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Int.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.neg v)
    done

let kernel_neg_int16 (a : (int, int16_elt) t) (out : (int, int16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Int.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Int.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Int.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Int.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Int.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.neg v)
    done

let kernel_neg_uint16 (a : (int, uint16_elt) t) (out : (int, uint16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Int.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Int.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Int.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Int.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Int.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.neg v)
    done

let kernel_neg_int32 (a : (int32, int32_elt) t) (out : (int32, int32_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Int32.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Int32.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Int32.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Int32.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Int32.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.neg v)
    done

let kernel_neg_int64 (a : (int64, int64_elt) t) (out : (int64, int64_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Int64.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Int64.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Int64.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Int64.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Int64.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.neg v)
    done

let kernel_neg_int (a : (int, int_elt) t) (out : (int, int_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Int.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Int.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Int.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Int.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Int.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.neg v)
    done

let kernel_neg_nativeint (a : (nativeint, nativeint_elt) t)
    (out : (nativeint, nativeint_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Nativeint.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Nativeint.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Nativeint.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Nativeint.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Nativeint.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.neg v)
    done

let kernel_neg_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Complex.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Complex.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Complex.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Complex.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Complex.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.neg v)
    done

let kernel_neg_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Complex.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Complex.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Complex.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Complex.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Complex.neg v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.neg v)
    done

(* Extended dtype kernel functions for neg *)
let kernel_neg_bfloat16 (a : (float, bfloat16_elt) t)
    (out : (float, bfloat16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.neg v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.neg v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.neg v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.neg v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.neg v);
      incr i
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (Float.neg v)
    done

let kernel_neg_bool (a : (bool, bool_elt) t)
    (out : (bool, bool_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (not v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (not v)
    done

let kernel_neg_int4 (a : (int, int4_signed_elt) t)
    (out : (int, int4_signed_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (-v0);
      Array1.unsafe_set out_buf (out_base + i1) (-v1);
      Array1.unsafe_set out_buf (out_base + i2) (-v2);
      Array1.unsafe_set out_buf (out_base + i3) (-v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (-v);
      incr i
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (-v)
    done

let kernel_neg_uint4 (a : (int, int4_unsigned_elt) t)
    (out : (int, int4_unsigned_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (-v0);
      Array1.unsafe_set out_buf (out_base + i1) (-v1);
      Array1.unsafe_set out_buf (out_base + i2) (-v2);
      Array1.unsafe_set out_buf (out_base + i3) (-v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (-v);
      incr i
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (-v)
    done

let kernel_neg_float8_e4m3 (a : (float, float8_e4m3_elt) t)
    (out : (float, float8_e4m3_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (Float.neg v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (Float.neg v)
    done

let kernel_neg_float8_e5m2 (a : (float, float8_e5m2_elt) t)
    (out : (float, float8_e5m2_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (Float.neg v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (Float.neg v)
    done

let kernel_neg_complex16 (a : (Complex.t, complex16_elt) t)
    (out : (Complex.t, complex16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (Complex.neg v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.neg v)
    done

let kernel_neg_qint8 (a : (int, qint8_elt) t)
    (out : (int, qint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (-v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (-v)
    done

let kernel_neg_quint8 (a : (int, quint8_elt) t)
    (out : (int, quint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (-v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (-v)
    done

let kernel_sqrt_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.sqrt v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.sqrt v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.sqrt v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.sqrt v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.sqrt v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sqrt v)
    done

let kernel_sqrt_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.sqrt v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.sqrt v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.sqrt v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.sqrt v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.sqrt v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sqrt v)
    done

let kernel_sqrt_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.sqrt v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.sqrt v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.sqrt v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.sqrt v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.sqrt v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sqrt v)
    done

let kernel_sqrt_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Complex.sqrt v0);
      Array1.unsafe_set out_buf (out_base + i1) (Complex.sqrt v1);
      Array1.unsafe_set out_buf (out_base + i2) (Complex.sqrt v2);
      Array1.unsafe_set out_buf (out_base + i3) (Complex.sqrt v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Complex.sqrt v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.sqrt v)
    done

let kernel_sqrt_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Complex.sqrt v0);
      Array1.unsafe_set out_buf (out_base + i1) (Complex.sqrt v1);
      Array1.unsafe_set out_buf (out_base + i2) (Complex.sqrt v2);
      Array1.unsafe_set out_buf (out_base + i3) (Complex.sqrt v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Complex.sqrt v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.sqrt v)
    done

(* Extended dtype kernel functions for sqrt *)
let kernel_sqrt_bfloat16 (a : (float, bfloat16_elt) t)
    (out : (float, bfloat16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.sqrt v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.sqrt v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.sqrt v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.sqrt v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.sqrt v);
      incr i
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sqrt v)
    done

let kernel_sqrt_float8_e4m3 (a : (float, float8_e4m3_elt) t)
    (out : (float, float8_e4m3_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (Float.sqrt v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sqrt v)
    done

let kernel_sqrt_float8_e5m2 (a : (float, float8_e5m2_elt) t)
    (out : (float, float8_e5m2_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (Float.sqrt v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sqrt v)
    done

let kernel_sqrt_complex16 (a : (Complex.t, complex16_elt) t)
    (out : (Complex.t, complex16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (Complex.sqrt v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.sqrt v)
    done

let kernel_recip_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (1.0 /. v0);
      Array1.unsafe_set out_buf (out_base + i1) (1.0 /. v1);
      Array1.unsafe_set out_buf (out_base + i2) (1.0 /. v2);
      Array1.unsafe_set out_buf (out_base + i3) (1.0 /. v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (1.0 /. v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin_offset_in_a_data = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin_offset_in_a_data) in
      Array1.unsafe_set out_buf (offset out + k) (1.0 /. v)
    done

let kernel_recip_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (1.0 /. v0);
      Array1.unsafe_set out_buf (out_base + i1) (1.0 /. v1);
      Array1.unsafe_set out_buf (out_base + i2) (1.0 /. v2);
      Array1.unsafe_set out_buf (out_base + i3) (1.0 /. v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (1.0 /. v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin_offset_in_a_data = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin_offset_in_a_data) in
      Array1.unsafe_set out_buf (offset out + k) (1.0 /. v)
    done

let kernel_recip_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (1.0 /. v0);
      Array1.unsafe_set out_buf (out_base + i1) (1.0 /. v1);
      Array1.unsafe_set out_buf (out_base + i2) (1.0 /. v2);
      Array1.unsafe_set out_buf (out_base + i3) (1.0 /. v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (1.0 /. v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin_offset_in_a_data = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin_offset_in_a_data) in
      Array1.unsafe_set out_buf (offset out + k) (1.0 /. v)
    done

let kernel_recip_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Complex.inv v0);
      Array1.unsafe_set out_buf (out_base + i1) (Complex.inv v1);
      Array1.unsafe_set out_buf (out_base + i2) (Complex.inv v2);
      Array1.unsafe_set out_buf (out_base + i3) (Complex.inv v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Complex.inv v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.inv v)
    done

let kernel_recip_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Complex.inv v0);
      Array1.unsafe_set out_buf (out_base + i1) (Complex.inv v1);
      Array1.unsafe_set out_buf (out_base + i2) (Complex.inv v2);
      Array1.unsafe_set out_buf (out_base + i3) (Complex.inv v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Complex.inv v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.inv v)
    done

(* Extended dtype kernel functions for recip *)
let kernel_recip_bfloat16 (a : (float, bfloat16_elt) t)
    (out : (float, bfloat16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (1.0 /. v0);
      Array1.unsafe_set out_buf (out_base + i1) (1.0 /. v1);
      Array1.unsafe_set out_buf (out_base + i2) (1.0 /. v2);
      Array1.unsafe_set out_buf (out_base + i3) (1.0 /. v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (1.0 /. v);
      incr i
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (1.0 /. v)
    done

let kernel_recip_float8_e4m3 (a : (float, float8_e4m3_elt) t)
    (out : (float, float8_e4m3_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (1.0 /. v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (1.0 /. v)
    done

let kernel_recip_float8_e5m2 (a : (float, float8_e5m2_elt) t)
    (out : (float, float8_e5m2_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (1.0 /. v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (1.0 /. v)
    done

let kernel_recip_complex16 (a : (Complex.t, complex16_elt) t)
    (out : (Complex.t, complex16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    for i = 0 to end_idx - start_idx - 1 do
      let v = Array1.unsafe_get a_buf (arg_base + i) in
      Array1.unsafe_set out_buf (out_base + i) (Complex.inv v)
    done)
  else
    let md_index = Array.make (Array.length (shape out)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      let linear_idx = Shape.ravel_index md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + linear_idx) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.inv v)
    done

let kernel_exp2_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.exp2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.exp2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.exp2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.exp2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.exp2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.exp2 v)
    done

let kernel_exp2_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.exp2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.exp2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.exp2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.exp2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.exp2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.exp2 v)
    done

let kernel_exp2_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.exp2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.exp2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.exp2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.exp2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.exp2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.exp2 v)
    done

let kernel_exp2_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (complex_exp2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (complex_exp2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (complex_exp2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (complex_exp2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (complex_exp2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_exp2 v)
    done

let kernel_exp2_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (complex_exp2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (complex_exp2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (complex_exp2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (complex_exp2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (complex_exp2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_exp2 v)
    done

let kernel_exp2_bfloat16 (a : (float, bfloat16_elt) t)
    (out : (float, bfloat16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.exp2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.exp2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.exp2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.exp2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.exp2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.exp2 v)
    done

let kernel_exp2_float8_e4m3 (a : (float, float8_e4m3_elt) t)
    (out : (float, float8_e4m3_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.exp2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.exp2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.exp2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.exp2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.exp2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.exp2 v)
    done

let kernel_exp2_float8_e5m2 (a : (float, float8_e5m2_elt) t)
    (out : (float, float8_e5m2_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.exp2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.exp2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.exp2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.exp2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.exp2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.exp2 v)
    done

let kernel_exp2_complex16 (a : (Complex.t, complex16_elt) t)
    (out : (Complex.t, complex16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (complex_exp2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (complex_exp2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (complex_exp2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (complex_exp2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (complex_exp2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_exp2 v)
    done

let kernel_log2_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.log2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.log2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.log2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.log2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.log2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.log2 v)
    done

let kernel_log2_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.log2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.log2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.log2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.log2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.log2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.log2 v)
    done

let kernel_log2_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.log2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.log2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.log2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.log2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.log2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.log2 v)
    done

let kernel_log2_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (complex_log2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (complex_log2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (complex_log2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (complex_log2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (complex_log2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_log2 v)
    done

let kernel_log2_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (complex_log2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (complex_log2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (complex_log2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (complex_log2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (complex_log2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_log2 v)
    done

let kernel_log2_bfloat16 (a : (float, bfloat16_elt) t)
    (out : (float, bfloat16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.log2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.log2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.log2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.log2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.log2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.log2 v)
    done

let kernel_log2_float8_e4m3 (a : (float, float8_e4m3_elt) t)
    (out : (float, float8_e4m3_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.log2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.log2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.log2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.log2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.log2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.log2 v)
    done

let kernel_log2_float8_e5m2 (a : (float, float8_e5m2_elt) t)
    (out : (float, float8_e5m2_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.log2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.log2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.log2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.log2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.log2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.log2 v)
    done

let kernel_log2_complex16 (a : (Complex.t, complex16_elt) t)
    (out : (Complex.t, complex16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (complex_log2 v0);
      Array1.unsafe_set out_buf (out_base + i1) (complex_log2 v1);
      Array1.unsafe_set out_buf (out_base + i2) (complex_log2 v2);
      Array1.unsafe_set out_buf (out_base + i3) (complex_log2 v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (complex_log2 v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_log2 v)
    done

let kernel_sin_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.sin v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.sin v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.sin v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.sin v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.sin v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sin v)
    done

let kernel_sin_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.sin v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.sin v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.sin v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.sin v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.sin v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sin v)
    done

let kernel_sin_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.sin v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.sin v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.sin v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.sin v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.sin v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sin v)
    done

let kernel_sin_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (complex_sin v0);
      Array1.unsafe_set out_buf (out_base + i1) (complex_sin v1);
      Array1.unsafe_set out_buf (out_base + i2) (complex_sin v2);
      Array1.unsafe_set out_buf (out_base + i3) (complex_sin v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (complex_sin v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_sin v)
    done

let kernel_sin_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (complex_sin v0);
      Array1.unsafe_set out_buf (out_base + i1) (complex_sin v1);
      Array1.unsafe_set out_buf (out_base + i2) (complex_sin v2);
      Array1.unsafe_set out_buf (out_base + i3) (complex_sin v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (complex_sin v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_sin v)
    done

let kernel_sin_bfloat16 (a : (float, bfloat16_elt) t)
    (out : (float, bfloat16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.sin v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.sin v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.sin v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.sin v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.sin v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sin v)
    done

let kernel_sin_float8_e4m3 (a : (float, float8_e4m3_elt) t)
    (out : (float, float8_e4m3_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.sin v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.sin v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.sin v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.sin v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.sin v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sin v)
    done

let kernel_sin_float8_e5m2 (a : (float, float8_e5m2_elt) t)
    (out : (float, float8_e5m2_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (Float.sin v0);
      Array1.unsafe_set out_buf (out_base + i1) (Float.sin v1);
      Array1.unsafe_set out_buf (out_base + i2) (Float.sin v2);
      Array1.unsafe_set out_buf (out_base + i3) (Float.sin v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (Float.sin v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sin v)
    done

let kernel_sin_complex16 (a : (Complex.t, complex16_elt) t)
    (out : (Complex.t, complex16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let arg_base = offset a + start_idx in
    let out_base = offset out + start_idx in
    let i = ref 0 in
    let n = end_idx - start_idx in
    while !i + 3 < n do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (arg_base + i0) in
      let v1 = Array1.unsafe_get a_buf (arg_base + i1) in
      let v2 = Array1.unsafe_get a_buf (arg_base + i2) in
      let v3 = Array1.unsafe_get a_buf (arg_base + i3) in
      Array1.unsafe_set out_buf (out_base + i0) (complex_sin v0);
      Array1.unsafe_set out_buf (out_base + i1) (complex_sin v1);
      Array1.unsafe_set out_buf (out_base + i2) (complex_sin v2);
      Array1.unsafe_set out_buf (out_base + i3) (complex_sin v3);
      i := !i + 4
    done;
    while !i < n do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (arg_base + idx) in
      Array1.unsafe_set out_buf (out_base + idx) (complex_sin v);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_sin v)
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
  | Bfloat16 -> kernel_neg_bfloat16 a c start_idx end_idx
  | Bool -> kernel_neg_bool a c start_idx end_idx
  | Int4_signed -> kernel_neg_int4 a c start_idx end_idx
  | Int4_unsigned -> kernel_neg_uint4 a c start_idx end_idx
  | Float8_e4m3 -> kernel_neg_float8_e4m3 a c start_idx end_idx
  | Float8_e5m2 -> kernel_neg_float8_e5m2 a c start_idx end_idx
  | Complex16 -> kernel_neg_complex16 a c start_idx end_idx
  | Qint8 -> kernel_neg_qint8 a c start_idx end_idx
  | Quint8 -> kernel_neg_quint8 a c start_idx end_idx
  | _ -> invalid_arg "kernel_neg: unsupported type"

let kernel_sqrt (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_sqrt_float16 a c start_idx end_idx
  | Float32 -> kernel_sqrt_float32 a c start_idx end_idx
  | Float64 -> kernel_sqrt_float64 a c start_idx end_idx
  | Complex32 -> kernel_sqrt_complex32 a c start_idx end_idx
  | Complex64 -> kernel_sqrt_complex64 a c start_idx end_idx
  | Bfloat16 -> kernel_sqrt_bfloat16 a c start_idx end_idx
  | Float8_e4m3 -> kernel_sqrt_float8_e4m3 a c start_idx end_idx
  | Float8_e5m2 -> kernel_sqrt_float8_e5m2 a c start_idx end_idx
  | Complex16 -> kernel_sqrt_complex16 a c start_idx end_idx
  | _ -> invalid_arg "kernel_sqrt: unsupported type"

let kernel_recip (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_recip_float16 a c start_idx end_idx
  | Float32 -> kernel_recip_float32 a c start_idx end_idx
  | Float64 -> kernel_recip_float64 a c start_idx end_idx
  | Complex32 -> kernel_recip_complex32 a c start_idx end_idx
  | Complex64 -> kernel_recip_complex64 a c start_idx end_idx
  | Bfloat16 -> kernel_recip_bfloat16 a c start_idx end_idx
  | Float8_e4m3 -> kernel_recip_float8_e4m3 a c start_idx end_idx
  | Float8_e5m2 -> kernel_recip_float8_e5m2 a c start_idx end_idx
  | Complex16 -> kernel_recip_complex16 a c start_idx end_idx
  | _ -> invalid_arg "kernel_recip: unsupported type"

let kernel_exp2 (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_exp2_float16 a c start_idx end_idx
  | Float32 -> kernel_exp2_float32 a c start_idx end_idx
  | Float64 -> kernel_exp2_float64 a c start_idx end_idx
  | Complex32 -> kernel_exp2_complex32 a c start_idx end_idx
  | Complex64 -> kernel_exp2_complex64 a c start_idx end_idx
  | Bfloat16 -> kernel_exp2_bfloat16 a c start_idx end_idx
  | Float8_e4m3 -> kernel_exp2_float8_e4m3 a c start_idx end_idx
  | Float8_e5m2 -> kernel_exp2_float8_e5m2 a c start_idx end_idx
  | Complex16 -> kernel_exp2_complex16 a c start_idx end_idx
  | _ -> invalid_arg "kernel_exp2: unsupported type"

let kernel_log2 (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_log2_float16 a c start_idx end_idx
  | Float32 -> kernel_log2_float32 a c start_idx end_idx
  | Float64 -> kernel_log2_float64 a c start_idx end_idx
  | Complex32 -> kernel_log2_complex32 a c start_idx end_idx
  | Complex64 -> kernel_log2_complex64 a c start_idx end_idx
  | Bfloat16 -> kernel_log2_bfloat16 a c start_idx end_idx
  | Float8_e4m3 -> kernel_log2_float8_e4m3 a c start_idx end_idx
  | Float8_e5m2 -> kernel_log2_float8_e5m2 a c start_idx end_idx
  | Complex16 -> kernel_log2_complex16 a c start_idx end_idx
  | _ -> invalid_arg "kernel_log: unsupported type"

let kernel_sin (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_sin_float16 a c start_idx end_idx
  | Float32 -> kernel_sin_float32 a c start_idx end_idx
  | Float64 -> kernel_sin_float64 a c start_idx end_idx
  | Complex32 -> kernel_sin_complex32 a c start_idx end_idx
  | Complex64 -> kernel_sin_complex64 a c start_idx end_idx
  | Bfloat16 -> kernel_sin_bfloat16 a c start_idx end_idx
  | Float8_e4m3 -> kernel_sin_float8_e4m3 a c start_idx end_idx
  | Float8_e5m2 -> kernel_sin_float8_e5m2 a c start_idx end_idx
  | Complex16 -> kernel_sin_complex16 a c start_idx end_idx
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
