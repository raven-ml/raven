open Bigarray
open Ndarray_core
open Internal

let[@inline] bool_to_int b = if b then 1 else 0

let[@inline] lognot_float (x : float) : float =
  let open Int64 in
  float_of_bits (lognot (bits_of_float x))

module Complex_ops = struct
  open Complex

  let abs z =
    let mag = norm z in
    { re = mag; im = 0. }

  let sign z =
    let mag = norm z in
    if mag = 0. then zero else div z { re = mag; im = 0. }

  let sin z =
    let a, b = (z.re, z.im) in
    { re = Stdlib.sin a *. Stdlib.cosh b; im = Stdlib.cos a *. Stdlib.sinh b }

  let cos z =
    let a, b = (z.re, z.im) in
    {
      re = Stdlib.cos a *. Stdlib.cosh b;
      im = -.(Stdlib.sin a *. Stdlib.sinh b);
    }

  let tan z = div (sin z) (cos z)

  let asin z =
    let i = Complex.i in
    let one = Complex.one in
    let iz = mul i z in
    let z_sq = mul z z in
    let one_minus_z_sq = sub one z_sq in
    let sqrt_val = sqrt one_minus_z_sq in
    let log_arg = add iz sqrt_val in
    let log_val = log log_arg in
    mul (neg i) log_val

  let acos z =
    let i = Complex.i in
    let one = Complex.one in
    let z_sq = mul z z in
    let z_sq_minus_one = sub z_sq one in
    let sqrt_val = sqrt z_sq_minus_one in
    let log_arg = add z sqrt_val in
    let log_val = log log_arg in
    mul (neg i) log_val

  let atan z =
    let i = Complex.i in
    let minus_i_half = { re = 0.; im = -0.5 } in

    let i_plus_z = add i z in
    let i_minus_z = sub i z in

    if i_minus_z = zero then
      let ratio = div i_plus_z i_minus_z in
      mul minus_i_half (log ratio)
    else
      let ratio = div i_plus_z i_minus_z in
      let log_val = log ratio in
      mul minus_i_half log_val

  let sinh z =
    let a, b = (z.re, z.im) in
    { re = Stdlib.sinh a *. Stdlib.cos b; im = Stdlib.cosh a *. Stdlib.sin b }

  let cosh z =
    let a, b = (z.re, z.im) in
    { re = Stdlib.cosh a *. Stdlib.cos b; im = Stdlib.sinh a *. Stdlib.sin b }

  let tanh z = div (sinh z) (cosh z)

  let asinh z =
    let one = Complex.one in
    let z_sq = mul z z in
    let z_sq_plus_one = add z_sq one in
    let sqrt_val = sqrt z_sq_plus_one in
    let log_arg = add z sqrt_val in
    log log_arg

  let acosh z =
    let one = Complex.one in
    let z_sq = mul z z in
    let z_sq_minus_one = sub z_sq one in
    let sqrt_val = sqrt z_sq_minus_one in
    let log_arg = add z sqrt_val in
    log log_arg

  let atanh z =
    let one = Complex.one in
    let half = { re = 0.5; im = 0. } in
    let one_plus_z = add one z in
    let one_minus_z = sub one z in

    if one_minus_z = zero then
      let ratio = div one_plus_z one_minus_z in
      mul half (log ratio)
    else
      let ratio = div one_plus_z one_minus_z in
      let log_val = log ratio in
      mul half log_val

  let lognot a = { re = lognot_float a.re; im = lognot_float a.im }

  let floor a =
    let re = Float.floor a.re in
    let im = Float.floor a.im in
    { re; im }

  let ceil a =
    let re = Float.ceil a.re in
    let im = Float.ceil a.im in
    { re; im }

  let round a =
    let re = Float.round a.re in
    let im = Float.round a.im in
    { re; im }

  let is_nan a =
    let re = Float.is_nan a.re in
    let im = Float.is_nan a.im in
    re || im

  let is_infinite a =
    let re = Float.is_infinite a.re in
    let im = Float.is_infinite a.im in
    re || im

  let is_finite a =
    let re = Float.is_finite a.re in
    let im = Float.is_finite a.im in
    re && im
end

let[@inline] float_sign x = if x > 0. then 1. else if x < 0. then -1. else 0.
let[@inline] int_sign x = if x > 0 then 1 else if x < 0 then -1 else 0

let[@inline] int32_sign x =
  if x > Int32.zero then 1l else if x < Int32.zero then -1l else 0l

let[@inline] int64_sign x =
  if x > Int64.zero then 1L else if x < Int64.zero then -1L else 0L

let[@inline] nativeint_sign x =
  if x > Nativeint.zero then 1n else if x < Nativeint.zero then -1n else 0n

let kernel_neg_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.neg v)
    done

let kernel_neg_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.neg v)
    done

let kernel_neg_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.neg v)
    done

let kernel_neg_int8 (a : (int, int8_elt) t) (out : (int, int8_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.neg v)
    done

let kernel_neg_uint8 (a : (int, uint8_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.neg v)
    done

let kernel_neg_int16 (a : (int, int16_elt) t) (out : (int, int16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.neg v)
    done

let kernel_neg_uint16 (a : (int, uint16_elt) t) (out : (int, uint16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.neg v)
    done

let kernel_neg_int32 (a : (int32, int32_elt) t) (out : (int32, int32_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int32.neg v)
    done

let kernel_neg_int64 (a : (int64, int64_elt) t) (out : (int64, int64_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int64.neg v)
    done

let kernel_neg_int (a : (int, int_elt) t) (out : (int, int_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.neg v)
    done

let kernel_neg_nativeint (a : (nativeint, nativeint_elt) t)
    (out : (nativeint, nativeint_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Nativeint.neg v)
    done

let kernel_neg_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.neg v)
    done

let kernel_neg_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.neg v)
    done

let kernel_abs_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.abs v0);
      Array1.unsafe_set out_buf i1 (Float.abs v1);
      Array1.unsafe_set out_buf i2 (Float.abs v2);
      Array1.unsafe_set out_buf i3 (Float.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.abs v)
    done

let kernel_abs_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.abs v0);
      Array1.unsafe_set out_buf i1 (Float.abs v1);
      Array1.unsafe_set out_buf i2 (Float.abs v2);
      Array1.unsafe_set out_buf i3 (Float.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.abs v)
    done

let kernel_abs_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.abs v0);
      Array1.unsafe_set out_buf i1 (Float.abs v1);
      Array1.unsafe_set out_buf i2 (Float.abs v2);
      Array1.unsafe_set out_buf i3 (Float.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.abs v)
    done

let kernel_abs_int8 (a : (int, int8_elt) t) (out : (int, int8_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.abs v0);
      Array1.unsafe_set out_buf i1 (Int.abs v1);
      Array1.unsafe_set out_buf i2 (Int.abs v2);
      Array1.unsafe_set out_buf i3 (Int.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.abs v)
    done

let kernel_abs_uint8 (a : (int, uint8_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.abs v0);
      Array1.unsafe_set out_buf i1 (Int.abs v1);
      Array1.unsafe_set out_buf i2 (Int.abs v2);
      Array1.unsafe_set out_buf i3 (Int.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.abs v)
    done

let kernel_abs_int16 (a : (int, int16_elt) t) (out : (int, int16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.abs v0);
      Array1.unsafe_set out_buf i1 (Int.abs v1);
      Array1.unsafe_set out_buf i2 (Int.abs v2);
      Array1.unsafe_set out_buf i3 (Int.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.abs v)
    done

let kernel_abs_uint16 (a : (int, uint16_elt) t) (out : (int, uint16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.abs v0);
      Array1.unsafe_set out_buf i1 (Int.abs v1);
      Array1.unsafe_set out_buf i2 (Int.abs v2);
      Array1.unsafe_set out_buf i3 (Int.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.abs v)
    done

let kernel_abs_int32 (a : (int32, int32_elt) t) (out : (int32, int32_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int32.abs v0);
      Array1.unsafe_set out_buf i1 (Int32.abs v1);
      Array1.unsafe_set out_buf i2 (Int32.abs v2);
      Array1.unsafe_set out_buf i3 (Int32.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int32.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int32.abs v)
    done

let kernel_abs_int64 (a : (int64, int64_elt) t) (out : (int64, int64_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int64.abs v0);
      Array1.unsafe_set out_buf i1 (Int64.abs v1);
      Array1.unsafe_set out_buf i2 (Int64.abs v2);
      Array1.unsafe_set out_buf i3 (Int64.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int64.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int64.abs v)
    done

let kernel_abs_int (a : (int, int_elt) t) (out : (int, int_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.abs v0);
      Array1.unsafe_set out_buf i1 (Int.abs v1);
      Array1.unsafe_set out_buf i2 (Int.abs v2);
      Array1.unsafe_set out_buf i3 (Int.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.abs v)
    done

let kernel_abs_nativeint (a : (nativeint, nativeint_elt) t)
    (out : (nativeint, nativeint_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.abs v0);
      Array1.unsafe_set out_buf i1 (Nativeint.abs v1);
      Array1.unsafe_set out_buf i2 (Nativeint.abs v2);
      Array1.unsafe_set out_buf i3 (Nativeint.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Nativeint.abs v)
    done

let kernel_abs_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.abs v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.abs v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.abs v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.abs v)
    done

let kernel_abs_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.abs v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.abs v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.abs v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.abs v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.abs v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.abs v)
    done

let kernel_sign_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (float_sign v0);
      Array1.unsafe_set out_buf i1 (float_sign v1);
      Array1.unsafe_set out_buf i2 (float_sign v2);
      Array1.unsafe_set out_buf i3 (float_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (float_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (float_sign v)
    done

let kernel_sign_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (float_sign v0);
      Array1.unsafe_set out_buf i1 (float_sign v1);
      Array1.unsafe_set out_buf i2 (float_sign v2);
      Array1.unsafe_set out_buf i3 (float_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (float_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (float_sign v)
    done

let kernel_sign_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (float_sign v0);
      Array1.unsafe_set out_buf i1 (float_sign v1);
      Array1.unsafe_set out_buf i2 (float_sign v2);
      Array1.unsafe_set out_buf i3 (float_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (float_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (float_sign v)
    done

let kernel_sign_int8 (a : (int, int8_elt) t) (out : (int, int8_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (int_sign v0);
      Array1.unsafe_set out_buf i1 (int_sign v1);
      Array1.unsafe_set out_buf i2 (int_sign v2);
      Array1.unsafe_set out_buf i3 (int_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (int_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (int_sign v)
    done

let kernel_sign_uint8 (a : (int, uint8_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (int_sign v0);
      Array1.unsafe_set out_buf i1 (int_sign v1);
      Array1.unsafe_set out_buf i2 (int_sign v2);
      Array1.unsafe_set out_buf i3 (int_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (int_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (int_sign v)
    done

let kernel_sign_int16 (a : (int, int16_elt) t) (out : (int, int16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (int_sign v0);
      Array1.unsafe_set out_buf i1 (int_sign v1);
      Array1.unsafe_set out_buf i2 (int_sign v2);
      Array1.unsafe_set out_buf i3 (int_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (int_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (int_sign v)
    done

let kernel_sign_uint16 (a : (int, uint16_elt) t) (out : (int, uint16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (int_sign v0);
      Array1.unsafe_set out_buf i1 (int_sign v1);
      Array1.unsafe_set out_buf i2 (int_sign v2);
      Array1.unsafe_set out_buf i3 (int_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (int_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (int_sign v)
    done

let kernel_sign_int32 (a : (int32, int32_elt) t) (out : (int32, int32_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (int32_sign v0);
      Array1.unsafe_set out_buf i1 (int32_sign v1);
      Array1.unsafe_set out_buf i2 (int32_sign v2);
      Array1.unsafe_set out_buf i3 (int32_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (int32_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (int32_sign v)
    done

let kernel_sign_int64 (a : (int64, int64_elt) t) (out : (int64, int64_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (int64_sign v0);
      Array1.unsafe_set out_buf i1 (int64_sign v1);
      Array1.unsafe_set out_buf i2 (int64_sign v2);
      Array1.unsafe_set out_buf i3 (int64_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (int64_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (int64_sign v)
    done

let kernel_sign_int (a : (int, int_elt) t) (out : (int, int_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (int_sign v0);
      Array1.unsafe_set out_buf i1 (int_sign v1);
      Array1.unsafe_set out_buf i2 (int_sign v2);
      Array1.unsafe_set out_buf i3 (int_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (int_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (int_sign v)
    done

let kernel_sign_nativeint (a : (nativeint, nativeint_elt) t)
    (out : (nativeint, nativeint_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (nativeint_sign v0);
      Array1.unsafe_set out_buf i1 (nativeint_sign v1);
      Array1.unsafe_set out_buf i2 (nativeint_sign v2);
      Array1.unsafe_set out_buf i3 (nativeint_sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (nativeint_sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (nativeint_sign v)
    done

let kernel_sign_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.sign v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.sign v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.sign v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.sign v)
    done

let kernel_sign_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.sign v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.sign v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.sign v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.sign v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.sign v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.sign v)
    done

let kernel_bit_not_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (lognot_float v0);
      Array1.unsafe_set out_buf i1 (lognot_float v1);
      Array1.unsafe_set out_buf i2 (lognot_float v2);
      Array1.unsafe_set out_buf i3 (lognot_float v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (lognot_float v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (lognot_float v)
    done

let kernel_bit_not_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (lognot_float v0);
      Array1.unsafe_set out_buf i1 (lognot_float v1);
      Array1.unsafe_set out_buf i2 (lognot_float v2);
      Array1.unsafe_set out_buf i3 (lognot_float v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (lognot_float v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (lognot_float v)
    done

let kernel_bit_not_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (lognot_float v0);
      Array1.unsafe_set out_buf i1 (lognot_float v1);
      Array1.unsafe_set out_buf i2 (lognot_float v2);
      Array1.unsafe_set out_buf i3 (lognot_float v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (lognot_float v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (lognot_float v)
    done

let kernel_bit_not_int8 (a : (int, int8_elt) t) (out : (int, int8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.lognot v0);
      Array1.unsafe_set out_buf i1 (Int.lognot v1);
      Array1.unsafe_set out_buf i2 (Int.lognot v2);
      Array1.unsafe_set out_buf i3 (Int.lognot v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.lognot v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.lognot v)
    done

let kernel_bit_not_uint8 (a : (int, uint8_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.lognot v0);
      Array1.unsafe_set out_buf i1 (Int.lognot v1);
      Array1.unsafe_set out_buf i2 (Int.lognot v2);
      Array1.unsafe_set out_buf i3 (Int.lognot v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.lognot v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.lognot v)
    done

let kernel_bit_not_int16 (a : (int, int16_elt) t) (out : (int, int16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.lognot v0);
      Array1.unsafe_set out_buf i1 (Int.lognot v1);
      Array1.unsafe_set out_buf i2 (Int.lognot v2);
      Array1.unsafe_set out_buf i3 (Int.lognot v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.lognot v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.lognot v)
    done

let kernel_bit_not_uint16 (a : (int, uint16_elt) t) (out : (int, uint16_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.lognot v0);
      Array1.unsafe_set out_buf i1 (Int.lognot v1);
      Array1.unsafe_set out_buf i2 (Int.lognot v2);
      Array1.unsafe_set out_buf i3 (Int.lognot v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.lognot v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.lognot v)
    done

let kernel_bit_not_int32 (a : (int32, int32_elt) t) (out : (int32, int32_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int32.lognot v0);
      Array1.unsafe_set out_buf i1 (Int32.lognot v1);
      Array1.unsafe_set out_buf i2 (Int32.lognot v2);
      Array1.unsafe_set out_buf i3 (Int32.lognot v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int32.lognot v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int32.lognot v)
    done

let kernel_bit_not_int64 (a : (int64, int64_elt) t) (out : (int64, int64_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int64.lognot v0);
      Array1.unsafe_set out_buf i1 (Int64.lognot v1);
      Array1.unsafe_set out_buf i2 (Int64.lognot v2);
      Array1.unsafe_set out_buf i3 (Int64.lognot v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int64.lognot v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int64.lognot v)
    done

let kernel_bit_not_int (a : (int, int_elt) t) (out : (int, int_elt) t) start_idx
    end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Int.lognot v0);
      Array1.unsafe_set out_buf i1 (Int.lognot v1);
      Array1.unsafe_set out_buf i2 (Int.lognot v2);
      Array1.unsafe_set out_buf i3 (Int.lognot v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Int.lognot v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Int.lognot v)
    done

let kernel_bit_not_nativeint (a : (nativeint, nativeint_elt) t)
    (out : (nativeint, nativeint_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.lognot v0);
      Array1.unsafe_set out_buf i1 (Nativeint.lognot v1);
      Array1.unsafe_set out_buf i2 (Nativeint.lognot v2);
      Array1.unsafe_set out_buf i3 (Nativeint.lognot v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.lognot v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Nativeint.lognot v)
    done

let kernel_bit_not_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.lognot v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.lognot v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.lognot v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.lognot v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.lognot v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.lognot v)
    done

let kernel_bit_not_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.lognot v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.lognot v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.lognot v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.lognot v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.lognot v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.lognot v)
    done

let kernel_sqrt_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sqrt v)
    done

let kernel_sqrt_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sqrt v)
    done

let kernel_sqrt_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sqrt v)
    done

let kernel_sqrt_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.sqrt v)
    done

let kernel_sqrt_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.sqrt v)
    done

let kernel_exp_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.exp v0);
      Array1.unsafe_set out_buf i1 (Float.exp v1);
      Array1.unsafe_set out_buf i2 (Float.exp v2);
      Array1.unsafe_set out_buf i3 (Float.exp v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.exp v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.exp v)
    done

let kernel_exp_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.exp v0);
      Array1.unsafe_set out_buf i1 (Float.exp v1);
      Array1.unsafe_set out_buf i2 (Float.exp v2);
      Array1.unsafe_set out_buf i3 (Float.exp v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.exp v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.exp v)
    done

let kernel_exp_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.exp v0);
      Array1.unsafe_set out_buf i1 (Float.exp v1);
      Array1.unsafe_set out_buf i2 (Float.exp v2);
      Array1.unsafe_set out_buf i3 (Float.exp v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.exp v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.exp v)
    done

let kernel_exp_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex.exp v0);
      Array1.unsafe_set out_buf i1 (Complex.exp v1);
      Array1.unsafe_set out_buf i2 (Complex.exp v2);
      Array1.unsafe_set out_buf i3 (Complex.exp v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex.exp v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.exp v)
    done

let kernel_exp_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex.exp v0);
      Array1.unsafe_set out_buf i1 (Complex.exp v1);
      Array1.unsafe_set out_buf i2 (Complex.exp v2);
      Array1.unsafe_set out_buf i3 (Complex.exp v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex.exp v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.exp v)
    done

let kernel_log_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.log v0);
      Array1.unsafe_set out_buf i1 (Float.log v1);
      Array1.unsafe_set out_buf i2 (Float.log v2);
      Array1.unsafe_set out_buf i3 (Float.log v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.log v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.log v)
    done

let kernel_log_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.log v0);
      Array1.unsafe_set out_buf i1 (Float.log v1);
      Array1.unsafe_set out_buf i2 (Float.log v2);
      Array1.unsafe_set out_buf i3 (Float.log v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.log v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.log v)
    done

let kernel_log_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.log v0);
      Array1.unsafe_set out_buf i1 (Float.log v1);
      Array1.unsafe_set out_buf i2 (Float.log v2);
      Array1.unsafe_set out_buf i3 (Float.log v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.log v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.log v)
    done

let kernel_log_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex.log v0);
      Array1.unsafe_set out_buf i1 (Complex.log v1);
      Array1.unsafe_set out_buf i2 (Complex.log v2);
      Array1.unsafe_set out_buf i3 (Complex.log v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex.log v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.log v)
    done

let kernel_log_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex.log v0);
      Array1.unsafe_set out_buf i1 (Complex.log v1);
      Array1.unsafe_set out_buf i2 (Complex.log v2);
      Array1.unsafe_set out_buf i3 (Complex.log v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex.log v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex.log v)
    done

let kernel_sin_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sin v)
    done

let kernel_sin_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sin v)
    done

let kernel_sin_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
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
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sin v)
    done

let kernel_sin_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.sin v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.sin v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.sin v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.sin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.sin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.sin v)
    done

let kernel_sin_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.sin v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.sin v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.sin v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.sin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.sin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.sin v)
    done

let kernel_cos_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.cos v0);
      Array1.unsafe_set out_buf i1 (Float.cos v1);
      Array1.unsafe_set out_buf i2 (Float.cos v2);
      Array1.unsafe_set out_buf i3 (Float.cos v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.cos v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.cos v)
    done

let kernel_cos_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.cos v0);
      Array1.unsafe_set out_buf i1 (Float.cos v1);
      Array1.unsafe_set out_buf i2 (Float.cos v2);
      Array1.unsafe_set out_buf i3 (Float.cos v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.cos v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.cos v)
    done

let kernel_cos_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.cos v0);
      Array1.unsafe_set out_buf i1 (Float.cos v1);
      Array1.unsafe_set out_buf i2 (Float.cos v2);
      Array1.unsafe_set out_buf i3 (Float.cos v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.cos v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.cos v)
    done

let kernel_cos_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.cos v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.cos v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.cos v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.cos v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.cos v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.cos v)
    done

let kernel_cos_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.cos v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.cos v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.cos v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.cos v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.cos v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.cos v)
    done

let kernel_tan_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.tan v0);
      Array1.unsafe_set out_buf i1 (Float.tan v1);
      Array1.unsafe_set out_buf i2 (Float.tan v2);
      Array1.unsafe_set out_buf i3 (Float.tan v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.tan v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.tan v)
    done

let kernel_tan_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.tan v0);
      Array1.unsafe_set out_buf i1 (Float.tan v1);
      Array1.unsafe_set out_buf i2 (Float.tan v2);
      Array1.unsafe_set out_buf i3 (Float.tan v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.tan v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.tan v)
    done

let kernel_tan_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.tan v0);
      Array1.unsafe_set out_buf i1 (Float.tan v1);
      Array1.unsafe_set out_buf i2 (Float.tan v2);
      Array1.unsafe_set out_buf i3 (Float.tan v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.tan v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.tan v)
    done

let kernel_tan_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.tan v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.tan v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.tan v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.tan v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.tan v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.tan v)
    done

let kernel_tan_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.tan v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.tan v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.tan v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.tan v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.tan v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.tan v)
    done

let kernel_asin_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.asin v0);
      Array1.unsafe_set out_buf i1 (Float.asin v1);
      Array1.unsafe_set out_buf i2 (Float.asin v2);
      Array1.unsafe_set out_buf i3 (Float.asin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.asin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.asin v)
    done

let kernel_asin_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.asin v0);
      Array1.unsafe_set out_buf i1 (Float.asin v1);
      Array1.unsafe_set out_buf i2 (Float.asin v2);
      Array1.unsafe_set out_buf i3 (Float.asin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.asin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.asin v)
    done

let kernel_asin_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.asin v0);
      Array1.unsafe_set out_buf i1 (Float.asin v1);
      Array1.unsafe_set out_buf i2 (Float.asin v2);
      Array1.unsafe_set out_buf i3 (Float.asin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.asin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.asin v)
    done

let kernel_asin_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.asin v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.asin v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.asin v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.asin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.asin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.asin v)
    done

let kernel_asin_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.asin v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.asin v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.asin v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.asin v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.asin v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.asin v)
    done

let kernel_acos_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.acos v0);
      Array1.unsafe_set out_buf i1 (Float.acos v1);
      Array1.unsafe_set out_buf i2 (Float.acos v2);
      Array1.unsafe_set out_buf i3 (Float.acos v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.acos v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.acos v)
    done

let kernel_acos_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.acos v0);
      Array1.unsafe_set out_buf i1 (Float.acos v1);
      Array1.unsafe_set out_buf i2 (Float.acos v2);
      Array1.unsafe_set out_buf i3 (Float.acos v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.acos v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.acos v)
    done

let kernel_acos_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.acos v0);
      Array1.unsafe_set out_buf i1 (Float.acos v1);
      Array1.unsafe_set out_buf i2 (Float.acos v2);
      Array1.unsafe_set out_buf i3 (Float.acos v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.acos v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.acos v)
    done

let kernel_acos_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.acos v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.acos v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.acos v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.acos v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.acos v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.acos v)
    done

let kernel_acos_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.acos v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.acos v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.acos v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.acos v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.acos v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.acos v)
    done

let kernel_atan_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.atan v0);
      Array1.unsafe_set out_buf i1 (Float.atan v1);
      Array1.unsafe_set out_buf i2 (Float.atan v2);
      Array1.unsafe_set out_buf i3 (Float.atan v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.atan v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.atan v)
    done

let kernel_atan_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.atan v0);
      Array1.unsafe_set out_buf i1 (Float.atan v1);
      Array1.unsafe_set out_buf i2 (Float.atan v2);
      Array1.unsafe_set out_buf i3 (Float.atan v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.atan v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.atan v)
    done

let kernel_atan_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.atan v0);
      Array1.unsafe_set out_buf i1 (Float.atan v1);
      Array1.unsafe_set out_buf i2 (Float.atan v2);
      Array1.unsafe_set out_buf i3 (Float.atan v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.atan v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.atan v)
    done

let kernel_atan_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.atan v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.atan v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.atan v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.atan v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.atan v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.atan v)
    done

let kernel_atan_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.atan v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.atan v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.atan v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.atan v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.atan v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.atan v)
    done

let kernel_sinh_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.sinh v0);
      Array1.unsafe_set out_buf i1 (Float.sinh v1);
      Array1.unsafe_set out_buf i2 (Float.sinh v2);
      Array1.unsafe_set out_buf i3 (Float.sinh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.sinh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sinh v)
    done

let kernel_sinh_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.sinh v0);
      Array1.unsafe_set out_buf i1 (Float.sinh v1);
      Array1.unsafe_set out_buf i2 (Float.sinh v2);
      Array1.unsafe_set out_buf i3 (Float.sinh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.sinh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sinh v)
    done

let kernel_sinh_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.sinh v0);
      Array1.unsafe_set out_buf i1 (Float.sinh v1);
      Array1.unsafe_set out_buf i2 (Float.sinh v2);
      Array1.unsafe_set out_buf i3 (Float.sinh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.sinh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.sinh v)
    done

let kernel_sinh_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.sinh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.sinh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.sinh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.sinh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.sinh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.sinh v)
    done

let kernel_sinh_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.sinh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.sinh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.sinh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.sinh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.sinh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.sinh v)
    done

let kernel_cosh_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.cosh v0);
      Array1.unsafe_set out_buf i1 (Float.cosh v1);
      Array1.unsafe_set out_buf i2 (Float.cosh v2);
      Array1.unsafe_set out_buf i3 (Float.cosh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.cosh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.cosh v)
    done

let kernel_cosh_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.cosh v0);
      Array1.unsafe_set out_buf i1 (Float.cosh v1);
      Array1.unsafe_set out_buf i2 (Float.cosh v2);
      Array1.unsafe_set out_buf i3 (Float.cosh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.cosh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.cosh v)
    done

let kernel_cosh_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.cosh v0);
      Array1.unsafe_set out_buf i1 (Float.cosh v1);
      Array1.unsafe_set out_buf i2 (Float.cosh v2);
      Array1.unsafe_set out_buf i3 (Float.cosh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.cosh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.cosh v)
    done

let kernel_cosh_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.cosh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.cosh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.cosh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.cosh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.cosh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.cosh v)
    done

let kernel_cosh_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.cosh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.cosh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.cosh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.cosh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.cosh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.cosh v)
    done

let kernel_tanh_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.tanh v0);
      Array1.unsafe_set out_buf i1 (Float.tanh v1);
      Array1.unsafe_set out_buf i2 (Float.tanh v2);
      Array1.unsafe_set out_buf i3 (Float.tanh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.tanh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.tanh v)
    done

let kernel_tanh_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.tanh v0);
      Array1.unsafe_set out_buf i1 (Float.tanh v1);
      Array1.unsafe_set out_buf i2 (Float.tanh v2);
      Array1.unsafe_set out_buf i3 (Float.tanh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.tanh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.tanh v)
    done

let kernel_tanh_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.tanh v0);
      Array1.unsafe_set out_buf i1 (Float.tanh v1);
      Array1.unsafe_set out_buf i2 (Float.tanh v2);
      Array1.unsafe_set out_buf i3 (Float.tanh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.tanh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.tanh v)
    done

let kernel_tanh_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.tanh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.tanh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.tanh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.tanh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.tanh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.tanh v)
    done

let kernel_tanh_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.tanh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.tanh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.tanh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.tanh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.tanh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.tanh v)
    done

let kernel_asinh_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.asinh v0);
      Array1.unsafe_set out_buf i1 (Float.asinh v1);
      Array1.unsafe_set out_buf i2 (Float.asinh v2);
      Array1.unsafe_set out_buf i3 (Float.asinh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.asinh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.asinh v)
    done

let kernel_asinh_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.asinh v0);
      Array1.unsafe_set out_buf i1 (Float.asinh v1);
      Array1.unsafe_set out_buf i2 (Float.asinh v2);
      Array1.unsafe_set out_buf i3 (Float.asinh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.asinh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.asinh v)
    done

let kernel_asinh_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.asinh v0);
      Array1.unsafe_set out_buf i1 (Float.asinh v1);
      Array1.unsafe_set out_buf i2 (Float.asinh v2);
      Array1.unsafe_set out_buf i3 (Float.asinh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.asinh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.asinh v)
    done

let kernel_asinh_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.asinh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.asinh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.asinh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.asinh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.asinh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.asinh v)
    done

let kernel_asinh_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.asinh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.asinh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.asinh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.asinh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.asinh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.asinh v)
    done

let kernel_acosh_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.acosh v0);
      Array1.unsafe_set out_buf i1 (Float.acosh v1);
      Array1.unsafe_set out_buf i2 (Float.acosh v2);
      Array1.unsafe_set out_buf i3 (Float.acosh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.acosh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.acosh v)
    done

let kernel_acosh_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.acosh v0);
      Array1.unsafe_set out_buf i1 (Float.acosh v1);
      Array1.unsafe_set out_buf i2 (Float.acosh v2);
      Array1.unsafe_set out_buf i3 (Float.acosh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.acosh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.acosh v)
    done

let kernel_acosh_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.acosh v0);
      Array1.unsafe_set out_buf i1 (Float.acosh v1);
      Array1.unsafe_set out_buf i2 (Float.acosh v2);
      Array1.unsafe_set out_buf i3 (Float.acosh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.acosh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.acosh v)
    done

let kernel_acosh_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.acosh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.acosh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.acosh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.acosh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.acosh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.acosh v)
    done

let kernel_acosh_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.acosh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.acosh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.acosh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.acosh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.acosh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.acosh v)
    done

let kernel_atanh_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.atanh v0);
      Array1.unsafe_set out_buf i1 (Float.atanh v1);
      Array1.unsafe_set out_buf i2 (Float.atanh v2);
      Array1.unsafe_set out_buf i3 (Float.atanh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.atanh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.atanh v)
    done

let kernel_atanh_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.atanh v0);
      Array1.unsafe_set out_buf i1 (Float.atanh v1);
      Array1.unsafe_set out_buf i2 (Float.atanh v2);
      Array1.unsafe_set out_buf i3 (Float.atanh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.atanh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.atanh v)
    done

let kernel_atanh_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.atanh v0);
      Array1.unsafe_set out_buf i1 (Float.atanh v1);
      Array1.unsafe_set out_buf i2 (Float.atanh v2);
      Array1.unsafe_set out_buf i3 (Float.atanh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.atanh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.atanh v)
    done

let kernel_atanh_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.atanh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.atanh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.atanh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.atanh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.atanh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.atanh v)
    done

let kernel_atanh_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.atanh v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.atanh v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.atanh v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.atanh v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.atanh v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.atanh v)
    done

let kernel_floor_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.floor v0);
      Array1.unsafe_set out_buf i1 (Float.floor v1);
      Array1.unsafe_set out_buf i2 (Float.floor v2);
      Array1.unsafe_set out_buf i3 (Float.floor v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.floor v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.floor v)
    done

let kernel_floor_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.floor v0);
      Array1.unsafe_set out_buf i1 (Float.floor v1);
      Array1.unsafe_set out_buf i2 (Float.floor v2);
      Array1.unsafe_set out_buf i3 (Float.floor v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.floor v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.floor v)
    done

let kernel_floor_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.floor v0);
      Array1.unsafe_set out_buf i1 (Float.floor v1);
      Array1.unsafe_set out_buf i2 (Float.floor v2);
      Array1.unsafe_set out_buf i3 (Float.floor v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.floor v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.floor v)
    done

let kernel_floor_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.floor v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.floor v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.floor v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.floor v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.floor v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.floor v)
    done

let kernel_floor_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.floor v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.floor v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.floor v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.floor v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.floor v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.floor v)
    done

let kernel_ceil_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.ceil v0);
      Array1.unsafe_set out_buf i1 (Float.ceil v1);
      Array1.unsafe_set out_buf i2 (Float.ceil v2);
      Array1.unsafe_set out_buf i3 (Float.ceil v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.ceil v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.ceil v)
    done

let kernel_ceil_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.ceil v0);
      Array1.unsafe_set out_buf i1 (Float.ceil v1);
      Array1.unsafe_set out_buf i2 (Float.ceil v2);
      Array1.unsafe_set out_buf i3 (Float.ceil v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.ceil v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.ceil v)
    done

let kernel_ceil_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.ceil v0);
      Array1.unsafe_set out_buf i1 (Float.ceil v1);
      Array1.unsafe_set out_buf i2 (Float.ceil v2);
      Array1.unsafe_set out_buf i3 (Float.ceil v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.ceil v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.ceil v)
    done

let kernel_ceil_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.ceil v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.ceil v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.ceil v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.ceil v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.ceil v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.ceil v)
    done

let kernel_ceil_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.ceil v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.ceil v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.ceil v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.ceil v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.ceil v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.ceil v)
    done

let kernel_round_float16 (a : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.round v0);
      Array1.unsafe_set out_buf i1 (Float.round v1);
      Array1.unsafe_set out_buf i2 (Float.round v2);
      Array1.unsafe_set out_buf i3 (Float.round v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.round v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.round v)
    done

let kernel_round_float32 (a : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.round v0);
      Array1.unsafe_set out_buf i1 (Float.round v1);
      Array1.unsafe_set out_buf i2 (Float.round v2);
      Array1.unsafe_set out_buf i3 (Float.round v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.round v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.round v)
    done

let kernel_round_float64 (a : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Float.round v0);
      Array1.unsafe_set out_buf i1 (Float.round v1);
      Array1.unsafe_set out_buf i2 (Float.round v2);
      Array1.unsafe_set out_buf i3 (Float.round v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Float.round v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Float.round v)
    done

let kernel_round_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.round v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.round v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.round v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.round v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.round v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.round v)
    done

let kernel_round_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (Complex_ops.round v0);
      Array1.unsafe_set out_buf i1 (Complex_ops.round v1);
      Array1.unsafe_set out_buf i2 (Complex_ops.round v2);
      Array1.unsafe_set out_buf i3 (Complex_ops.round v3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (Complex_ops.round v);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (Complex_ops.round v)
    done

let kernel_isnan_float16 (a : (float, float16_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.is_nan v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.is_nan v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.is_nan v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.is_nan v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.is_nan v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.is_nan v))
    done

let kernel_isnan_float32 (a : (float, float32_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.is_nan v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.is_nan v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.is_nan v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.is_nan v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.is_nan v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.is_nan v))
    done

let kernel_isnan_float64 (a : (float, float64_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.is_nan v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.is_nan v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.is_nan v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.is_nan v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.is_nan v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.is_nan v))
    done

let kernel_isnan_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Complex_ops.is_nan v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Complex_ops.is_nan v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Complex_ops.is_nan v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Complex_ops.is_nan v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Complex_ops.is_nan v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Complex_ops.is_nan v))
    done

let kernel_isnan_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Complex_ops.is_nan v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Complex_ops.is_nan v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Complex_ops.is_nan v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Complex_ops.is_nan v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Complex_ops.is_nan v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Complex_ops.is_nan v))
    done

let kernel_isinf_float16 (a : (float, float16_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.is_infinite v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.is_infinite v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.is_infinite v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.is_infinite v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.is_infinite v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.is_infinite v))
    done

let kernel_isinf_float32 (a : (float, float32_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.is_infinite v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.is_infinite v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.is_infinite v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.is_infinite v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.is_infinite v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.is_infinite v))
    done

let kernel_isinf_float64 (a : (float, float64_elt) t) (out : (int, uint8_elt) t)
    start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.is_infinite v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.is_infinite v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.is_infinite v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.is_infinite v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.is_infinite v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.is_infinite v))
    done

let kernel_isinf_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Complex_ops.is_infinite v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Complex_ops.is_infinite v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Complex_ops.is_infinite v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Complex_ops.is_infinite v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Complex_ops.is_infinite v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Complex_ops.is_infinite v))
    done

let kernel_isinf_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Complex_ops.is_infinite v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Complex_ops.is_infinite v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Complex_ops.is_infinite v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Complex_ops.is_infinite v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Complex_ops.is_infinite v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Complex_ops.is_infinite v))
    done

let kernel_isfinite_float16 (a : (float, float16_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.is_finite v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.is_finite v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.is_finite v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.is_finite v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.is_finite v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.is_finite v))
    done

let kernel_isfinite_float32 (a : (float, float32_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.is_finite v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.is_finite v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.is_finite v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.is_finite v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.is_finite v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.is_finite v))
    done

let kernel_isfinite_float64 (a : (float, float64_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.is_finite v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.is_finite v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.is_finite v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.is_finite v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.is_finite v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.is_finite v))
    done

let kernel_isfinite_complex32 (a : (Complex.t, complex32_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Complex_ops.is_finite v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Complex_ops.is_finite v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Complex_ops.is_finite v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Complex_ops.is_finite v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Complex_ops.is_finite v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Complex_ops.is_finite v))
    done

let kernel_isfinite_complex64 (a : (Complex.t, complex64_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, out_buf = (buffer a, buffer out) in
  if is_c_contiguous a && is_c_contiguous out then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let v0 = Array1.unsafe_get a_buf (offset a + i0) in
      let v1 = Array1.unsafe_get a_buf (offset a + i1) in
      let v2 = Array1.unsafe_get a_buf (offset a + i2) in
      let v3 = Array1.unsafe_get a_buf (offset a + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Complex_ops.is_finite v0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Complex_ops.is_finite v1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Complex_ops.is_finite v2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Complex_ops.is_finite v3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let v = Array1.unsafe_get a_buf (offset a + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Complex_ops.is_finite v));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let v = Array1.unsafe_get a_buf (offset a + a_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Complex_ops.is_finite v))
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

let kernel_abs (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_abs_float16 a c start_idx end_idx
  | Float32 -> kernel_abs_float32 a c start_idx end_idx
  | Float64 -> kernel_abs_float64 a c start_idx end_idx
  | Complex32 -> kernel_abs_complex32 a c start_idx end_idx
  | Complex64 -> kernel_abs_complex64 a c start_idx end_idx
  | Int8_signed -> kernel_abs_int8 a c start_idx end_idx
  | Int8_unsigned -> kernel_abs_uint8 a c start_idx end_idx
  | Int16_signed -> kernel_abs_int16 a c start_idx end_idx
  | Int16_unsigned -> kernel_abs_uint16 a c start_idx end_idx
  | Int32 -> kernel_abs_int32 a c start_idx end_idx
  | Int64 -> kernel_abs_int64 a c start_idx end_idx
  | Int -> kernel_abs_int a c start_idx end_idx
  | Nativeint -> kernel_abs_nativeint a c start_idx end_idx
  | _ -> invalid_arg "kernel_abs: unsupported type"

let kernel_sign (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_sign_float16 a c start_idx end_idx
  | Float32 -> kernel_sign_float32 a c start_idx end_idx
  | Float64 -> kernel_sign_float64 a c start_idx end_idx
  | Int8_signed -> kernel_sign_int8 a c start_idx end_idx
  | Int8_unsigned -> kernel_sign_uint8 a c start_idx end_idx
  | Int16_signed -> kernel_sign_int16 a c start_idx end_idx
  | Int16_unsigned -> kernel_sign_uint16 a c start_idx end_idx
  | Int32 -> kernel_sign_int32 a c start_idx end_idx
  | Int64 -> kernel_sign_int64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_sign: unsupported type"

let kernel_sqrt (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_sqrt_float16 a c start_idx end_idx
  | Float32 -> kernel_sqrt_float32 a c start_idx end_idx
  | Float64 -> kernel_sqrt_float64 a c start_idx end_idx
  | Complex32 -> kernel_sqrt_complex32 a c start_idx end_idx
  | Complex64 -> kernel_sqrt_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_sqrt: unsupported type"

let kernel_exp (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_exp_float16 a c start_idx end_idx
  | Float32 -> kernel_exp_float32 a c start_idx end_idx
  | Float64 -> kernel_exp_float64 a c start_idx end_idx
  | Complex32 -> kernel_exp_complex32 a c start_idx end_idx
  | Complex64 -> kernel_exp_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_exp: unsupported type"

let kernel_log (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_log_float16 a c start_idx end_idx
  | Float32 -> kernel_log_float32 a c start_idx end_idx
  | Float64 -> kernel_log_float64 a c start_idx end_idx
  | Complex32 -> kernel_log_complex32 a c start_idx end_idx
  | Complex64 -> kernel_log_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_log: unsupported type"

let kernel_sin (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_sin_float16 a c start_idx end_idx
  | Float32 -> kernel_sin_float32 a c start_idx end_idx
  | Float64 -> kernel_sin_float64 a c start_idx end_idx
  | Complex32 -> kernel_sin_complex32 a c start_idx end_idx
  | Complex64 -> kernel_sin_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_sin: unsupported type"

let kernel_cos (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_cos_float16 a c start_idx end_idx
  | Float32 -> kernel_cos_float32 a c start_idx end_idx
  | Float64 -> kernel_cos_float64 a c start_idx end_idx
  | Complex32 -> kernel_cos_complex32 a c start_idx end_idx
  | Complex64 -> kernel_cos_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_cos: unsupported type"

let kernel_tan (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_tan_float16 a c start_idx end_idx
  | Float32 -> kernel_tan_float32 a c start_idx end_idx
  | Float64 -> kernel_tan_float64 a c start_idx end_idx
  | Complex32 -> kernel_tan_complex32 a c start_idx end_idx
  | Complex64 -> kernel_tan_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_tan: unsupported type"

let kernel_asin (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_asin_float16 a c start_idx end_idx
  | Float32 -> kernel_asin_float32 a c start_idx end_idx
  | Float64 -> kernel_asin_float64 a c start_idx end_idx
  | Complex32 -> kernel_asin_complex32 a c start_idx end_idx
  | Complex64 -> kernel_asin_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_asin: unsupported type"

let kernel_acos (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_acos_float16 a c start_idx end_idx
  | Float32 -> kernel_acos_float32 a c start_idx end_idx
  | Float64 -> kernel_acos_float64 a c start_idx end_idx
  | Complex32 -> kernel_acos_complex32 a c start_idx end_idx
  | Complex64 -> kernel_acos_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_acos: unsupported type"

let kernel_atan (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_atan_float16 a c start_idx end_idx
  | Float32 -> kernel_atan_float32 a c start_idx end_idx
  | Float64 -> kernel_atan_float64 a c start_idx end_idx
  | Complex32 -> kernel_atan_complex32 a c start_idx end_idx
  | Complex64 -> kernel_atan_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_atan: unsupported type"

let kernel_sinh (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_sinh_float16 a c start_idx end_idx
  | Float32 -> kernel_sinh_float32 a c start_idx end_idx
  | Float64 -> kernel_sinh_float64 a c start_idx end_idx
  | Complex32 -> kernel_sinh_complex32 a c start_idx end_idx
  | Complex64 -> kernel_sinh_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_sinh: unsupported type"

let kernel_cosh (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_cosh_float16 a c start_idx end_idx
  | Float32 -> kernel_cosh_float32 a c start_idx end_idx
  | Float64 -> kernel_cosh_float64 a c start_idx end_idx
  | Complex32 -> kernel_cosh_complex32 a c start_idx end_idx
  | Complex64 -> kernel_cosh_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_cosh: unsupported type"

let kernel_tanh (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_tanh_float16 a c start_idx end_idx
  | Float32 -> kernel_tanh_float32 a c start_idx end_idx
  | Float64 -> kernel_tanh_float64 a c start_idx end_idx
  | Complex32 -> kernel_tanh_complex32 a c start_idx end_idx
  | Complex64 -> kernel_tanh_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_tanh: unsupported type"

let kernel_asinh (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_asinh_float16 a c start_idx end_idx
  | Float32 -> kernel_asinh_float32 a c start_idx end_idx
  | Float64 -> kernel_asinh_float64 a c start_idx end_idx
  | Complex32 -> kernel_asinh_complex32 a c start_idx end_idx
  | Complex64 -> kernel_asinh_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_asinh: unsupported type"

let kernel_acosh (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_acosh_float16 a c start_idx end_idx
  | Float32 -> kernel_acosh_float32 a c start_idx end_idx
  | Float64 -> kernel_acosh_float64 a c start_idx end_idx
  | Complex32 -> kernel_acosh_complex32 a c start_idx end_idx
  | Complex64 -> kernel_acosh_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_acosh: unsupported type"

let kernel_atanh (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_atanh_float16 a c start_idx end_idx
  | Float32 -> kernel_atanh_float32 a c start_idx end_idx
  | Float64 -> kernel_atanh_float64 a c start_idx end_idx
  | Complex32 -> kernel_atanh_complex32 a c start_idx end_idx
  | Complex64 -> kernel_atanh_complex64 a c start_idx end_idx
  | _ -> invalid_arg "kernel_atanh: unsupported type"

let kernel_bit_not (type a b) (a : (a, b) t) (c : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_bit_not_float16 a c start_idx end_idx
  | Float32 -> kernel_bit_not_float32 a c start_idx end_idx
  | Float64 -> kernel_bit_not_float64 a c start_idx end_idx
  | Int8_signed -> kernel_bit_not_int8 a c start_idx end_idx
  | Int8_unsigned -> kernel_bit_not_uint8 a c start_idx end_idx
  | Int16_signed -> kernel_bit_not_int16 a c start_idx end_idx
  | Int16_unsigned -> kernel_bit_not_uint16 a c start_idx end_idx
  | Int32 -> kernel_bit_not_int32 a c start_idx end_idx
  | Int64 -> kernel_bit_not_int64 a c start_idx end_idx
  | Int -> kernel_bit_not_int a c start_idx end_idx
  | Nativeint -> kernel_bit_not_nativeint a c start_idx end_idx
  | _ -> invalid_arg "kernel_bit_not: unsupported type"

let kernel_floor (type b) (a : (float, b) t) (c : (float, b) t) start_idx
    end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_floor_float16 a c start_idx end_idx
  | Float32 -> kernel_floor_float32 a c start_idx end_idx
  | Float64 -> kernel_floor_float64 a c start_idx end_idx

let kernel_ceil (type b) (a : (float, b) t) (c : (float, b) t) start_idx end_idx
    =
  match Array1.kind a.buffer with
  | Float16 -> kernel_ceil_float16 a c start_idx end_idx
  | Float32 -> kernel_ceil_float32 a c start_idx end_idx
  | Float64 -> kernel_ceil_float64 a c start_idx end_idx

let kernel_round (type b) (a : (float, b) t) (c : (float, b) t) start_idx
    end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_round_float16 a c start_idx end_idx
  | Float32 -> kernel_round_float32 a c start_idx end_idx
  | Float64 -> kernel_round_float64 a c start_idx end_idx

let kernel_isnan (type b) (a : (float, b) t) (c : (int, uint8_elt) t) start_idx
    end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_isnan_float16 a c start_idx end_idx
  | Float32 -> kernel_isnan_float32 a c start_idx end_idx
  | Float64 -> kernel_isnan_float64 a c start_idx end_idx

let kernel_isinf (type b) (a : (float, b) t) (c : (int, uint8_elt) t) start_idx
    end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_isinf_float16 a c start_idx end_idx
  | Float32 -> kernel_isinf_float32 a c start_idx end_idx
  | Float64 -> kernel_isinf_float64 a c start_idx end_idx

let kernel_isfinite (type b) (a : (float, b) t) (c : (int, uint8_elt) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_isfinite_float16 a c start_idx end_idx
  | Float32 -> kernel_isfinite_float32 a c start_idx end_idx
  | Float64 -> kernel_isfinite_float64 a c start_idx end_idx

let neg context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_neg a out start_idx end_idx)

let abs context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_abs a out start_idx end_idx)

let sign context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_sign a out start_idx end_idx)

let sqrt context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_sqrt a out start_idx end_idx)

let exp context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_exp a out start_idx end_idx)

let log context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_log a out start_idx end_idx)

let sin context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_sin a out start_idx end_idx)

let cos context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_cos a out start_idx end_idx)

let tan context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_tan a out start_idx end_idx)

let asin context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_asin a out start_idx end_idx)

let acos context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_acos a out start_idx end_idx)

let atan context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_atan a out start_idx end_idx)

let sinh context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_sinh a out start_idx end_idx)

let cosh context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_cosh a out start_idx end_idx)

let tanh context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_tanh a out start_idx end_idx)

let asinh context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_asinh a out start_idx end_idx)

let acosh context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_acosh a out start_idx end_idx)

let atanh context a out =
  let size = Array.fold_left ( * ) 1 (shape a) in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_atanh a out start_idx end_idx)

let bit_not context a out =
  let n = size out in
  Parallel.parallel_for context.pool 0 n (fun s e -> kernel_bit_not a out s e)

let floor context a out =
  let n = size out in
  Parallel.parallel_for context.pool 0 n (fun s e -> kernel_floor a out s e)

let ceil context a out =
  let n = size out in
  Parallel.parallel_for context.pool 0 n (fun s e -> kernel_ceil a out s e)

let round context a out =
  let n = size out in
  Parallel.parallel_for context.pool 0 n (fun s e -> kernel_round a out s e)

let isnan context a out =
  let n = size out in
  Parallel.parallel_for context.pool 0 n (fun s e -> kernel_isnan a out s e)

let isinf context a out =
  let n = size out in
  Parallel.parallel_for context.pool 0 n (fun s e -> kernel_isinf a out s e)

let isfinite context a out =
  let n = size out in
  Parallel.parallel_for context.pool 0 n (fun s e -> kernel_isfinite a out s e)
