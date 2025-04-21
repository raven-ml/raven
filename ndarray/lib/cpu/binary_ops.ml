open Bigarray
open Ndarray_core
open Internal

let bool_to_int b = if b then 1 else 0

let rec int_pow (a : int) = function
  | 0 -> 1
  | 1 -> a
  | n ->
      let b = int_pow a (n / 2) in
      b * b * if n mod 2 = 0 then 1 else a

let rec int32_pow (a : int32) =
  let open Int32 in
  function
  | 0l -> 1l
  | 1l -> a
  | n ->
      let b = int32_pow a (div n 2l) in
      mul b (mul b (if rem n 2l = 0l then 1l else a))

let rec int64_pow (a : int64) =
  let open Int64 in
  function
  | 0L -> 1L
  | 1L -> a
  | n ->
      let b = int64_pow a (div n 2L) in
      mul b (mul b (if rem n 2L = 0L then 1L else a))

let rec nativeint_pow (a : nativeint) =
  let open Nativeint in
  function
  | 0n -> 1n
  | 1n -> a
  | n ->
      let b = nativeint_pow a (div n 2n) in
      mul b (mul b (if rem n 2n = 0n then 1n else a))

let trunc f = if f >= 0. then Float.floor f else Float.ceil f

let complex_rem (x : Complex.t) (y : Complex.t) : Complex.t =
  let q = Complex.div x y in
  let qf = Complex.{ re = trunc q.re; im = trunc q.im } in
  Complex.sub x (Complex.mul y qf)

let complex_max x y = if x.Complex.re > y.Complex.re then x else y
let complex_min x y = if x.Complex.re < y.Complex.re then x else y

let logand_float (x : float) (y : float) : float =
  let open Int64 in
  float_of_bits (logand (bits_of_float x) (bits_of_float y))

let logand_complex (a : Complex.t) (b : Complex.t) : Complex.t =
  Complex.{ re = logand_float a.re b.re; im = logand_float a.im b.im }

let logor_float (x : float) (y : float) : float =
  let open Int64 in
  float_of_bits (logor (bits_of_float x) (bits_of_float y))

let logor_complex (a : Complex.t) (b : Complex.t) : Complex.t =
  Complex.{ re = logor_float a.re b.re; im = logor_float a.im b.im }

let logxor_float (x : float) (y : float) : float =
  let open Int64 in
  float_of_bits (logxor (bits_of_float x) (bits_of_float y))

let logxor_complex (a : Complex.t) (b : Complex.t) : Complex.t =
  Complex.{ re = logxor_float a.re b.re; im = logxor_float a.im b.im }

let kernel_add_float16 (a : (float, float16_elt) t) (b : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.add a_val b_val)
    done

let kernel_add_float32 (a : (float, float32_elt) t) (b : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.add a_val b_val)
    done

let kernel_add_float64 (a : (float, float64_elt) t) (b : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.add a_val b_val)
    done

let kernel_add_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.add a_val b_val)
    done

let kernel_add_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.add a_val b_val)
    done

let kernel_add_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.add a_val b_val)
    done

let kernel_add_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.add a_val b_val)
    done

let kernel_add_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int32.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int32.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int32.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int32.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int32.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int32.add a_val b_val)
    done

let kernel_add_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int64.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int64.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int64.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int64.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int64.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int64.add a_val b_val)
    done

let kernel_add_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.add a_val b_val)
    done

let kernel_add_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Nativeint.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Nativeint.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Nativeint.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Nativeint.add a_val b_val)
    done

let kernel_add_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Complex.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Complex.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Complex.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Complex.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Complex.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.add a_val b_val)
    done

let kernel_add_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Complex.add a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Complex.add a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Complex.add a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Complex.add a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Complex.add a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.add a_val b_val)
    done

let kernel_sub_float16 (a : (float, float16_elt) t) (b : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.sub a_val b_val)
    done

let kernel_sub_float32 (a : (float, float32_elt) t) (b : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.sub a_val b_val)
    done

let kernel_sub_float64 (a : (float, float64_elt) t) (b : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.sub a_val b_val)
    done

let kernel_sub_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.sub a_val b_val)
    done

let kernel_sub_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.sub a_val b_val)
    done

let kernel_sub_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.sub a_val b_val)
    done

let kernel_sub_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.sub a_val b_val)
    done

let kernel_sub_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int32.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int32.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int32.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int32.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int32.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int32.sub a_val b_val)
    done

let kernel_sub_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int64.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int64.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int64.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int64.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int64.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int64.sub a_val b_val)
    done

let kernel_sub_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.sub a_val b_val)
    done

let kernel_sub_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Nativeint.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Nativeint.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Nativeint.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Nativeint.sub a_val b_val)
    done

let kernel_sub_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Complex.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Complex.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Complex.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Complex.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Complex.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.sub a_val b_val)
    done

let kernel_sub_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Complex.sub a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Complex.sub a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Complex.sub a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Complex.sub a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Complex.sub a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.sub a_val b_val)
    done

let kernel_mul_float16 (a : (float, float16_elt) t) (b : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.mul a_val b_val)
    done

let kernel_mul_float32 (a : (float, float32_elt) t) (b : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.mul a_val b_val)
    done

let kernel_mul_float64 (a : (float, float64_elt) t) (b : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.mul a_val b_val)
    done

let kernel_mul_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.mul a_val b_val)
    done

let kernel_mul_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.mul a_val b_val)
    done

let kernel_mul_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.mul a_val b_val)
    done

let kernel_mul_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.mul a_val b_val)
    done

let kernel_mul_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int32.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int32.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int32.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int32.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int32.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int32.mul a_val b_val)
    done

let kernel_mul_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int64.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int64.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int64.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int64.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int64.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int64.mul a_val b_val)
    done

let kernel_mul_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.mul a_val b_val)
    done

let kernel_mul_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Nativeint.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Nativeint.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Nativeint.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Nativeint.mul a_val b_val)
    done

let kernel_mul_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Complex.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Complex.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Complex.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Complex.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Complex.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.mul a_val b_val)
    done

let kernel_mul_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Complex.mul a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Complex.mul a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Complex.mul a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Complex.mul a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Complex.mul a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.mul a_val b_val)
    done

let kernel_div_float16 (a : (float, float16_elt) t) (b : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.div a_val b_val)
    done

let kernel_div_float32 (a : (float, float32_elt) t) (b : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.div a_val b_val)
    done

let kernel_div_float64 (a : (float, float64_elt) t) (b : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.div a_val b_val)
    done

let kernel_div_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.div a_val b_val)
    done

let kernel_div_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.div a_val b_val)
    done

let kernel_div_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.div a_val b_val)
    done

let kernel_div_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.div a_val b_val)
    done

let kernel_div_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int32.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int32.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int32.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int32.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int32.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int32.div a_val b_val)
    done

let kernel_div_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int64.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int64.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int64.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int64.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int64.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int64.div a_val b_val)
    done

let kernel_div_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.div a_val b_val)
    done

let kernel_div_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Nativeint.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Nativeint.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Nativeint.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Nativeint.div a_val b_val)
    done

let kernel_div_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Complex.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Complex.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Complex.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Complex.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Complex.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.div a_val b_val)
    done

let kernel_div_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Complex.div a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Complex.div a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Complex.div a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Complex.div a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Complex.div a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.div a_val b_val)
    done

let kernel_pow_float16 (a : (float, float16_elt) t) (b : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.pow a_val b_val)
    done

let kernel_pow_float32 (a : (float, float32_elt) t) (b : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.pow a_val b_val)
    done

let kernel_pow_float64 (a : (float, float64_elt) t) (b : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.pow a_val b_val)
    done

let kernel_pow_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (int_pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (int_pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (int_pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (int_pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (int_pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (int_pow a_val b_val)
    done

let kernel_pow_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (int_pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (int_pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (int_pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (int_pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (int_pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (int_pow a_val b_val)
    done

let kernel_pow_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (int_pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (int_pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (int_pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (int_pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (int_pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (int_pow a_val b_val)
    done

let kernel_pow_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (int_pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (int_pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (int_pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (int_pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (int_pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (int_pow a_val b_val)
    done

let kernel_pow_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (int32_pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (int32_pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (int32_pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (int32_pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (int32_pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (int32_pow a_val b_val)
    done

let kernel_pow_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (int64_pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (int64_pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (int64_pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (int64_pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (int64_pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (int64_pow a_val b_val)
    done

let kernel_pow_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (int_pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (int_pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (int_pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (int_pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (int_pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (int_pow a_val b_val)
    done

let kernel_pow_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (nativeint_pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (nativeint_pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (nativeint_pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (nativeint_pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (nativeint_pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (nativeint_pow a_val b_val)
    done

let kernel_pow_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Complex.pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Complex.pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Complex.pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Complex.pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Complex.pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.pow a_val b_val)
    done

let kernel_pow_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Complex.pow a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Complex.pow a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Complex.pow a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Complex.pow a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Complex.pow a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.pow a_val b_val)
    done

let kernel_equal_float16 (a : (float, float16_elt) t)
    (b : (float, float16_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.equal a_val b_val))
    done

let kernel_equal_float32 (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.equal a_val b_val))
    done

let kernel_equal_float64 (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Float.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Float.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Float.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Float.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Float.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Float.equal a_val b_val))
    done

let kernel_equal_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Int.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Int.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Int.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Int.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Int.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Int.equal a_val b_val))
    done

let kernel_equal_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Int.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Int.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Int.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Int.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Int.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Int.equal a_val b_val))
    done

let kernel_equal_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Int.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Int.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Int.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Int.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Int.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Int.equal a_val b_val))
    done

let kernel_equal_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Int.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Int.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Int.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Int.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Int.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Int.equal a_val b_val))
    done

let kernel_equal_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Int32.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Int32.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Int32.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Int32.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Int32.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Int32.equal a_val b_val))
    done

let kernel_equal_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Int64.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Int64.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Int64.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Int64.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Int64.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Int64.equal a_val b_val))
    done

let kernel_equal_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Int.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Int.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Int.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Int.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Int.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Int.equal a_val b_val))
    done

let kernel_equal_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (Nativeint.equal a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (Nativeint.equal a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (Nativeint.equal a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (Nativeint.equal a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (Nativeint.equal a_val b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (Nativeint.equal a_val b_val))
    done

let kernel_equal_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 = b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 = b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 = b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 = b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val = b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val = b_val))
    done

let kernel_equal_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 = b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 = b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 = b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 = b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val = b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val = b_val))
    done

let kernel_rem_float16 (a : (float, float16_elt) t) (b : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.rem a_val b_val)
    done

let kernel_rem_float32 (a : (float, float32_elt) t) (b : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.rem a_val b_val)
    done

let kernel_rem_float64 (a : (float, float64_elt) t) (b : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.rem a_val b_val)
    done

let kernel_rem_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.rem a_val b_val)
    done

let kernel_rem_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.rem a_val b_val)
    done

let kernel_rem_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.rem a_val b_val)
    done

let kernel_rem_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.rem a_val b_val)
    done

let kernel_rem_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int32.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int32.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int32.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int32.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int32.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int32.rem a_val b_val)
    done

let kernel_rem_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int64.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int64.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int64.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int64.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int64.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int64.rem a_val b_val)
    done

let kernel_rem_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.rem a_val b_val)
    done

let kernel_rem_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Nativeint.rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Nativeint.rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Nativeint.rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Nativeint.rem a_val b_val)
    done

let kernel_rem_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (complex_rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (complex_rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (complex_rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (complex_rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (complex_rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (complex_rem a_val b_val)
    done

let kernel_rem_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (complex_rem a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (complex_rem a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (complex_rem a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (complex_rem a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (complex_rem a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (complex_rem a_val b_val)
    done

let kernel_max_float16 (a : (float, float16_elt) t) (b : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.max a_val b_val)
    done

let kernel_max_float32 (a : (float, float32_elt) t) (b : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.max a_val b_val)
    done

let kernel_max_float64 (a : (float, float64_elt) t) (b : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.max a_val b_val)
    done

let kernel_max_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.max a_val b_val)
    done

let kernel_max_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.max a_val b_val)
    done

let kernel_max_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.max a_val b_val)
    done

let kernel_max_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.max a_val b_val)
    done

let kernel_max_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int32.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int32.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int32.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int32.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int32.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int32.max a_val b_val)
    done

let kernel_max_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int64.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int64.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int64.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int64.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int64.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int64.max a_val b_val)
    done

let kernel_max_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.max a_val b_val)
    done

let kernel_max_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Nativeint.max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Nativeint.max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Nativeint.max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Nativeint.max a_val b_val)
    done

let kernel_max_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (complex_max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (complex_max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (complex_max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (complex_max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (complex_max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (complex_max a_val b_val)
    done

let kernel_max_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (complex_max a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (complex_max a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (complex_max a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (complex_max a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (complex_max a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (complex_max a_val b_val)
    done

let kernel_min_float16 (a : (float, float16_elt) t) (b : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.min a_val b_val)
    done

let kernel_min_float32 (a : (float, float32_elt) t) (b : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.min a_val b_val)
    done

let kernel_min_float64 (a : (float, float64_elt) t) (b : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Float.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Float.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Float.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Float.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Float.min a_val b_val)
    done

let kernel_min_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.min a_val b_val)
    done

let kernel_min_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.min a_val b_val)
    done

let kernel_min_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.min a_val b_val)
    done

let kernel_min_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.min a_val b_val)
    done

let kernel_min_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int32.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int32.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int32.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int32.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int32.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int32.min a_val b_val)
    done

let kernel_min_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int64.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int64.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int64.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int64.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int64.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int64.min a_val b_val)
    done

let kernel_min_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.min a_val b_val)
    done

let kernel_min_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Nativeint.min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Nativeint.min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Nativeint.min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Nativeint.min a_val b_val)
    done

let kernel_min_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (complex_min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (complex_min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (complex_min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (complex_min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (complex_min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (complex_min a_val b_val)
    done

let kernel_min_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (complex_min a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (complex_min a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (complex_min a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (complex_min a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (complex_min a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (complex_min a_val b_val)
    done

let kernel_greater_float16 (a : (float, float16_elt) t)
    (b : (float, float16_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_float32 (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_float64 (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0.re > b_val0.re));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1.re > b_val1.re));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2.re > b_val2.re));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3.re > b_val3.re));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val.re > b_val.re));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val.re > b_val.re))
    done

let kernel_greater_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 > b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 > b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 > b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 > b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val > b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val > b_val))
    done

let kernel_greater_equal_float16 (a : (float, float16_elt) t)
    (b : (float, float16_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_float32 (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_float64 (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_uint16 (a : (int, uint16_elt) t)
    (b : (int, uint16_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_int32 (a : (int32, int32_elt) t)
    (b : (int32, int32_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_int64 (a : (int64, int64_elt) t)
    (b : (int64, int64_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 >= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 >= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 >= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 >= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val >= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0.re >= b_val0.re));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1.re >= b_val1.re));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2.re >= b_val2.re));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3.re >= b_val3.re));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val.re >= b_val.re));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_greater_equal_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0.re >= b_val0.re));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1.re >= b_val1.re));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2.re >= b_val2.re));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3.re >= b_val3.re));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val.re >= b_val.re));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val >= b_val))
    done

let kernel_less_float16 (a : (float, float16_elt) t)
    (b : (float, float16_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_float32 (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_float64 (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 < b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 < b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 < b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 < b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val < b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0.re < b_val0.re));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1.re < b_val1.re));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2.re < b_val2.re));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3.re < b_val3.re));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val.re < b_val.re));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val.re < b_val.re))
    done

let kernel_less_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0.re < b_val0.re));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1.re < b_val1.re));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2.re < b_val2.re));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3.re < b_val3.re));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val.re < b_val.re));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val < b_val))
    done

let kernel_less_equal_float16 (a : (float, float16_elt) t)
    (b : (float, float16_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_float32 (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_float64 (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_int32 (a : (int32, int32_elt) t)
    (b : (int32, int32_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_int64 (a : (int64, int64_elt) t)
    (b : (int64, int64_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0 <= b_val0));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1 <= b_val1));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2 <= b_val2));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3 <= b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val <= b_val));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val <= b_val))
    done

let kernel_less_equal_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0.re <= b_val0.re));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1.re <= b_val1.re));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2.re <= b_val2.re));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3.re <= b_val3.re));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val.re <= b_val.re));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val.re <= b_val.re))
    done

let kernel_less_equal_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (int, uint8_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (bool_to_int (a_val0.re <= b_val0.re));
      Array1.unsafe_set out_buf i1 (bool_to_int (a_val1.re <= b_val1.re));
      Array1.unsafe_set out_buf i2 (bool_to_int (a_val2.re <= b_val2.re));
      Array1.unsafe_set out_buf i3 (bool_to_int (a_val3.re <= b_val3.re));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (a_val.re <= b_val.re));
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (bool_to_int (a_val.re <= b_val.re))
    done

let kernel_bit_and_float16 (a : (float, float16_elt) t)
    (b : (float, float16_elt) t) (out : (float, float16_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logand_float a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logand_float a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logand_float a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logand_float a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logand_float a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logand_float a_val b_val)
    done

let kernel_bit_and_float32 (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (float, float32_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logand_float a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logand_float a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logand_float a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logand_float a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logand_float a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logand_float a_val b_val)
    done

let kernel_bit_and_float64 (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (float, float64_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logand_float a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logand_float a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logand_float a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logand_float a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logand_float a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logand_float a_val b_val)
    done

let kernel_bit_and_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logand a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logand a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logand a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logand a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logand a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logand a_val b_val)
    done

let kernel_bit_and_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logand a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logand a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logand a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logand a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logand a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logand a_val b_val)
    done

let kernel_bit_and_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logand a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logand a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logand a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logand a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logand a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logand a_val b_val)
    done

let kernel_bit_and_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logand a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logand a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logand a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logand a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logand a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logand a_val b_val)
    done

let kernel_bit_and_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int32.logand a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int32.logand a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int32.logand a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int32.logand a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int32.logand a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int32.logand a_val b_val)
    done

let kernel_bit_and_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int64.logand a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int64.logand a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int64.logand a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int64.logand a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int64.logand a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int64.logand a_val b_val)
    done

let kernel_bit_and_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logand a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logand a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logand a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logand a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logand a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logand a_val b_val)
    done

let kernel_bit_and_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.logand a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Nativeint.logand a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Nativeint.logand a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Nativeint.logand a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.logand a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Nativeint.logand a_val b_val)
    done

let kernel_bit_and_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logand_complex a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logand_complex a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logand_complex a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logand_complex a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logand_complex a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logand_complex a_val b_val)
    done

let kernel_bit_and_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logand_complex a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logand_complex a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logand_complex a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logand_complex a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logand_complex a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logand_complex a_val b_val)
    done

let kernel_bit_or_float16 (a : (float, float16_elt) t)
    (b : (float, float16_elt) t) (out : (float, float16_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logor_float a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logor_float a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logor_float a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logor_float a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logor_float a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logor_float a_val b_val)
    done

let kernel_bit_or_float32 (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (float, float32_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logor_float a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logor_float a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logor_float a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logor_float a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logor_float a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logor_float a_val b_val)
    done

let kernel_bit_or_float64 (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (float, float64_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logor_float a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logor_float a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logor_float a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logor_float a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logor_float a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logor_float a_val b_val)
    done

let kernel_bit_or_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logor a_val b_val)
    done

let kernel_bit_or_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logor a_val b_val)
    done

let kernel_bit_or_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logor a_val b_val)
    done

let kernel_bit_or_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logor a_val b_val)
    done

let kernel_bit_or_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int32.logor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int32.logor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int32.logor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int32.logor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int32.logor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int32.logor a_val b_val)
    done

let kernel_bit_or_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int64.logor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int64.logor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int64.logor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int64.logor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int64.logor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int64.logor a_val b_val)
    done

let kernel_bit_or_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logor a_val b_val)
    done

let kernel_bit_or_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.logor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Nativeint.logor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Nativeint.logor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Nativeint.logor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.logor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Nativeint.logor a_val b_val)
    done

let kernel_bit_or_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logor_complex a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logor_complex a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logor_complex a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logor_complex a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logor_complex a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logor_complex a_val b_val)
    done

let kernel_bit_or_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logor_complex a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logor_complex a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logor_complex a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logor_complex a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logor_complex a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logor_complex a_val b_val)
    done

let kernel_bit_xor_float16 (a : (float, float16_elt) t)
    (b : (float, float16_elt) t) (out : (float, float16_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logxor_float a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logxor_float a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logxor_float a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logxor_float a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logxor_float a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logxor_float a_val b_val)
    done

let kernel_bit_xor_float32 (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (float, float32_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logxor_float a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logxor_float a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logxor_float a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logxor_float a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logxor_float a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logxor_float a_val b_val)
    done

let kernel_bit_xor_float64 (a : (float, float64_elt) t)
    (b : (float, float64_elt) t) (out : (float, float64_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logxor_float a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logxor_float a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logxor_float a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logxor_float a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logxor_float a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logxor_float a_val b_val)
    done

let kernel_bit_xor_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logxor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logxor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logxor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logxor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logxor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logxor a_val b_val)
    done

let kernel_bit_xor_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logxor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logxor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logxor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logxor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logxor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logxor a_val b_val)
    done

let kernel_bit_xor_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logxor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logxor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logxor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logxor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logxor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logxor a_val b_val)
    done

let kernel_bit_xor_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logxor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logxor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logxor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logxor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logxor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logxor a_val b_val)
    done

let kernel_bit_xor_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int32.logxor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int32.logxor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int32.logxor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int32.logxor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int32.logxor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int32.logxor a_val b_val)
    done

let kernel_bit_xor_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int64.logxor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int64.logxor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int64.logxor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int64.logxor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int64.logxor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int64.logxor a_val b_val)
    done

let kernel_bit_xor_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Int.logxor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Int.logxor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Int.logxor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Int.logxor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Int.logxor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Int.logxor a_val b_val)
    done

let kernel_bit_xor_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (out : (nativeint, nativeint_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (Nativeint.logxor a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (Nativeint.logxor a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (Nativeint.logxor a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (Nativeint.logxor a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Nativeint.logxor a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Nativeint.logxor a_val b_val)
    done

let kernel_bit_xor_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (out : (Complex.t, complex32_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logxor_complex a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logxor_complex a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logxor_complex a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logxor_complex a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logxor_complex a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logxor_complex a_val b_val)
    done

let kernel_bit_xor_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (out : (Complex.t, complex64_elt) t)
    start_idx end_idx =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  if is_c_contiguous a && is_c_contiguous b then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      Array1.unsafe_set out_buf i0 (logxor_complex a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (logxor_complex a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (logxor_complex a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (logxor_complex a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (logxor_complex a_val b_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape out) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (logxor_complex a_val b_val)
    done

let kernel_add (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_add_float16 a b out start_idx end_idx
  | Float32 -> kernel_add_float32 a b out start_idx end_idx
  | Float64 -> kernel_add_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_add_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_add_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_add_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_add_uint16 a b out start_idx end_idx
  | Int32 -> kernel_add_int32 a b out start_idx end_idx
  | Int64 -> kernel_add_int64 a b out start_idx end_idx
  | Int -> kernel_add_int a b out start_idx end_idx
  | Nativeint -> kernel_add_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_add_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_add_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_add: unsupported type"

let kernel_sub (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_sub_float16 a b out start_idx end_idx
  | Float32 -> kernel_sub_float32 a b out start_idx end_idx
  | Float64 -> kernel_sub_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_sub_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_sub_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_sub_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_sub_uint16 a b out start_idx end_idx
  | Int32 -> kernel_sub_int32 a b out start_idx end_idx
  | Int64 -> kernel_sub_int64 a b out start_idx end_idx
  | Int -> kernel_sub_int a b out start_idx end_idx
  | Nativeint -> kernel_sub_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_sub_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_sub_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_sub: unsupported type"

let kernel_mul (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_mul_float16 a b out start_idx end_idx
  | Float32 -> kernel_mul_float32 a b out start_idx end_idx
  | Float64 -> kernel_mul_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_mul_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_mul_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_mul_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_mul_uint16 a b out start_idx end_idx
  | Int32 -> kernel_mul_int32 a b out start_idx end_idx
  | Int64 -> kernel_mul_int64 a b out start_idx end_idx
  | Int -> kernel_mul_int a b out start_idx end_idx
  | Nativeint -> kernel_mul_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_mul_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_mul_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_mul: unsupported type"

let kernel_div (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_div_float16 a b out start_idx end_idx
  | Float32 -> kernel_div_float32 a b out start_idx end_idx
  | Float64 -> kernel_div_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_div_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_div_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_div_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_div_uint16 a b out start_idx end_idx
  | Int32 -> kernel_div_int32 a b out start_idx end_idx
  | Int64 -> kernel_div_int64 a b out start_idx end_idx
  | Int -> kernel_div_int a b out start_idx end_idx
  | Nativeint -> kernel_div_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_div_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_div_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_div: unsupported type"

let kernel_pow (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_pow_float16 a b out start_idx end_idx
  | Float32 -> kernel_pow_float32 a b out start_idx end_idx
  | Float64 -> kernel_pow_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_pow_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_pow_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_pow_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_pow_uint16 a b out start_idx end_idx
  | Int32 -> kernel_pow_int32 a b out start_idx end_idx
  | Int64 -> kernel_pow_int64 a b out start_idx end_idx
  | Int -> kernel_pow_int a b out start_idx end_idx
  | Nativeint -> kernel_pow_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_pow_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_pow_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_pow: unsupported type"

let kernel_equal (type a b) (a : (a, b) t) (b : (a, b) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_equal_float16 a b out start_idx end_idx
  | Float32 -> kernel_equal_float32 a b out start_idx end_idx
  | Float64 -> kernel_equal_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_equal_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_equal_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_equal_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_equal_uint16 a b out start_idx end_idx
  | Int32 -> kernel_equal_int32 a b out start_idx end_idx
  | Int64 -> kernel_equal_int64 a b out start_idx end_idx
  | Int -> kernel_equal_int a b out start_idx end_idx
  | Nativeint -> kernel_equal_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_equal_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_equal_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_equal: unsupported type"

let kernel_rem (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_rem_float16 a b out start_idx end_idx
  | Float32 -> kernel_rem_float32 a b out start_idx end_idx
  | Float64 -> kernel_rem_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_rem_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_rem_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_rem_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_rem_uint16 a b out start_idx end_idx
  | Int32 -> kernel_rem_int32 a b out start_idx end_idx
  | Int64 -> kernel_rem_int64 a b out start_idx end_idx
  | Int -> kernel_rem_int a b out start_idx end_idx
  | Nativeint -> kernel_rem_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_rem_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_rem_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_rem: unsupported type"

let kernel_max (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_max_float16 a b out start_idx end_idx
  | Float32 -> kernel_max_float32 a b out start_idx end_idx
  | Float64 -> kernel_max_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_max_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_max_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_max_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_max_uint16 a b out start_idx end_idx
  | Int32 -> kernel_max_int32 a b out start_idx end_idx
  | Int64 -> kernel_max_int64 a b out start_idx end_idx
  | Int -> kernel_max_int a b out start_idx end_idx
  | Nativeint -> kernel_max_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_max_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_max_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_max: unsupported type"

let kernel_min (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_min_float16 a b out start_idx end_idx
  | Float32 -> kernel_min_float32 a b out start_idx end_idx
  | Float64 -> kernel_min_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_min_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_min_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_min_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_min_uint16 a b out start_idx end_idx
  | Int32 -> kernel_min_int32 a b out start_idx end_idx
  | Int64 -> kernel_min_int64 a b out start_idx end_idx
  | Int -> kernel_min_int a b out start_idx end_idx
  | Nativeint -> kernel_min_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_min_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_min_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_min: unsupported type"

let kernel_greater (type a b) (a : (a, b) t) (b : (a, b) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_greater_float16 a b out start_idx end_idx
  | Float32 -> kernel_greater_float32 a b out start_idx end_idx
  | Float64 -> kernel_greater_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_greater_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_greater_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_greater_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_greater_uint16 a b out start_idx end_idx
  | Int32 -> kernel_greater_int32 a b out start_idx end_idx
  | Int64 -> kernel_greater_int64 a b out start_idx end_idx
  | Int -> kernel_greater_int a b out start_idx end_idx
  | Nativeint -> kernel_greater_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_greater_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_greater_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_greater: unsupported type"

let kernel_greater_equal (type a b) (a : (a, b) t) (b : (a, b) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_greater_equal_float16 a b out start_idx end_idx
  | Float32 -> kernel_greater_equal_float32 a b out start_idx end_idx
  | Float64 -> kernel_greater_equal_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_greater_equal_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_greater_equal_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_greater_equal_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_greater_equal_uint16 a b out start_idx end_idx
  | Int32 -> kernel_greater_equal_int32 a b out start_idx end_idx
  | Int64 -> kernel_greater_equal_int64 a b out start_idx end_idx
  | Int -> kernel_greater_equal_int a b out start_idx end_idx
  | Nativeint -> kernel_greater_equal_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_greater_equal_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_greater_equal_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_greater_equal: unsupported type"

let kernel_less (type a b) (a : (a, b) t) (b : (a, b) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_less_float16 a b out start_idx end_idx
  | Float32 -> kernel_less_float32 a b out start_idx end_idx
  | Float64 -> kernel_less_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_less_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_less_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_less_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_less_uint16 a b out start_idx end_idx
  | Int32 -> kernel_less_int32 a b out start_idx end_idx
  | Int64 -> kernel_less_int64 a b out start_idx end_idx
  | Int -> kernel_less_int a b out start_idx end_idx
  | Nativeint -> kernel_less_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_less_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_less_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_less: unsupported type"

let kernel_less_equal (type a b) (a : (a, b) t) (b : (a, b) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_less_equal_float16 a b out start_idx end_idx
  | Float32 -> kernel_less_equal_float32 a b out start_idx end_idx
  | Float64 -> kernel_less_equal_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_less_equal_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_less_equal_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_less_equal_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_less_equal_uint16 a b out start_idx end_idx
  | Int32 -> kernel_less_equal_int32 a b out start_idx end_idx
  | Int64 -> kernel_less_equal_int64 a b out start_idx end_idx
  | Int -> kernel_less_equal_int a b out start_idx end_idx
  | Nativeint -> kernel_less_equal_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_less_equal_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_less_equal_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_less_equal: unsupported type"

let kernel_bit_and (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_bit_and_float16 a b out start_idx end_idx
  | Float32 -> kernel_bit_and_float32 a b out start_idx end_idx
  | Float64 -> kernel_bit_and_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_bit_and_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_bit_and_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_bit_and_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_bit_and_uint16 a b out start_idx end_idx
  | Int32 -> kernel_bit_and_int32 a b out start_idx end_idx
  | Int64 -> kernel_bit_and_int64 a b out start_idx end_idx
  | Int -> kernel_bit_and_int a b out start_idx end_idx
  | Nativeint -> kernel_bit_and_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_bit_and_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_bit_and_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_bit_and: unsupported type"

let kernel_bit_or (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_bit_or_float16 a b out start_idx end_idx
  | Float32 -> kernel_bit_or_float32 a b out start_idx end_idx
  | Float64 -> kernel_bit_or_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_bit_or_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_bit_or_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_bit_or_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_bit_or_uint16 a b out start_idx end_idx
  | Int32 -> kernel_bit_or_int32 a b out start_idx end_idx
  | Int64 -> kernel_bit_or_int64 a b out start_idx end_idx
  | Int -> kernel_bit_or_int a b out start_idx end_idx
  | Nativeint -> kernel_bit_or_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_bit_or_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_bit_or_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_bit_or: unsupported type"

let kernel_bit_xor (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_bit_xor_float16 a b out start_idx end_idx
  | Float32 -> kernel_bit_xor_float32 a b out start_idx end_idx
  | Float64 -> kernel_bit_xor_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_bit_xor_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_bit_xor_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_bit_xor_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_bit_xor_uint16 a b out start_idx end_idx
  | Int32 -> kernel_bit_xor_int32 a b out start_idx end_idx
  | Int64 -> kernel_bit_xor_int64 a b out start_idx end_idx
  | Int -> kernel_bit_xor_int a b out start_idx end_idx
  | Nativeint -> kernel_bit_xor_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_bit_xor_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_bit_xor_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_bit_xor: unsupported type"

let add (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_add a b out start_idx end_idx)

let sub (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_sub a b out start_idx end_idx)

let mul (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_mul a b out start_idx end_idx)

let div (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_div a b out start_idx end_idx)

let pow (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_pow a b out start_idx end_idx)

let equal (type a b) context (a : (a, b) t) (b : (a, b) t)
    (out : (int, int8_unsigned_elt) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_equal a b out start_idx end_idx)

let rem (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_rem a b out start_idx end_idx)

let max (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_max a b out start_idx end_idx)

let min (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_min a b out start_idx end_idx)

let greater (type a b) context (a : (a, b) t) (b : (a, b) t)
    (out : (int, int8_unsigned_elt) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_greater a b out start_idx end_idx)

let greater_equal (type a b) context (a : (a, b) t) (b : (a, b) t)
    (out : (int, int8_unsigned_elt) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_greater_equal a b out start_idx end_idx)

let less (type a b) context (a : (a, b) t) (b : (a, b) t)
    (out : (int, int8_unsigned_elt) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_less a b out start_idx end_idx)

let less_equal (type a b) context (a : (a, b) t) (b : (a, b) t)
    (out : (int, int8_unsigned_elt) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_less_equal a b out start_idx end_idx)

let bit_and (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_bit_and a b out start_idx end_idx)

let bit_or (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_bit_or a b out start_idx end_idx)

let bit_xor (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_bit_xor a b out start_idx end_idx)
