open Bigarray
open Nx_core.Dtype
module Shape = Nx_core.Shape
open Internal

let[@inline] bool_to_int b = if b then 1 else 0

let rec int_pow (a : int) = function
  | 0 -> 1
  | 1 -> a
  | n ->
      let b = int_pow a (n / 2) in
      b * b * if n mod 2 = 0 then 1 else a

let rec int32_pow (a : int32) = function
  | 0l -> 1l
  | 1l -> a
  | n ->
      let b = int32_pow a (Int32.div n 2l) in
      Int32.mul b (Int32.mul b (if Int32.rem n 2l = 0l then 1l else a))

let rec int64_pow (a : int64) = function
  | 0L -> 1L
  | 1L -> a
  | n ->
      let b = int64_pow a (Int64.div n 2L) in
      Int64.mul b (Int64.mul b (if Int64.rem n 2L = 0L then 1L else a))

let rec nativeint_pow (a : nativeint) = function
  | 0n -> 1n
  | 1n -> a
  | n ->
      let b = nativeint_pow a (Nativeint.div n 2n) in
      Nativeint.mul b
        (Nativeint.mul b (if Nativeint.rem n 2n = 0n then 1n else a))

let[@inline] trunc f = if f >= 0. then Float.floor f else Float.ceil f

let[@inline] complex_modulo x y =
  let q = Complex.div x y in
  let qf = Complex.{ re = trunc q.re; im = trunc q.im } in
  Complex.sub x (Complex.mul y qf)

let[@inline] complex_max x y = if x.Complex.re > y.Complex.re then x else y

let[@inline] complex_idiv x y =
  let q = Complex.div x y in
  { Complex.re = Float.trunc q.re; im = Float.trunc q.im }

let[@inline] logand_float (x : float) (y : float) : float =
  let open Int64 in
  float_of_bits (logand (bits_of_float x) (bits_of_float y))

let[@inline] logand_complex (a : Complex.t) (b : Complex.t) : Complex.t =
  Complex.{ re = logand_float a.re b.re; im = logand_float a.im b.im }

let[@inline] logor_float (x : float) (y : float) : float =
  Int64.float_of_bits
    (Int64.logor (Int64.bits_of_float x) (Int64.bits_of_float y))

let[@inline] logor_complex (a : Complex.t) (b : Complex.t) : Complex.t =
  Complex.{ re = logor_float a.re b.re; im = logor_float a.im b.im }

let[@inline] logxor_float (x : float) (y : float) : float =
  Int64.float_of_bits
    (Int64.logxor (Int64.bits_of_float x) (Int64.bits_of_float y))

let[@inline] logxor_complex (a : Complex.t) (b : Complex.t) : Complex.t =
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.add a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.sub a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.mul a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.mul a_val b_val)
    done

let kernel_fdiv_float16 (a : (float, float16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.div a_val b_val)
    done

let kernel_fdiv_float32 (a : (float, float32_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.div a_val b_val)
    done

let kernel_fdiv_float64 (a : (float, float64_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.div a_val b_val)
    done

let kernel_fdiv_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.div a_val b_val)
    done

let kernel_fdiv_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.div a_val b_val)
    done

let kernel_fdiv_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.div a_val b_val)
    done

let kernel_fdiv_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.div a_val b_val)
    done

let kernel_fdiv_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.div a_val b_val)
    done

let kernel_fdiv_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.div a_val b_val)
    done

let kernel_fdiv_int (a : (int, int_elt) t) (b : (int, int_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.div a_val b_val)
    done

let kernel_fdiv_nativeint (a : (nativeint, nativeint_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.div a_val b_val)
    done

let kernel_fdiv_complex32 (a : (Complex.t, complex32_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.div a_val b_val)
    done

let kernel_fdiv_complex64 (a : (Complex.t, complex64_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.div a_val b_val)
    done

let kernel_idiv_float16 (a : (float, float16_elt) t)
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
      Array1.unsafe_set out_buf i0 (Float.trunc (Float.div a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (Float.trunc (Float.div a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (Float.trunc (Float.div a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (Float.trunc (Float.div a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.trunc (Float.div a_val b_val));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.trunc (Float.div a_val b_val))
    done

let kernel_idiv_float32 (a : (float, float32_elt) t)
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
      Array1.unsafe_set out_buf i0 (Float.trunc (Float.div a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (Float.trunc (Float.div a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (Float.trunc (Float.div a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (Float.trunc (Float.div a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.trunc (Float.div a_val b_val));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.trunc (Float.div a_val b_val))
    done

let kernel_idiv_float64 (a : (float, float64_elt) t)
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
      Array1.unsafe_set out_buf i0 (Float.trunc (Float.div a_val0 b_val0));
      Array1.unsafe_set out_buf i1 (Float.trunc (Float.div a_val1 b_val1));
      Array1.unsafe_set out_buf i2 (Float.trunc (Float.div a_val2 b_val2));
      Array1.unsafe_set out_buf i3 (Float.trunc (Float.div a_val3 b_val3));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (Float.trunc (Float.div a_val b_val));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.trunc (Float.div a_val b_val))
    done

let kernel_idiv_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.div a_val b_val)
    done

let kernel_idiv_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.div a_val b_val)
    done

let kernel_idiv_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.div a_val b_val)
    done

let kernel_idiv_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.div a_val b_val)
    done

let kernel_idiv_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.div a_val b_val)
    done

let kernel_idiv_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.div a_val b_val)
    done

let kernel_idiv_int (a : (int, int_elt) t) (b : (int, int_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.div a_val b_val)
    done

let kernel_idiv_nativeint (a : (nativeint, nativeint_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.div a_val b_val)
    done

let kernel_idiv_complex32 (a : (Complex.t, complex32_elt) t)
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
      Array1.unsafe_set out_buf i0 (complex_idiv a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (complex_idiv a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (complex_idiv a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (complex_idiv a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (complex_idiv a_val b_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_idiv a_val b_val)
    done

let kernel_idiv_complex64 (a : (Complex.t, complex64_elt) t)
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
      Array1.unsafe_set out_buf i0 (complex_idiv a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (complex_idiv a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (complex_idiv a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (complex_idiv a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (complex_idiv a_val b_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_idiv a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (int_pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (int_pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (int_pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (int_pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (int32_pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (int64_pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (int_pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (nativeint_pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Complex.pow a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (Float.equal a_val b_val))
    done

let kernel_modulo_float16 (a : (float, float16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.rem a_val b_val)
    done

let kernel_modulo_float32 (a : (float, float32_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.rem a_val b_val)
    done

let kernel_modulo_float64 (a : (float, float64_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.rem a_val b_val)
    done

let kernel_modulo_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.rem a_val b_val)
    done

let kernel_modulo_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.rem a_val b_val)
    done

let kernel_modulo_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.rem a_val b_val)
    done

let kernel_modulo_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.rem a_val b_val)
    done

let kernel_modulo_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.rem a_val b_val)
    done

let kernel_modulo_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.rem a_val b_val)
    done

let kernel_modulo_int (a : (int, int_elt) t) (b : (int, int_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.rem a_val b_val)
    done

let kernel_modulo_nativeint (a : (nativeint, nativeint_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.rem a_val b_val)
    done

let kernel_modulo_complex32 (a : (Complex.t, complex32_elt) t)
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
      Array1.unsafe_set out_buf i0 (complex_modulo a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (complex_modulo a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (complex_modulo a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (complex_modulo a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (complex_modulo a_val b_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_modulo a_val b_val)
    done

let kernel_modulo_complex64 (a : (Complex.t, complex64_elt) t)
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
      Array1.unsafe_set out_buf i0 (complex_modulo a_val0 b_val0);
      Array1.unsafe_set out_buf i1 (complex_modulo a_val1 b_val1);
      Array1.unsafe_set out_buf i2 (complex_modulo a_val2 b_val2);
      Array1.unsafe_set out_buf i3 (complex_modulo a_val3 b_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (complex_modulo a_val b_val);
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_modulo a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Float.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_max a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (complex_max a_val b_val)
    done

let kernel_cmplt_float16 (a : (float, float16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_float32 (a : (float, float32_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_float64 (a : (float, float64_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_int (a : (int, int_elt) t) (b : (int, int_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_nativeint (a : (nativeint, nativeint_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmplt_complex32 (a : (Complex.t, complex32_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val.re < b_val.re))
    done

let kernel_cmplt_complex64 (a : (Complex.t, complex64_elt) t)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (a_val < b_val))
    done

let kernel_cmpne_float16 (a : (float, float16_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_float32 (a : (float, float32_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_float64 (a : (float, float64_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_int (a : (int, int_elt) t) (b : (int, int_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_nativeint (a : (nativeint, nativeint_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_complex32 (a : (Complex.t, complex32_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
    done

let kernel_cmpne_complex64 (a : (Complex.t, complex64_elt) t)
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
      Array1.unsafe_set out_buf i0 (bool_to_int (not (a_val0 = b_val0)));
      Array1.unsafe_set out_buf i1 (bool_to_int (not (a_val1 = b_val1)));
      Array1.unsafe_set out_buf i2 (bool_to_int (not (a_val2 = b_val2)));
      Array1.unsafe_set out_buf i3 (bool_to_int (not (a_val3 = b_val3)));
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      Array1.unsafe_set out_buf idx (bool_to_int (not (a_val = b_val)));
      incr i
    done)
  else
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (bool_to_int (not (a_val = b_val)))
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logand_float a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logand_float a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logand_float a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logand a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logand a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logand a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logand a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.logand a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.logand a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logand a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.logand a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logand_complex a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logand_complex a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logor_float a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logor_float a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logor_float a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.logor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.logor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.logor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logor_complex a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logor_complex a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logxor_float a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logxor_float a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logxor_float a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logxor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logxor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logxor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logxor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int32.logxor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int64.logxor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Int.logxor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (Nativeint.logxor a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logxor_complex a_val b_val)
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
    (* Pre-allocate work array to avoid allocations in loop *)
    let md_index = Array.make (Array.length (shape out)) 0 in
    let a_idx = Array.make (Array.length (shape a)) 0 in
    let b_idx = Array.make (Array.length (shape b)) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k (shape out) md_index;
      Shape.broadcast_index_into md_index (shape a) a_idx;
      let a_lin = Shape.ravel_index a_idx (strides a) in
      Shape.broadcast_index_into md_index (shape b) b_idx;
      let b_lin = Shape.ravel_index b_idx (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf (offset out + k) (logxor_complex a_val b_val)
    done

let kernel_add (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind (buffer a) with
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
  match Array1.kind (buffer a) with
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
  match Array1.kind (buffer a) with
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

let kernel_fdiv (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind (buffer a) with
  | Float16 -> kernel_fdiv_float16 a b out start_idx end_idx
  | Float32 -> kernel_fdiv_float32 a b out start_idx end_idx
  | Float64 -> kernel_fdiv_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_fdiv_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_fdiv_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_fdiv_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_fdiv_uint16 a b out start_idx end_idx
  | Int32 -> kernel_fdiv_int32 a b out start_idx end_idx
  | Int64 -> kernel_fdiv_int64 a b out start_idx end_idx
  | Int -> kernel_fdiv_int a b out start_idx end_idx
  | Nativeint -> kernel_fdiv_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_fdiv_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_fdiv_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_fdiv: unsupported type"

let kernel_idiv (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind (buffer a) with
  | Float16 -> kernel_idiv_float16 a b out start_idx end_idx
  | Float32 -> kernel_idiv_float32 a b out start_idx end_idx
  | Float64 -> kernel_idiv_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_idiv_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_idiv_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_idiv_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_idiv_uint16 a b out start_idx end_idx
  | Int32 -> kernel_idiv_int32 a b out start_idx end_idx
  | Int64 -> kernel_idiv_int64 a b out start_idx end_idx
  | Int -> kernel_idiv_int a b out start_idx end_idx
  | Nativeint -> kernel_idiv_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_idiv_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_idiv_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_idiv: unsupported type"

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

let kernel_modulo (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_modulo_float16 a b out start_idx end_idx
  | Float32 -> kernel_modulo_float32 a b out start_idx end_idx
  | Float64 -> kernel_modulo_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_modulo_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_modulo_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_modulo_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_modulo_uint16 a b out start_idx end_idx
  | Int32 -> kernel_modulo_int32 a b out start_idx end_idx
  | Int64 -> kernel_modulo_int64 a b out start_idx end_idx
  | Int -> kernel_modulo_int a b out start_idx end_idx
  | Nativeint -> kernel_modulo_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_modulo_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_modulo_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_modulo: unsupported type"

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

let kernel_cmplt (type a b) (a : (a, b) t) (b : (a, b) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_cmplt_float16 a b out start_idx end_idx
  | Float32 -> kernel_cmplt_float32 a b out start_idx end_idx
  | Float64 -> kernel_cmplt_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_cmplt_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_cmplt_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_cmplt_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_cmplt_uint16 a b out start_idx end_idx
  | Int32 -> kernel_cmplt_int32 a b out start_idx end_idx
  | Int64 -> kernel_cmplt_int64 a b out start_idx end_idx
  | Int -> kernel_cmplt_int a b out start_idx end_idx
  | Nativeint -> kernel_cmplt_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_cmplt_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_cmplt_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_cmplt: unsupported type"

let kernel_cmpne (type a b) (a : (a, b) t) (b : (a, b) t)
    (out : (int, uint8_elt) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_cmpne_float16 a b out start_idx end_idx
  | Float32 -> kernel_cmpne_float32 a b out start_idx end_idx
  | Float64 -> kernel_cmpne_float64 a b out start_idx end_idx
  | Int8_signed -> kernel_cmpne_int8 a b out start_idx end_idx
  | Int8_unsigned -> kernel_cmpne_uint8 a b out start_idx end_idx
  | Int16_signed -> kernel_cmpne_int16 a b out start_idx end_idx
  | Int16_unsigned -> kernel_cmpne_uint16 a b out start_idx end_idx
  | Int32 -> kernel_cmpne_int32 a b out start_idx end_idx
  | Int64 -> kernel_cmpne_int64 a b out start_idx end_idx
  | Int -> kernel_cmpne_int a b out start_idx end_idx
  | Nativeint -> kernel_cmpne_nativeint a b out start_idx end_idx
  | Complex32 -> kernel_cmpne_complex32 a b out start_idx end_idx
  | Complex64 -> kernel_cmpne_complex64 a b out start_idx end_idx
  | _ -> invalid_arg "kernel_cmpne: unsupported type"

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

let fdiv (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_fdiv a b out start_idx end_idx)

let idiv (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_idiv a b out start_idx end_idx)

let pow (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_pow a b out start_idx end_idx)

let modulo (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_modulo a b out start_idx end_idx)

let max (type a b) context (a : (a, b) t) (b : (a, b) t) (out : (a, b) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_max a b out start_idx end_idx)

let cmplt (type a b) context (a : (a, b) t) (b : (a, b) t)
    (out : (int, int8_unsigned_elt) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_cmplt a b out start_idx end_idx)

let cmpne (type a b) context (a : (a, b) t) (b : (a, b) t)
    (out : (int, int8_unsigned_elt) t) =
  let size = size out in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_cmpne a b out start_idx end_idx)

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
