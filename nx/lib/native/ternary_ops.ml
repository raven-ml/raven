open Bigarray
open Nx_core
open Internal

let[@inline] int_fma a b c = (a * b) + c
let[@inline] float_fma a b c = (a *. b) +. c
let[@inline] int32_fma a b c = Int32.mul a b |> Int32.add c
let[@inline] int64_fma a b c = Int64.mul a b |> Int64.add c
let[@inline] nativeint_fma a b c = Nativeint.mul a b |> Nativeint.add c
let[@inline] complex_fma a b c = Complex.mul a b |> Complex.add c

let kernel_fma_float16 (a : (float, float16_elt) t) (b : (float, float16_elt) t)
    (c : (float, float16_elt) t) (out : (float, float16_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (Float.fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (Float.fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (Float.fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (Float.fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (Float.fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (Float.fma a_val b_val c_val)
    done

let kernel_fma_float32 (a : (float, float32_elt) t) (b : (float, float32_elt) t)
    (c : (float, float32_elt) t) (out : (float, float32_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (Float.fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (Float.fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (Float.fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (Float.fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (Float.fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (Float.fma a_val b_val c_val)
    done

let kernel_fma_float64 (a : (float, float64_elt) t) (b : (float, float64_elt) t)
    (c : (float, float64_elt) t) (out : (float, float64_elt) t) start_idx
    end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (Float.fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (Float.fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (Float.fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (Float.fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (Float.fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (Float.fma a_val b_val c_val)
    done

let kernel_fma_int8 (a : (int, int8_elt) t) (b : (int, int8_elt) t)
    (c : (int, int8_elt) t) (out : (int, int8_elt) t) start_idx end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (int_fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (int_fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (int_fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (int_fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (int_fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (int_fma a_val b_val c_val)
    done

let kernel_fma_uint8 (a : (int, uint8_elt) t) (b : (int, uint8_elt) t)
    (c : (int, uint8_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (int_fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (int_fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (int_fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (int_fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (int_fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (int_fma a_val b_val c_val)
    done

let kernel_fma_int16 (a : (int, int16_elt) t) (b : (int, int16_elt) t)
    (c : (int, int16_elt) t) (out : (int, int16_elt) t) start_idx end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (int_fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (int_fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (int_fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (int_fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (int_fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (int_fma a_val b_val c_val)
    done

let kernel_fma_uint16 (a : (int, uint16_elt) t) (b : (int, uint16_elt) t)
    (c : (int, uint16_elt) t) (out : (int, uint16_elt) t) start_idx end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (int_fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (int_fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (int_fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (int_fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (int_fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (int_fma a_val b_val c_val)
    done

let kernel_fma_int32 (a : (int32, int32_elt) t) (b : (int32, int32_elt) t)
    (c : (int32, int32_elt) t) (out : (int32, int32_elt) t) start_idx end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (int32_fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (int32_fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (int32_fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (int32_fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (int32_fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (int32_fma a_val b_val c_val)
    done

let kernel_fma_int64 (a : (int64, int64_elt) t) (b : (int64, int64_elt) t)
    (c : (int64, int64_elt) t) (out : (int64, int64_elt) t) start_idx end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (int64_fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (int64_fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (int64_fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (int64_fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (int64_fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (int64_fma a_val b_val c_val)
    done

let kernel_fma_int (a : (int, int_elt) t) (b : (int, int_elt) t)
    (c : (int, int_elt) t) (out : (int, int_elt) t) start_idx end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (int_fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (int_fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (int_fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (int_fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (int_fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (int_fma a_val b_val c_val)
    done

let kernel_fma_nativeint (a : (nativeint, nativeint_elt) t)
    (b : (nativeint, nativeint_elt) t) (c : (nativeint, nativeint_elt) t)
    (out : (nativeint, nativeint_elt) t) start_idx end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (nativeint_fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (nativeint_fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (nativeint_fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (nativeint_fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (nativeint_fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (nativeint_fma a_val b_val c_val)
    done

let kernel_fma_complex32 (a : (Complex.t, complex32_elt) t)
    (b : (Complex.t, complex32_elt) t) (c : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (complex_fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (complex_fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (complex_fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (complex_fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (complex_fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (complex_fma a_val b_val c_val)
    done

let kernel_fma_complex64 (a : (Complex.t, complex64_elt) t)
    (b : (Complex.t, complex64_elt) t) (c : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let a_buf, b_buf, c_buf, out_buf =
    (buffer a, buffer b, buffer c, buffer out)
  in
  if is_c_contiguous a && is_c_contiguous b && is_c_contiguous c then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let a_val0 = Array1.unsafe_get a_buf (offset a + i0) in
      let b_val0 = Array1.unsafe_get b_buf (offset b + i0) in
      let c_val0 = Array1.unsafe_get c_buf (offset c + i0) in
      let a_val1 = Array1.unsafe_get a_buf (offset a + i1) in
      let b_val1 = Array1.unsafe_get b_buf (offset b + i1) in
      let c_val1 = Array1.unsafe_get c_buf (offset c + i1) in
      let a_val2 = Array1.unsafe_get a_buf (offset a + i2) in
      let b_val2 = Array1.unsafe_get b_buf (offset b + i2) in
      let c_val2 = Array1.unsafe_get c_buf (offset c + i2) in
      let a_val3 = Array1.unsafe_get a_buf (offset a + i3) in
      let b_val3 = Array1.unsafe_get b_buf (offset b + i3) in
      let c_val3 = Array1.unsafe_get c_buf (offset c + i3) in
      Array1.unsafe_set out_buf i0 (complex_fma a_val0 b_val0 c_val0);
      Array1.unsafe_set out_buf i1 (complex_fma a_val1 b_val1 c_val1);
      Array1.unsafe_set out_buf i2 (complex_fma a_val2 b_val2 c_val2);
      Array1.unsafe_set out_buf i3 (complex_fma a_val3 b_val3 c_val3);
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let a_val = Array1.unsafe_get a_buf (offset a + idx) in
      let b_val = Array1.unsafe_get b_buf (offset b + idx) in
      let c_val = Array1.unsafe_get c_buf (offset c + idx) in
      Array1.unsafe_set out_buf idx (complex_fma a_val b_val c_val);
      incr i
    done)
  else
    for k = start_idx to end_idx - 1 do
      let md_index = linear_to_md_c_contig k (shape c) in
      let a_lin = md_to_linear md_index (strides a) in
      let b_lin = md_to_linear md_index (strides b) in
      let c_lin = md_to_linear md_index (strides c) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      let c_val = Array1.unsafe_get c_buf (offset c + c_lin) in
      Array1.unsafe_set out_buf k (complex_fma a_val b_val c_val)
    done

let kernel_fma (type a b) (a : (a, b) t) (b : (a, b) t) (c : (a, b) t)
    (out : (a, b) t) start_idx end_idx =
  match Array1.kind a.buffer with
  | Float16 -> kernel_fma_float16 a b c out start_idx end_idx
  | Float32 -> kernel_fma_float32 a b c out start_idx end_idx
  | Float64 -> kernel_fma_float64 a b c out start_idx end_idx
  | Int8_signed -> kernel_fma_int8 a b c out start_idx end_idx
  | Int8_unsigned -> kernel_fma_uint8 a b c out start_idx end_idx
  | Int16_signed -> kernel_fma_int16 a b c out start_idx end_idx
  | Int16_unsigned -> kernel_fma_uint16 a b c out start_idx end_idx
  | Int32 -> kernel_fma_int32 a b c out start_idx end_idx
  | Int64 -> kernel_fma_int64 a b c out start_idx end_idx
  | Int -> kernel_fma_int a b c out start_idx end_idx
  | Nativeint -> kernel_fma_nativeint a b c out start_idx end_idx
  | Complex32 -> kernel_fma_complex32 a b c out start_idx end_idx
  | Complex64 -> kernel_fma_complex64 a b c out start_idx end_idx
  | _ -> invalid_arg "kernel_fma: unsupported type"

let fma (type a b) context (a : (a, b) t) (b : (a, b) t) (c : (a, b) t)
    (out : (a, b) t) =
  let size = size c in
  Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
      kernel_fma a b c out start_idx end_idx)
