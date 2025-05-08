open Bigarray
open Nx_core.Dtype
open Nx_core.View
open Internal

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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
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
      let md_index = offset_to_index_contig k (shape out) in
      let a_lin = index_to_offset md_index (strides a) in
      let b_lin = index_to_offset md_index (strides b) in
      let a_val = Array1.unsafe_get a_buf (offset a + a_lin) in
      let b_val = Array1.unsafe_get b_buf (offset b + b_lin) in
      Array1.unsafe_set out_buf k (Complex.div a_val b_val)
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

let kernel_div (type a b) (a : (a, b) t) (b : (a, b) t) (out : (a, b) t)
    start_idx end_idx =
  match Array1.kind (buffer a) with
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
