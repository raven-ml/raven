open Bigarray_ext
open Nx_core.Dtype
module Shape = Nx_core.Shape
module View = Nx_core.View
open Internal

let kernel_where_float16 (cond : (int, uint8_elt) t)
    (x : (float, float16_elt) t) (y : (float, float16_elt) t)
    (out : (float, float16_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_float32 (cond : (int, uint8_elt) t)
    (x : (float, float32_elt) t) (y : (float, float32_elt) t)
    (out : (float, float32_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_float64 (cond : (int, uint8_elt) t)
    (x : (float, float64_elt) t) (y : (float, float64_elt) t)
    (out : (float, float64_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_int8 (cond : (int, uint8_elt) t) (x : (int, int8_elt) t)
    (y : (int, int8_elt) t) (out : (int, int8_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_uint8 (cond : (int, uint8_elt) t) (x : (int, uint8_elt) t)
    (y : (int, uint8_elt) t) (out : (int, uint8_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_int16 (cond : (int, uint8_elt) t) (x : (int, int16_elt) t)
    (y : (int, int16_elt) t) (out : (int, int16_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_uint16 (cond : (int, uint8_elt) t) (x : (int, uint16_elt) t)
    (y : (int, uint16_elt) t) (out : (int, uint16_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_int32 (cond : (int, uint8_elt) t) (x : (int32, int32_elt) t)
    (y : (int32, int32_elt) t) (out : (int32, int32_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_int64 (cond : (int, uint8_elt) t) (x : (int64, int64_elt) t)
    (y : (int64, int64_elt) t) (out : (int64, int64_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_int (cond : (int, uint8_elt) t) (x : (int, int_elt) t)
    (y : (int, int_elt) t) (out : (int, int_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_nativeint (cond : (int, uint8_elt) t)
    (x : (nativeint, nativeint_elt) t) (y : (nativeint, nativeint_elt) t)
    (out : (nativeint, nativeint_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_complex32 (cond : (int, uint8_elt) t)
    (x : (Complex.t, complex32_elt) t) (y : (Complex.t, complex32_elt) t)
    (out : (Complex.t, complex32_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_complex64 (cond : (int, uint8_elt) t)
    (x : (Complex.t, complex64_elt) t) (y : (Complex.t, complex64_elt) t)
    (out : (Complex.t, complex64_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in

      let cond_val0 = Array1.unsafe_get cond_buf k0 in
      let x_val0 = Array1.unsafe_get x_buf k0 in
      let y_val0 = Array1.unsafe_get y_buf k0 in
      Array1.unsafe_set out_buf k0 (if cond_val0 <> 0 then x_val0 else y_val0);

      let cond_val1 = Array1.unsafe_get cond_buf k1 in
      let x_val1 = Array1.unsafe_get x_buf k1 in
      let y_val1 = Array1.unsafe_get y_buf k1 in
      Array1.unsafe_set out_buf k1 (if cond_val1 <> 0 then x_val1 else y_val1);

      let cond_val2 = Array1.unsafe_get cond_buf k2 in
      let x_val2 = Array1.unsafe_get x_buf k2 in
      let y_val2 = Array1.unsafe_get y_buf k2 in
      Array1.unsafe_set out_buf k2 (if cond_val2 <> 0 then x_val2 else y_val2);

      let cond_val3 = Array1.unsafe_get cond_buf k3 in
      let x_val3 = Array1.unsafe_get x_buf k3 in
      let y_val3 = Array1.unsafe_get y_buf k3 in
      Array1.unsafe_set out_buf k3 (if cond_val3 <> 0 then x_val3 else y_val3);

      k := !k + 4
    done;
    while !k < end_idx do
      let current_k = !k in
      let cond_val = Array1.unsafe_get cond_buf current_k in
      let x_val = Array1.unsafe_get x_buf current_k in
      let y_val = Array1.unsafe_get y_buf current_k in
      Array1.unsafe_set out_buf current_k
        (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    (* Pre-allocate work arrays to avoid allocations in loop *)
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;

      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

(* Extended dtypes for kernel_where *)
let kernel_where_bfloat16 (cond : (int, uint8_elt) t)
    (x : (float, bfloat16_elt) t) (y : (float, bfloat16_elt) t)
    (out : (float, bfloat16_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k + 0 and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in
      let cond0 = Array1.unsafe_get cond_buf (cond_off + k0) in
      let cond1 = Array1.unsafe_get cond_buf (cond_off + k1) in
      let cond2 = Array1.unsafe_get cond_buf (cond_off + k2) in
      let cond3 = Array1.unsafe_get cond_buf (cond_off + k3) in
      let x0 = Array1.unsafe_get x_buf (x_off + k0) in
      let x1 = Array1.unsafe_get x_buf (x_off + k1) in
      let x2 = Array1.unsafe_get x_buf (x_off + k2) in
      let x3 = Array1.unsafe_get x_buf (x_off + k3) in
      let y0 = Array1.unsafe_get y_buf (y_off + k0) in
      let y1 = Array1.unsafe_get y_buf (y_off + k1) in
      let y2 = Array1.unsafe_get y_buf (y_off + k2) in
      let y3 = Array1.unsafe_get y_buf (y_off + k3) in
      Array1.unsafe_set out_buf (offset out + k0) (if cond0 <> 0 then x0 else y0);
      Array1.unsafe_set out_buf (offset out + k1) (if cond1 <> 0 then x1 else y1);
      Array1.unsafe_set out_buf (offset out + k2) (if cond2 <> 0 then x2 else y2);
      Array1.unsafe_set out_buf (offset out + k3) (if cond3 <> 0 then x3 else y3);
      k := !k + 4
    done;
    while !k < end_idx do
      let cond_val = Array1.unsafe_get cond_buf (cond_off + !k) in
      let x_val = Array1.unsafe_get x_buf (x_off + !k) in
      let y_val = Array1.unsafe_get y_buf (y_off + !k) in
      Array1.unsafe_set out_buf (offset out + !k) (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in

    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;
      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_bool (cond : (int, uint8_elt) t)
    (x : (bool, bool_elt) t) (y : (bool, bool_elt) t)
    (out : (bool, bool_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k + 0 and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in
      let cond0 = Array1.unsafe_get cond_buf (cond_off + k0) in
      let cond1 = Array1.unsafe_get cond_buf (cond_off + k1) in
      let cond2 = Array1.unsafe_get cond_buf (cond_off + k2) in
      let cond3 = Array1.unsafe_get cond_buf (cond_off + k3) in
      let x0 = Array1.unsafe_get x_buf (x_off + k0) in
      let x1 = Array1.unsafe_get x_buf (x_off + k1) in
      let x2 = Array1.unsafe_get x_buf (x_off + k2) in
      let x3 = Array1.unsafe_get x_buf (x_off + k3) in
      let y0 = Array1.unsafe_get y_buf (y_off + k0) in
      let y1 = Array1.unsafe_get y_buf (y_off + k1) in
      let y2 = Array1.unsafe_get y_buf (y_off + k2) in
      let y3 = Array1.unsafe_get y_buf (y_off + k3) in
      Array1.unsafe_set out_buf (offset out + k0) (if cond0 <> 0 then x0 else y0);
      Array1.unsafe_set out_buf (offset out + k1) (if cond1 <> 0 then x1 else y1);
      Array1.unsafe_set out_buf (offset out + k2) (if cond2 <> 0 then x2 else y2);
      Array1.unsafe_set out_buf (offset out + k3) (if cond3 <> 0 then x3 else y3);
      k := !k + 4
    done;
    while !k < end_idx do
      let cond_val = Array1.unsafe_get cond_buf (cond_off + !k) in
      let x_val = Array1.unsafe_get x_buf (x_off + !k) in
      let y_val = Array1.unsafe_get y_buf (y_off + !k) in
      Array1.unsafe_set out_buf (offset out + !k) (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in

    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;
      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_int4 (cond : (int, uint8_elt) t)
    (x : (int, int4_elt) t) (y : (int, int4_elt) t)
    (out : (int, int4_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k + 0 and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in
      let cond0 = Array1.unsafe_get cond_buf (cond_off + k0) in
      let cond1 = Array1.unsafe_get cond_buf (cond_off + k1) in
      let cond2 = Array1.unsafe_get cond_buf (cond_off + k2) in
      let cond3 = Array1.unsafe_get cond_buf (cond_off + k3) in
      let x0 = Array1.unsafe_get x_buf (x_off + k0) in
      let x1 = Array1.unsafe_get x_buf (x_off + k1) in
      let x2 = Array1.unsafe_get x_buf (x_off + k2) in
      let x3 = Array1.unsafe_get x_buf (x_off + k3) in
      let y0 = Array1.unsafe_get y_buf (y_off + k0) in
      let y1 = Array1.unsafe_get y_buf (y_off + k1) in
      let y2 = Array1.unsafe_get y_buf (y_off + k2) in
      let y3 = Array1.unsafe_get y_buf (y_off + k3) in
      Array1.unsafe_set out_buf (offset out + k0) (if cond0 <> 0 then x0 else y0);
      Array1.unsafe_set out_buf (offset out + k1) (if cond1 <> 0 then x1 else y1);
      Array1.unsafe_set out_buf (offset out + k2) (if cond2 <> 0 then x2 else y2);
      Array1.unsafe_set out_buf (offset out + k3) (if cond3 <> 0 then x3 else y3);
      k := !k + 4
    done;
    while !k < end_idx do
      let cond_val = Array1.unsafe_get cond_buf (cond_off + !k) in
      let x_val = Array1.unsafe_get x_buf (x_off + !k) in
      let y_val = Array1.unsafe_get y_buf (y_off + !k) in
      Array1.unsafe_set out_buf (offset out + !k) (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in

    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;
      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_uint4 (cond : (int, uint8_elt) t)
    (x : (int, uint4_elt) t) (y : (int, uint4_elt) t)
    (out : (int, uint4_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k + 0 and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in
      let cond0 = Array1.unsafe_get cond_buf (cond_off + k0) in
      let cond1 = Array1.unsafe_get cond_buf (cond_off + k1) in
      let cond2 = Array1.unsafe_get cond_buf (cond_off + k2) in
      let cond3 = Array1.unsafe_get cond_buf (cond_off + k3) in
      let x0 = Array1.unsafe_get x_buf (x_off + k0) in
      let x1 = Array1.unsafe_get x_buf (x_off + k1) in
      let x2 = Array1.unsafe_get x_buf (x_off + k2) in
      let x3 = Array1.unsafe_get x_buf (x_off + k3) in
      let y0 = Array1.unsafe_get y_buf (y_off + k0) in
      let y1 = Array1.unsafe_get y_buf (y_off + k1) in
      let y2 = Array1.unsafe_get y_buf (y_off + k2) in
      let y3 = Array1.unsafe_get y_buf (y_off + k3) in
      Array1.unsafe_set out_buf (offset out + k0) (if cond0 <> 0 then x0 else y0);
      Array1.unsafe_set out_buf (offset out + k1) (if cond1 <> 0 then x1 else y1);
      Array1.unsafe_set out_buf (offset out + k2) (if cond2 <> 0 then x2 else y2);
      Array1.unsafe_set out_buf (offset out + k3) (if cond3 <> 0 then x3 else y3);
      k := !k + 4
    done;
    while !k < end_idx do
      let cond_val = Array1.unsafe_get cond_buf (cond_off + !k) in
      let x_val = Array1.unsafe_get x_buf (x_off + !k) in
      let y_val = Array1.unsafe_get y_buf (y_off + !k) in
      Array1.unsafe_set out_buf (offset out + !k) (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in

    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;
      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_float8_e4m3 (cond : (int, uint8_elt) t)
    (x : (float, float8_e4m3_elt) t) (y : (float, float8_e4m3_elt) t)
    (out : (float, float8_e4m3_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k + 0 and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in
      let cond0 = Array1.unsafe_get cond_buf (cond_off + k0) in
      let cond1 = Array1.unsafe_get cond_buf (cond_off + k1) in
      let cond2 = Array1.unsafe_get cond_buf (cond_off + k2) in
      let cond3 = Array1.unsafe_get cond_buf (cond_off + k3) in
      let x0 = Array1.unsafe_get x_buf (x_off + k0) in
      let x1 = Array1.unsafe_get x_buf (x_off + k1) in
      let x2 = Array1.unsafe_get x_buf (x_off + k2) in
      let x3 = Array1.unsafe_get x_buf (x_off + k3) in
      let y0 = Array1.unsafe_get y_buf (y_off + k0) in
      let y1 = Array1.unsafe_get y_buf (y_off + k1) in
      let y2 = Array1.unsafe_get y_buf (y_off + k2) in
      let y3 = Array1.unsafe_get y_buf (y_off + k3) in
      Array1.unsafe_set out_buf (offset out + k0) (if cond0 <> 0 then x0 else y0);
      Array1.unsafe_set out_buf (offset out + k1) (if cond1 <> 0 then x1 else y1);
      Array1.unsafe_set out_buf (offset out + k2) (if cond2 <> 0 then x2 else y2);
      Array1.unsafe_set out_buf (offset out + k3) (if cond3 <> 0 then x3 else y3);
      k := !k + 4
    done;
    while !k < end_idx do
      let cond_val = Array1.unsafe_get cond_buf (cond_off + !k) in
      let x_val = Array1.unsafe_get x_buf (x_off + !k) in
      let y_val = Array1.unsafe_get y_buf (y_off + !k) in
      Array1.unsafe_set out_buf (offset out + !k) (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in

    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;
      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_float8_e5m2 (cond : (int, uint8_elt) t)
    (x : (float, float8_e5m2_elt) t) (y : (float, float8_e5m2_elt) t)
    (out : (float, float8_e5m2_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k + 0 and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in
      let cond0 = Array1.unsafe_get cond_buf (cond_off + k0) in
      let cond1 = Array1.unsafe_get cond_buf (cond_off + k1) in
      let cond2 = Array1.unsafe_get cond_buf (cond_off + k2) in
      let cond3 = Array1.unsafe_get cond_buf (cond_off + k3) in
      let x0 = Array1.unsafe_get x_buf (x_off + k0) in
      let x1 = Array1.unsafe_get x_buf (x_off + k1) in
      let x2 = Array1.unsafe_get x_buf (x_off + k2) in
      let x3 = Array1.unsafe_get x_buf (x_off + k3) in
      let y0 = Array1.unsafe_get y_buf (y_off + k0) in
      let y1 = Array1.unsafe_get y_buf (y_off + k1) in
      let y2 = Array1.unsafe_get y_buf (y_off + k2) in
      let y3 = Array1.unsafe_get y_buf (y_off + k3) in
      Array1.unsafe_set out_buf (offset out + k0) (if cond0 <> 0 then x0 else y0);
      Array1.unsafe_set out_buf (offset out + k1) (if cond1 <> 0 then x1 else y1);
      Array1.unsafe_set out_buf (offset out + k2) (if cond2 <> 0 then x2 else y2);
      Array1.unsafe_set out_buf (offset out + k3) (if cond3 <> 0 then x3 else y3);
      k := !k + 4
    done;
    while !k < end_idx do
      let cond_val = Array1.unsafe_get cond_buf (cond_off + !k) in
      let x_val = Array1.unsafe_get x_buf (x_off + !k) in
      let y_val = Array1.unsafe_get y_buf (y_off + !k) in
      Array1.unsafe_set out_buf (offset out + !k) (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in

    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;
      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_complex16 (cond : (int, uint8_elt) t)
    (x : (Complex.t, complex16_elt) t) (y : (Complex.t, complex16_elt) t)
    (out : (Complex.t, complex16_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k + 0 and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in
      let cond0 = Array1.unsafe_get cond_buf (cond_off + k0) in
      let cond1 = Array1.unsafe_get cond_buf (cond_off + k1) in
      let cond2 = Array1.unsafe_get cond_buf (cond_off + k2) in
      let cond3 = Array1.unsafe_get cond_buf (cond_off + k3) in
      let x0 = Array1.unsafe_get x_buf (x_off + k0) in
      let x1 = Array1.unsafe_get x_buf (x_off + k1) in
      let x2 = Array1.unsafe_get x_buf (x_off + k2) in
      let x3 = Array1.unsafe_get x_buf (x_off + k3) in
      let y0 = Array1.unsafe_get y_buf (y_off + k0) in
      let y1 = Array1.unsafe_get y_buf (y_off + k1) in
      let y2 = Array1.unsafe_get y_buf (y_off + k2) in
      let y3 = Array1.unsafe_get y_buf (y_off + k3) in
      Array1.unsafe_set out_buf (offset out + k0) (if cond0 <> 0 then x0 else y0);
      Array1.unsafe_set out_buf (offset out + k1) (if cond1 <> 0 then x1 else y1);
      Array1.unsafe_set out_buf (offset out + k2) (if cond2 <> 0 then x2 else y2);
      Array1.unsafe_set out_buf (offset out + k3) (if cond3 <> 0 then x3 else y3);
      k := !k + 4
    done;
    while !k < end_idx do
      let cond_val = Array1.unsafe_get cond_buf (cond_off + !k) in
      let x_val = Array1.unsafe_get x_buf (x_off + !k) in
      let y_val = Array1.unsafe_get y_buf (y_off + !k) in
      Array1.unsafe_set out_buf (offset out + !k) (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in

    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;
      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_qint8 (cond : (int, uint8_elt) t)
    (x : (int, qint8_elt) t) (y : (int, qint8_elt) t)
    (out : (int, qint8_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k + 0 and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in
      let cond0 = Array1.unsafe_get cond_buf (cond_off + k0) in
      let cond1 = Array1.unsafe_get cond_buf (cond_off + k1) in
      let cond2 = Array1.unsafe_get cond_buf (cond_off + k2) in
      let cond3 = Array1.unsafe_get cond_buf (cond_off + k3) in
      let x0 = Array1.unsafe_get x_buf (x_off + k0) in
      let x1 = Array1.unsafe_get x_buf (x_off + k1) in
      let x2 = Array1.unsafe_get x_buf (x_off + k2) in
      let x3 = Array1.unsafe_get x_buf (x_off + k3) in
      let y0 = Array1.unsafe_get y_buf (y_off + k0) in
      let y1 = Array1.unsafe_get y_buf (y_off + k1) in
      let y2 = Array1.unsafe_get y_buf (y_off + k2) in
      let y3 = Array1.unsafe_get y_buf (y_off + k3) in
      Array1.unsafe_set out_buf (offset out + k0) (if cond0 <> 0 then x0 else y0);
      Array1.unsafe_set out_buf (offset out + k1) (if cond1 <> 0 then x1 else y1);
      Array1.unsafe_set out_buf (offset out + k2) (if cond2 <> 0 then x2 else y2);
      Array1.unsafe_set out_buf (offset out + k3) (if cond3 <> 0 then x3 else y3);
      k := !k + 4
    done;
    while !k < end_idx do
      let cond_val = Array1.unsafe_get cond_buf (cond_off + !k) in
      let x_val = Array1.unsafe_get x_buf (x_off + !k) in
      let y_val = Array1.unsafe_get y_buf (y_off + !k) in
      Array1.unsafe_set out_buf (offset out + !k) (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in

    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;
      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where_quint8 (cond : (int, uint8_elt) t)
    (x : (int, quint8_elt) t) (y : (int, quint8_elt) t)
    (out : (int, quint8_elt) t) start_idx end_idx =
  let cond_buf = buffer cond in
  let x_buf = buffer x in
  let y_buf = buffer y in
  let out_buf = buffer out in

  let out_s = shape out in

  let cond_s = shape cond in
  let x_s = shape x in
  let y_s = shape y in

  let cond_st = strides cond in
  let x_st = strides x in
  let y_st = strides y in

  let cond_off = offset cond in
  let x_off = offset x in
  let y_off = offset y in

  let can_use_direct_indexing =
    Shape.equal out_s cond_s && Shape.equal out_s x_s && Shape.equal out_s y_s
    && is_c_contiguous cond && is_c_contiguous x && is_c_contiguous y
  in

  if can_use_direct_indexing then (
    let k = ref start_idx in
    while !k + 3 < end_idx do
      let k0 = !k + 0 and k1 = !k + 1 and k2 = !k + 2 and k3 = !k + 3 in
      let cond0 = Array1.unsafe_get cond_buf (cond_off + k0) in
      let cond1 = Array1.unsafe_get cond_buf (cond_off + k1) in
      let cond2 = Array1.unsafe_get cond_buf (cond_off + k2) in
      let cond3 = Array1.unsafe_get cond_buf (cond_off + k3) in
      let x0 = Array1.unsafe_get x_buf (x_off + k0) in
      let x1 = Array1.unsafe_get x_buf (x_off + k1) in
      let x2 = Array1.unsafe_get x_buf (x_off + k2) in
      let x3 = Array1.unsafe_get x_buf (x_off + k3) in
      let y0 = Array1.unsafe_get y_buf (y_off + k0) in
      let y1 = Array1.unsafe_get y_buf (y_off + k1) in
      let y2 = Array1.unsafe_get y_buf (y_off + k2) in
      let y3 = Array1.unsafe_get y_buf (y_off + k3) in
      Array1.unsafe_set out_buf (offset out + k0) (if cond0 <> 0 then x0 else y0);
      Array1.unsafe_set out_buf (offset out + k1) (if cond1 <> 0 then x1 else y1);
      Array1.unsafe_set out_buf (offset out + k2) (if cond2 <> 0 then x2 else y2);
      Array1.unsafe_set out_buf (offset out + k3) (if cond3 <> 0 then x3 else y3);
      k := !k + 4
    done;
    while !k < end_idx do
      let cond_val = Array1.unsafe_get cond_buf (cond_off + !k) in
      let x_val = Array1.unsafe_get x_buf (x_off + !k) in
      let y_val = Array1.unsafe_get y_buf (y_off + !k) in
      Array1.unsafe_set out_buf (offset out + !k) (if cond_val <> 0 then x_val else y_val);
      incr k
    done)
  else
    let out_multi_idx = Array.make (Array.length out_s) 0 in
    let cond_multi_idx = Array.make (Array.length cond_s) 0 in
    let x_multi_idx = Array.make (Array.length x_s) 0 in
    let y_multi_idx = Array.make (Array.length y_s) 0 in

    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_s out_multi_idx;
      Shape.broadcast_index_into out_multi_idx cond_s cond_multi_idx;
      Shape.broadcast_index_into out_multi_idx x_s x_multi_idx;
      Shape.broadcast_index_into out_multi_idx y_s y_multi_idx;

      let cond_phys_idx = Shape.ravel_index cond_multi_idx cond_st + cond_off in
      let x_phys_idx = Shape.ravel_index x_multi_idx x_st + x_off in
      let y_phys_idx = Shape.ravel_index y_multi_idx y_st + y_off in

      let cond_val = Array1.unsafe_get cond_buf cond_phys_idx in
      let x_val = Array1.unsafe_get x_buf x_phys_idx in
      let y_val = Array1.unsafe_get y_buf y_phys_idx in
      Array1.unsafe_set out_buf
        (offset out + k)
        (if cond_val <> 0 then x_val else y_val)
    done

let kernel_where (type a b) (cond : (int, uint8_elt) t) (if_true : (a, b) t)
    (if_false : (a, b) t) (out : (a, b) t) start_idx end_idx =
  match Array1.kind if_true.buffer with
  | Float16 -> kernel_where_float16 cond if_true if_false out start_idx end_idx
  | Float32 -> kernel_where_float32 cond if_true if_false out start_idx end_idx
  | Float64 -> kernel_where_float64 cond if_true if_false out start_idx end_idx
  | Int8_signed -> kernel_where_int8 cond if_true if_false out start_idx end_idx
  | Int8_unsigned ->
      kernel_where_uint8 cond if_true if_false out start_idx end_idx
  | Int16_signed ->
      kernel_where_int16 cond if_true if_false out start_idx end_idx
  | Int16_unsigned ->
      kernel_where_uint16 cond if_true if_false out start_idx end_idx
  | Int32 -> kernel_where_int32 cond if_true if_false out start_idx end_idx
  | Int64 -> kernel_where_int64 cond if_true if_false out start_idx end_idx
  | Int -> kernel_where_int cond if_true if_false out start_idx end_idx
  | Nativeint ->
      kernel_where_nativeint cond if_true if_false out start_idx end_idx
  | Complex32 ->
      kernel_where_complex32 cond if_true if_false out start_idx end_idx
  | Complex64 ->
      kernel_where_complex64 cond if_true if_false out start_idx end_idx
  | Bfloat16 -> kernel_where_bfloat16 cond if_true if_false out start_idx end_idx
  | Bool -> kernel_where_bool cond if_true if_false out start_idx end_idx
  | Int4_signed -> kernel_where_int4 cond if_true if_false out start_idx end_idx
  | Int4_unsigned -> kernel_where_uint4 cond if_true if_false out start_idx end_idx
  | Float8_e4m3 -> kernel_where_float8_e4m3 cond if_true if_false out start_idx end_idx
  | Float8_e5m2 -> kernel_where_float8_e5m2 cond if_true if_false out start_idx end_idx
  | Complex16 -> kernel_where_complex16 cond if_true if_false out start_idx end_idx
  | Qint8 -> kernel_where_qint8 cond if_true if_false out start_idx end_idx
  | Quint8 -> kernel_where_quint8 cond if_true if_false out start_idx end_idx
  | _ -> invalid_arg "kernel_where: unsupported type"

let where (type a b) context (cond : (int, uint8_elt) t) (if_true : (a, b) t)
    (if_false : (a, b) t) (out : (a, b) t) : unit =
  let total_elements = size out in

  Parallel.parallel_for context.pool 0 (total_elements - 1)
    (fun start_idx end_idx ->
      kernel_where cond if_true if_false out start_idx end_idx)
