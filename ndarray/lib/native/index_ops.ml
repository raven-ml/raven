open Bigarray
open Ndarray_core
open Internal

let where (type a b) context (cond : (int, uint8_elt) t) (x : (a, b) t)
    (y : (a, b) t) (out : (a, b) t) : unit =
  if dtype x <> dtype y then
    invalid_arg "where: data tensors x and y must have the same dtype";

  let cond_shape = shape cond in
  let x_shape = shape x in
  let y_shape = shape y in

  let out_shape =
    try broadcast_shapes cond_shape (broadcast_shapes x_shape y_shape)
    with Invalid_argument msg -> invalid_arg msg
  in

  let numel arr_shape =
    if Array.length arr_shape = 0 then 1 else Array.fold_left ( * ) 1 arr_shape
  in

  let total_elements = numel out_shape in

  if total_elements = 0 then ()
  else
    let cond_buf = buffer cond in
    let x_buf = buffer x in
    let y_buf = buffer y in
    let out_buf = buffer out in

    let cond_strides = strides cond in
    let x_strides = strides x in
    let y_strides = strides y in

    let cond_offset = offset cond in
    let x_offset = offset x in
    let y_offset = offset y in

    let linear_index_from_multi multi_idx strides offset =
      let ndim = Array.length multi_idx in
      let linear_idx = ref offset in
      let strides_len = Array.length strides in
      if ndim <> strides_len then
        failwith "Internal error: dimension mismatch in linear_index_from_multi";
      (* Should not happen *)
      for i = 0 to ndim - 1 do
        linear_idx := !linear_idx + (multi_idx.(i) * strides.(i))
      done;
      !linear_idx
    in

    let process_element lin_idx =
      let out_multi_idx = multi_index_from_linear lin_idx out_shape in

      let cond_multi_idx = compute_broadcast_index out_multi_idx cond_shape in
      let x_multi_idx = compute_broadcast_index out_multi_idx x_shape in
      let y_multi_idx = compute_broadcast_index out_multi_idx y_shape in

      let cond_lin_idx =
        linear_index_from_multi cond_multi_idx cond_strides cond_offset
      in
      let x_lin_idx = linear_index_from_multi x_multi_idx x_strides x_offset in
      let y_lin_idx = linear_index_from_multi y_multi_idx y_strides y_offset in

      let cond_val = Array1.unsafe_get cond_buf cond_lin_idx in

      let value_to_set =
        if cond_val <> 0 then Array1.unsafe_get x_buf x_lin_idx
        else Array1.unsafe_get y_buf y_lin_idx
      in
      Array1.unsafe_set out_buf lin_idx value_to_set
    in

    let parallel_element_threshold = 10000 in
    if total_elements < parallel_element_threshold || total_elements = 0 then
      for i = 0 to total_elements - 1 do
        process_element i
      done
    else
      Parallel.parallel_for context.pool 0 (total_elements - 1)
        (fun idx_start idx_end ->
          for i = idx_start to idx_end do
            process_element i
          done)
