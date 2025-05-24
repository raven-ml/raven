open Bigarray
open Descriptor

module Make (B : Backend_intf.S) = struct
  let data = B.buffer
  let shape t = shape (B.descriptor t)
  let dtype t = dtype (B.descriptor t)
  let strides t = strides (B.descriptor t)
  let stride i t = stride i (B.descriptor t)
  let dims t = dims (B.descriptor t)
  let dim i t = dim i (B.descriptor t)
  let ndim t = ndim (B.descriptor t)
  let itemsize t = itemsize (B.descriptor t)
  let size t = size (B.descriptor t)
  let nbytes t = nbytes (B.descriptor t)
  let offset t = offset (B.descriptor t)
  let layout t = layout (B.descriptor t)

  let get_item _ctx indices arr =
    let ndim = Array.length (shape arr) in
    let nindices = Array.length indices in
    let buffer = B.buffer arr in

    if ndim = 0 then
      if nindices = 0 then Array1.unsafe_get buffer (offset arr)
      else
        invalid_arg
          (Format.asprintf
             "get_item: Cannot provide indices (got %d) for a 0D tensor"
             nindices)
    else if nindices = 1 then
      let k = indices.(0) in
      let s = size arr in
      if s = 0 then
        invalid_arg
          (Format.asprintf "get_item: Cannot index into an empty array (size 0)")
      else
        let flat_k = if k < 0 then s + k else k in
        if flat_k < 0 || flat_k >= s then
          invalid_arg
            (Format.asprintf
               "get_item: flat index %d (raw %d) is out of bounds for array \
                with size %d"
               flat_k k s)
        else
          let md_index = Descriptor.linear_to_md_c_contig flat_k (shape arr) in
          let physical_linear_index =
            offset arr + Descriptor.md_to_linear md_index (strides arr)
          in
          Array1.unsafe_get buffer physical_linear_index
    else if nindices = ndim then (
      for i = 0 to ndim - 1 do
        let idx = indices.(i) in
        let dim_size = (shape arr).(i) in
        if idx < 0 || idx >= dim_size then
          invalid_arg
            (Format.asprintf
               "get_item: Index %d at dimension %d is out of bounds for shape \
                %a"
               idx i pp_int_array (shape arr))
      done;
      let physical_linear_index =
        offset arr + Descriptor.md_to_linear indices (strides arr)
      in
      Array1.unsafe_get buffer physical_linear_index)
    else
      invalid_arg
        (Format.asprintf
           "get_item: Incorrect number of indices (%d) provided for array with \
            %d dimensions. Expected 0 (for 0D), 1 (flat), or %d (multi-dim)."
           nindices ndim ndim)

  let set_item _ctx indices value arr =
    let ndim = Array.length (shape arr) in
    let nindices = Array.length indices in
    let buffer = B.buffer arr in

    if ndim = 0 then
      if nindices = 0 then
        let physical_linear_index = offset arr in
        Array1.unsafe_set buffer physical_linear_index value
      else
        invalid_arg
          (Format.asprintf
             "set_item: Cannot provide indices (got %d) for a 0D tensor \
              assignment"
             nindices)
    else if nindices = 1 then
      let k = indices.(0) in
      let s = size arr in
      if s = 0 then
        invalid_arg
          (Format.asprintf "set_item: Cannot index into an empty array (size 0)")
      else
        let flat_k = if k < 0 then s + k else k in
        if flat_k < 0 || flat_k >= s then
          invalid_arg
            (Format.asprintf
               "set_item: flat index %d (raw %d) is out of bounds for array \
                with size %d"
               flat_k k s)
        else
          let md_index = Descriptor.linear_to_md_c_contig flat_k (shape arr) in
          let physical_linear_index =
            offset arr + Descriptor.md_to_linear md_index (strides arr)
          in
          Array1.unsafe_set buffer physical_linear_index value
    else if nindices = ndim then (
      for i = 0 to ndim - 1 do
        let idx = indices.(i) in
        let dim_size = (shape arr).(i) in
        if idx < 0 || idx >= dim_size then
          invalid_arg
            (Format.asprintf
               "set_item: Index %d at dimension %d is out of bounds for shape \
                %a"
               idx i pp_int_array (shape arr))
      done;
      let physical_linear_index =
        offset arr + Descriptor.md_to_linear indices (strides arr)
      in
      Array1.unsafe_set buffer physical_linear_index value)
    else
      invalid_arg
        (Format.asprintf
           "set_item: Incorrect number of indices (%d) provided for array with \
            %d dimensions. Expected 0 (for 0D), 1 (flat), or %d (multi-dim)."
           nindices ndim ndim)

  let get _ctx indices arr =
    let desc = B.descriptor arr in
    let original_shape = desc.shape in
    let original_strides = desc.strides in
    let original_offset = desc.offset in
    let original_ndim = Array.length original_shape in
    let nindices = Array.length indices in

    if nindices > original_ndim then
      invalid_arg
        (Format.asprintf
           "get: Too many indices (%d) for array with %d dimensions" nindices
           original_ndim);

    let offset_increment = ref 0 in
    for i = 0 to nindices - 1 do
      let idx = indices.(i) in
      let dim_size = original_shape.(i) in
      (* Handle zero-sized dimensions correctly *)
      if
        idx < 0
        || (dim_size > 0 && idx >= dim_size)
        || (dim_size == 0 && idx != 0)
      then
        invalid_arg
          (Format.asprintf
             "get: Index %d at dimension %d is out of bounds for shape %a" idx i
             pp_int_array original_shape)
      else if dim_size > 0 then
        (* Only add to offset if dimension is not size 0 *)
        offset_increment := !offset_increment + (idx * original_strides.(i))
    done;

    let new_ndim = original_ndim - nindices in
    let new_shape = Array.sub original_shape nindices new_ndim in
    let new_strides = Array.sub original_strides nindices new_ndim in
    let new_offset = original_offset + !offset_increment in
    let size = Array.fold_left ( * ) 1 new_shape in

    let final_layout =
      if new_ndim = 0 || size <= 1 then C_contiguous
      else
        let is_c =
          check_c_contiguity_from_shape_strides new_shape new_strides
        in
        if is_c then C_contiguous else Strided
    in

    let new_descriptor =
      {
        desc with
        shape = new_shape;
        strides = new_strides;
        offset = new_offset;
        layout = final_layout;
      }
    in
    B.view new_descriptor arr

  let set _ctx indices value_arr target_arr =
    let target_desc = B.descriptor target_arr in
    let target_buffer = B.buffer target_arr in
    let original_shape = target_desc.shape in
    let original_strides = target_desc.strides in
    let original_offset = target_desc.offset in
    let original_ndim = Array.length original_shape in
    let nindices = Array.length indices in

    if nindices > original_ndim then
      invalid_arg
        (Format.asprintf
           "set: Too many indices (%d) for array with %d dimensions" nindices
           original_ndim);

    let view_offset_increment = ref 0 in
    for i = 0 to nindices - 1 do
      let idx = indices.(i) in
      let dim_size = original_shape.(i) in
      if dim_size = 0 then
        if idx = 0 then ()
        else
          invalid_arg
            (Format.asprintf
               "set: Index %d at dimension %d is out of bounds for zero-sized \
                dimension"
               idx i)
      else if idx < 0 || idx >= dim_size then
        invalid_arg
          (Format.asprintf
             "set: Index %d at dimension %d is out of bounds for shape %a" idx i
             pp_int_array original_shape)
      else
        view_offset_increment :=
          !view_offset_increment + (idx * original_strides.(i))
    done;

    let view_ndim = original_ndim - nindices in
    let view_shape = Array.sub original_shape nindices view_ndim in
    let view_strides = Array.sub original_strides nindices view_ndim in
    let view_offset = original_offset + !view_offset_increment in

    let value_desc = B.descriptor value_arr in
    let value_buffer = B.buffer value_arr in

    if value_desc.dtype <> target_desc.dtype then
      invalid_arg "set: Mismatched dtypes";

    if value_desc.shape <> view_shape then
      invalid_arg
        (Format.asprintf
           "set: Shape mismatch for assignment. Value shape %a cannot be \
            assigned to target view shape %a"
           pp_int_array value_desc.shape pp_int_array view_shape);

    let num_elements_to_copy = Descriptor.size value_desc in
    if num_elements_to_copy > 0 then
      iter_multi_indices value_desc.shape (fun md_index ->
          let linear_idx_val =
            md_to_linear md_index value_desc.strides + value_desc.offset
          in
          let linear_idx_target =
            md_to_linear md_index view_strides + view_offset
          in
          let value = Array1.unsafe_get value_buffer linear_idx_val in
          Array1.unsafe_set target_buffer linear_idx_target value)
end
