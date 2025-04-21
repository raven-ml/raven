open Descriptor

module Make (B : Backend_intf.S) = struct
  let map context f arr =
    let src_desc = B.descriptor arr in
    let src_buffer = B.buffer arr in
    let shape = src_desc.shape in
    let dtype = src_desc.dtype in
    let sz = size src_desc in

    let dest_buffer = Buffer.create_buffer dtype sz in
    let dest_strides = compute_c_strides shape in
    let dest_desc =
      {
        src_desc with
        layout = C_contiguous;
        strides = dest_strides;
        offset = 0;
      }
    in

    (if sz > 0 then
       let flat_idx = ref 0 in
       iter_multi_indices shape (fun md_index ->
           let src_linear_offset =
             md_to_linear md_index src_desc.strides + src_desc.offset
           in
           let value =
             Bigarray.Array1.unsafe_get src_buffer src_linear_offset
           in
           let mapped_value = f value in
           Bigarray.Array1.unsafe_set dest_buffer !flat_idx mapped_value;
           incr flat_idx));
    B.from_buffer context dest_desc dest_buffer

  let iter _context f arr =
    let desc = B.descriptor arr in
    let buffer = B.buffer arr in
    let shape = desc.shape in
    if size desc > 0 then
      iter_multi_indices shape (fun md_index ->
          let linear_offset =
            md_to_linear md_index desc.strides + desc.offset
          in
          try
            let value = Bigarray.Array1.unsafe_get buffer linear_offset in
            f value
          with _ ->
            Printf.eprintf "Error accessing index %s linear index %s\n%!"
              (Array.fold_left
                 (fun acc i -> acc ^ string_of_int i ^ ", ")
                 "" md_index)
              (string_of_int linear_offset))

  let fold _context f acc arr =
    let desc = B.descriptor arr in
    let buffer = B.buffer arr in
    let shape = desc.shape in
    let acc_ref = ref acc in
    if size desc > 0 then
      iter_multi_indices shape (fun md_index ->
          let linear_offset =
            md_to_linear md_index desc.strides + desc.offset
          in
          let value = Bigarray.Array1.unsafe_get buffer linear_offset in
          acc_ref := f !acc_ref value);
    !acc_ref
end
