open Bigarray
open Descriptor

module Make (B : Backend_intf.S) = struct
  let astype context dtype t =
    let new_shape = Array.copy (shape (B.descriptor t)) in
    let new_strides = compute_c_strides new_shape in
    let buffer = Astype.astype dtype (B.descriptor t) (B.buffer t) in
    let desc =
      {
        dtype;
        shape = new_shape;
        layout = C_contiguous;
        strides = new_strides;
        offset = 0;
      }
    in
    B.from_buffer context desc buffer

  let to_bigarray _context t =
    let desc = B.descriptor t in
    let src_buffer = B.buffer t in
    let shape = desc.shape in
    let dtype = desc.dtype in
    let kind = Buffer.kind_of_dtype dtype in
    let size = Array.fold_left ( * ) 1 desc.shape in

    let dest_genarray = Genarray.create kind c_layout shape in
    let dest_buffer = reshape_1 dest_genarray size in

    let flat_idx = ref 0 in
    iter_multi_indices shape (fun md_index ->
        let src_linear_offset =
          md_to_linear md_index desc.strides + desc.offset
        in
        let value = Bigarray.Array1.unsafe_get src_buffer src_linear_offset in
        (* Since dest is C-contiguous, the flat_idx corresponds to the C-order
           linear index *)
        Bigarray.Array1.unsafe_set dest_buffer !flat_idx value;
        incr flat_idx);
    dest_genarray

  let of_bigarray context ba =
    let size = Array.fold_left ( * ) 1 (Genarray.dims ba) in
    let host_buffer = reshape_1 ba size in
    let shape = Genarray.dims ba in
    let kind = Genarray.kind ba in
    let dtype = Buffer.dtype_of_kind kind in
    let strides = compute_c_strides shape in
    let descriptor =
      { dtype; shape; layout = C_contiguous; strides; offset = 0 }
    in
    B.from_buffer context descriptor host_buffer

  let to_array _context t =
    let desc = B.descriptor t in
    let buffer = B.buffer t in
    let shape = desc.shape in
    let sz = size desc in
    let ocaml_array = Array.make sz (zero desc.dtype) in
    (if sz > 0 then
       let flat_idx = ref 0 in
       iter_multi_indices shape (fun md_index ->
           let linear_offset =
             md_to_linear md_index desc.strides + desc.offset
           in
           let value = Bigarray.Array1.unsafe_get buffer linear_offset in
           ocaml_array.(!flat_idx) <- value;
           incr flat_idx));
    ocaml_array

  let dtype_to_string (type a b) _ctx (dtype : (a, b) dtype) =
    match dtype with
    | Float16 -> "float16"
    | Float32 -> "float32"
    | Float64 -> "float64"
    | Int8 -> "int8"
    | Int16 -> "int16"
    | Int32 -> "int32"
    | Int64 -> "int64"
    | UInt8 -> "uint8"
    | UInt16 -> "uint16"
    | Complex32 -> "complex32"
    | Complex64 -> "complex64"

  let pp_dtype context fmt dtype =
    Format.fprintf fmt "%s" (dtype_to_string context dtype)

  let shape_to_string _ctx shape =
    let shape_str =
      Array.map string_of_int shape |> Array.to_list |> String.concat "x"
    in
    Printf.sprintf "[%s]" shape_str

  let pp_shape context fmt shape =
    Format.fprintf fmt "%s" (shape_to_string context shape)

  let pp (type a b) _ctx fmt (arr : (a, b) B.b_t) =
    let open Format in
    let desc = B.descriptor arr in
    let buffer = B.buffer arr in
    let dtype = desc.dtype in
    let shape = desc.shape in
    let ndim = Array.length shape in
    let sz = size desc in

    let pp_element fmt (elt : a) =
      match dtype with
      | Float16 -> fprintf fmt "%g" elt
      | Float32 -> fprintf fmt "%g" elt
      | Float64 -> fprintf fmt "%g" elt
      | Int8 -> fprintf fmt "%d" elt
      | Int16 -> fprintf fmt "%d" elt
      | Int32 -> fprintf fmt "%ld" elt
      | Int64 -> fprintf fmt "%Ld" elt
      | UInt8 -> fprintf fmt "%d" elt
      | UInt16 -> fprintf fmt "%d" elt
      | Complex32 -> fprintf fmt "(%g+%gi)" elt.re elt.im
      | Complex64 -> fprintf fmt "(%g+%gi)" elt.re elt.im
    in

    if sz = 0 && ndim > 0 then fprintf fmt "[]"
    else if ndim = 0 then
      if sz > 0 then
        let value = Bigarray.Array1.unsafe_get buffer desc.offset in
        pp_element fmt value
      else fprintf fmt "<empty scalar>"
    else
      let rec pp_slice fmt current_indices =
        let current_ndim = List.length current_indices in
        if current_ndim = ndim then
          let md_index = Array.of_list current_indices in
          let linear_offset =
            md_to_linear md_index desc.strides + desc.offset
          in
          if linear_offset < 0 || linear_offset >= Bigarray.Array1.dim buffer
          then
            fprintf fmt "<OOB:%d/%d>" linear_offset (Bigarray.Array1.dim buffer)
          else
            let value = Bigarray.Array1.unsafe_get buffer linear_offset in
            pp_element fmt value
        else
          let axis = current_ndim in
          let dim_size = shape.(axis) in
          fprintf fmt "[";
          if dim_size > 0 then (
            if axis < ndim - 1 then pp_open_vbox fmt 0 else pp_open_hbox fmt ();
            for i = 0 to dim_size - 1 do
              if i > 0 then (
                fprintf fmt ",";
                if axis = ndim - 1 then fprintf fmt " " else pp_print_cut fmt ());
              pp_slice fmt (current_indices @ [ i ])
            done;
            pp_close_box fmt ());
          fprintf fmt "]"
      in
      if sz > 0 then pp_slice fmt [] else fprintf fmt "[]"

  let pp_info context fmt arr =
    let open Format in
    let desc = B.descriptor arr in

    fprintf fmt "@[<v 0>";
    fprintf fmt "Ndarray Info:@,";
    fprintf fmt "  Shape: %a@," (pp_shape context) desc.shape;
    fprintf fmt "  Dtype: %a@," (pp_dtype context) desc.dtype;
    fprintf fmt "  Strides: [%s]@,"
      (String.concat "; "
         (Array.to_list (Array.map string_of_int desc.strides)));
    fprintf fmt "  Offset: %d@," desc.offset;
    fprintf fmt "  Size: %d@," (size desc)

  let print_info ctx arr =
    pp_info ctx Format.std_formatter arr;
    Format.pp_print_newline Format.std_formatter ();
    Format.pp_print_flush Format.std_formatter ()

  let print ctx arr =
    pp ctx Format.std_formatter arr;
    Format.pp_print_newline Format.std_formatter ();
    Format.pp_print_flush Format.std_formatter ()

  let to_string ctx arr =
    let buf = Stdlib.Buffer.create 1024 in
    (* Increased buffer size *)
    let fmt = Format.formatter_of_buffer buf in
    pp ctx fmt arr;
    Format.pp_print_flush fmt ();
    Stdlib.Buffer.contents buf

  let to_string_info ctx arr =
    let buf = Stdlib.Buffer.create 1024 in
    let fmt = Format.formatter_of_buffer buf in
    pp_info ctx fmt arr;
    Format.pp_print_flush fmt ();
    Stdlib.Buffer.contents buf
end
