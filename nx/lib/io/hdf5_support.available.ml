(* HDF5 support is available *)

open Bigarray_ext
open Utils

let hdf5_available = true

(* Convert HDF5 dataset to packed_nx *)
let h5_to_packed_nx dataset_id =
  let open Hdf5_raw in
  (* Get dataspace to determine shape *)
  let space_id = H5d.get_space dataset_id in
  let dims, _maxdims = H5s.get_simple_extent_dims space_id in

  (* Get datatype *)
  let dtype_id = H5d.get_type dataset_id in
  let type_class = H5t.get_class dtype_id in
  let type_size = H5t.get_size dtype_id in

  (* Read the data based on type *)
  let result =
    match type_class with
    | H5t.Class.FLOAT ->
        if type_size = 4 then (
          let arr = Genarray.create Float32 C_layout dims in
          H5d.read_bigarray dataset_id H5t.native_float H5s.all H5s.all arr;
          P (Nx.of_bigarray arr))
        else if type_size = 8 then (
          let arr = Genarray.create Float64 C_layout dims in
          H5d.read_bigarray dataset_id H5t.native_double H5s.all H5s.all arr;
          P (Nx.of_bigarray arr))
        else fail_msg "Unsupported float size: %d bytes" type_size
    | H5t.Class.INTEGER ->
        (* Handle different integer sizes - default to common types *)
        if type_size = 1 then (
          let arr = Genarray.create Int8_unsigned C_layout dims in
          H5d.read_bigarray dataset_id H5t.native_uint8 H5s.all H5s.all arr;
          P (Nx.of_bigarray arr))
        else if type_size = 2 then (
          let arr = Genarray.create Int16_signed C_layout dims in
          H5d.read_bigarray dataset_id H5t.native_int16 H5s.all H5s.all arr;
          P (Nx.of_bigarray arr))
        else if type_size = 4 then (
          let arr = Genarray.create Int32 C_layout dims in
          H5d.read_bigarray dataset_id H5t.native_int32 H5s.all H5s.all arr;
          P (Nx.of_bigarray arr))
        else if type_size = 8 then (
          let arr = Genarray.create Int64 C_layout dims in
          H5d.read_bigarray dataset_id H5t.native_int64 H5s.all H5s.all arr;
          P (Nx.of_bigarray arr))
        else fail_msg "Unsupported integer size: %d bytes" type_size
    | _ -> fail_msg "Unsupported HDF5 datatype class"
  in

  (* Clean up *)
  H5t.close dtype_id;
  H5s.close space_id;
  result

let load_h5_dataset ~path ~dataset =
  let open Hdf5_raw in
  try
    (* Open file *)
    let file_id = H5f.open_ path [ H5f.Acc.RDONLY ] in

    (* Open dataset *)
    let dataset_id =
      try H5d.open_ file_id dataset
      with _ ->
        H5f.close file_id;
        fail_msg "H5 Load Error (%s): Dataset '%s' not found" path dataset
    in

    (* Read the data *)
    let result = h5_to_packed_nx dataset_id in

    (* Clean up *)
    H5d.close dataset_id;
    H5f.close file_id;
    result
  with
  | Unix.Unix_error (e, _, _) ->
      fail_msg "H5 Load Error (%s): %s" path (Unix.error_message e)
  | Sys_error msg -> fail_msg "H5 Load System Error (%s): %s" path msg
  | Failure msg -> fail_msg "H5 Load Failure (%s): %s" path msg
  | ex ->
      let err_msg = Printexc.to_string ex in
      fail_msg "Unexpected H5 Load Error (%s): %s" path err_msg

let save_h5_dataset (type a b) (nx : (a, b) Nx.t) ~path ~dataset =
  let open Hdf5_raw in
  try
    (* Create or open file *)
    let file_id =
      if Sys.file_exists path then H5f.open_ path [ H5f.Acc.RDWR ]
      else H5f.create path [ H5f.Acc.TRUNC ]
    in

    (* Get array info *)
    let genarray = Nx.to_bigarray nx in
    let dims = Genarray.dims genarray in
    (* Note: dims is already int array, which is same as Hsize.t *)
    let kind = Genarray.kind genarray in

    (* Create dataspace *)
    let space_id = H5s.create_simple dims in

    (* Determine HDF5 datatype based on OCaml array kind *)
    (* We need to use runtime type determination due to GADT constraints *)
    let h5_type =
      match kind with
      | Float32 -> H5t.native_float
      | Float64 -> H5t.native_double
      | Int8_signed -> H5t.native_int8
      | Int8_unsigned -> H5t.native_uint8
      | Int16_signed -> H5t.native_int16
      | Int16_unsigned -> H5t.native_uint16
      | Int32 -> H5t.native_int32
      | Int64 -> H5t.native_int64
      | _ -> fail_msg "Unsupported array type for HDF5 save"
    in

    (* Check if dataset exists and delete if so *)
    let dataset_exists =
      try
        let ds = H5d.open_ file_id dataset in
        H5d.close ds;
        true
      with _ -> false
    in

    if dataset_exists then H5l.delete file_id dataset;

    (* Create dataset *)
    let dataset_id = H5d.create file_id dataset h5_type space_id in

    (* Write data *)
    H5d.write_bigarray dataset_id h5_type H5s.all H5s.all genarray;

    (* Clean up *)
    H5d.close dataset_id;
    H5s.close space_id;
    H5f.close file_id
  with
  | Unix.Unix_error (e, _, _) ->
      fail_msg "H5 Save Error (%s): %s" path (Unix.error_message e)
  | Sys_error msg -> fail_msg "H5 Save System Error (%s): %s" path msg
  | Failure msg -> fail_msg "H5 Save Failure (%s): %s" path msg
  | ex ->
      let err_msg = Printexc.to_string ex in
      fail_msg "Unexpected H5 Save Error (%s): %s" path err_msg

type h5_archive = (string, packed_nx) Hashtbl.t

let load_h5_all path =
  let open Hdf5_raw in
  let archive = Hashtbl.create 16 in
  try
    (* Open file *)
    let file_id = H5f.open_ path [ H5f.Acc.RDONLY ] in

    (* Simple implementation: try to iterate through root group *)
    (* For a complete implementation, we'd need recursive group traversal *)
    let root_info = H5g.get_info file_id in

    for i = 0 to root_info.H5g.Info.nlinks - 1 do
      try
        (* Get object name by index - use correct module names *)
        let obj_name =
          H5l.get_name_by_idx file_id "." H5_raw.Index.NAME
            H5_raw.Iter_order.INC i
        in

        (* Try to open as dataset *)
        try
          let dataset_id = H5d.open_ file_id obj_name in
          let packed = h5_to_packed_nx dataset_id in
          Hashtbl.add archive obj_name packed;
          H5d.close dataset_id
        with _ ->
          (* Not a dataset, might be a group - skip for now *)
          ()
      with _ -> ()
    done;

    H5f.close file_id;
    archive
  with
  | Unix.Unix_error (e, _, _) ->
      fail_msg "H5 Load Error (%s): %s" path (Unix.error_message e)
  | Sys_error msg -> fail_msg "H5 Load System Error (%s): %s" path msg
  | Failure msg -> fail_msg "H5 Load Failure (%s): %s" path msg
  | ex ->
      let err_msg = Printexc.to_string ex in
      fail_msg "Unexpected H5 Load Error (%s): %s" path err_msg

let save_h5_all items path =
  let open Hdf5_raw in
  try
    (* Create new file *)
    let file_id = H5f.create path [ H5f.Acc.TRUNC ] in

    (* Save each item as a dataset *)
    List.iter
      (fun (name, P nx) ->
        (* Use polymorphic save that works with packed type *)
        let genarray = Nx.to_bigarray nx in
        let dims = Genarray.dims genarray in
        let kind = Genarray.kind genarray in

        (* Create dataspace *)
        let space_id = H5s.create_simple dims in

        (* Determine HDF5 datatype *)
        let h5_type =
          match kind with
          | Float32 -> H5t.native_float
          | Float64 -> H5t.native_double
          | Int8_signed -> H5t.native_int8
          | Int8_unsigned -> H5t.native_uint8
          | Int16_signed -> H5t.native_int16
          | Int16_unsigned -> H5t.native_uint16
          | Int32 -> H5t.native_int32
          | Int64 -> H5t.native_int64
          | _ -> fail_msg "Unsupported array type for HDF5 save"
        in

        (* Create and write dataset *)
        let dataset_id = H5d.create file_id name h5_type space_id in
        H5d.write_bigarray dataset_id h5_type H5s.all H5s.all genarray;

        (* Clean up *)
        H5d.close dataset_id;
        H5s.close space_id)
      items;

    H5f.close file_id
  with
  | Unix.Unix_error (e, _, _) ->
      fail_msg "H5 Save Error (%s): %s" path (Unix.error_message e)
  | Sys_error msg -> fail_msg "H5 Save System Error (%s): %s" path msg
  | Failure msg -> fail_msg "H5 Save Failure (%s): %s" path msg
  | ex ->
      let err_msg = Printexc.to_string ex in
      fail_msg "Unexpected H5 Save Error (%s): %s" path err_msg
