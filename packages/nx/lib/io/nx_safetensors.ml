(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Error

let strf = Printf.sprintf

let check_overwrite overwrite path =
  if (not overwrite) && Sys.file_exists path then
    failwith (strf "file already exists: %s" path)

(* Loading *)

type kind = K : ('a, 'b) Nx_dtype.t -> kind

(* A dtype nx lacks is handed out as its bytes. *)
let kind_of_dtype : Safetensors.dtype -> kind = function
  | BOOL -> K Bool
  | U8 | F8_E8M0 | F4 | F6_E2M3 | F6_E3M2 -> K UInt8
  | I8 -> K Int8
  | F8_E5M2 -> K Float8_e5m2
  | F8_E4M3 -> K Float8_e4m3
  | I16 -> K Int16
  | U16 -> K UInt16
  | F16 -> K Float16
  | BF16 -> K BFloat16
  | I32 -> K Int32
  | U32 -> K UInt32
  | F32 -> K Float32
  | F64 -> K Float64
  | I64 -> K Int64
  | U64 -> K UInt64

(* [tensor mapping kind shape ~off ~len] is the entry of [len] bytes at byte
   [off] of [mapping]. It is a view of [mapping] when its address suits [kind],
   and a copy in the machine's byte order otherwise. *)
let tensor (type a b) mapping (kind : (a, b) Nx_dtype.t) shape ~off ~len =
  let size = Nx_dtype.itemsize kind in
  let bytes = Nx_buffer.of_bigarray1 (Bigarray.Array1.sub mapping off len) in
  let aligned =
    let mask = Nativeint.of_int (size - 1) in
    Nativeint.logand (Nx_buffer.unsafe_data_ptr bytes) mask = 0n
  in
  let buffer =
    if len = 0 then Nx_buffer.create kind 0
    else if aligned && not Sys.big_endian then Nx_buffer.reinterpret kind bytes
    else begin
      let n = len / size in
      let buffer = Nx_buffer.create kind n in
      let dst = Bigarray.reshape_1 (Nx_buffer.to_genarray buffer [| n |]) n in
      Nx_io_codec.blit_bytes ~src:mapping ~src_off:off ~dst ~dst_off:0 ~len;
      if Sys.big_endian then
        Nx_io_codec.byteswap dst ~element_size:size ~elements:n;
      buffer
    end
  in
  Nx.P (Nx.of_buffer buffer ~shape)

let read_exactly fd n =
  let buf = Bytes.create n in
  let rec go off =
    if off < n then
      match Unix.read fd buf off (n - off) with
      | 0 -> failwith "unexpected end of file"
      | k -> go (off + k)
  in
  go 0;
  Bytes.unsafe_to_string buf

(* The file is validated before it is mapped: a page of a mapping that lies past
   the end of its file faults when touched, and no handler catches that. *)
let map_validated path fd =
  let stat = Unix.LargeFile.fstat fd in
  if stat.st_kind <> Unix.S_REG then failwith "not a regular file";
  let file_len = Int64.to_int stat.st_size in
  let prefix = Safetensors.header_len_bytes in
  if file_len < prefix then
    fail_msg "%d bytes is too short for a SafeTensors file" file_len;
  let header_len = String.get_int64_le (read_exactly fd prefix) 0 in
  if
    Int64.compare header_len 0L < 0
    || Int64.compare header_len (Int64.of_int Safetensors.max_header_size) > 0
  then
    fail_msg "header length %Lu exceeds the limit of %d bytes" header_len
      Safetensors.max_header_size;
  let header_len = Int64.to_int header_len in
  if prefix + header_len > file_len then
    fail_msg "header length %d exceeds the file's %d bytes" header_len file_len;
  match Safetensors.parse_header (read_exactly fd header_len) with
  | Error err -> failwith (Safetensors.string_of_error err)
  | Ok (metadata, data_len) ->
      let expected = prefix + header_len + data_len in
      if expected <> file_len then
        fail_msg "header describes a file of %d bytes but the file has %d"
          expected file_len;
      let mapping =
        Unix.map_file fd Bigarray.int8_unsigned Bigarray.c_layout false [| -1 |]
        |> Bigarray.array1_of_genarray
      in
      if Bigarray.Array1.dim mapping <> expected then
        fail_msg "file changed size while loading: mapped %d bytes of %d"
          (Bigarray.Array1.dim mapping)
          expected;
      (* The path is recorded as opened from any directory. *)
      let path =
        if Filename.is_relative path then Filename.concat (Sys.getcwd ()) path
        else path
      in
      Nx_buffer.register_file
        { path; size = file_len; mtime = stat.st_mtime; inode = stat.st_ino }
        (Nx_buffer.of_bigarray1 mapping);
      (metadata, prefix + header_len, mapping)

let load_safetensors path =
  try
    let fd =
      Unix.openfile path [ Unix.O_RDONLY; Unix.O_NONBLOCK; Unix.O_CLOEXEC ] 0
    in
    let (metadata : Safetensors.metadata), data_start, mapping =
      Fun.protect
        ~finally:(fun () -> Unix.close fd)
        (fun () -> map_validated path fd)
    in
    let archive = Hashtbl.create (Array.length metadata.tensors) in
    Hashtbl.iter
      (fun name index ->
        let info = metadata.tensors.(index) in
        let start, stop = info.data_offsets in
        let len = stop - start in
        let (K kind) = kind_of_dtype info.dtype in
        (* Elements narrower than a byte have no shape in bytes. *)
        let shape =
          match info.dtype with
          | F4 | F6_E2M3 | F6_E3M2 -> [| len |]
          | _ -> Array.of_list info.shape
        in
        Hashtbl.replace archive name
          (tensor mapping kind shape ~off:(data_start + start) ~len))
      metadata.index_map;
    archive
  with
  | Failure msg -> fail_msg "%s: %s" path msg
  | Unix.Unix_error (e, _, _) -> fail_msg "%s: %s" path (Unix.error_message e)

(* Saving *)

(* The SafeTensors dtype of [t] and its elements' bytes, in row-major order and
   little-endian, as stored: a float's bits are copied, never read as a
   float. *)
let tensor_to_bytes (type a b) (t : (a, b) Nx.t) =
  let dtype : Safetensors.dtype =
    match Nx.dtype t with
    | Bool -> BOOL
    | Int8 -> I8
    | UInt8 -> U8
    | Int16 -> I16
    | UInt16 -> U16
    | Int32 -> I32
    | UInt32 -> U32
    | Int64 -> I64
    | UInt64 -> U64
    | Float8_e4m3 -> F8_E4M3
    | Float8_e5m2 -> F8_E5M2
    | Float16 -> F16
    | BFloat16 -> BF16
    | Float32 -> F32
    | Float64 -> F64
    | dtype ->
        fail_msg "unsupported dtype for safetensors: %s"
          (Nx_dtype.to_string dtype)
  in
  let size = Nx.itemsize t in
  let bytes = Bytes.create (Nx.nbytes t) in
  Nx_buffer.blit_to_bytes (Nx.to_buffer t) bytes;
  if Sys.big_endian then
    for e = 0 to Nx.numel t - 1 do
      for i = 0 to (size / 2) - 1 do
        let lo = (e * size) + i and hi = (e * size) + size - 1 - i in
        let c = Bytes.get bytes lo in
        Bytes.set bytes lo (Bytes.get bytes hi);
        Bytes.set bytes hi c
      done
    done;
  (dtype, Bytes.unsafe_to_string bytes)

let replace_or_keep temp path =
  Unix.chmod temp Temp_file.mode;
  try Unix.rename temp path
  with Unix.Unix_error _ -> (
    Gc.full_major ();
    try Unix.rename temp path
    with Unix.Unix_error (e, _, _) ->
      fail_msg "cannot replace %s (%s): the tensors were written to %s" path
        (Unix.error_message e) temp)

let save_safetensors ?(overwrite = true) path items =
  check_overwrite overwrite path;
  let names = Hashtbl.create (List.length items) in
  List.iter
    (fun (name, _) ->
      if Hashtbl.mem names name then fail_msg "tensor %S is named twice" name;
      Hashtbl.add names name ())
    items;
  let tensor_views =
    List.map
      (fun (name, Nx.P arr) ->
        let shape = Array.to_list (Nx.shape arr) in
        let dtype, data = tensor_to_bytes arr in
        match Safetensors.tensor_view_new ~dtype ~shape ~data with
        | Ok view -> (name, view)
        | Error err ->
            fail_msg "failed to create tensor view for '%s': %s" name
              (Safetensors.string_of_error err))
      items
  in
  try
    let temp = Temp_file.sibling path in
    match Safetensors.serialize_to_file tensor_views None temp with
    | Ok () -> replace_or_keep temp path
    | Error err ->
        Temp_file.remove_if_exists temp;
        failwith (Safetensors.string_of_error err)
  with
  | Sys_error msg -> failwith msg
  | Unix.Unix_error (e, _, _) -> fail_msg "%s: %s" path (Unix.error_message e)
