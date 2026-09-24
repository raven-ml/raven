(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Error

let strf = Printf.sprintf

(* Little-endian byte encoding *)

let write_i32_le bytes off v =
  Bytes.set bytes off (Char.chr (Int32.to_int (Int32.logand v 0xffl)));
  Bytes.set bytes (off + 1)
    (Char.chr (Int32.to_int (Int32.logand (Int32.shift_right v 8) 0xffl)));
  Bytes.set bytes (off + 2)
    (Char.chr (Int32.to_int (Int32.logand (Int32.shift_right v 16) 0xffl)));
  Bytes.set bytes (off + 3)
    (Char.chr (Int32.to_int (Int32.logand (Int32.shift_right v 24) 0xffl)))

let check_overwrite overwrite path =
  if (not overwrite) && Sys.file_exists path then
    failwith (strf "file already exists: %s" path)

(* Byte-swap 16-bit elements in [buf] from native to little-endian or back *)
let swap_16 buf n =
  for i = 0 to n - 1 do
    let pos = i * 2 in
    let b0 = Bytes.get buf pos in
    Bytes.set buf pos (Bytes.get buf (pos + 1));
    Bytes.set buf (pos + 1) b0
  done

(* Loading *)

type kind = K : ('a, 'b) Nx_buffer.kind -> kind

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
let tensor (type a b) mapping (kind : (a, b) Nx_buffer.kind) shape ~off ~len =
  let size = Nx_buffer.kind_size_in_bytes kind in
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

let tensor_to_bytes (type a b) (arr : (a, b) Nx.t) =
  let n = Array.fold_left ( * ) 1 (Nx.shape arr) in
  (* Nx.flatten rejects rank-0 tensors; reshape to [| n |] handles all ranks *)
  let buf = Nx.to_buffer (Nx.reshape [| n |] arr) in
  (* Little-endian encoders; [get] closes over [buf] with its element type
     refined by the branch that calls the encoder. *)
  let le8 (get : int -> char) =
    let bytes = Bytes.create n in
    for i = 0 to n - 1 do
      Bytes.set bytes i (get i)
    done;
    Bytes.unsafe_to_string bytes
  in
  let le16 (get : int -> int) =
    let bytes = Bytes.create (n * 2) in
    for i = 0 to n - 1 do
      let v = get i land 0xffff in
      Bytes.set bytes (i * 2) (Char.chr (v land 0xff));
      Bytes.set bytes ((i * 2) + 1) (Char.chr (v lsr 8))
    done;
    Bytes.unsafe_to_string bytes
  in
  let le32 (get : int -> int32) =
    let bytes = Bytes.create (n * 4) in
    for i = 0 to n - 1 do
      write_i32_le bytes (i * 4) (get i)
    done;
    Bytes.unsafe_to_string bytes
  in
  let le64 (get : int -> int64) =
    let bytes = Bytes.create (n * 8) in
    for i = 0 to n - 1 do
      Safetensors.write_u64_le bytes (i * 8) (get i)
    done;
    Bytes.unsafe_to_string bytes
  in
  match Nx_buffer.kind buf with
  | Float32 ->
      let get i = Int32.bits_of_float (Nx_buffer.unsafe_get buf i) in
      (Safetensors.F32, le32 get)
  | Float64 ->
      let get i = Int64.bits_of_float (Nx_buffer.unsafe_get buf i) in
      (Safetensors.F64, le64 get)
  | Int32 -> (Safetensors.I32, le32 (Nx_buffer.unsafe_get buf))
  | UInt32 -> (Safetensors.U32, le32 (Nx_buffer.unsafe_get buf))
  | Int64 -> (Safetensors.I64, le64 (Nx_buffer.unsafe_get buf))
  | UInt64 -> (Safetensors.U64, le64 (Nx_buffer.unsafe_get buf))
  | Int16 -> (Safetensors.I16, le16 (Nx_buffer.unsafe_get buf))
  | UInt16 -> (Safetensors.U16, le16 (Nx_buffer.unsafe_get buf))
  | Int8 ->
      let get i = Char.chr (Nx_buffer.unsafe_get buf i land 0xff) in
      (Safetensors.I8, le8 get)
  | UInt8 ->
      let get i = Char.chr (Nx_buffer.unsafe_get buf i land 0xff) in
      (Safetensors.U8, le8 get)
  | Bool ->
      let get i = if Nx_buffer.unsafe_get buf i then '\001' else '\000' in
      (Safetensors.BOOL, le8 get)
  | Float8_e4m3 | Float8_e5m2 ->
      let tag =
        match Nx_buffer.kind buf with
        | Float8_e4m3 -> Safetensors.F8_E4M3
        | _ -> Safetensors.F8_E5M2
      in
      let bytes = Bytes.create n in
      Nx_buffer.blit_to_bytes ~src_off:0 ~dst_off:0 ~len:n buf bytes;
      (tag, Bytes.unsafe_to_string bytes)
  | Float16 | BFloat16 ->
      let tag =
        match Nx_buffer.kind buf with
        | Float16 -> Safetensors.F16
        | _ -> Safetensors.BF16
      in
      let bytes = Bytes.create (n * 2) in
      Nx_buffer.blit_to_bytes ~src_off:0 ~dst_off:0 ~len:n buf bytes;
      if Sys.big_endian then swap_16 bytes n;
      (tag, Bytes.unsafe_to_string bytes)
  | _ ->
      fail_msg "unsupported dtype for safetensors: %s"
        (Nx_buffer.kind_name (Nx_buffer.kind buf))

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
