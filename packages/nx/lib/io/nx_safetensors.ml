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

module B = Nx_device.Buffer

(* The [n] bytes of [file] from byte [pos]. *)
let read_string file ~pos n =
  let b = B.create Nx_device.host Nx_dtype.Scalar.UInt8 n in
  B.copy ~src:(B.view file ~offset:pos Nx_dtype.Scalar.UInt8 n) ~dst:b;
  let chars = B.bigarray Bigarray.char b in
  String.init n (Bigarray.Array1.get chars)

(* The header of [file] and the offset of its data, once the file's length is
   the one the header describes. *)
let read_header file =
  let file_len = B.length file in
  let prefix = Safetensors.header_len_bytes in
  if file_len < prefix then
    fail_msg "%d bytes is too short for a SafeTensors file" file_len;
  let header_len = String.get_int64_le (read_string file ~pos:0 prefix) 0 in
  if
    Int64.compare header_len 0L < 0
    || Int64.compare header_len (Int64.of_int Safetensors.max_header_size) > 0
  then
    fail_msg "header length %Lu exceeds the limit of %d bytes" header_len
      Safetensors.max_header_size;
  let header_len = Int64.to_int header_len in
  if prefix + header_len > file_len then
    fail_msg "header length %d exceeds the file's %d bytes" header_len file_len;
  match Safetensors.parse_header (read_string file ~pos:prefix header_len) with
  | Error err -> failwith (Safetensors.string_of_error err)
  | Ok (metadata, data_len) ->
      let expected = prefix + header_len + data_len in
      if expected <> file_len then
        fail_msg "header describes a file of %d bytes but the file has %d"
          expected file_len;
      (metadata, prefix + header_len)

let load_safetensors path =
  match B.of_file path with
  | Error why -> failwith why
  | Ok file -> (
      try
        let (metadata : Safetensors.metadata), data_start = read_header file in
        Hashtbl.fold
          (fun name index archive ->
            if name = "" then failwith "a tensor has an empty name";
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
            Archive.add name
              (Storage.mapped file kind shape ~off:(data_start + start) ~len)
              archive)
          metadata.index_map Archive.empty
      with
      | Failure msg -> fail_msg "%s: %s" path msg
      | Sys_error msg -> failwith msg)

(* Saving *)

(* [tensor_data name t] is the SafeTensors dtype of [t] and a buffer of its
   elements' bytes, in row-major order and little-endian, as stored: on a
   little-endian host, a value with one device's buffer is written from that
   device, from its own storage when its elements are a contiguous run of it.
   Any other value, a traced one included, is read to the host. A float's bits
   are copied, never read as a float. [save_safetensors] claims the buffer while
   it writes it. It fails naming the entry [name] if SafeTensors has no dtype
   for [t]'s. *)
let tensor_data (type a b) name (t : (a, b) Nx.t) =
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
    | Bit -> fail_msg "%s: %s" name (no_bit "SafeTensors")
    | dtype ->
        fail_msg "%s: SafeTensors has no %s" name (Nx_dtype.to_string dtype)
  in
  let size = Nx_dtype.itemsize (Nx.dtype t) in
  if Sys.big_endian && size > 1 then begin
    Storage.reading ~by:"Nx_io.save_safetensors" t @@ fun elements ->
    let swapped =
      B.create Nx_device.host (B.dtype elements) (B.length elements)
    in
    B.copy ~src:elements ~dst:swapped;
    Nx_io_codec.byteswap (Storage.bytes swapped) ~element_size:size
      ~elements:(Nx.numel t);
    (dtype, swapped)
  end
  else
    match Nx.shards t with
    | [ _ ], _ -> (dtype, Nx.to_buffer t)
    | _ | (exception Invalid_argument _) ->
        (dtype, Nx.Op.eval (Read { by = "Nx_io.save_safetensors"; x = t }))

let replace_or_keep temp path =
  let failed e =
    fail_msg "cannot replace %s (%s): the tensors were written to %s" path
      (Unix.error_message e) temp
  in
  try Unix.rename temp path with
  | Unix.Unix_error _ when Sys.win32 -> (
      (* Windows refuses to replace a file while a view of its pages is mapped:
         a value loaded from [path] and borrowed may hold one until it is
         collected. *)
      Gc.full_major ();
      Nx_device.synchronize Nx_device.disk;
      try Unix.rename temp path with Unix.Unix_error (e, _, _) -> failed e)
  | Unix.Unix_error (e, _, _) -> failed e

(* A new file of [n] bytes beside [path], and its name. *)
let sibling path n =
  let held name =
    match Unix.lstat name with
    | _ -> true
    | exception Unix.Unix_error _ -> false
  in
  let rec go attempts =
    let temp = Temp_file.name path in
    match B.create_file temp n with
    | Ok file -> (temp, file)
    | Error _ when attempts > 1 && held temp -> go (attempts - 1)
    | Error why -> raise (Sys_error why)
  in
  go 1000

(* The bytes of a file of [header] then [parts], each part at its offset after
   the header. *)
let size header parts =
  String.length header
  + List.fold_left (fun n (off, src) -> Int.max n (off + B.nbytes src)) 0 parts

(* Writes the header and the tensors' bytes into [file]. *)
let write file header parts =
  let hlen = String.length header in
  let bytes = B.create Nx_device.host Nx_dtype.Scalar.UInt8 hlen in
  let chars = B.bigarray Bigarray.char bytes in
  String.iteri (Bigarray.Array1.set chars) header;
  B.copy ~src:bytes ~dst:(B.view file ~offset:0 Nx_dtype.Scalar.UInt8 hlen);
  List.iter
    (fun (off, src) ->
      B.copy ~src
        ~dst:
          (B.view file ~offset:(hlen + off) Nx_dtype.Scalar.UInt8 (B.nbytes src)))
    parts

let save_safetensors ?(overwrite = true) path archive =
  check_overwrite overwrite path;
  let items = Archive.bindings archive in
  let tensors =
    List.map
      (fun (name, Nx.P arr) ->
        let shape = Array.to_list (Nx.shape arr) in
        let dtype, src = tensor_data name arr in
        match
          Safetensors.tensor_view_new ~dtype ~shape ~nbytes:(B.nbytes src)
        with
        | Ok view -> (name, (view, src))
        | Error err ->
            fail_msg "failed to create tensor view for '%s': %s" name
              (Safetensors.string_of_error err))
      items
  in
  let header, offsets =
    match
      Safetensors.layout (List.map (fun (n, (v, _)) -> (n, v)) tensors) None
    with
    | Ok layout -> layout
    | Error err -> failwith (Safetensors.string_of_error err)
  in
  let srcs = Hashtbl.create (List.length tensors) in
  List.iter (fun (name, (_, src)) -> Hashtbl.add srcs name src) tensors;
  let parts =
    List.map (fun (name, off) -> (off, Hashtbl.find srcs name)) offsets
  in
  try
    let temp, file = sibling path (size header parts) in
    match
      Storage.claiming (List.map snd parts) (fun () -> write file header parts);
      B.flush file
    with
    | () -> replace_or_keep temp path
    | exception e ->
        Temp_file.remove_if_exists temp;
        raise e
  with
  | Sys_error msg -> failwith msg
  | Unix.Unix_error (e, _, _) -> fail_msg "%s: %s" path (Unix.error_message e)
