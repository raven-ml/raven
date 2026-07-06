(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Error
open Packed_nx

let strf = Printf.sprintf

(* Little-endian byte encoding/decoding *)

let read_i32_le s off =
  let b0 = Char.code s.[off] in
  let b1 = Char.code s.[off + 1] in
  let b2 = Char.code s.[off + 2] in
  let b3 = Char.code s.[off + 3] in
  Int32.(
    logor
      (shift_left (of_int b3) 24)
      (logor
         (shift_left (of_int b2) 16)
         (logor (shift_left (of_int b1) 8) (of_int b0))))

let read_u16_le s off = Char.code s.[off] lor (Char.code s.[off + 1] lsl 8)

let write_i32_le bytes off v =
  Bytes.set bytes off (Char.chr (Int32.to_int (Int32.logand v 0xffl)));
  Bytes.set bytes (off + 1)
    (Char.chr (Int32.to_int (Int32.logand (Int32.shift_right v 8) 0xffl)));
  Bytes.set bytes (off + 2)
    (Char.chr (Int32.to_int (Int32.logand (Int32.shift_right v 16) 0xffl)));
  Bytes.set bytes (off + 3)
    (Char.chr (Int32.to_int (Int32.logand (Int32.shift_right v 24) 0xffl)))

(* Error conversion *)

let wrap_exn f =
  try f () with
  | Sys_error msg -> Error (Io_error msg)
  | ex -> Error (Other (Printexc.to_string ex))

let check_overwrite overwrite path =
  if (not overwrite) && Sys.file_exists path then
    failwith (strf "file already exists: %s" path)

(* Tensor construction helpers *)

let make_tensor kind shape n f =
  let ba = Nx_buffer.create kind n in
  for i = 0 to n - 1 do
    Nx_buffer.unsafe_set ba i (f i)
  done;
  Nx.reshape shape (Nx.of_buffer ba ~shape:[| n |])

(* Byte-swap 16-bit elements in [buf] from native to little-endian or back *)
let swap_16 buf n =
  for i = 0 to n - 1 do
    let pos = i * 2 in
    let b0 = Bytes.get buf pos in
    Bytes.set buf pos (Bytes.get buf (pos + 1));
    Bytes.set buf (pos + 1) b0
  done

(* Load 16-bit LE data into a tensor, byte-swapping on big-endian *)
let blit_tensor_16le kind shape n data offset =
  let byte_len = n * 2 in
  let ba = Nx_buffer.create kind n in
  let tmp = Bytes.create byte_len in
  if Sys.big_endian then begin
    for i = 0 to n - 1 do
      let src = offset + (i * 2) in
      let dst = i * 2 in
      Bytes.set tmp dst data.[src + 1];
      Bytes.set tmp (dst + 1) data.[src]
    done
  end
  else Bytes.blit_string data offset tmp 0 byte_len;
  Nx_buffer.blit_from_bytes ~src_off:0 ~dst_off:0 ~len:n tmp ba;
  Nx.reshape shape (Nx.of_buffer ba ~shape:[| n |])

(* Load 8-bit data into a tensor, preserving exact bit patterns *)
let blit_tensor_8 kind shape n data offset =
  let ba = Nx_buffer.create kind n in
  let tmp = Bytes.create n in
  Bytes.blit_string data offset tmp 0 n;
  Nx_buffer.blit_from_bytes ~src_off:0 ~dst_off:0 ~len:n tmp ba;
  Nx.reshape shape (Nx.of_buffer ba ~shape:[| n |])

(* Loading *)

let load_tensor (view : Safetensors.tensor_view) =
  let shape = Array.of_list view.shape in
  let n = Array.fold_left ( * ) 1 shape in
  match view.dtype with
  | F32 ->
      let f i =
        Int32.float_of_bits (read_i32_le view.data (view.offset + (i * 4)))
      in
      Some (P (make_tensor Float32 shape n f))
  | F64 ->
      let f i =
        Int64.float_of_bits
          (Safetensors.read_u64_le view.data (view.offset + (i * 8)))
      in
      Some (P (make_tensor Float64 shape n f))
  | I32 ->
      let f i = read_i32_le view.data (view.offset + (i * 4)) in
      Some (P (make_tensor Int32 shape n f))
  | U32 ->
      let f i = read_i32_le view.data (view.offset + (i * 4)) in
      Some (P (make_tensor Uint32 shape n f))
  | I64 ->
      let f i = Safetensors.read_u64_le view.data (view.offset + (i * 8)) in
      Some (P (make_tensor Int64 shape n f))
  | U64 ->
      let f i = Safetensors.read_u64_le view.data (view.offset + (i * 8)) in
      Some (P (make_tensor Uint64 shape n f))
  | I16 ->
      let f i =
        let v = read_u16_le view.data (view.offset + (i * 2)) in
        if v >= 0x8000 then v - 0x10000 else v
      in
      Some (P (make_tensor Int16_signed shape n f))
  | U16 ->
      let f i = read_u16_le view.data (view.offset + (i * 2)) in
      Some (P (make_tensor Int16_unsigned shape n f))
  | I8 ->
      let f i =
        let v = Char.code view.data.[view.offset + i] in
        if v >= 0x80 then v - 0x100 else v
      in
      Some (P (make_tensor Int8_signed shape n f))
  | U8 ->
      let f i = Char.code view.data.[view.offset + i] in
      Some (P (make_tensor Int8_unsigned shape n f))
  | BOOL ->
      let f i = Char.code view.data.[view.offset + i] <> 0 in
      Some (P (make_tensor Bool shape n f))
  | F8_E4M3 ->
      Some (P (blit_tensor_8 Float8_e4m3 shape n view.data view.offset))
  | F8_E5M2 ->
      Some (P (blit_tensor_8 Float8_e5m2 shape n view.data view.offset))
  | F16 ->
      if view.offset land 1 <> 0 then
        fail_msg "unaligned float16 tensor offset: %d" view.offset;
      Some (P (blit_tensor_16le Float16 shape n view.data view.offset))
  | BF16 ->
      if view.offset land 1 <> 0 then
        fail_msg "unaligned bfloat16 tensor offset: %d" view.offset;
      Some (P (blit_tensor_16le Bfloat16 shape n view.data view.offset))
  | _ -> None

let load_safetensors path =
  wrap_exn @@ fun () ->
  let ic = open_in_bin path in
  let buf =
    Fun.protect ~finally:(fun () -> close_in ic) @@ fun () ->
    let len = in_channel_length ic in
    really_input_string ic len
  in
  match Safetensors.deserialize buf with
  | Error err -> Error (Format_error (Safetensors.string_of_error err))
  | Ok st ->
      let tensors = Safetensors.tensors st in
      let result = Hashtbl.create (List.length tensors) in
      List.iter
        (fun (name, view) ->
          match load_tensor view with
          | Some packed -> Hashtbl.add result name packed
          | None ->
              Printf.eprintf
                "warning: skipping tensor '%s' with unsupported dtype %s\n" name
                (Safetensors.dtype_to_string view.dtype))
        tensors;
      Ok result

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
  | Uint32 -> (Safetensors.U32, le32 (Nx_buffer.unsafe_get buf))
  | Int64 -> (Safetensors.I64, le64 (Nx_buffer.unsafe_get buf))
  | Uint64 -> (Safetensors.U64, le64 (Nx_buffer.unsafe_get buf))
  | Int16_signed -> (Safetensors.I16, le16 (Nx_buffer.unsafe_get buf))
  | Int16_unsigned -> (Safetensors.U16, le16 (Nx_buffer.unsafe_get buf))
  | Int8_signed ->
      let get i = Char.chr (Nx_buffer.unsafe_get buf i land 0xff) in
      (Safetensors.I8, le8 get)
  | Int8_unsigned ->
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
  | Float16 | Bfloat16 ->
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
        (Nx_core.Dtype.of_buffer_kind (Nx_buffer.kind buf)
        |> Nx_core.Dtype.to_string)

let save_safetensors ?(overwrite = true) path items =
  wrap_exn @@ fun () ->
  check_overwrite overwrite path;
  let tensor_views =
    List.map
      (fun (name, P arr) ->
        let shape = Array.to_list (Nx.shape arr) in
        let dtype, data = tensor_to_bytes arr in
        match Safetensors.tensor_view_new ~dtype ~shape ~data with
        | Ok view -> (name, view)
        | Error err ->
            fail_msg "failed to create tensor view for '%s': %s" name
              (Safetensors.string_of_error err))
      items
  in
  match Safetensors.serialize_to_file tensor_views None path with
  | Ok () -> Ok ()
  | Error err -> Error (Format_error (Safetensors.string_of_error err))
