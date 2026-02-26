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

let blit_tensor kind shape n data_bytes src_off =
  let ba = Nx_buffer.create kind n in
  Nx_buffer.blit_from_bytes ~src_off ~dst_off:0 ~len:n data_bytes ba;
  Nx.reshape shape (Nx.of_buffer ba ~shape:[| n |])

(* Loading *)

let load_tensor (view : Safetensors.tensor_view) =
  let shape = Array.of_list view.shape in
  let n = Array.fold_left ( * ) 1 shape in
  let data_bytes = Bytes.unsafe_of_string view.data in
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
  | F16 ->
      if view.offset land 1 <> 0 then
        fail_msg "unaligned float16 tensor offset: %d" view.offset;
      Some (P (blit_tensor Float16 shape n data_bytes (view.offset / 2)))
  | BF16 ->
      if view.offset land 1 <> 0 then
        fail_msg "unaligned bfloat16 tensor offset: %d" view.offset;
      Some (P (blit_tensor Bfloat16 shape n data_bytes (view.offset / 2)))
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
  let buf = Nx.to_buffer (Nx.flatten arr) in
  match Nx_buffer.kind buf with
  | Float32 ->
      let bytes = Bytes.create (n * 4) in
      for i = 0 to n - 1 do
        write_i32_le bytes (i * 4)
          (Int32.bits_of_float (Nx_buffer.unsafe_get buf i))
      done;
      (Safetensors.F32, Bytes.unsafe_to_string bytes)
  | Float64 ->
      let bytes = Bytes.create (n * 8) in
      for i = 0 to n - 1 do
        Safetensors.write_u64_le bytes (i * 8)
          (Int64.bits_of_float (Nx_buffer.unsafe_get buf i))
      done;
      (Safetensors.F64, Bytes.unsafe_to_string bytes)
  | Int32 ->
      let bytes = Bytes.create (n * 4) in
      for i = 0 to n - 1 do
        write_i32_le bytes (i * 4) (Nx_buffer.unsafe_get buf i)
      done;
      (Safetensors.I32, Bytes.unsafe_to_string bytes)
  | Float16 ->
      let bytes = Bytes.create (n * 2) in
      Nx_buffer.blit_to_bytes ~src_off:0 ~dst_off:0 ~len:n buf bytes;
      (Safetensors.F16, Bytes.unsafe_to_string bytes)
  | Bfloat16 ->
      let bytes = Bytes.create (n * 2) in
      Nx_buffer.blit_to_bytes ~src_off:0 ~dst_off:0 ~len:n buf bytes;
      (Safetensors.BF16, Bytes.unsafe_to_string bytes)
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
