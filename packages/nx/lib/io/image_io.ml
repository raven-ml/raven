(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

open Bigarray

type bytes = (int, int8_unsigned_elt, c_layout) Array1.t

external png_probe : bytes -> int * int = "caml_nx_io_png_probe"
external png_decode : bytes -> bytes -> bool -> unit = "caml_nx_io_png_decode"
external jpeg_probe : bytes -> int * int = "caml_nx_io_jpeg_probe"
external jpeg_decode : bytes -> bytes -> bool -> unit = "caml_nx_io_jpeg_decode"

external png_encode : Unix.file_descr -> bytes -> int -> int -> int -> unit
  = "caml_nx_io_png_encode"

external jpeg_encode : Unix.file_descr -> bytes -> int -> int -> int -> unit
  = "caml_nx_io_jpeg_encode"

let map_file fd size =
  if size = 0 then Array1.create int8_unsigned c_layout 0
  else
    Unix.map_file fd int8_unsigned c_layout false [| size |]
    |> Bigarray.array1_of_genarray

let checked_pixels width height channels =
  if width <= 0 || height <= 0 then failwith "image dimensions must be positive";
  if width > max_int / height || width * height > max_int / channels then
    failwith "image dimensions are too large";
  width * height * channels

let load_image ~grayscale path =
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  let src = map_file fd (Unix.fstat fd).st_size in
  let probe, decode =
    if
      Array1.dim src >= 8
      && Array1.unsafe_get src 0 = 0x89
      && Array1.unsafe_get src 1 = 0x50
      && Array1.unsafe_get src 2 = 0x4e
      && Array1.unsafe_get src 3 = 0x47
      && Array1.unsafe_get src 4 = 0x0d
      && Array1.unsafe_get src 5 = 0x0a
      && Array1.unsafe_get src 6 = 0x1a
      && Array1.unsafe_get src 7 = 0x0a
    then (png_probe, png_decode)
    else if
      Array1.dim src >= 2
      && Array1.unsafe_get src 0 = 0xff
      && Array1.unsafe_get src 1 = 0xd8
    then (jpeg_probe, jpeg_decode)
    else failwith "unsupported image stream: expected PNG or JPEG"
  in
  let width, height = probe src in
  let channels = if grayscale then 1 else 3 in
  let length = checked_pixels width height channels in
  let buffer = Nx_buffer.create Nx_buffer.UInt8 length in
  let dst = Nx_buffer.to_bigarray1 buffer in
  decode src dst grayscale;
  let shape =
    if grayscale then [| height; width |] else [| height; width; 3 |]
  in
  Nx.of_buffer buffer ~shape

let remove_if_exists path = try Sys.remove path with Sys_error _ -> ()

let replace_with_temp temp path =
  match
    Unix.chmod temp 0o640;
    Unix.rename temp path
  with
  | () -> ()
  | exception exn ->
      remove_if_exists temp;
      raise exn

let encode_to_path ~encode ~exclusive path data ~width ~height ~channels =
  let flags =
    if exclusive then [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_EXCL ]
    else [ Unix.O_WRONLY; Unix.O_TRUNC ]
  in
  let fd = Unix.openfile path flags 0o640 in
  match
    Fun.protect
      ~finally:(fun () -> Unix.close fd)
      (fun () -> encode fd data width height channels)
  with
  | () -> ()
  | exception exn ->
      remove_if_exists path;
      raise exn

let save_png ~overwrite path data ~width ~height ~channels =
  ignore (checked_pixels width height channels);
  if not overwrite then
    encode_to_path ~encode:png_encode ~exclusive:true path data ~width ~height
      ~channels
  else
    let temp =
      Filename.temp_file ~temp_dir:(Filename.dirname path)
        (Filename.basename path ^ ".")
        ".tmp"
    in
    match
      encode_to_path ~encode:png_encode ~exclusive:false temp data ~width
        ~height ~channels
    with
    | () -> replace_with_temp temp path
    | exception exn ->
        remove_if_exists temp;
        raise exn

let save_jpeg ~overwrite path data ~width ~height ~channels =
  ignore (checked_pixels width height channels);
  if channels <> 1 && channels <> 3 then
    invalid_arg "JPEG output requires one or three channels";
  if not overwrite then
    encode_to_path ~encode:jpeg_encode ~exclusive:true path data ~width ~height
      ~channels
  else
    let temp =
      Filename.temp_file ~temp_dir:(Filename.dirname path)
        (Filename.basename path ^ ".")
        ".tmp"
    in
    match
      encode_to_path ~encode:jpeg_encode ~exclusive:false temp data ~width
        ~height ~channels
    with
    | () -> replace_with_temp temp path
    | exception exn ->
        remove_if_exists temp;
        raise exn
