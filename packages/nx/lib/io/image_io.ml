(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

open Bigarray

type bytes = (int, int8_unsigned_elt, c_layout) Array1.t

external png_parse : bytes -> int * int = "caml_nx_io_png_probe"
external png_idat : bytes -> bytes * int = "caml_nx_io_png_idat"

external png_unfilter : bytes -> bytes -> bytes -> bool -> unit
  = "caml_nx_io_png_decode"

external png_filter : bytes -> int -> int -> int -> string
  = "caml_nx_io_png_filter"

external jpeg_probe : bytes -> int * int = "caml_nx_io_jpeg_probe"
external jpeg_decode : bytes -> bytes -> bool -> unit = "caml_nx_io_jpeg_decode"

external jpeg_encode : Unix.file_descr -> bytes -> int -> int -> int -> unit
  = "caml_nx_io_jpeg_encode"

(* PNG image data is a zlib stream of filtered scanlines, cut into IDAT
   chunks. *)

module Crc32 = Compress_deflate.Crc32

let be32 src at =
  (Array1.get src at lsl 24)
  lor (Array1.get src (at + 1) lsl 16)
  lor (Array1.get src (at + 2) lsl 8)
  lor Array1.get src (at + 3)

(* Checks the CRC-32 of each chunk before [png_parse] reads them. A chunk that
   runs past the end of [src] stops the check, and [png_parse] reports it. *)
let png_probe src =
  let n = Array1.dim src in
  let rec check off =
    if n - off >= 12 then begin
      let length = be32 src off in
      if length <= n - off - 12 then begin
        let crc = Crc32.bigbytes (Array1.sub src (off + 4) (length + 4)) in
        if crc <> be32 src (off + 8 + length) then
          failwith "PNG chunk CRC mismatch";
        check (off + 12 + length)
      end
    end
  in
  check 8;
  png_parse src

let png_decode src dst grayscale =
  let idat, length = png_idat src in
  let filtered = Array1.create int8_unsigned c_layout length in
  (match Compress_deflate.Zlib.decompress idat filtered with
  | Ok () -> ()
  | Error e -> failwith ("invalid PNG zlib stream: " ^ e));
  png_unfilter src filtered dst grayscale

(* [png_chunk w kind data length] writes the chunk [kind] of the first [length]
   bytes of [data] on [w]. *)
let png_chunk w kind data length =
  let open Bytesrw in
  let header = Bytes.create 8 in
  Bytes.set_int32_be header 0 (Int32.of_int length);
  Bytes.blit_string kind 0 header 4 4;
  let data = Bytes.Slice.make_or_eod data ~first:0 ~length in
  let crc = Crc32.slice (Bytes.Slice.make header ~first:4 ~length:4) in
  let crc = Crc32.slice ~crc data in
  Bytes.Writer.write w (Bytes.Slice.make header ~first:0 ~length:8);
  if not (Bytes.Slice.is_eod data) then Bytes.Writer.write w data;
  Bytes.set_int32_be header 0 (Int32.of_int crc);
  Bytes.Writer.write w (Bytes.Slice.make header ~first:0 ~length:4)

let idat_length = 1 lsl 20

(* [idat_writer w n] cuts the zlib stream written to it, of [n] bytes of data,
   into IDAT chunks of [idat_length] bytes, the last one shorter, on [w]. Such a
   stream is at most [n + 5 * (n / 65535 + 1) + 6] bytes long. *)
let idat_writer w n =
  let open Bytesrw in
  let capacity = Int.min idat_length (n + (5 * ((n / 65535) + 1)) + 6) in
  let chunk = Bytes.create capacity and length = ref 0 in
  let flush () =
    if !length > 0 then begin
      png_chunk w "IDAT" chunk !length;
      length := 0
    end
  in
  let rec write s =
    if Bytes.Slice.is_eod s then flush ()
    else begin
      let n = Int.min (Bytes.Slice.length s) (capacity - !length) in
      Bytes.blit (Bytes.Slice.bytes s) (Bytes.Slice.first s) chunk !length n;
      length := !length + n;
      if !length = capacity then flush ();
      if n < Bytes.Slice.length s then write (Bytes.Slice.drop_first_or_eod n s)
    end
  in
  Bytes.Writer.make write

(* Writes the PNG image [data] on [w], without ending [w]. A [ppm] other than 0
   is written as a pHYs chunk of [ppm] pixels per metre on both axes, and [srgb]
   as an sRGB chunk with the perceptual rendering intent. *)
let png_encode ~ppm ~srgb w data width height channels =
  let filtered = png_filter data width height channels in
  Bytesrw.Bytes.Writer.write_string w "\137PNG\r\n\026\n";
  let header = Bytes.make 13 '\000' in
  Bytes.set_int32_be header 0 (Int32.of_int width);
  Bytes.set_int32_be header 4 (Int32.of_int height);
  Bytes.set_uint8 header 8 8;
  Bytes.set_uint8 header 9 (match channels with 1 -> 0 | 3 -> 2 | _ -> 6);
  png_chunk w "IHDR" header 13;
  if srgb then png_chunk w "sRGB" (Bytes.make 1 '\000') 1;
  if ppm <> 0 then begin
    let phys = Bytes.make 9 '\001' in
    Bytes.set_int32_be phys 0 (Int32.of_int ppm);
    Bytes.set_int32_be phys 4 (Int32.of_int ppm);
    png_chunk w "pHYs" phys 9
  end;
  let idat = idat_writer w (String.length filtered) in
  let z = Compress_deflate.Zlib.compress_writes () ~eod:true idat in
  Bytesrw.Bytes.Writer.write_string z filtered;
  Bytesrw.Bytes.Writer.write_eod z;
  png_chunk w "IEND" Bytes.empty 0

let write_png fd data width height channels =
  png_encode ~ppm:0 ~srgb:false
    (Bytesrw_unix.bytes_writer_of_fd fd)
    data width height channels

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
  let dst = Array1.create int8_unsigned c_layout length in
  decode src dst grayscale;
  let shape =
    if grayscale then [| height; width |] else [| height; width; 3 |]
  in
  Nx.of_bigarray (reshape (genarray_of_array1 dst) shape)

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
      Temp_file.remove_if_exists path;
      raise exn

let save_png ~overwrite path data ~width ~height ~channels =
  ignore (checked_pixels width height channels);
  if not overwrite then
    encode_to_path ~encode:write_png ~exclusive:true path data ~width ~height
      ~channels
  else
    let temp = Temp_file.sibling path in
    match
      encode_to_path ~encode:write_png ~exclusive:false temp data ~width ~height
        ~channels
    with
    | () -> Temp_file.replace temp path
    | exception exn ->
        Temp_file.remove_if_exists temp;
        raise exn

let encode_png data ~width ~height ~channels ~ppm ~srgb =
  ignore (checked_pixels width height channels);
  let b = Buffer.create 1024 in
  png_encode ~ppm ~srgb
    (Bytesrw.Bytes.Writer.of_buffer b)
    data width height channels;
  Buffer.contents b

let save_jpeg ~overwrite path data ~width ~height ~channels =
  ignore (checked_pixels width height channels);
  if channels <> 1 && channels <> 3 then
    invalid_arg "JPEG output requires one or three channels";
  if not overwrite then
    encode_to_path ~encode:jpeg_encode ~exclusive:true path data ~width ~height
      ~channels
  else
    let temp = Temp_file.sibling path in
    match
      encode_to_path ~encode:jpeg_encode ~exclusive:false temp data ~width
        ~height ~channels
    with
    | () -> Temp_file.replace temp path
    | exception exn ->
        Temp_file.remove_if_exists temp;
        raise exn
