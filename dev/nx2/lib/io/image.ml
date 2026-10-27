(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray

type bytes = (int, int8_unsigned_elt, c_layout) Array1.t

external png_parse : bytes -> int * int * int = "caml_nx_io_png_probe"
external png_idat : bytes -> bytes * int = "caml_nx_io_png_idat"

external png_unfilter : bytes -> bytes -> bytes -> unit
  = "caml_nx_io_png_decode"

external png_filter : bytes -> int -> int -> int -> string
  = "caml_nx_io_png_filter"

external jpeg_probe : bytes -> int * int * int = "caml_nx_io_jpeg_probe"
external jpeg_decode : bytes -> bytes -> unit = "caml_nx_io_jpeg_decode"

external jpeg_stream : bytes -> int -> int -> int -> string
  = "caml_nx_io_jpeg_encode"

module Crc32 = Compress_deflate.Crc32

(* PNG image data is a zlib stream of filtered scanlines, cut into IDAT
   chunks. *)

let be32 src at =
  (Array1.get src at lsl 24)
  lor (Array1.get src (at + 1) lsl 16)
  lor (Array1.get src (at + 2) lsl 8)
  lor Array1.get src (at + 3)

let png_max = 2147483647

(* The chunk type IEND, which ends the image, as a big-endian integer. *)
let iend = 0x49454e44

(* Checks the length and CRC-32 of each chunk up to IEND before [png_parse]
   reads them, then the sides [png_parse] reads. A chunk that runs past the end
   of [src] stops the check, and [png_parse] reports it. *)
let png_probe src =
  let n = Array1.dim src in
  let rec check off =
    if n - off >= 12 then begin
      let length = be32 src off in
      if length > png_max then failwith "invalid PNG chunk length";
      if length <= n - off - 12 then begin
        let crc = Crc32.bigbytes (Array1.sub src (off + 4) (length + 4)) in
        if crc <> be32 src (off + 8 + length) then
          failwith "PNG chunk CRC mismatch";
        if be32 src (off + 4) <> iend then check (off + 12 + length)
      end
    end
  in
  check 8;
  let width, height, channels = png_parse src in
  if width > png_max || height > png_max then
    failwith "invalid PNG image header";
  (width, height, channels)

let png_decode src dst =
  let idat, length = png_idat src in
  let filtered = Array1.create int8_unsigned c_layout length in
  (match Compress_deflate.Zlib.decompress_into idat filtered with
  | Ok () -> ()
  | Error e -> failwith ("invalid PNG zlib stream: " ^ e));
  png_unfilter src filtered dst

(* [png_chunk b kind data first length] adds the chunk [kind] of the [length]
   bytes of [data] at [first] to [b]. *)
let png_chunk b kind data first length =
  Buffer.add_int32_be b (Int32.of_int length);
  Buffer.add_string b kind;
  Buffer.add_substring b data first length;
  let crc = Crc32.string ~crc:(Crc32.string kind) ~first ~length data in
  Buffer.add_int32_be b (Int32.of_int crc)

let idat_length = 1 lsl 20

(* The zlib stream is cut into IDAT chunks of [idat_length] bytes, the last one
   shorter. A [ppm] other than 0 is written as a pHYs chunk of [ppm] pixels per
   metre on both axes, and [srgb] as an sRGB chunk with the perceptual rendering
   intent. *)
let png_encode ~ppm ~srgb data ~width ~height ~channels =
  let b = Buffer.create 1024 in
  let z =
    Compress_deflate.Zlib.compress (png_filter data width height channels)
  in
  Buffer.add_string b "\137PNG\r\n\026\n";
  let header = Bytes.make 13 '\000' in
  Bytes.set_int32_be header 0 (Int32.of_int width);
  Bytes.set_int32_be header 4 (Int32.of_int height);
  Bytes.set_uint8 header 8 8;
  (* The colour type of grey, grey and alpha, RGB and RGBA. *)
  let color = match channels with 1 -> 0 | 2 -> 4 | 3 -> 2 | _ -> 6 in
  Bytes.set_uint8 header 9 color;
  png_chunk b "IHDR" (Bytes.to_string header) 0 13;
  if srgb then png_chunk b "sRGB" "\000" 0 1;
  if ppm <> 0 then begin
    let phys = Bytes.make 9 '\001' in
    Bytes.set_int32_be phys 0 (Int32.of_int ppm);
    Bytes.set_int32_be phys 4 (Int32.of_int ppm);
    png_chunk b "pHYs" (Bytes.to_string phys) 0 9
  end;
  let rec idat first =
    if first < String.length z then begin
      let length = Int.min idat_length (String.length z - first) in
      png_chunk b "IDAT" z first length;
      idat (first + length)
    end
  in
  idat 0;
  png_chunk b "IEND" "" 0 0;
  Buffer.contents b

let is_png src =
  Array1.dim src >= 8
  && List.for_all2
       (fun i c -> Array1.unsafe_get src i = c)
       [ 0; 1; 2; 3; 4; 5; 6; 7 ]
       [ 0x89; 0x50; 0x4e; 0x47; 0x0d; 0x0a; 0x1a; 0x0a ]

let is_jpeg src =
  Array1.dim src >= 2
  && Array1.unsafe_get src 0 = 0xff
  && Array1.unsafe_get src 1 = 0xd8

let jpeg_encode data ~width ~height ~channels =
  jpeg_stream data width height channels

(* The probe and the decoder of the stream [src], recognised by its first
   bytes. *)
let codec src =
  if is_png src then (png_probe, png_decode)
  else if is_jpeg src then (jpeg_probe, jpeg_decode)
  else failwith "no PNG or JPEG stream"

let probe src =
  let probe, _ = codec src in
  let width, height, channels = probe src in
  if width <= 0 || height <= 0 then failwith "the image has no pixel";
  if width > max_int / height || width * height > max_int / channels then
    failwith "the image is too large";
  (height, width, channels)

let decode src dst =
  let _, decode = codec src in
  decode src dst
