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

(* [png_filter_row pixels row y channels] writes row [y] of [pixels], filtered,
   into [row], whose length is one more than the row's. [y] must be a row of
   [pixels]: the stub reads it unchecked. *)
external png_filter_row : bytes -> bytes -> int -> int -> unit
  = "caml_nx_io_png_filter_row"
[@@noalloc]

external jpeg_probe : bytes -> int * int * int = "caml_nx_io_jpeg_probe"
external jpeg_decode : bytes -> bytes -> unit = "caml_nx_io_jpeg_decode"

external jpeg_encode : bytes -> width:int -> height:int -> channels:int -> bytes
  = "caml_nx_io_jpeg_encode"

module Crc32 = Lz.Crc32

(* PNG image data is a zlib stream of filtered scanlines, cut into IDAT
   chunks. *)

let png_signature = "\137PNG\r\n\026\n"

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
  (* Bytes after the zlib stream are the container's, and are ignored. *)
  (match Lz.Zlib.decode idat filtered with
  | Ok { Lz.produced; _ } when produced = length -> ()
  | Ok { Lz.produced; _ } ->
      Fail.failf
        "invalid PNG zlib stream: %d bytes of scanlines, the image has %d"
        produced length
  | Error e ->
      Fail.failf "invalid PNG zlib stream: %s"
        (Format.asprintf "%a" Lz.pp_error e));
  png_unfilter src filtered dst

(* [png_chunk kind data length] is the chunk [kind] of the first [length] bytes
   of [data]: its length, type, data and CRC-32. *)
let png_chunk kind data length =
  let b = Bytes.create (length + 12) in
  Bytes.set_int32_be b 0 (Int32.of_int length);
  Bytes.blit_string kind 0 b 4 4;
  Bytes.blit_string data 0 b 8 length;
  let crc = Crc32.string ~crc:(Crc32.string kind) ~length data in
  Bytes.set_int32_be b (length + 8) (Int32.of_int crc);
  Bytes.unsafe_to_string b

let idat_length = 1 lsl 20

(* Each row is filtered into [row], then given to the zlib encoder, whose runs
   fill [idat]. A full [idat] is emitted as an IDAT chunk, and what remains at
   the end as a last, shorter one. The bytes do not depend on how the encoder's
   input is sliced, so they are the whole stream's cut every [idat_length]
   bytes. A [ppm] other than 0 is written as a pHYs chunk of [ppm] pixels per
   metre on both axes, and [srgb] as an sRGB chunk with the perceptual rendering
   intent. *)
let png_encode ~ppm ~srgb data ~width ~height ~channels emit =
  if Array1.dim data <> width * height * channels then
    invalid_arg "Image.png_encode: the pixels do not fill the image";
  emit png_signature;
  let header = Bytes.make 13 '\000' in
  Bytes.set_int32_be header 0 (Int32.of_int width);
  Bytes.set_int32_be header 4 (Int32.of_int height);
  Bytes.set_uint8 header 8 8;
  (* The colour type of grey, grey and alpha, RGB and RGBA. *)
  let color = match channels with 1 -> 0 | 2 -> 4 | 3 -> 2 | _ -> 6 in
  Bytes.set_uint8 header 9 color;
  emit (png_chunk "IHDR" (Bytes.unsafe_to_string header) 13);
  if srgb then emit (png_chunk "sRGB" "\000" 1);
  if ppm <> 0 then begin
    let phys = Bytes.make 9 '\001' in
    Bytes.set_int32_be phys 0 (Int32.of_int ppm);
    Bytes.set_int32_be phys 4 (Int32.of_int ppm);
    emit (png_chunk "pHYs" (Bytes.unsafe_to_string phys) 9)
  end;
  let idat = Bytes.create idat_length in
  let fill = ref 0 in
  (* [png_chunk] reads [idat] as a string and copies it, keeping nothing. *)
  let flush () =
    emit (png_chunk "IDAT" (Bytes.unsafe_to_string idat) !fill);
    fill := 0
  in
  (* Copies [run] from [first] to [idat], emitting each IDAT chunk it fills. *)
  let rec copy run first =
    let n = Int.min (Array1.dim run - first) (idat_length - !fill) in
    for i = 0 to n - 1 do
      Bytes.unsafe_set idat (!fill + i)
        (Char.unsafe_chr (Array1.unsafe_get run (first + i)))
    done;
    fill := !fill + n;
    if !fill = idat_length then flush ();
    if first + n < Array1.dim run then copy run (first + n)
  in
  let add run = copy run 0 in
  let encoder = Lz.Zlib.encoder () in
  let rec give ~last src pos =
    let n = Array1.dim src - pos in
    match Lz.Encoder.step encoder ~last (Array1.sub src pos n) add with
    | Lz.More k when k < n -> give ~last src (pos + k)
    | Lz.More _ | Lz.Done _ -> ()
  in
  let row = Array1.create int8_unsigned c_layout ((width * channels) + 1) in
  for y = 0 to height - 1 do
    png_filter_row data row y channels;
    give ~last:false row 0
  done;
  give ~last:true (Array1.sub row 0 0) 0;
  if !fill > 0 then flush ();
  emit (png_chunk "IEND" "" 0)

let is_png src =
  let n = String.length png_signature in
  Array1.dim src >= n
  && String.init n (fun i -> Char.chr (Array1.unsafe_get src i)) = png_signature

let is_jpeg src =
  Array1.dim src >= 2
  && Array1.unsafe_get src 0 = 0xff
  && Array1.unsafe_get src 1 = 0xd8

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
