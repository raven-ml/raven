(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PNG and JPEG codecs over bytes. The C stubs check every span and size before
    they release the runtime lock, and touch only bigarray storage, copied
    scalars and their own allocations while it is released. *)

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val decode : grayscale:bool -> bytes -> bytes * int * int
(** [decode ~grayscale src] is the pixels of the PNG or JPEG stream [src], with
    its height and width: one channel with [grayscale], three otherwise, row by
    row from the top.

    Raises [Failure] if [src] is no such stream, or a malformed one. *)

val png_encode :
  ppm:int ->
  srgb:bool ->
  bytes ->
  width:int ->
  height:int ->
  channels:int ->
  string
(** [png_encode ~ppm ~srgb pixels ~width ~height ~channels] is a PNG file of
    [pixels], 8 bits per sample. A [ppm] other than [0] is a pHYs chunk of [ppm]
    pixels per metre, and [srgb] an sRGB chunk. [channels] is 1, 3 or 4 and the
    image has a pixel. *)

val jpeg_encode : Unix.file_descr -> bytes -> int -> int -> int -> unit
(** [jpeg_encode fd pixels width height channels] writes a baseline JPEG file of
    [pixels] to [fd], at quality 90. [channels] is 1 or 3 and the image has a
    pixel.

    Raises [Unix.Unix_error] if the write fails. *)
