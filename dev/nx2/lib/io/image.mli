(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PNG and JPEG codecs over bytes. The C stubs check every span and size before
    they release the runtime lock, and touch only bigarray storage, copied
    scalars and their own allocations while it is released. *)

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val probe : bytes -> int * int * int
(** [probe src] is the height, width and channels of the image of the PNG or
    JPEG stream [src], as {!Nx_io.load_image} states them.

    Raises [Failure] if [src] is no such stream, if its header is malformed, or
    if the image has no pixel or more than [max_int] samples. *)

val decode : bytes -> bytes -> unit
(** [decode src dst] writes the pixels of [src] into [dst], row by row from the
    top, with the channels {!probe} gives.

    Raises [Failure] if [src] is malformed, or if its pixels do not fill [dst]
    exactly. *)

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
    pixels per metre, and [srgb] an sRGB chunk. [channels] is 1 to 4 and the
    image has a pixel. *)

val jpeg_encode : bytes -> width:int -> height:int -> channels:int -> string
(** [jpeg_encode pixels ~width ~height ~channels] is a baseline JPEG file of
    [pixels], at quality 90. [channels] is 1 or 3 and the image has a pixel. *)
