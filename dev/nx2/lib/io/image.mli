(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PNG and JPEG codecs over bytes.

    The C stubs capture each bigarray's pointer and length before they release
    the runtime lock. While it is released they touch only bigarray storage,
    their locals and memory they allocated with [malloc]. *)

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val png_max : int
(** [png_max] is [2{^31} - 1], the largest PNG integer (PNG-3, 7.1). It bounds a
    chunk's length, an image's sides and the pHYs chunk's pixels per metre. *)

val probe : bytes -> int * int * int
(** [probe src] is the height, width and channels of the image of the PNG or
    JPEG stream [src], as {!Nx_io.load_image} states them.

    Raises [Failure] if [src] is neither a PNG nor a JPEG stream, if its header
    is malformed, or if the image has no pixel or more than [max_int] samples.
*)

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
  (string -> unit) ->
  unit
(** [png_encode ~ppm ~srgb pixels ~width ~height ~channels emit] calls [emit] on
    each piece of a PNG file of [pixels], 8 bits per sample, in order: the
    signature, then each chunk. A [ppm] other than [0] is a pHYs chunk of [ppm]
    pixels per metre, and [srgb] an sRGB chunk. [channels] must be 1 to 4, and
    each side must be 1 to {!png_max}. Beyond [pixels], it holds a row, the zlib
    encoder's state of about 1 MiB, a buffer of 1 MiB of image data and the
    chunk [emit] takes.

    Raises [Invalid_argument] if [pixels] does not hold
    [width * height * channels] bytes. *)

val jpeg_encode : bytes -> width:int -> height:int -> channels:int -> bytes
(** [jpeg_encode pixels ~width ~height ~channels] is a baseline JPEG file of
    [pixels], at quality 90, in memory of its own. Encoding grows a buffer from
    16 KiB by doubling, to at most twice the file's length or 16 KiB, whichever
    is larger, and shrinks it to the file.

    Raises [Failure] unless [channels] is 1 or 3, each side is 1 to 65535 and
    [pixels] holds [width * height * channels] bytes. *)
