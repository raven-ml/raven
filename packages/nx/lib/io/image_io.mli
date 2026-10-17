(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PNG and JPEG codecs, the implementation of {!Nx_io.load_image},
    {!Nx_io.save_image} and {!Nx_io.encode_png}.

    Pixels are 8-bit samples in row-major order, interleaved by channel.
    The functions below check that the width, height and channel count
    are positive and that their product fits an [int], and raise
    [Failure] otherwise. {!save_jpeg} checks its channel count first. *)

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for pixel and file bytes. *)

val load_image :
  grayscale:bool -> string -> (int, Bigarray.int8_unsigned_elt) Nx.t
(** [load_image ~grayscale path] is {!Nx_io.load_image}[ ~grayscale path].
    It reads the file by its magic bytes, whatever its extension. *)

val save_png :
  overwrite:bool ->
  string ->
  bytes ->
  width:int ->
  height:int ->
  channels:int ->
  unit
(** [save_png ~overwrite path data ~width ~height ~channels] writes the
    PNG of [data] to [path] through {!Temp_file.write}. [channels] must be
    1, 3 or 4 and [data] must hold [width * height * channels] bytes. *)

val encode_png :
  bytes ->
  width:int ->
  height:int ->
  channels:int ->
  ppm:int ->
  srgb:bool ->
  string
(** [encode_png data ~width ~height ~channels ~ppm ~srgb] is the PNG of
    [data], as {!save_png} writes it. A [ppm] other than [0] adds a [pHYs]
    chunk of [ppm] pixels per metre on both axes, and [srgb] an [sRGB]
    chunk with the perceptual rendering intent. *)

val save_jpeg :
  overwrite:bool ->
  string ->
  bytes ->
  width:int ->
  height:int ->
  channels:int ->
  unit
(** [save_jpeg ~overwrite path data ~width ~height ~channels] writes the
    JPEG of [data] to [path] through {!Temp_file.write}. [data] must hold
    [width * height * channels] bytes. Raises [Invalid_argument] if
    [channels] is not 1 or 3. *)
