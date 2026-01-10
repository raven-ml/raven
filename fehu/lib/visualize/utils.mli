(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Internal helpers for render conversions and drawing. *)

type rgb8 =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val convert_to_rgb8 : Fehu.Render.image -> Fehu.Render.image * rgb8
(** Convert an image to RGB8 format with a fresh mutable buffer. *)

val rgb24_bytes_of_image : Fehu.Render.image -> Bytes.t
(** Convert an image to packed RGB24 bytes (row-major). *)

val compose_grid :
  rows:int -> cols:int -> Fehu.Render.image array -> Fehu.Render.image
(** Compose images into a grid with row-major indexing. *)

val blit_rgb8 :
  src:rgb8 ->
  src_width:int ->
  src_height:int ->
  dst:rgb8 ->
  dst_width:int ->
  x:int ->
  y:int ->
  unit
(** Copy an RGB8 image into a destination buffer at [(x, y)]. *)

val set_pixel_rgb8 :
  buffer:rgb8 -> width:int -> x:int -> y:int -> r:int -> g:int -> b:int -> unit
(** Set a single pixel in an RGB8 buffer. *)

val fill_rect_rgb8 :
  buffer:rgb8 ->
  width:int ->
  x:int ->
  y:int ->
  w:int ->
  h:int ->
  r:int ->
  g:int ->
  b:int ->
  unit
(** Fill a rectangle in an RGB8 buffer. *)
