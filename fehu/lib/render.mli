(** Typed render payloads shared across Fehu environments and utilities.

    Rendered frames can carry pixel buffers, text, or vector graphics.
    Environments should prefer returning {!Render.t} values to maximize
    interoperability with recorders and sinks. *)

module Pixel : sig
  type format =
    [ `RGB8  (** 8-bit RGB pixels stored as packed bytes *)
    | `RGBA8  (** 8-bit RGBA pixels *)
    | `GRAY8  (** 8-bit grayscale pixels *)
    | `RGBf  (** 32-bit float RGB pixels *)
    | `RGBAf  (** 32-bit float RGBA pixels *) ]
  (** Supported pixel formats for image buffers. *)
end

type image = {
  width : int;
  height : int;
  pixel_format : Pixel.format;
  data_u8 :
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    option;
  data_f32 :
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t option;
}
(** Image payload backed by contiguous buffers.

    Exactly one of [data_u8] or [data_f32] should be [Some] depending on the
    pixel format; the other must be [None]. *)

type text = string
(** Text payload for ANSI render modes. *)

type svg = string
(** Serialized SVG payload. *)

type frame =
  | Image of image
  | Text of text
  | Svg of svg
  | None  (** Render frame variant. *)

type t = frame
(** Alias for render frames. *)

val image_u8 :
  width:int ->
  height:int ->
  pixel_format:Pixel.format ->
  data:(char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  unit ->
  image
(** Helper to construct an 8-bit image payload. *)

val image_f32 :
  width:int ->
  height:int ->
  pixel_format:Pixel.format ->
  data:(float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  unit ->
  image
(** Helper to construct a float32 image payload. *)
