(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Minimal Cairo bindings.

    Thin bindings covering image and PDF surface creation, path drawing, text
    rendering, and PNG output. Designed for the Hugin rendering backend; not a
    general-purpose Cairo binding.

    All functions raise [Failure] on Cairo errors and [Invalid_argument] on
    destroyed handles. *)

(** {1:types Handle types} *)

type t
(** The type for Cairo drawing contexts. *)

type surface
(** The type for Cairo surfaces. *)

(** {1:enums Enumerations} *)

type font_weight = Normal | Bold  (** The type for font weight. *)
type line_cap = Butt | Round | Square  (** The type for line cap style. *)

type line_join =
  | Join_miter
  | Join_round
  | Join_bevel  (** The type for line join style. *)

type antialias =
  | Antialias_default
  | Antialias_none
  | Antialias_gray
  | Antialias_subpixel  (** The type for antialiasing mode. *)

(** {1:text_extents Text extents} *)

type text_extents = {
  x_bearing : float;
  y_bearing : float;
  width : float;
  height : float;
  x_advance : float;
  y_advance : float;
}
(** The type for text extent measurements. *)

(** {1:context Context creation} *)

val create : surface -> t
(** [create surface] is a new drawing context targeting [surface]. *)

(** {1:state State} *)

val save : t -> unit
(** [save t] pushes the current graphics state onto the stack. *)

val restore : t -> unit
(** [restore t] pops the graphics state from the stack. *)

(** {1:transform Transformations} *)

val translate : t -> float -> float -> unit
(** [translate t tx ty] translates the user-space origin by [(tx, ty)]. *)

val scale : t -> float -> float -> unit
(** [scale t sx sy] scales the user-space axes by [(sx, sy)]. *)

val rotate : t -> float -> unit
(** [rotate t angle] rotates the user-space axes by [angle] radians. *)

(** {1:source Source} *)

val set_source_rgba : t -> float -> float -> float -> float -> unit
(** [set_source_rgba t r g b a] sets the source to the given RGBA color. *)

val set_source_surface : t -> surface -> x:float -> y:float -> unit
(** [set_source_surface t s ~x ~y] sets [s] as the source, offset by [(x, y)].
*)

(** {1:stroke_fill Stroke and fill parameters} *)

val set_line_width : t -> float -> unit
(** [set_line_width t w] sets the stroke line width. *)

val set_line_cap : t -> line_cap -> unit
(** [set_line_cap t cap] sets the line cap style. *)

val set_line_join : t -> line_join -> unit
(** [set_line_join t join] sets the line join style. *)

val set_dash : t -> float array -> unit
(** [set_dash t dashes] sets the dash pattern. An empty array disables dashing.
*)

val set_antialias : t -> antialias -> unit
(** [set_antialias t aa] sets the antialiasing mode. *)

(** {1:font Font} *)

val select_font_face : t -> string -> font_weight -> unit
(** [select_font_face t family weight] selects a toy font face. Slant is always
    upright. *)

val set_font_size : t -> float -> unit
(** [set_font_size t size] sets the font size in user-space units. *)

val text_extents : t -> string -> text_extents
(** [text_extents t s] is the extents of [s] with the current font. *)

val show_text : t -> string -> unit
(** [show_text t s] renders [s] at the current point. *)

(** {1:path Path operations} *)

val move_to : t -> float -> float -> unit
(** [move_to t x y] begins a new sub-path at [(x, y)]. *)

val line_to : t -> float -> float -> unit
(** [line_to t x y] adds a line segment to [(x, y)]. *)

val arc : t -> float -> float -> r:float -> a1:float -> a2:float -> unit
(** [arc t xc yc ~r ~a1 ~a2] adds a circular arc centered at [(xc, yc)] with
    radius [r] from angle [a1] to [a2] (in radians). *)

val rectangle : t -> float -> float -> w:float -> h:float -> unit
(** [rectangle t x y ~w ~h] adds a closed rectangle sub-path. *)

(** {1:path_mod Path module} *)

module Path : sig
  val close : t -> unit
  (** [close t] closes the current sub-path with a line to its start. *)

  val clear : t -> unit
  (** [clear t] clears the current path. *)
end

(** {1:drawing Drawing operations} *)

val fill : t -> unit
(** [fill t] fills the current path and clears it. *)

val fill_preserve : t -> unit
(** [fill_preserve t] fills the current path without clearing it. *)

val stroke : t -> unit
(** [stroke t] strokes the current path and clears it. *)

val paint : t -> unit
(** [paint t] paints the current source everywhere within the current clip. *)

val clip : t -> unit
(** [clip t] establishes a new clip region by intersecting the current clip with
    the current path, then clears the path. *)

(** {1:surface Surface operations} *)

module Surface : sig
  val finish : surface -> unit
  (** [finish s] finalizes the surface and releases external resources. *)

  val flush : surface -> unit
  (** [flush s] completes any pending drawing operations. *)
end

(** {1:image Image surface} *)

module Image : sig
  val create : w:int -> h:int -> surface
  (** [create ~w ~h] is a new ARGB32 image surface of dimensions [w] x [h].

      Raises [Failure] if allocation fails. *)

  val create_for_data8 :
    (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    w:int ->
    h:int ->
    stride:int ->
    surface
  (** [create_for_data8 data ~w ~h ~stride] wraps existing pixel [data] as an
      ARGB32 image surface. [data] must remain live for the lifetime of the
      surface. *)

  val stride_for_width : int -> int
  (** [stride_for_width w] is the minimum stride in bytes for an ARGB32 image of
      width [w], respecting Cairo alignment requirements. *)
end

(** {1:pdf PDF surface} *)

module Pdf : sig
  val create : string -> w:float -> h:float -> surface
  (** [create filename ~w ~h] is a new PDF surface writing to [filename].
      Dimensions are in points (1 point = 1/72 inch). *)
end

(** {1:png PNG output} *)

module Png : sig
  val write : surface -> string -> unit
  (** [write surface filename] writes [surface] as a PNG file. *)

  val write_to_stream : surface -> (string -> unit) -> unit
  (** [write_to_stream surface f] writes [surface] as PNG data, calling [f] with
      each chunk. *)
end
