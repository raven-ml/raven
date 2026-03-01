(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Handle types *)

type t
type surface

(* Enums *)

type font_weight = Normal | Bold
type line_cap = Butt | Round | Square
type line_join = Join_miter | Join_round | Join_bevel

type antialias =
  | Antialias_default
  | Antialias_none
  | Antialias_gray
  | Antialias_subpixel

(* Text extents *)

type text_extents = {
  x_bearing : float;
  y_bearing : float;
  width : float;
  height : float;
  x_advance : float;
  y_advance : float;
}

(* Context *)

external create : surface -> t = "caml_ucairo_create"

(* State *)

external save : t -> unit = "caml_ucairo_save"
external restore : t -> unit = "caml_ucairo_restore"

(* Transformations *)

external translate : t -> float -> float -> unit = "caml_ucairo_translate"
external scale : t -> float -> float -> unit = "caml_ucairo_scale"
external rotate : t -> float -> unit = "caml_ucairo_rotate"

(* Source *)

external set_source_rgba : t -> float -> float -> float -> float -> unit
  = "caml_ucairo_set_source_rgba"

external set_source_surface : t -> surface -> x:float -> y:float -> unit
  = "caml_ucairo_set_source_surface"

(* Stroke/fill parameters *)

external set_line_width : t -> float -> unit = "caml_ucairo_set_line_width"
external raw_set_line_cap : t -> int -> unit = "caml_ucairo_set_line_cap"

let set_line_cap t cap =
  raw_set_line_cap t (match cap with Butt -> 0 | Round -> 1 | Square -> 2)

external raw_set_line_join : t -> int -> unit = "caml_ucairo_set_line_join"

let set_line_join t join =
  raw_set_line_join t
    (match join with Join_miter -> 0 | Join_round -> 1 | Join_bevel -> 2)

external set_dash : t -> float array -> unit = "caml_ucairo_set_dash"
external raw_set_antialias : t -> int -> unit = "caml_ucairo_set_antialias"

let set_antialias t aa =
  raw_set_antialias t
    (match aa with
    | Antialias_default -> 0
    | Antialias_none -> 1
    | Antialias_gray -> 2
    | Antialias_subpixel -> 3)

(* Font *)

external raw_select_font_face : t -> string -> int -> unit
  = "caml_ucairo_select_font_face"

let select_font_face t family weight =
  raw_select_font_face t family (match weight with Normal -> 0 | Bold -> 1)

external set_font_size : t -> float -> unit = "caml_ucairo_set_font_size"
external text_extents : t -> string -> text_extents = "caml_ucairo_text_extents"
external show_text : t -> string -> unit = "caml_ucairo_show_text"

(* Path *)

external move_to : t -> float -> float -> unit = "caml_ucairo_move_to"
external line_to : t -> float -> float -> unit = "caml_ucairo_line_to"

external arc : t -> float -> float -> r:float -> a1:float -> a2:float -> unit
  = "caml_ucairo_arc_bytecode" "caml_ucairo_arc_native"

external rectangle : t -> float -> float -> w:float -> h:float -> unit
  = "caml_ucairo_rectangle"

module Path = struct
  external close : t -> unit = "caml_ucairo_path_close"
  external clear : t -> unit = "caml_ucairo_path_clear"
end

(* Drawing *)

external fill : t -> unit = "caml_ucairo_fill"
external fill_preserve : t -> unit = "caml_ucairo_fill_preserve"
external stroke : t -> unit = "caml_ucairo_stroke"
external paint : t -> unit = "caml_ucairo_paint"
external clip : t -> unit = "caml_ucairo_clip"

(* Surface *)

module Surface = struct
  external finish : surface -> unit = "caml_ucairo_surface_finish"
  external flush : surface -> unit = "caml_ucairo_surface_flush"
end

(* Image *)

module Image = struct
  external create : w:int -> h:int -> surface = "caml_ucairo_image_create"

  external create_for_data8 :
    (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    w:int ->
    h:int ->
    stride:int ->
    surface = "caml_ucairo_image_create_for_data8"

  external stride_for_width : int -> int = "caml_ucairo_image_stride_for_width"
  [@@noalloc]
end

(* PDF *)

module Pdf = struct
  external create : string -> w:float -> h:float -> surface
    = "caml_ucairo_pdf_create"
end

(* PNG *)

module Png = struct
  external write : surface -> string -> unit = "caml_ucairo_png_write"

  external write_to_stream : surface -> (string -> unit) -> unit
    = "caml_ucairo_png_write_to_stream"
end
