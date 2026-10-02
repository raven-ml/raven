(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Figures: marks and their composition, as the user describes them. *)

module Text := Hugin_next_text.Text
module Picture := Hugin_next_vg.Picture
module Scale := Hugin_next_kit.Scale

(** {1:marks Marks} *)

type rows = Rows.t
type reducer = M4 | Cells | Raster

type binding =
  | B : {
      role : ('d, 'r) Role.t;
      ch : ('d, 'r) Channel.t;
      imply : 'd Scale.t option;
      guide : bool option;
    }
      -> binding

type mark = {
  kind : string;
  reduce : reducer option;
  coord : Coord.t option;
  swatch : (rows -> Picture.t) option;
  bindings : binding list;
  draw : rows -> Picture.t;
  shape : int array;  (** The shape the channels broadcast to. *)
}

val equal_mark : mark -> mark -> bool
val find_binding : ('d, 'r) Role.t -> binding list -> binding option

val mark_shape : string -> ?shape:int array -> binding list -> int array
(** [mark_shape fn ~shape bindings] is the shape of a mark of [bindings]: that
    of [shape] and the channels broadcast together. Raises [Invalid_argument],
    naming [fn], if they do not, if [shape] has a negative dimension, or if a
    [dim] or an [index] does not fit it. *)

val make_mark :
  string ->
  name:string ->
  ?reduce:reducer ->
  ?coord:Coord.t ->
  ?shape:int array ->
  ?swatch:(rows -> Picture.t) ->
  binding list ->
  (rows -> Picture.t) ->
  mark
(** [make_mark fn ~name ~reduce ~coord ~shape ~swatch bindings draw] is the mark
    [name] of [bindings], whose shape is that of [shape] and the channels
    broadcast together. Raises [Invalid_argument], naming [fn], as [Mark.v]
    does. *)

(** {1:figures Figures} *)

type sharing = [ `Shared | `Independent ]
type side = [ `Left | `Right | `Top | `Bottom ]
type guide_kind = Axis of { grid : bool } | Legend

type guide = {
  kind : guide_kind;
  scale : string;
  side : side option;
  show : bool;
}

type t =
  | Mark of mark
  | Layer of t list
  | Grid of {
      rows : t list list;
      widths : float list option;
      heights : float list option;
    }
  | Span of { rows : int; cols : int; f : t }
  | Share of (string * sharing) list * t
  | Title of { align : Text.Layout.halign; text : Text.t; f : t }
  | Coord_sys of Coord.t * t
  | Name of string * t
  | Bind : 'a View.key * ('a -> t) -> t
  | Guide of guide

val equal : t -> t -> bool
val equal_side : side -> side -> bool
val pp_side : Format.formatter -> side -> unit
val equal_guide : guide -> guide -> bool
val pp_guide : Format.formatter -> guide -> unit
val is_axis : guide -> bool
val equal_halign : Text.Layout.halign -> Text.Layout.halign -> bool

(** {1:composing Composing} *)

val layer : t list -> t
val grid : ?widths:float list -> ?heights:float list -> t list list -> t
val span : ?rows:int -> ?cols:int -> t -> t
val share : (string * sharing) list -> t -> t
val title : ?align:Text.Layout.halign -> Text.t -> t -> t
val coord : Coord.t -> t -> t
val name : string -> t -> t
val bind : 'a View.key -> ('a -> t) -> t
val axis : ?side:side -> ?grid:bool -> ?show:bool -> string -> t
val legend : ?side:side -> ?show:bool -> string -> t
