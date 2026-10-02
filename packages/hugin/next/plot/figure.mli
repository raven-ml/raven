(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Figures: marks and their composition, as the user describes them. *)

module Text := Hugin_next_text.Text
module Picture := Hugin_next_vg.Picture
module Scale := Hugin_next_kit.Scale

(** {1:marks Marks} *)

(** Rows are inhabited once figures are drawn: until then no draw function is
    called. *)
type rows = |

type reducer = M4 | Cells | Raster

type binding =
  | B : {
      role : ('d, 'r) Role.t;
      ch : ('d, 'r) Channel.t;
      imply : float Scale.t option;
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
val find_binding : string -> binding list -> binding option

val make_mark :
  string ->
  name:string ->
  ?reduce:reducer ->
  ?coord:Coord.t ->
  ?swatch:(rows -> Picture.t) ->
  ?base:int array ->
  binding list ->
  (rows -> Picture.t) ->
  mark
(** [make_mark fn ~name ~reduce ~coord ~swatch ~base bindings draw] is the mark
    [name] of [bindings], whose shape is that of [base] and the channels
    broadcast together. Raises [Invalid_argument], naming [fn], as [Mark.v]
    does. *)

(** {1:figures Figures} *)

type sharing = [ `Shared | `Independent ]
type side = [ `Left | `Right | `Top | `Bottom ]

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
  | Axis of { side : side option; grid : bool; show : bool; scale : string }
  | Legend of { side : side option; show : bool; scale : string }

val equal : t -> t -> bool
val equal_side : side -> side -> bool
val pp_side : Format.formatter -> side -> unit
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
