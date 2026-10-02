(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The rows of a mark in one panel, as its draw function sees them. *)

module P2 := Hugin_next_gg.P2
module Path := Hugin_next_gg.Path
module Color := Hugin_next_gg.Color
module Text := Hugin_next_text.Text
module Picture := Hugin_next_vg.Picture
module Scale := Hugin_next_kit.Scale

(** {1:columns Columns} *)

(** The values of one binding, one per row. *)
type col =
  | Col : {
      role : ('d, 'r) Role.t;
      values : 'r array;  (** In the range, missing values as {!Mark.get}. *)
      norm : float array option;  (** Normalised, if it reads a scale. *)
      fn : (float -> 'r) option;  (** The range, if it reads a scale. *)
      ticks : float array option;  (** The frozen ticks of its scale. *)
      cats : int array option;
          (** On a band scale, each row's index in its domain, [-1] where
              missing. *)
      band : float option;  (** On a band scale, its bandwidth. *)
      zero : float option;
          (** On a continuous scale, the normalised value of [0.] clamped into
              its domain. *)
    }
      -> col

(** {1:rows Rows} *)

(** A fitted scale of any kind. *)
type fitted = Fitted : 'd Scale.t -> fitted

type t = {
  id : Common.id;
  shape : int array;
  index : int array;
  theme : Theme.t;
  projection : Coord.projection;
  axes : fitted option * fitted option;
      (** The fitted scales that the panel's channels on x and y read. *)
  cols : col list;
  dropped : bool array;
  warn : string -> unit;
}

val select : t -> int array -> t
(** [select r ks] is the rows [ks] of [r], in that order. *)

(** {1:observing Observing} *)

val length : t -> int
val get : t -> ('d, 'r) Role.t -> 'r array option
val normalized : t -> ('d, 'r) Role.t -> float array option
val range : t -> ('d, 'r) Role.t -> (float -> 'r) option
val ticks : t -> ('d, 'r) Role.t -> float array option
val scale : t -> [ `X | `Y ] -> 'd Scale.kind -> 'd Scale.t option
val positions : t -> float array * float array
val points : t -> float array * float array
val extent : t -> [ `X | `Y ] -> float array * float array
val project : t -> Path.t -> Path.t
val series : t -> t list

val glyphs : Color.t -> P2.t -> Text.Layout.t -> Picture.t
(** [glyphs c at l] is the glyph runs of [l] with its anchor at [at], in [c]
    where its text sets no colour. *)

val text :
  ?halign:Text.Layout.halign ->
  ?valign:Text.Layout.valign ->
  t ->
  Color.t ->
  P2.t ->
  Text.t ->
  Picture.t
