(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Guides: axes, facet headers, legends and titles, each laid out along one
    side of a node of the layout, in the frame of that side.

    The frame of a side has the page's orientation and its origin at the side's
    start: the top-left corner of the node's data hull for [`Left] and [`Top],
    the top-right for [`Right], and the bottom-left for [`Bottom]. A guide's
    depth is how far it reaches beyond its side; its reach past an end of the
    side is a protrusion on the adjacent side. *)

module P2 := Hugin_next_gg.P2
module Box2 := Hugin_next_gg.Box2
module Text := Hugin_next_text.Text
module Ticks := Hugin_next_kit.Ticks

val gap_em : float
(** [gap_em] is the gap, in em, between grid cells and between a legend and what
    it stands beside. *)

type sides = { left : float; right : float; top : float; bottom : float }
(** Lengths beyond each side of a box, in points. *)

val no_sides : sides
(** [no_sides] is zero on every side. *)

val horizontal : Figure.side -> bool
(** [horizontal s] is [true] iff [s] is [`Top] or [`Bottom]. *)

(** {1:elements Elements} *)

type placed = { text : Text.t; set : Text.Layout.t; at : P2.t; data : bool }
(** A text set with its anchor at [at]. [data] tells category labels, drawn
    whatever glyphs they lack, from figure text, which must have every glyph. *)

val text_box : placed -> Box2.t
(** [text_box p] is the box of [p]. *)

type element =
  | Text of placed  (** A title or header, in ink. *)
  | Label of placed  (** A tick or legend label. *)
  | Rules of (P2.t * P2.t) list  (** Axis lines and ticks. *)
  | Grid_lines of (P2.t * P2.t) list  (** Drawn under the marks. *)
  | Bar of { box : Box2.t; scale : int; vertical : bool }
      (** A colour bar of the scale [scale], which runs up if [vertical]. *)
  | Swatch of {
      box : Box2.t;
      scale : int;
      entry : int;
      entries : int;
      u : float;  (** The entry's normalised value. *)
    }

(** {1:specs Guide specifications} *)

type axis_part =
  | Ticks of { labelled : bool }
      (** The axis proper; unlabelled where the next panel on its side labels
          it. *)
  | Header of int  (** A facet header, its category's index. *)
  | Scale_title of Figure.side
      (** The title of the axes or headers on that side of the panels it serves.
      *)

type kind =
  | Axis of { guide : Figure.guide; scale : int; part : axis_part }
      (** A part of the axis of the position or facet scale [scale], its index
          in the resolved figure's scales. *)
  | Legend of { guide : Figure.guide; scale : int }
  | Title of { align : Text.Layout.halign; text : Text.t }

type tier = Proper | Headers | Scale_titles | Legends | Figure_titles

val tier : kind -> tier
(** [tier k] is the tier of a guide of kind [k]: its rank outward from the side,
    the axis proper innermost. *)

type spec = { id : Common.id; side : Figure.side; kind : kind }

(** {1:laid Laid-out guides} *)

type t = {
  spec : spec;
  length : float;  (** Of the side it was laid along. *)
  bounds : Box2.t option;  (** Of every element it may draw, in its frame. *)
  least : float;  (** The length its side needs. *)
  wrap : float;  (** The length a horizontal legend's rows wrapped at. *)
  elements : element list;
}

type cx
(** Measurements and the scales and ticks guides show. *)

val cx : ?reused:cx -> Theme.t -> Resolved.fitted array -> cx
(** [cx ~reused theme scales] measures text in [theme]. The measurements of
    [reused] are taken over if its theme is equal. *)

val with_ticks : Ticks.t array -> cx -> cx
(** [with_ticks ts cx] is [cx] with the ticks of each scale. *)

val choose : cx -> (spec * float) list -> Ticks.t array
(** [choose cx gs] is the ticks of each scale, chosen once against every guide
    of [gs] showing it, at its length, so that labels overlap on none: every
    category for a facet scale and a categorical legend. *)

val lay :
  cx ->
  spec ->
  length:float ->
  across:float ->
  wrap:float ->
  span:float * float ->
  t
(** [lay cx g ~length ~across ~wrap ~span] is [g] laid out along a side [length]
    long of a data hull [across] deep, with the ticks of [cx]. A horizontal
    legend sets as many entries per row as fit in [wrap]; a title aligns to
    [span], an interval along the side's line. A hidden guide draws nothing. *)

val protrusion : t -> sides
(** [protrusion g] is the depth of [g] on its side and its reach past the ends
    on the adjacent sides, from its bounds. *)

val move : P2.t -> t -> t
(** [move p g] is [g] with its frame's origin at [p]. *)

val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
