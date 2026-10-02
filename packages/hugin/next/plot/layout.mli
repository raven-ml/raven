(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Laid-out figures: panels, guides and titles placed on the page. *)

module P2 := Hugin_next_gg.P2
module Box2 := Hugin_next_gg.Box2
module Text := Hugin_next_text.Text
module Ticks := Hugin_next_kit.Ticks

(** {1:lengths Derived lengths, in em} *)

val tick_em : float
val swatch_em : float
val pad_em : float

(** {1:laid_out Laid-out figures} *)

type panel = { id : Common.id; box : Box2.t; projection : Coord.projection }

type placed = {
  text : Text.t;
  set : Text.Layout.t;
  at : P2.t;
  turned : bool;
  data : bool;
}
(** A text set at a point of the page, upright or turned a quarter turn
    counterclockwise. *)

type axis_out = {
  ax_id : Common.id;
  ax_panel : Common.id;
  ax_scale : int;  (** Its index in the resolved figure's scales. *)
  ax_side : Figure.side;
  ax_offset : float;  (** Its distance from its panel's side. *)
  ax_grid : bool;
  ax_labels : placed list;  (** Those drawn. *)
  ax_title : placed option;
}

type header_out = { hd_id : Common.id; hd_panel : Common.id; hd_label : placed }
type legend_entry = { u : float; swatch : Box2.t; label : placed }

type legend_body =
  | Bar of { bar : Box2.t; labels : placed list }
  | Entries of legend_entry list

type legend_out = {
  lg_id : Common.id;
  lg_scale : int;
  lg_side : Figure.side;
  lg_title : placed option;
  lg_body : legend_body;
}

type t

val layout : ?prev:t -> ?theme:Theme.t -> Size.t -> Resolved.t -> t
val size : t -> float * float
val panels : t -> panel list
val warnings : t -> Common.warning list

(** {1:parts Parts} *)

val resolved : t -> Resolved.t
val theme : t -> Theme.t
val coords : t -> (panel * Coord.t) list
val frozen : t -> Ticks.t array
val axes : t -> axis_out list
val headers : t -> header_out list
val legends : t -> legend_out list
val titles : t -> placed list
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
