(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Layout items: the tree of panels, grids, headings and legends that layout
    sizes and places, built from a resolved figure. *)

module Text := Hugin_next_text.Text
module Ticks := Hugin_next_kit.Ticks

(** {1:items Items} *)

type axis_spec = {
  a_id : Common.id;
  a_scale : int;  (** Its index in the resolved figure's scales. *)
  a_on : [ `Axis of Role.axis | `Header of Role.axis ];
  a_side : Figure.side;
  a_guide : Figure.guide;  (** Explicit, or else the default. *)
  a_labelled : bool;  (** False where the next panel on its side labels it. *)
  a_category : string option;  (** The panel's category, for a header. *)
}

type leaf = {
  l_id : Common.id;
  l_coord : Coord.t;
  l_ratio : float option;  (** The height of its data area over its width. *)
  l_axes : axis_spec list;
}

type legend_spec = {
  ls_id : Common.id;
  ls_scale : int;
  ls_side : Figure.side;
  ls_bar : bool;
}

type track = Flex of float | Fixed

type item =
  | Leaf of leaf
  | Grid of grid
  | Heading of {
      owner : Common.id;  (** The titled node, or a facet scale's axis. *)
      align : Text.Layout.halign;
      head : Text.t;
      hside : Figure.side;
    }
  | Legend of legend_spec

and grid = {
  gid : Common.id;
  gcols : track array;
  grows : track array;
  gcells : gcell list;
  gbody : int option;  (** The cell headings align on. *)
}

and gcell = { r0 : int; c0 : int; nr : int; nc : int; it : item }

(** The guides that show a scale's ticks. *)
type use =
  | Axis_of of Role.axis
  | Header_of
  | Legend_of of {
      bar : bool;
      side : Figure.side;
      block : Common.id;
      show : bool;
    }

(** {1:texts Guide texts} *)

val tick_text : Ticks.tick -> Text.t
val categorical : Resolved.fitted -> bool

val guide_title : Resolved.fitted -> string option -> Text.t option
(** [guide_title s note] is the distinct titles of the channels reading [s] in
    the order of the figure, separated by commas, then [note]. *)

val category_text : Resolved.fitted -> string -> Text.t
(** [category_text s name] is the text that shows the category [name] of [s]. *)

(** {1:building Building items} *)

val blocks : Resolved.t -> (Common.id * Common.id list) list
(** [blocks r] is each node of [r] that layout places as a whole, with the
    panels it holds, outer nodes first. *)

val uses_of : Resolved.t -> (Common.id * Common.id list) list -> use list array
(** [uses_of r blocks] is, per scale of [r], the guides that show it. *)

val build : Resolved.t -> Resolved.fitted array -> use list array -> item
(** [build r scales uses] is the item of the figure [r]. *)
