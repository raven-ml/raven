(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arranging expanded figures: shares give scopes, layers broadcast over grids,
    and every panel's content becomes a list of occurrences. *)

module Text := Hugin_next_text.Text

(** {1:scopes Scopes} *)

(** The scope of a scale. *)
type key =
  | Figure
  | Node of Common.id
  | Cell of Common.id
  | Panels_of of Common.id * Common.id
      (** Per panel of the mark, in the cell. *)
  | Panel of Common.id * Common.id  (** The mark, the facet panel. *)

val equal_key : key -> key -> bool

type shares = (string * key) list

type env = {
  shares : shares;  (** Innermost first. *)
  pending : string list;  (** Independent names awaiting the node's children. *)
  cell : Common.id;  (** The innermost grid cell, or the root. *)
}

val key_of : env -> string -> key
(** [key_of env name] is the scope of the scale [name] in [env]. *)

(** {1:contents Contents} *)

type occ = {
  mid : Common.id;
  mark : Figure.mark;
  order : int;
  shares : shares;
  per_panel : string list;
}
(** An occurrence: a mark in one cell. *)

type axis_item = {
  gid : Common.id;
  side : Figure.side option;
  grid : bool;
  show : bool;
  scale : string;
}

type legend_item = {
  lid : Common.id;
  lside : Figure.side option;
  lshow : bool;
  lscale : string;
  lkey : key;  (** The scope of the scales it stands for. *)
}

type guide_item = G_axis of axis_item | G_legend of legend_item

val equal_axis : axis_item -> axis_item -> bool
val equal_legend : legend_item -> legend_item -> bool

type content = {
  occs : occ list;
  guides : guide_item list;
  coords : (Common.id * Coord.t) list;
  held : (Common.id * shares) list;
      (** The nodes that lie in the content, each with the scopes a channel
          there reads. *)
}

type shaped = { titles : (Text.Layout.halign * Text.t) list; body : body }
and body = Single of content | Arr of arr

and arr = {
  aid : Common.id;
  nrows : int;
  ncols : int;
  cells : cell list;
  widths : float list option;
  heights : float list option;
}

and cell = {
  row : int;
  col : int;
  rows : int;
  cols : int;
  cid : Common.id;
  s : shaped;
}

(** {1:arranging Arranging} *)

val arrange : Common.id list ref -> env -> in_cell:bool -> Expand.node -> shaped
(** [arrange nodes env ~in_cell n] is [n] arranged, with the id of each core
    node added to [nodes]. *)

val panels : Common.id -> shaped -> (Common.id * content) list
(** [panels root s] is the cells of [s] that hold content, with their ids, in
    reading order. *)
