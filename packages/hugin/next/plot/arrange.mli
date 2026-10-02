(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arranging figures: binds are evaluated and ids assigned, shares give scopes,
    layers broadcast over grids, and every panel's content becomes a list of
    occurrences. *)

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

type content = {
  occs : occ list;
  guides : (Common.id * Figure.guide * key) list;
      (** With the scope of the scale it names. *)
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

type read = Read : View.ident * 'a View.sort -> read  (** A key read. *)

type t = { shaped : shaped; order : Common.id list; reads : read list }
(** [order] is every node in figure order, [reads] the keys the binds read. *)

val arrange : View.t -> Figure.t -> t
(** [arrange view f] is [f] arranged, its binds reading [view]. *)

val legends : shaped -> (Common.id * Figure.guide * key) list
(** [legends s] is the legends of the contents of [s], in reading order. *)

val panels : shaped -> (Common.id * content) list
(** [panels s] is the cells of [s] that hold content, with their ids, in reading
    order. *)
