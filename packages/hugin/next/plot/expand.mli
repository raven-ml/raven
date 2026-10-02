(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Expanding figures: binds evaluated, ids assigned. Wrappers have their
    child's id. *)

module Text := Hugin_next_text.Text

type node = { id : Common.id; n : enode }

and enode =
  | E_mark of { mark : Figure.mark; order : int }
  | E_layer of node list
  | E_grid of {
      rows : node list list;
      widths : float list option;
      heights : float list option;
    }
  | E_span of { rows : int; cols : int; f : node }
  | E_share of (string * Figure.sharing) list * node
  | E_title of Text.Layout.halign * Text.t * node
  | E_coord of Coord.t * node
  | E_axis of {
      side : Figure.side option;
      grid : bool;
      show : bool;
      scale : string;
    }
  | E_legend of { side : Figure.side option; show : bool; scale : string }

type read =
  | Read : View.ident * 'a View.sort -> read  (** A key the figure reads. *)

type expansion = {
  view : View.t;
  mutable reads : read list;
  mutable marks : int;
}

val force : expansion -> Figure.t -> Figure.t
(** [force st f] evaluates the binds at the head of [f], through wrappers. *)

val names : Figure.t -> string list
(** [names f] is the names given to [f] by the wrappers at its head. *)

val expand : expansion -> Common.id -> Figure.t -> node
(** [expand st id f] is [f] expanded, with the id [id]. Marks are numbered in
    the order of the figure. *)
