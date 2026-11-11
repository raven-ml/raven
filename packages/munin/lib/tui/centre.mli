(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The centre: the banner, the hero, and the depth's view, charts, strips
    or zoom; and the text elements the whole screen is made of. *)

(** {1:text Text} *)

val frag :
  ?bold:bool -> Theme.t -> Theme.token -> string -> Mosaic.Text.fragment
(** [frag th tok s] is [s] in [tok]. *)

val line : ?grow:bool -> Mosaic.Text.fragment list -> Model.msg Mosaic.t
(** [line frags] is one line of [frags], cut at its width; growing to the
    width it is given when [grow] (default [true]). *)

val cell : int -> Mosaic.Text.fragment list -> Model.msg Mosaic.t
(** [cell w frags] is [frags] in [w] cells, cut. *)

val rcell : int -> Mosaic.Text.fragment list -> Model.msg Mosaic.t
(** [rcell w frags] is [frags] at the right end of [w] cells. *)

val right : Mosaic.Text.fragment list -> Model.msg Mosaic.t
(** [right frags] is [frags] at the right end of the width given. *)

val split :
  Mosaic.Text.fragment list -> Mosaic.Text.fragment list -> Model.msg Mosaic.t
(** [split left right] is one line of [left], cut, with [right] at its
    right end. *)

val rule : Theme.t -> Model.msg Mosaic.t
(** [rule th] is a horizontal rule that takes the width given. *)

val section : Theme.t -> string -> Model.msg Mosaic.t
(** [section th title] is [title] followed by a rule. *)

val glyph : Theme.t -> Facts.run -> Mosaic.Text.fragment
(** [glyph th f] is the run's state glyph in its colour. *)

val state_token : Facts.run -> Theme.token
(** [state_token f] is the colour of the run's state. *)

val arrow : Munin.Metric.goal option -> string
(** [arrow g] is [↓] to minimise, [↑] to maximise, [""] without a goal. *)

(** {1:centre The centre} *)

val draw : Theme.t -> Plot.t -> Mosaic.Canvas.t -> dt:float -> unit
(** [draw th p] is a canvas's drawing of [p], over the whole canvas. *)

val view : Theme.t -> Model.t -> Model.panes -> Facts.run -> Model.msg Mosaic.t
(** [view th m p f] is the centre of the focus [f]. *)
