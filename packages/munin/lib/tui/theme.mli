(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The dashboard's colours as tokens. The screens name tokens, never
    colours.

    A token has a colour for a dark and for a light terminal. With direct
    colour it is that colour, with 256 colours the nearest entry of the
    palette's cube and ramp, and with 16 or 8 colours an entry of the 16 or
    the terminal's default, the dim tokens faint and the brightest bold. Without
    colour only bold and faint remain, and the cursor's row is reverse
    video. Faint colours are mixed toward the terminal's background. *)

type look = {
  scheme : [ `Dark | `Light ];
  depth : Ansi.Styling.t;  (** What the screen is written at. *)
  background : (int * int * int) option;
      (** The terminal's background, when it reported it. *)
}
(** The type for what the screens' colours depend on: plain data. *)

type token =
  | Raise  (** The bars, the help box, the code of the empty screen. *)
  | Sel  (** The cursor's row. *)
  | Line  (** Rules. *)
  | Line2  (** The empty part of a gauge, a run not drawn. *)
  | Dim
  | Mid
  | Fg
  | Hi
  | Live  (** Also better. *)
  | Fail  (** Also NaN, worse. *)
  | Warn  (** Heat, a dirty tree, the star. *)

type t
(** The type for a look's styles. *)

val make : look -> t
(** [make look] is [look]'s styles. *)

val coloured : t -> bool
(** [coloured t] is [false] without colour. *)

val fg : ?bold:bool -> t -> token -> Ansi.Style.t
(** [fg t tok] is text in [tok]. *)

val bg : t -> token -> Ansi.Color.t option
(** [bg t tok] is [tok] as a background, [None] where the depth has none. *)

val ground : t -> Ansi.Color.t
(** [ground t] is the terminal's own background, at every depth: what an
    element drawn over others paints, so that none of them shows through. *)

val runs : int
(** [runs] is the number of run colours; a run's colour is taken modulo
    it. *)

val run : ?bold:bool -> t -> int -> Ansi.Style.t
(** [run t i] is text in the run colour [i]. *)

val faint : t -> token -> Ansi.Style.t
(** [faint t tok] is [tok] mixed halfway toward the background. *)

val faint_run : t -> int -> Ansi.Style.t
(** [faint_run t i] is the run colour [i] mixed halfway toward the
    background. *)

val tint : t -> Ansi.Color.t option
(** [tint t] is the banner's background: [Fail] mixed far toward the
    background. *)

val sel : t -> Ansi.Style.t
(** [sel t] is the cursor's row: its background, or reverse video without
    colour. *)
