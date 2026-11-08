(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Windows: the frames of [Expr.window], one per row.

    It implements [Talon.Expr.window]'s semantics over the rows of one batch cut
    into segments ({!Reduce}): each row's window holds rows of its own segment
    alone. *)

type t
(** The type for the windows of a frame's rows. *)

val v :
  Reduce.segments ->
  on:Column.t option ->
  before:Column.t option ->
  after:Column.t option ->
  exclude:bool ->
  t
(** [v s ~on ~before ~after ~exclude] is the window of each row of [s], along
    the key [on] or the rows' places, with the bounds [before] and [after], the
    rows of the row's key taken out if [exclude]. [on] is a number or a
    temporal value, and the bounds are of its type or meet it, or durations of
    its unit or a coarser one; integers without [on]. *)

val windows : t -> int
(** [windows w] is the number of windows of [w]: one per row whose key and
    bounds are not null, in row order. *)

val row : t -> int -> int
(** [row w i] is the row of window [i]. *)

val frame : t -> Reduce.frame
(** [frame w] is the windows [w] as frames ({!Reduce.frame}), one per window,
    in window order. *)

val to_rows : t -> Column.t -> Column.t
(** [to_rows w c] is the values [c], one per window, on the windows' rows,
    and null on the rows that have no window. *)
