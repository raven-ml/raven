(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A metric's series reduced to the dot columns of a plot, as the charts draw
   it: the laws of the Charts section of [Munin_tui].

   A value is immutable. [extend] reads only the samples past those a value
   reduced, and copies the per-column state, which is linear in the plot's dot
   columns. *)

type event = [ `Nan | `Pos_inf | `Neg_inf ]
type plot = { lo : float; hi : float; dots : int }
type t

val column : plot -> float -> int
(** [column p x] is the dot column of [x], [-1] left of the domain and [p.dots]
    right of it, binned as matrix bins a position. *)

val reduce : ?tau:float -> plot -> x:(int -> float) -> Munin.Metric.series -> t
val extend : t -> x:(int -> float) -> Munin.Metric.series -> t
val points : t -> (float * float) array
val level : t -> (float * float) array
val events : t -> (int * event) array

val extent : t -> (float * float) option
(** [extent c] is the least and greatest finite values reduced in the plot's x
    domain, smoothed when [c] smooths. *)
