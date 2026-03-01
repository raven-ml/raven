(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tick generation for axes.

    {b Internal module.} Produces nicely-spaced tick positions and formatted
    labels for linear, logarithmic, square-root, asinh, and symlog scales. *)

val generate :
  [ `Linear | `Log | `Sqrt | `Asinh | `Symlog of float ] ->
  lo:float ->
  hi:float ->
  ?max_ticks:int ->
  unit ->
  (float * string) list
(** [generate kind ~lo ~hi ()] is a list of [(value, label)] pairs for ticks
    spanning [[lo, hi]]. [max_ticks] defaults to [8]. *)
