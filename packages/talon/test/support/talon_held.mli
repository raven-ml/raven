(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The host memory that talon's columns hold, for talon's suites. *)

open Talon

val settled : unit -> int
(** [settled ()] is the bytes the host holds ({!Rig.allocated}) after four
    rounds of collection, each of which returns to it what it collected. *)

val made : ?inputs:Column.t list -> Column.t -> int
(** [made ~inputs c] is the bytes of [c]'s arrays, at every depth of its
    layout ({!Column.layout}), that lie in no buffer of [inputs]' arrays,
    each buffer counted once. *)

val check : ?inputs:Column.t list -> (unit -> Column.t) -> unit
(** [check ~inputs make] asserts that the column [make ()] is host memory: the
    {!settled} bytes grow by at least its {!made} bytes while it is live, and
    return to what they were once it is collected. [make] runs once
    uncounted first. *)
