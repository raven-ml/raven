(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The host memory that talon's columns hold, and the collector's work that
    dropping it brings, for talon's suites. *)

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

(** {1:cycles Major cycles} *)

type cycles = {
  majors : int;  (** Major cycles completed. *)
  minors : int;  (** Minor collections. *)
  promoted : float;  (** Words promoted to the major heap. *)
  major_words : float;  (** Words allocated in the major heap, promoted ones
      included. *)
}
(** The type for what a run gave the collector. *)

val cycles : (unit -> unit) -> cycles
(** [cycles f] is what [f ()] gives the collector, counted from a full major
    collection: the cycles a pacing law bounds, and what else drives them, for
    its failure to print. *)

val pp_cycles : cycles -> string
(** [pp_cycles c] is [c] as one line of text. *)
