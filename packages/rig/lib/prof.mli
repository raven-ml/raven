(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The profiles being taken, which every module records into.

    {!Rig.Profile.take} opens a profile here and stops it when its function
    returns or raises. Every event recorded in between goes into each profile
    open at the time. The open profiles are one atomic list and each
    profile's events one atomic cell, so any domain records without a lock.
    Each event gets a number from one counter of the process, which orders
    events of equal time ({!Profile}).

    A profile detaches its events in one exchange when it stops, which no
    allocation precedes, so an asynchronous exception cannot leave it open. A
    recorder that read the list before the stop and adds afterwards finds the
    cell closed, and its event is dropped. A stopped profile may stay in the
    list, where an exception that ended {!stop} after its exchange left it,
    until the next {!start} or {!stop} removes it. *)

type t = {
  events : (int * Def.event) list option Atomic.t;
      (** Numbered in the order recorded, newest first; [None] once stopped. *)
  pending : intake Atomic.t;
      (** Its {!pending} entries, newest first. *)
}
(** The type for profiles. *)

(** The type for a profile's entries: it takes them while [take]'s function
    runs, then only gives up those it took to [take]'s reads, until its
    stop. *)
and intake = Open of pending list | Reading of pending list | Closed

(** The type for the events of [device]'s work up to [value] that [read]
    adds to a profile once the device reached it, such as a submission's
    span. Each profile that timed the work holds an entry of its own; [drop]
    ends the entry, read or not, and the last [drop] of the work's entries
    lets go of what they read. *)
and pending = {
  device : Def.device;
  value : int;
  read : t -> unit;
  drop : unit -> unit;
}

val active : unit -> t list
(** [active ()] is the open profiles, oldest first, and stopped ones that
    remain in the list. *)

val enabled : unit -> bool
(** [enabled ()] is [true] iff a profile is open. Recorders test it first, so
    a program that takes no profile builds no event. *)

val start : unit -> t
(** [start ()] opens a new profile, which every event recorded until {!stop}
    goes into. *)

val stop : t -> (int * Def.event) list
(** [stop p] closes [p], drops its pending entries, and is its events,
    newest first. Calls after the first are [[]]. *)

val pend : t -> pending -> unit
(** [pend p e] adds [e] to [p]'s pending entries, or drops it once {!seal}
    or {!stop} ran on [p]. *)

val seal : t -> unit
(** [seal p] makes [p] take no more entries: [take]'s function returned. *)

val add : t list -> Def.event -> unit
(** [add ps e] records [e] under one number in each profile of [ps] still
    open. A recorder that read {!active} earlier, as a submit does for the
    span it records once its value is reached, adds to those profiles
    alone. *)

val record : Def.event -> unit
(** [record e] is [add (active ()) e]. *)

val now : unit -> int
(** [now ()] is {!Rig.Profile.now}. *)
