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

    A profile detaches its events in one exchange when it stops. A recorder
    that read the list before the stop and adds afterwards finds the cell
    closed, and its event is dropped. *)

type t = {
  events : (int * Def.event) list option Atomic.t;
      (** Numbered in the order recorded, newest first; [None] once stopped. *)
}
(** The type for profiles. *)

val active : unit -> t list
(** [active ()] is the open profiles, oldest first. *)

val enabled : unit -> bool
(** [enabled ()] is [true] iff a profile is open. Recorders test it first, so
    a program that takes no profile builds no event. *)

val start : unit -> t
(** [start ()] opens a new profile, which every event recorded until {!stop}
    goes into. *)

val stop : t -> (int * Def.event) list
(** [stop p] closes [p] and is its events, newest first. Calls after the
    first are [[]]. *)

val add : t list -> Def.event -> unit
(** [add ps e] records [e] under one number in each profile of [ps] still
    open. A recorder that read {!active} earlier, as a submit does for the
    span it records once its value is reached, adds to those profiles
    alone. *)

val record : Def.event -> unit
(** [record e] is [add (active ()) e]. *)

val now : unit -> int
(** [now ()] is {!Rig.Profile.now}. *)
