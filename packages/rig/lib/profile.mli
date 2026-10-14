(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Profiles: the implementation of {!Rig.Profile}, and the profiles being
    taken, which every module records into.

    {!take} opens a profile and stops it when its function returns or
    raises. Every event recorded in between goes into each profile open at
    the time. The open profiles are one atomic list and each profile's
    events one atomic cell, so any domain records without a lock. Each event
    gets a number from one counter of the process, which orders events of
    equal time.

    Events that the host sees as they happen, such as a load, an allocation
    or a copy the host ran, are recorded immediately. A submit that a
    profile times leaves the profile a {!pending} entry that reads the times
    its driver wrote. When {!take}'s function returns, its profile takes no
    more entries, and [take] reads those it took into it alone, one at a
    time, then stops it.
    An entry leaves the profile as [take] starts to wait for its value, and
    is dropped however its read ends. A device lost meanwhile has its
    entries dropped.

    A profile detaches its events in one exchange when it stops, which no
    allocation precedes, so an asynchronous exception cannot leave it open,
    then drops its entries one at a time, and a [take] whose function raises
    keeps no entry. A recorder that read the list before the stop and adds
    afterwards finds the cell closed, and its event is dropped. A stopped
    profile may stay in the list, where an exception that ended the stop
    after its exchange left it with the entries it did not drop, until the
    next start or stop drops them and removes it. *)

open Def

(** {!Rig.Profile.event}. *)
type event = Def.event =
  | Span of {
      device : device;
      lane : string;
      name : string;
      start : int;
      stop : int;
    }
  | Allocation of { device : device; time : int; allocated : int }
  | Load of { image : image; binary : string; time : int }
  | Counters of {
      device : device;
      name : string;
      start : int;
      stop : int;
      counters : (string * int array) list;
    }
  | Trace of {
      device : device;
      name : string;
      start : int;
      stop : int;
      part : int;
      data : string;
    }
  | Overwritten of { device : device; time : int; runs : int }
  | Copy of { src : device; dst : device; bytes : int; start : int; stop : int }

type t
(** The type for profiles. *)

(** The type for the events of [device]'s work up to [value] that [read]
    adds to a profile once the device reached it, such as a submission's
    span. Each profile that timed the work holds an entry of its own; [drop]
    ends the entry, read or not, and the last [drop] of the work's entries
    lets go of what they read. *)
and pending = {
  device : device;
  value : int;
  read : t -> unit;
  drop : unit -> unit;
}

val active : unit -> t list
(** [active ()] is the open profiles, oldest first, and stopped ones that
    remain in the list. *)

val pend : t -> pending -> unit
(** [pend p e] adds [e] to [p]'s pending entries, or drops it once the
    function of [p]'s {!take} has returned. *)

val add : t list -> event -> unit
(** [add ps e] records [e] under one number in each profile of [ps] still
    open. A recorder that read {!active} earlier, as a submit does for the
    span it records once its value is reached, adds to those profiles
    alone. *)

val take : (unit -> 'a) -> 'a * event list
(** [take f] is {!Rig.Profile.take}[ f]. *)

val enabled : unit -> bool
(** [enabled ()] is {!Rig.Profile.enabled}[ ()]: whether a profile is
    open. Recorders test it first, so a program that takes no profile builds
    no event. *)

val now : unit -> int
(** [now ()] is {!Rig.Profile.now}[ ()]. *)

val record : event -> unit
(** [record e] is {!Rig.Profile.record}[ e], [add (active ()) e]. *)

val span : string -> (unit -> 'a) -> 'a
(** [span name f] is {!Rig.Profile.span}[ name f]. *)

val output_chrome_trace : out_channel -> event list -> unit
(** [output_chrome_trace oc events] is {!Rig.Profile.output_chrome_trace}[ oc
    events]. *)
