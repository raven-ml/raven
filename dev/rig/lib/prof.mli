(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The profiles being taken, which every module records into.

    Any domain may call any function. The profiles being taken are one atomic
    list, and so are a profile's events. *)

type t = {
  counters : string list;
  trace : bool;
  events : (int * Def.event) list Atomic.t;
      (** Numbered in the order recorded, newest first. *)
}

val active : unit -> t list
(** [active ()] is the profiles being taken, in the order they started. *)

val enabled : unit -> bool

val start : counters:string list -> trace:bool -> t
(** [start ~counters ~trace] is a new profile, which every record goes into
    until {!stop}. *)

val stop : t -> unit

val add : t list -> Def.event -> unit
(** [add ps e] records [e] in each of [ps], under one number. *)

val add_all : t list -> Def.event list -> unit

val record : Def.event -> unit
(** [record e] is [add (active ()) e]. *)

val now : unit -> int
(** {!Rig.Profile.now}. *)
