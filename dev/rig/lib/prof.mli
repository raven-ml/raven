(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The profiles being taken, which every module records into. *)

type t = {
  counters : string list;
  trace : bool;
  lock : Lock.t;
  mutable events : (int * Def.event) list;  (** Numbered, newest first. *)
}

val active : unit -> t list
(* [active ()] is the profiles being taken. *)

val enabled : unit -> bool
val start : counters:string list -> trace:bool -> t
val stop : t -> unit
val add : t list -> Def.event -> unit
(* [add ps e] records [e] in each of [ps]. *)

val add_all : t list -> Def.event list -> unit
val record : Def.event -> unit
(* [record e] is [add (active ()) e]. *)

val now : unit -> int
