(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Locks: a C mutex and a condition, which a forked child makes anew, so a lock
   that a thread of the parent held at the fork is free in the child. Every lock
   of the core's state is one. Any domain may call any function. *)

type t [@@immediate]

val create : unit -> t
val hold : t -> unit
val release : t -> unit
(* [hold l] waits for [l] and holds it, [release l] gives it back: for a section
   that raises nothing and allocates no closure. *)

val protect : t -> (unit -> 'a) -> 'a
(* [protect l f] is [f ()] run holding [l]. *)

val busy : t -> bool
(* [busy l] is [true] if a call holds [l]. *)

val wait : t -> unit
(* [wait l], holding [l], gives it up until a {!broadcast} on [l], or a spurious
   wake-up, then takes it back. *)

val broadcast : t -> unit
(* [broadcast l] wakes every {!wait} on [l]. *)
