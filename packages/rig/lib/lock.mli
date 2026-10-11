(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Locks: a C mutex with a condition variable, held by a custom block.

    Every lock in rig's OCaml code is one: each device's ({!Def.live}), and
    those of the tables in {!Dev}, {!Memory} and {!Copy}. {!hold} and
    {!release} bracket a section without a closure.

    Any domain may call any function. A call that blocks, waiting for a lock
    or in {!wait}, releases the domain's runtime lock meanwhile and runs no
    signal handler. A lock is not reentrant, so a thread that holds it and
    takes it again blocks for good. A lock is freed once it is collected, so
    a caller holds it, or what holds it, for as long as it uses it.

    A forked child makes every lock anew, so a lock that a thread of the
    parent held at the fork is free in the child. *)

type t
(** The type for locks. *)

val create : unit -> t
(** [create ()] is a new lock. Raises [Stdlib.Out_of_memory] if the C heap
    has no room for it. *)

val hold : t -> unit
(** [hold l] waits for [l] and holds it, for a section that raises nothing
    and allocates no closure, which {!release} ends. *)

val release : t -> unit
(** [release l] gives back [l], which the caller holds. *)

val protect : t -> (unit -> 'a) -> 'a
(** [protect l f] is [f ()] run holding [l], which it gives back if [f]
    raises. *)

val busy : t -> bool
(** [busy l] is [true] if a call held [l] when [busy] tried to take it. *)

val wait : t -> unit
(** [wait l], holding [l], gives it up until a {!broadcast} on [l] or a
    spurious wake-up, then takes it back. *)

val broadcast : t -> unit
(** [broadcast l] wakes every {!wait} on [l]. *)
