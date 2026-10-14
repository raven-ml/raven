(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Locks: a C mutex with a condition variable, held by a custom block.

    Every lock in rig's OCaml code is one: each device's ({!Def.live}), and
    those of the tables in {!Dev}, {!Memory} and {!Copy}.

    An exception raised while a lock is held must not leave it held, or the
    next call that takes it waits for good. An asynchronous exception, such
    as [Sys.Break] or [Out_of_memory], arises where OCaml code allocates or
    polls, and [Stack_overflow] where a function starts. So a section
    between {!hold} and {!release} allocates nothing, loops nowhere and calls
    no OCaml function, and every other section runs in {!protect}.

    Any domain may call any function. A call that blocks, waiting for a lock
    or in {!wait}, releases the domain's runtime lock meanwhile and runs no
    signal handler. A thread holds at most one lock, and a take while it
    holds one raises [Invalid_argument]. Such a take comes from a signal
    handler, finaliser or [Gc.Memprof] callback that runs inside a section
    and calls rig, which would otherwise wait for good on the lock it
    interrupted, or take two in no fixed order. C mutexes are taken only
    inside one stub, after any lock. A lock is freed once it is collected,
    so a caller holds it, or what holds it, for as long as it uses it.

    A forked child makes every lock anew, so a lock that a thread of the
    parent held at the fork is free in the child. *)

type t
(** The type for locks. *)

val create : unit -> t
(** [create ()] is a new lock. Raises [Stdlib.Out_of_memory] if the C heap
    has no room for it. *)

val hold : t -> unit
(** [hold l] waits for [l] and holds it, for a section that {!release} ends,
    which allocates nothing, loops nowhere and calls no OCaml function. *)

external release : t -> unit = "caml_rig_lock_give"
[@@noalloc]
(** [release l] gives back [l], which the caller holds. It is a call to C,
    so a section ends with no call to an OCaml function. *)

val protect : t -> ('a -> 'b) -> 'a -> 'b
(** [protect l f x] is [f x] run holding [l]. If [f] raises, [protect] gives
    back [l] before it allocates. Callers pass a closed function, defined at
    the top level, and its argument, so the call allocates no closure. *)

val busy : t -> bool
(** [busy l] is [true] if a call held [l] when [busy] tried to take it, or
    the thread holds a lock. *)

val wait : t -> unit
(** [wait l], holding [l], gives it up until a {!broadcast} on [l], a spurious
    wake-up or 200 ms, which no step of the wall clock lengthens, then takes
    it back. A caller must wait in a loop on its condition, so that it
    returns to OCaml at least every 200 ms, where a pending signal's handler
    runs ({!Rig.section-domains}). *)

external broadcast : t -> unit = "caml_rig_lock_broadcast"
[@@noalloc]
(** [broadcast l] wakes every {!wait} on [l]. It is a call to C, as
    {!release} is. *)
