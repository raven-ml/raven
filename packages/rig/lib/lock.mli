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
    or for a broadcast in {!await}, releases the domain's runtime lock and
    runs no signal handler while it blocks. Each lock counts among the locks
    its thread holds ([rig_hold] in [rig_edge.h]) from before its take to
    after its give. A thread takes one only while it holds no lock. A call
    made inside a lock ({!Rig.section-domains}) is refused as it starts
    ({!Dev.check_held}). A take while the count is not [0], which only a
    call that skipped that check makes, raises [Invalid_argument], having
    taken nothing, since it would wait for good on the lock it interrupted,
    or take two in no fixed order. C mutexes are taken only inside one
    stub, after any lock. A lock is freed once it is collected, so a caller
    holds it, or what holds it, for as long as it uses it.

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

val await : t -> ('a -> 'b option) -> 'a -> 'b
(** [await l f x] is the [v] of the first [Some v] that [f x], run holding
    [l], answers. After each [None] it gives up [l] until a {!broadcast} on
    [l], a spurious wake-up or 200 ms, which no step of the wall clock
    lengthens, and returns to OCaml holding no lock before it takes [l]
    again, so a pending signal's handler runs there outside every lock
    ({!Rig.section-domains}). If [f] raises, [await] gives back [l] before it
    allocates. *)

external broadcast : t -> unit = "caml_rig_lock_broadcast"
[@@noalloc]
(** [broadcast l] wakes every {!await} that waits on [l]. It is a call to C, as
    {!release} is. *)
