(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Locks: a C mutex and a condition. Every lock of rig's state is one.

    Any domain may call any function. A call that waits for a lock, and {!wait},
    release the runtime while they block and run no signal handler. A lock is
    not reentrant: a call that holds it and takes it again blocks for good. A
    lock is never freed.

    A forked child makes every lock anew, so a lock that a thread of the parent
    held at the fork is free in the child. *)

type t [@@immediate]

val create : unit -> t
(** [create ()] is a new lock. Raises [Stdlib.Out_of_memory] if memory runs out.
*)

val hold : t -> unit
(** [hold l] waits for [l] and holds it: for a section that raises nothing and
    allocates no closure, which {!release} ends. *)

val release : t -> unit
(** [release l] gives back [l], which the caller holds. *)

val protect : t -> (unit -> 'a) -> 'a
(** [protect l f] is [f ()] run holding [l], which it gives back if [f] raises.
*)

val busy : t -> bool
(** [busy l] is [true] if a call holds [l]. *)

val wait : t -> unit
(** [wait l], holding [l], gives it up until a {!broadcast} on [l], or a
    spurious wake-up, then takes it back. *)

val broadcast : t -> unit
(** [broadcast l] wakes every {!wait} on [l]. *)
