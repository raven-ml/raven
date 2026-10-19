(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The domain lock as the calling thread holds it. *)

val watch : (unit -> unit) -> (releases:int * held:int * cpu:int)
(** [watch f] runs [f ()] on the calling thread. [releases] is the count of
    times the thread released the domain lock within it. [held] is the
    longest stretch for which it held the lock, from [f]'s start or a
    reacquisition to the next release or [f]'s end, in nanoseconds of the
    thread's CPU time. [cpu] is [f]'s, in nanoseconds of the process's CPU
    time: the work of all its threads. A loaded host's scheduler lengthens
    neither. *)

val copy : unit -> int
(** [copy ()] is the least of five copies of 1 MiB by memcpy on the calling
    thread, in nanoseconds of its CPU time: what nx's lock bound allows a
    held stretch, on an unloaded core. *)

val bound : unit -> int
(** [bound ()] is what a held stretch may take on a loaded host, in
    nanoseconds of CPU time: four times {!copy}, as an efficiency core runs
    it up to about three times as long as the performance core [copy] may
    have run on. *)

val occupy : unit -> bool
(** [occupy ()] runs, with the calling domain's lock released, a job on two
    of the pool's threads whose units each wait until a thread within
    {!watch} releases the domain lock, or 10 s pass. It is [true] iff a
    release ended them. While it runs, the pool begins no other job on
    several threads: such a job waits for it to end. *)

val occupied : unit -> bool
(** [occupied ()] is [true] once an {!occupy}'s job runs its units, until
    it returns. *)
