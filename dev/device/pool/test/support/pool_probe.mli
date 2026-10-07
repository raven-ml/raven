(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the pool's suites and bench share: probes of [nx_pool.h], whose
    bodies record what the pool did (pool_probe_stubs.c), and the suites'
    waits.

    A probe that waits for another call of its job, which [nx_pool.h] forbids
    a body, gives up after 10 s, so a pool that breaks a promise fails the test
    instead of hanging it. *)

(** {1:recorded Recorded jobs} *)

type call = { lo : int64; hi : int64; worker : int; thread : int }
(** A body's call: its range [\[lo, hi)], its worker index, and its thread,
    numbered in the order of the threads' first calls, the caller's 0. *)

type job = { calls : call list; count : int; overlaps : int }
(** A job's calls in the order they began, the number of calls, and the calls
    that began while another call of their worker ran. *)

val record : threads:int -> total:int64 -> chunks:int64 -> job
(** [record ~threads ~total ~chunks] runs that job and is its calls. Each call
    lasts 1 us, so that several threads take part and an overlap shows. *)

val visibility : jobs:int -> threads:int -> int * int
(** [visibility ~jobs ~threads] runs [jobs] jobs of [threads] threads, each
    reading what the caller wrote before it and writing what the caller reads
    after it, and is (values the bodies read stale, values the caller read
    stale). *)

val nested : threads:int -> outer:int -> inner:int -> int * int * int * int
(** [nested ~threads ~outer ~inner] runs a job of [outer] units whose every
    call runs a job of [inner] units, and is (outer units run, inner jobs not
    run in one call, inner calls off their body's thread or not worker 0,
    inner units not run once). *)

val balance : int -> bool
(** [balance chunks] is whether, in a job of [chunks] chunks of one unit on
    two threads, the units after the call that runs unit 0 ran while it
    lasted. *)

(** {1:observed Jobs observed from another domain} *)

val reset : unit -> unit
(** [reset ()] clears what {!hold} and {!counted} recorded. *)

val hold : only_worker:bool -> bool
(** [hold ~only_worker] runs a job of two chunks on two threads whose chunks,
    or with [only_worker] the worker's once both began, wait for
    {!hold_release}. It is whether they were released before 10 s. *)

val hold_arrived : unit -> int
(** [hold_arrived ()] is the chunks of {!hold}'s job that began. *)

val hold_release : unit -> unit
(** [hold_release ()] releases {!hold}'s chunks. *)

val counted : threads:int -> total:int64 -> chunks:int64 -> unit
(** [counted ~threads ~total ~chunks] runs that job, counting its calls. *)

val counted_calls : unit -> int
(** [counted_calls ()] is the calls {!counted} made since {!reset}. *)

(** {1:host The host}

    The probes of a host fact are [-1] where the host lacks it. *)

val cores : unit -> int
(** [cores ()] is [nx_pool_cores ()]. *)

val performance_cores : unit -> int
(** [performance_cores ()] is [nx_pool_performance_cores ()]. *)

val sysctl : string -> int
(** [sysctl name] is the integer the sysctl [name] reads (macOS). *)

val active_processors : unit -> int
(** [active_processors ()] is the processors active in every group
    (Windows). *)

val pinned_cores : unit -> int * int
(** [pinned_cores ()] pins the calling thread to one CPU of its affinity, reads
    the cores, restores the affinity and reads them again (Linux). *)

val needs_two_cores : unit -> unit
(** [needs_two_cores ()] skips the running test on a host of one core. It calls
    [nx_pool_cores] only when run: test_pool's affinity child must call
    nothing of the pool before it pins itself. *)

(** {1:waits Waits} *)

val timeout : float
(** [timeout] is each test's limit, 30 s, past the 10 s that a probe or a
    poll waits before it fails with what it waited for. *)

val settle : float -> (unit -> 'a) -> ('a -> bool) -> 'a
(** [settle seconds read ok] reads [read ()] every millisecond until [ok] holds
    of the value or [seconds] have passed, and is the last value read. *)

val within : float -> string -> (unit -> bool) -> unit
(** [within seconds why ready] polls [ready] for at most [seconds], and fails
    the test with [why] if it never holds. *)

val finishes : string -> (unit -> 'a) -> 'a
(** [finishes what f] is [f ()], run on a domain of its own, and fails the test
    with [what] if it has not returned after 10 s: a job that waits forever
    fails the test instead of hanging it. *)

val while_held : (unit -> 'a) -> 'a
(** [while_held f] is [f ()], run while another domain's job of two threads
    runs: every chunk of it waits until [f] has returned. *)
