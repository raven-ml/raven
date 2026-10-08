(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the NVIDIA path's suites share. *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] returns once the process holds the machine's GPU lock, which
    it keeps until it exits, or at once if the machine has no NVIDIA GPU. The
    lock is [flock] on [/tmp/raven-rig-gpu.lock], the file every suite and bench
    that acts on a GPU of the machine locks; its holder writes its executable
    and process id into it. A suite calls [hold_gpu] before [Windtrap.run], so
    that the wait counts against no test's timeout, and a test calls it again
    before it opens the GPU. A bench calls it before [Thumper.run], so that the
    workers it forks run under the lock: [hold_gpu] starts no vendor library,
    which a process must not start before it forks. It returns at once, taking
    nothing, if the variable [RIG_GPU_LOCK_HELD] is set: the process that
    started this one holds the lock for it, as a timing run takes it before
    the host's timing locks.

    Raises [Failure] naming the holder if another process still holds the lock
    after 300 s, or naming the errno if the file cannot be locked. *)

val files : unit -> int
(** [files ()] is the number of files the process has open. *)

val limit_for : int -> int
(** [limit_for k] is the limit on the process's files ([RLIMIT_NOFILE]) under
    which it may open [k] files more than it has open now. *)

val with_limit : int -> (unit -> 'a) -> 'a
(** [with_limit n f] is [f ()], run with the process's soft limit on its files
    lowered to [n], and set back after it. Only the calling process is limited.
*)

val occupy : int -> int -> bool
(** [occupy at n] maps [n] inaccessible bytes at the address [at] of the
    process, unless something is mapped there: [true] if it did. *)

val vacate : int -> int -> unit
(** [vacate at n] unmaps the [n] bytes at [at]. *)
