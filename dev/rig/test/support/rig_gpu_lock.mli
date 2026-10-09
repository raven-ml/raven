(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The machine's GPU lock.

    Every suite and bench that acts on a GPU of the machine takes it, so that
    they run on the GPUs in turn. The caller decides whether the machine has
    the GPU it acts on, from files alone, and takes the lock only then. *)

val hold : unit -> unit
(** [hold ()] returns once the process holds the machine's GPU lock, which it
    keeps until it exits. The lock is [flock] on [/tmp/raven-rig-gpu.lock],
    shared by every checkout and user of the machine; its holder writes its
    executable and process id into the file. It returns at once, taking
    nothing, if the variable [RIG_GPU_LOCK_HELD] is set: the process that
    started this one holds the lock for it, as a timing run takes it before
    the host's timing locks. It starts no vendor library, so a bench may call
    it before it forks its workers, which then run under the lock.

    A suite calls it before [Windtrap.run], so that the wait counts against no
    test's timeout. While another process holds the lock, it prints the
    file's note once on [stderr]: the holder's, or an earlier holder's when the
    holder took the lock with the shell's [flock].

    Raises [Failure] naming the holder if the lock is still held after 300 s,
    or naming the errno if the file cannot be locked. *)
