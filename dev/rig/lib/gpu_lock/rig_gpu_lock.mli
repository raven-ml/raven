(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The machine's GPU lock.

    Programs that act on the machine's GPUs, such as test suites and benches,
    take the lock so that they run on the GPUs in turn. The lock is [flock] on
    the file [/tmp/raven-rig-gpu.lock], shared by every checkout, project and
    user of the machine: the first process to take it creates the file
    writable by every user. Its holder writes its executable and process id
    into the file.

    The library links no driver. A program decides whether the machine has the
    GPU it acts on, without opening it, and takes the lock only then. *)

val hold : unit -> unit
(** [hold ()] returns once the process holds the lock, which it keeps until it
    exits. It returns at once, taking nothing, if the variable
    [RIG_GPU_LOCK_HELD] is set: the process that started this one holds the
    lock for it, as in
    [flock /tmp/raven-rig-gpu.lock env RIG_GPU_LOCK_HELD=1 ./bench.exe]. It
    starts no vendor library, so a program may call it before it forks workers,
    which then run under the lock.

    While another process holds the lock, it prints the file's note once on
    [stderr]: the holder's, or an earlier holder's when the holder took the
    lock with the shell's [flock].

    Raises [Failure] naming the holder if the lock is still held after 300 s,
    or naming the errno if the file cannot be locked, as on Windows. *)
