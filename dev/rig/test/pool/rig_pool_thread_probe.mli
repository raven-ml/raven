(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Probes of [rig_pool.h]'s threads (rig_pool_thread_probe_stubs.c): what a body
    sees on a worker, and the workers' life. POSIX only. *)

(** {1:burst A burst of jobs} *)

val burst : unit -> unit
(** [burst ()] runs a job on every core, then jobs of two chunks on two threads
    back to back until {!burst_stop}. *)

val burst_stop : unit -> unit
(** [burst_stop ()] ends {!burst}. *)

(** {1:bodies Bodies on a worker} *)

val worker_mask : unit -> bool * (string * bool) list
(** [worker_mask ()] is whether a worker ran a chunk, and the signals of a table
    with whether that worker blocks them. *)

val faults : string list
(** [faults] are the signals a body may raise itself, in the order of the bits
    of {!constructor-Faults}' answer. *)

(** {1:children Children made by fork} *)

(** What a child made by fork answers, right after a job of the parent's. *)
type scenario =
  | Threads
      (** The process's threads before any job, after a job of one thread, after
          the first job of two, after jobs on every core. *)
  | Job
      (** Whether a worker ran a chunk; the units of a job on every core not run
          once. *)
  | Stack  (** Whether a worker ran a chunk that used 7 MiB of stack. *)
  | Faults
      (** Whether a worker ran a chunk; a bit per signal of {!faults} that it
          raised and whose handler ran before [raise] returned. *)
  | Limited
      (** 1 once a thread limit of 0 kept a thread from being made and was
          lifted, else 0; the units not run once by a job on every core under
          the limit and by one after it; the calls of both on a worker; the
          threads after them. *)

val in_child : scenario -> string * int array
(** [in_child s] is how the child ended, ["exit 0"] once it answered, and its
    values. *)

val fork : unit -> unit
(** [fork ()] forks a child that exits at once, and waits for it. *)

val limits_threads : unit -> bool
(** [limits_threads ()] is whether the system can limit a process's own threads:
    Linux counts each thread against [RLIMIT_NPROC]. *)

(** {1:threads Threads} *)

val running_threads : unit -> int
(** [running_threads ()] is the threads of the process, other than the calling
    one, that are running now, or [-1] where the system does not say. *)
