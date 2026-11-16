(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Probes of [rig_pool.h]'s threads (rig_pool_thread_probe_stubs.c): what a
    body sees on a worker, and the workers' life. POSIX only. *)

(** {1:bodies Bodies on a worker} *)

val worker_mask : Rig_pool_probe.kind -> bool * (string * bool) list
(** [worker_mask kind] is whether a worker of a job of [kind] ran a chunk, and
    the signals of a table with whether that worker blocks them. *)

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
  | Stack
      (** Whether a worker ran a chunk that used 7 MiB of stack, or 192 KiB
          for a blocking job. *)
  | Faults
      (** Whether a worker ran a chunk; a bit per signal of {!faults} that it
          raised and whose handler ran before [raise] returned. *)
  | Limited
      (** 1 once a thread limit of 0 kept a thread from being made and was
          lifted, else 0; the units not run once by a job on every core under
          the limit and by one after it; the calls of both on a worker; the
          threads after them. *)
  | Grown
      (** The process's threads before any job, after a job of two threads,
          after another of two, after one of three, after one of the cores
          and four more. *)
  | Limited_eight
      (** As [Limited], for jobs of eight threads: 1 once the limit bound and
          was lifted; the units not run once by both jobs; the calls on a
          worker under the limit; whether a worker then ran a chunk of a job
          of two; the threads after them. *)

val in_child : Rig_pool_probe.kind -> scenario -> string * int array
(** [in_child kind s] is how the child ended, ["exit 0"] once it answered, and
    its values, every job of the parent's and the child's of [kind]. *)

val fork_beside_blocking : int -> bool * string
(** [fork_beside_blocking n] forks while another thread's blocking job of [n]
    threads runs, every chunk held until fork returns. It is whether fork
    returned while that job still ran, and how the child ended: ["exit 0"]
    once [n] threads of the child's own ran the chunks of a blocking job of
    [n] at once. *)

(** {1:cpu CPU after a job} *)

val spin_cpu : Rig_pool_probe.kind -> runs:int -> int array * int array
(** [spin_cpu kind ~runs] runs [runs] jobs of [kind], at most 64, each of two
    chunks on two threads: the caller's, which ends once the worker's began,
    and the worker's, which sleeps 5 ms. It is, per job, the caller's CPU
    time from its chunk's end to the job's, and the worker's CPU time in the
    20 ms after the job returned, in ns; [-1] for both where no worker ran a
    chunk. Both are empty where the system gives no thread's CPU clock:
    Linux gives one. *)

val limits_threads : unit -> bool
(** [limits_threads ()] is whether the system can limit a process's own threads:
    Linux counts each thread against [RLIMIT_NPROC]. *)

(** {1:threads Threads} *)

val running_threads : unit -> int
(** [running_threads ()] is the threads of the process, other than the calling
    one, that are running now, or [-1] where the system does not say. *)
