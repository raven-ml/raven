(*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The host's worker threads.

    One set of threads per process computes on the host. Its jobs are C
    functions: the kernels of nx.cpu and the host programs a compiler emits
    run on it through the C interface [nx_pool.h], which states how a job is
    cut into chunks and run. Sharing one set keeps two jobs from taking more
    threads than the host has cores.

    From OCaml, the pool tells how many threads a job may use, so that a
    caller cuts its work to fit. *)

val cores : unit -> int
(** [cores ()] is the number of cores the process may occupy at once,
    between [1] and [64], and the most threads a job runs on, the caller
    included. On Linux it is the CPUs of the process's affinity mask, bounded
    by its cgroup's CPU quota; on macOS, the physical cores; elsewhere, the
    online CPUs. It is computed once per process. *)

val performance_cores : unit -> int
(** [performance_cores ()] is the number of {!cores} that run compute-bound
    work at full speed, between [1] and [cores ()]. On a host with cores of
    two speeds, such as Apple silicon's performance and efficiency cores, it
    counts the fast ones: a share of compute-bound work given to a slow core
    takes two to three times as long and delays the job's end. On a host with
    one kind of core it is [cores ()]. *)
