/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's worker threads.

   One set of threads per process computes on the host: nx.cpu's kernels and
   the host programs a compiler emits share it, so that two of them running
   at once never take more threads than the host has cores.

   A job is a range of units [0, total) cut into contiguous chunks. The
   calling thread and the workers claim the chunks one at a time, in index
   order, until none remains, so a thread that finishes early takes the
   chunks a slower one would have run. The job returns once every chunk has.

   The pool never touches the OCaml runtime: a chunk runs C, and the calling
   thread may hold the runtime or have released it. The workers are made at
   the first job that needs them and live until the process exits. They
   block signals, so a signal sent to the process reaches one of the
   program's own threads. */

#ifndef NX_POOL_H
#define NX_POOL_H

#include <stdint.h>

/* Cores */

/* nx_pool_cores () is the number of cores the process may occupy at once,
   between 1 and 64: on Linux, the CPUs of its affinity mask bounded by its
   cgroup's CPU quota; on macOS, the physical cores; elsewhere, the online
   CPUs. It is the most threads a job runs on, the caller included. */
int nx_pool_cores(void);

/* nx_pool_performance_cores () is the number of those cores that run
   compute-bound work at full speed, between 1 and nx_pool_cores (). On a
   host with cores of two speeds (Apple silicon's performance and efficiency
   cores) it counts the fast ones; a chunk that a slow core claims takes
   two to three times as long and delays the job's end. On a host with one
   kind of core it is nx_pool_cores (). */
int nx_pool_performance_cores(void);

/* Jobs */

/* The type for the function a job calls on each chunk: [lo, hi) is the
   chunk's units, [worker] the index of the thread that runs it, and [ctx]
   the job's context. */
typedef void (*nx_pool_body)(int64_t lo, int64_t hi, int worker, void *ctx);

/* nx_pool_run (threads, total, chunks, body, ctx) cuts [0, total) into
   [chunks] contiguous chunks and calls body (lo, hi, worker, ctx) once for
   each, on at most [threads] threads, the calling thread among them. It
   returns once every call has returned; their writes are then visible to
   the caller.

   Chunk [i] holds the units floor (i * total / chunks) to
   floor ((i + 1) * total / chunks) - 1, so chunks differ in size by at most
   one unit. [chunks] is first bounded to [1, total], so no chunk is empty;
   [threads] is bounded to [1, nx_pool_cores ()] and to [chunks]. A job of
   [total <= 0] calls nothing.

   [worker] is in [0, threads), the calling thread's being 0. A thread keeps
   its index for the whole job and claims a chunk only after its last one
   returned, so per-worker scratch indexed by [worker] is never shared.
   Chunks run in any order and at once: a chunk must not wait for another,
   since every chunk may run on one thread.

   The pool runs one job at a time. A job begun while another thread's job
   runs waits for it to end. A job begun from a body, on a thread that
   already runs a chunk, runs alone on that thread with index 0, so a body
   may begin jobs of its own.

   Threads that wait for work spin for a short while before they sleep, so
   jobs that follow each other closely start without a system call. A
   caller that holds the OCaml runtime keeps its domain's other threads
   from running until the job ends: release it around long jobs. If the
   workers cannot be made, the calling thread runs every chunk. In a child
   made by fork, the pool starts anew at its first job; fork waits for a
   running job to end. */
void nx_pool_run(int threads, int64_t total, int64_t chunks, nx_pool_body body,
                 void *ctx);

#endif
