/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's worker threads.

   One set of threads per process computes on the host: nx.cpu's kernels and
   the host programs a compiler emits share it, so that two of them running
   at once never take more threads than the host has cores.

   A job is a range of units [0, total) cut into contiguous chunks, which the
   calling thread and the workers claim one at a time until none remains. A
   thread that finishes early takes the chunks a slower one would have run.
   The job returns once every chunk has.

   The pool never touches the OCaml runtime: a chunk runs C, and the calling
   thread may hold the runtime or have released it. The workers are made at
   the first job that needs them and live until the process exits. A chunk
   on a worker has 8 MiB of stack; on the calling thread, the caller's. On
   macOS the workers run at the main thread's QoS class. They block the
   signals sent to the process, which then reach one of the program's own
   threads; a fault in a chunk is delivered on the thread that runs it. */

#ifndef NX_POOL_H
#define NX_POOL_H

#include <stdint.h>

/* Cores */

/* nx_pool_cores () is the number of cores the process may occupy at once,
   at least 1, and the most threads a job runs on, the caller included. On
   Linux it is the CPUs of the process's affinity mask, bounded by the CPU
   quota of its cgroup and the cgroup's ancestors, rounded up; on macOS, the
   physical cores; on Windows, the active processors; elsewhere, the online
   CPUs. It is computed at the first call; a later change of affinity or
   quota is not seen. */
int nx_pool_cores(void);

/* nx_pool_performance_cores () is the number of those cores that run
   compute-bound work at full speed, between 1 and nx_pool_cores (). On
   macOS it counts the performance cores (hw.perflevel0), since a chunk that
   an efficiency core claims takes two to three times as long and delays the
   job's end; elsewhere it is nx_pool_cores (). It is computed at the first
   call. */
int nx_pool_performance_cores(void);

/* Jobs */

/* The type for the function a job calls on each chunk: [lo, hi) is the
   chunk's units, [worker] the index of the thread that runs it, and [ctx]
   the job's context. */
typedef void (*nx_pool_body)(int64_t lo, int64_t hi, int worker, void *ctx);

/* nx_pool_run (threads, total, chunks, body, ctx) cuts [0, total) into
   [chunks] contiguous chunks and calls body (lo, hi, worker, ctx) once for
   each, on at most [threads] threads, the calling thread among them. It
   returns once every call has returned.

   Chunk [i] holds the units floor (i * total / chunks) to
   floor ((i + 1) * total / chunks) - 1, so chunks differ in size by at most
   one unit. [chunks] is first bounded to [1, total], so no chunk is empty;
   [threads] is bounded to [1, nx_pool_cores ()] and to [chunks]. A job of
   [total <= 0] calls nothing. Chunks are claimed in index order, so a
   caller that puts its costliest chunks first has them start first.

   [worker] is below [threads] bounded as above, so for [threads >= 1] it is
   below [threads]; the calling thread's is 0. A thread keeps its index for
   the whole job and claims a chunk only after its last one returned, so
   per-worker scratch indexed by [worker] is never shared; an index may run
   no chunk. Chunks run at once: a chunk must not wait for another, since
   every chunk may run on one thread. Writes the caller made before the call
   are visible to every body, and the bodies' writes to the caller once it
   returns.

   A job of one thread runs at once on the calling thread. A job of more
   waits for another thread's job to end, since the pool runs one at a
   time. A job begun from a body, on a thread that runs a chunk, runs alone
   on that thread with index 0, so a body may begin jobs of its own. A
   caller that holds the OCaml runtime during a job of more than one thread
   keeps every domain of the program from collecting until the job ends,
   including the time it waits: release it around such jobs.

   Threads that wait for work spin for up to 100 us before they sleep, so
   that jobs following each other closely start without a system call. If
   the workers cannot be made, the calling thread runs every chunk. In a
   child made by fork, the pool starts anew at its first job; fork waits for
   a running job to end, and a body must not fork. */
void nx_pool_run(int threads, int64_t total, int64_t chunks, nx_pool_body body,
                 void *ctx);

#endif
