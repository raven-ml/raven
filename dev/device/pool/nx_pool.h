/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's worker threads.

   One pool of threads per process computes on the host. nx.cpu's kernels
   and the host programs a compiler emits share it, so that together they
   never take more threads than the host has cores.

   The pool runs jobs. A job is a range of units [0, total) cut into
   contiguous chunks and run on at most a given number of threads: the
   calling thread, which is worker 0, and the pool's workers. Each thread
   claims the next chunk in index order until none remains, so a thread
   that finishes early runs the chunks a slower one would have run. The job
   returns once every chunk has run.

   A caller sizes a job by two facts about the host: its cores, which bound
   the threads of every job, and its performance cores, those that run
   compute-bound work at full speed. */

#ifndef NX_POOL_H
#define NX_POOL_H

#include <stdint.h>

/* Cores */

/* nx_pool_cores () is the number of cores the process may occupy at once,
   the most threads a job runs on. 1 <= nx_pool_cores ().

   On Linux it is min (a, ceil q): a the CPUs of the process's affinity
   mask, q the smallest cpu.max quota / period of its cgroup v2 and the
   cgroup's ancestors (no bound without a quota). On macOS it is the
   physical cores; on Windows, the active processors; elsewhere, the online
   CPUs. It is computed at the first call: a later change of affinity or
   quota is not seen. */
int nx_pool_cores(void);

/* nx_pool_performance_cores () is the number of those cores that run
   compute-bound work at full speed.
   1 <= nx_pool_performance_cores () <= nx_pool_cores ().

   On macOS it counts the performance cores (hw.perflevel0), since a chunk
   that an efficiency core claims takes two to three times as long and
   delays the end of its job; elsewhere it is nx_pool_cores (). It is
   computed at the first call. */
int nx_pool_performance_cores(void);

/* Jobs */

/* The type for the function a job calls on each chunk: [lo, hi) are the
   chunk's units, [worker] the index of the thread that runs it, [ctx] the
   job's context. */
typedef void (*nx_pool_body)(int64_t lo, int64_t hi, int worker, void *ctx);

/* nx_pool_run (threads, total, chunks, body, ctx) runs the job [0, total)
   cut into [chunks] chunks on at most [threads] threads: it calls
   body (lo, hi, worker, ctx) once per chunk and returns once every call
   has returned. A job of total <= 0 calls nothing.

   Bounds. Integers out of range are bounded: for total >= 1 the job has
   c chunks and at most t threads,

     c = min (max (chunks, 1), total)
     t = min (max (threads, 1), nx_pool_cores (), c)

   Chunk i, for 0 <= i < c, is

     [lo, hi) = [floor (i * total / c), floor ((i + 1) * total / c))

   exactly, for every total and c. The chunks partition [0, total), none is
   empty, and floor (total / c) <= hi - lo <= ceil (total / c). Chunks are
   claimed in index order: a caller that puts its costliest chunks first
   has them start first.

   Worker index. 0 <= worker < t <= max (threads, 1), and the calling
   thread's worker is 0. A thread keeps its index for the whole job and
   claims a chunk only after its last one returned, so two calls with the
   same worker never overlap and scratch indexed by worker is never shared.
   An index may run no chunk: per-worker partials start at their identity.
   Writes the caller made before the call are visible to every body, and
   the bodies' writes to the caller once it returns.

   Bodies. Chunks run in parallel or one after another on one thread, in
   any interleaving: a body must not wait for another chunk. A body runs C
   and must not call the OCaml runtime, since a worker is not an OCaml
   thread. A body may begin a job of its own (see Scheduling). A body must
   not fork. On a worker a body has 8 MiB of stack; on the calling thread,
   the caller's. Workers block every signal except those a body raises
   itself (SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGTRAP, SIGABRT, SIGSYS), so a
   signal sent to the process reaches one of the program's own threads and
   a fault in a body is delivered on the thread that runs it.

   Scheduling. The pool runs one job of more than one thread at a time. A
   job of t = 1, or a job begun from a body, runs at once on the calling
   thread alone as worker 0. Any other job first waits for another thread's
   job to end. A caller that holds the OCaml runtime during a job of t > 1
   keeps every domain of the program from collecting until the job ends,
   including the time it waits: release the runtime around such jobs.

   Threads. The workers are made at the first job of t > 1 and live until
   the process exits. Those that cannot be made are missing from every job;
   with none, the calling thread runs every chunk. Threads that wait for
   work spin for up to 100 us before they sleep, so jobs that follow each
   other closely start without a system call. In a child made by fork, the
   pool starts anew at its first job; fork waits for a running job to end. */
void nx_pool_run(int threads, int64_t total, int64_t chunks, nx_pool_body body,
                 void *ctx);

#endif
