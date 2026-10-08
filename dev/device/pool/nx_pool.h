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
   claims the next chunks in index order until none remains, so a thread
   that finishes early runs the chunks a slower one would have run. The job
   returns once every chunk has run.

   A caller sizes a job by two facts about the host: its cores, which bound
   the threads of every job, and its performance cores, those that run
   compute-bound work at full speed.

   References. The host facts come from sched_getaffinity(2) and the
   cpu.max files of the Linux kernel's cgroup v2
   (Documentation/admin-guide/cgroup-v2.rst), from sysctl(3) on macOS, and
   from GetActiveProcessorCount on Windows. */

#ifndef NX_POOL_H
#define NX_POOL_H

#include <stdint.h>

/* Cores */

/* nx_pool_cores () is the number of cores the process may occupy at once,
   the most threads a job runs on. 1 <= nx_pool_cores ().

   On Linux it is min (a, ceil q): a the CPUs of the affinity mask of the
   thread that makes the first call, q the smallest cpu.max quota / period
   of the process's cgroup v2 and the cgroup's ancestors (no bound without
   a quota). On macOS it is the physical cores; on Windows, the active
   processors; elsewhere, the online CPUs. Nothing else bounds it, however
   many cores the host has. It is computed at the first call: a later
   change of affinity or quota is not seen. */
int nx_pool_cores(void);

/* nx_pool_performance_cores () is the number of those cores that run
   compute-bound work at full speed.
   1 <= nx_pool_performance_cores () <= nx_pool_cores ().

   On macOS it counts the performance cores (hw.perflevel0) where the host
   reports them, since a chunk that an efficiency core claims takes two to
   three times as long and delays the end of its job; elsewhere, and on a
   Mac that does not report them, it is nx_pool_cores (). It is computed at
   the first call. */
int nx_pool_performance_cores(void);

/* Jobs */

/* The type for the function a job calls on its ranges: [lo, hi) are the
   range's units, [worker] the index of the thread that runs it, [ctx] the
   job's context. */
typedef void (*nx_pool_body)(int64_t lo, int64_t hi, int worker, void *ctx);

/* nx_pool_run (threads, total, chunks, body, ctx) runs the job [0, total)
   cut into [chunks] chunks on at most [threads] threads: it calls
   body (lo, hi, worker, ctx) on disjoint ranges [lo, hi) that cover
   [0, total), each made of whole consecutive chunks, and returns once every
   call has returned. A job of total <= 0 calls nothing; a job that runs on
   one thread makes one call, over [0, total).

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
   claims chunks only after its last call returned, so two calls with the
   same worker never overlap and scratch indexed by worker is never shared.
   An index may run no chunk: per-worker partials start at their identity.

   Bodies. Calls run in parallel or one after another on one thread, in any
   interleaving: a body must not wait for another call. Writes the caller
   made before the call are visible to every body, and the bodies' writes to
   the caller once it returns. A body runs C and must not call the OCaml
   runtime, since a worker is not an OCaml thread. A body may begin a job of
   its own (see Scheduling). A body must not fork. On a worker a body has
   8 MiB of stack, address space that memory backs as the body touches it; on
   the calling thread, the caller's. Workers block every signal except those
   a body raises itself (SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGTRAP, SIGABRT,
   SIGSYS), so a signal sent to the process reaches one of the program's own
   threads and a fault in a body is delivered on the thread that runs it. A
   profiler that samples threads by SIGPROF therefore never samples a
   worker.

   Scheduling. The pool runs one job of more than one thread at a time. A
   job of t = 1, or a job begun from a body of a job of more than one
   thread, runs at once on the calling thread alone as worker 0. Any other
   job first waits for another thread's job to end. A caller that holds the
   OCaml runtime during a job of t > 1 keeps every domain of the program
   from collecting until the job ends, including the time it waits: release
   the runtime around such jobs.

   Threads. The workers are made at the first job of t > 1 and live until
   the process exits. Those that cannot be made are missing from every job;
   with none, the calling thread runs every chunk. Threads that wait for
   work spin for up to 100 us before they sleep, so jobs that follow each
   other closely start without a system call. In a child made by fork, the
   pool starts anew at its first job; fork waits for a running job of more
   than one thread to end. */
void nx_pool_run(int threads, int64_t total, int64_t chunks, nx_pool_body body,
                 void *ctx);

#endif
