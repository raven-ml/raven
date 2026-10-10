/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Jobs: on how many threads a piece of work runs, and when it releases the
   runtime.

   A job's threads follow from its cost, the bytes memcpy would move in the
   time it takes: one thread per THREAD_BYTES. Starting a job of several
   threads costs 1 to 5 us on the M1 Max and kimchi, and at 384 KiB a
   thread a 1.5 MiB int8 to int4 pack runs in 6.2 us on four threads,
   against 7.7 on three at 512 KiB (kimchi). The threads are the
   performance cores while the bytes the job touches fit in the caches,
   and every core past CACHE_BYTES: a job the caches hold ran up to 1.6
   times as long on all ten of the M1 Max's cores as on its eight
   performance cores, and past the caches the efficiency cores add their
   share of the memory's bandwidth (a 64 MiB copy: 665 us on ten, 789 on
   eight).

   A job of more than one thread releases the runtime, since a domain that
   holds it through a job keeps every domain from collecting, and so does a
   job of one thread that costs more than HOLD_BYTES. The release is the
   one that runs no pending signal handler: a handler that raised there
   would leave the door's claims held. The reacquisition runs none either;
   they run once the external returns.

   A body may begin a job of its own, as a factorisation's GEMM updates
   do. That job runs where the runtime is already released, on the calling
   thread or on a worker, which is no OCaml thread: it neither releases nor
   reacquires the runtime. A thread-local flag, set while a released job's
   body runs on any thread, says so: rig_pool's own test of a body is not
   enough, since a released job of one thread runs its body inline, outside
   the pool. */

#include <caml/signals.h>

#include "cpu.h"

#define THREAD_BYTES (384 * 1024)
#define CACHE_BYTES (32 * 1024 * 1024)
#define HOLD_BYTES (1024 * 1024)

/* Chunks per thread: a thread that finishes its strip takes the chunks left
   in the others', so a thread that runs slower, an efficiency core's say,
   runs fewer. */
#define CHUNKS 8

int nx_cpu_threads(int64_t bytes, double cost) {
  double threads = cost / THREAD_BYTES;
  int cores =
      bytes > CACHE_BYTES ? rig_pool_cores() : rig_pool_performance_cores();
  if (threads > cores) return cores;
  return threads < 1 ? 1 : (int)threads;
}

/* Whether this thread runs a body of a job that released the runtime. */
static _Thread_local int released;

typedef struct {
  rig_pool_body body;
  void *ctx;
} job;

/* A released job's body, on whichever thread claims its range. */
static void run_released(int64_t lo, int64_t hi, int worker, void *ctx) {
  const job *j = ctx;
  int was = released;
  released = 1;
  j->body(lo, hi, worker, j->ctx);
  released = was;
}

void nx_cpu_job(int64_t total, int64_t bytes, double cost, rig_pool_body body,
                void *ctx) {
  if (total <= 0) return;
  int64_t threads = nx_cpu_threads(bytes, cost);
  if (threads > total) threads = total;
  if (threads <= 1 && cost < HOLD_BYTES) {
    body(0, total, 0, ctx);
    return;
  }
  job j = {body, ctx};
  if (released) {
    rig_pool_run((int)threads, total, threads * CHUNKS, run_released, &j);
    return;
  }
  caml_enter_blocking_section_no_pending();
  rig_pool_run((int)threads, total, threads * CHUNKS, run_released, &j);
  caml_leave_blocking_section();
}
