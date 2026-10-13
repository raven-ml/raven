/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Jobs: on how many threads a piece of work runs, when it releases the
   runtime, and the scratch its threads keep.

   A job's threads follow from its cost, the bytes memcpy would move in the
   time it takes: one thread per NX_CPU_THREAD_BYTES (cpu.h), the bytes a
   core moves in the time a job of several threads takes to start. On fudge,
   a Core Ultra 5 235, today's pool starts a job in 0.5 to 1.2 us: 128 KiB,
   at which a 1.5 MiB int8 to int4 pack takes 3.6 us against 5.8 at 384 KiB,
   a 64 Ki-element float32 copy 1.8 us against 3.7, a 64^3 float32 product
   1.8 us against 3.8. Work whose threads take longer to start, as the M1's
   matrix unit's do, states its own bytes a thread (nx_cpu_job_at). The
   threads are the performance cores while the bytes the job touches fit
   in the caches, and every core past CACHE_BYTES: a job the
   caches hold ran up to 1.6 times as long on all ten of the M1 Max's cores
   as on its eight performance cores, and past the caches the efficiency
   cores add their share of the memory's bandwidth (a 64 MiB copy: 665 us on
   ten, 789 on eight).

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

#include <pthread.h>
#include <stdlib.h>

#include <caml/signals.h>

#include "cpu.h"

#define CACHE_BYTES (32 * 1024 * 1024)
#define HOLD_BYTES (1024 * 1024)

int nx_cpu_threads(int64_t bytes, double cost) {
  return nx_cpu_threads_at(bytes, cost, NX_CPU_THREAD_BYTES);
}

int nx_cpu_threads_at(int64_t bytes, double cost, double thread_bytes) {
  double threads = cost / thread_bytes;
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
  nx_cpu_job_at(total, bytes, cost, NX_CPU_THREAD_BYTES, body, ctx);
}

void nx_cpu_job_at(int64_t total, int64_t bytes, double cost,
                   double thread_bytes, rig_pool_body body, void *ctx) {
  if (total <= 0) return;
  int64_t threads = nx_cpu_threads_at(bytes, cost, thread_bytes);
  if (threads > total) threads = total;
  if (threads <= 1 && cost < HOLD_BYTES) {
    body(0, total, 0, ctx);
    return;
  }
  job j = {body, ctx};
  int64_t chunks = threads * NX_CPU_CHUNKS;
  if (released) {
    rig_pool_run((int)threads, total, chunks, run_released, &j);
    return;
  }
  caml_enter_blocking_section_no_pending();
  rig_pool_run((int)threads, total, chunks, run_released, &j);
  caml_leave_blocking_section();
}

/* Scratch

   Each thread keeps KEPT buffers from call to call, each up to KEPT_BYTES:
   an allocation past 32 KiB goes to the system and back, 0.7 us on the
   M1, as long as packing a 128^3 product takes, and keeping the
   transforms' buffers saved 0.13 to 0.23 us a call of 64 to 1024 points
   (fft.c). KEPT is the most a kernel holds at once:
   Bluestein's rows, the four steps' grid and a pass's two lane buffers
   (fft.c). A thread's buffers are freed when it ends. */
#define KEPT 4
#define KEPT_BYTES (4 * 1024 * 1024)
#define ALIGN 128

typedef struct {
  void *p;
  int64_t n;
  int held;
} buffer;

typedef struct {
  buffer b[KEPT];
} kept;

static pthread_key_t kept_key;
static pthread_once_t kept_once = PTHREAD_ONCE_INIT;

/* The calling thread's buffers, or NULL: one load, where the key's lookup
   is two calls. */
static _Thread_local kept *kept_here;

static void kept_drop(void *v) {
  kept *k = v;
  for (int i = 0; i < KEPT; i++) free(k->b[i].p);
  free(k);
  kept_here = NULL;
}

static void kept_init(void) { pthread_key_create(&kept_key, kept_drop); }

/* The calling thread's buffers, made at its first take; NULL if they
   cannot be. */
static kept *mine(void) {
  if (kept_here != NULL) return kept_here;
  pthread_once(&kept_once, kept_init);
  kept *k = calloc(1, sizeof *k);
  if (k != NULL && pthread_setspecific(kept_key, k) == 0)
    return kept_here = k;
  free(k);
  return NULL;
}

static void *fresh(int64_t bytes) {
  int64_t n = bytes > 0 ? (bytes + ALIGN - 1) / ALIGN * ALIGN : ALIGN;
  return aligned_alloc(ALIGN, (size_t)n);
}

/* The first free buffer that fits, else the smallest free one grown, so
   the larger ones stay for the calls that need them. */
void *nx_cpu_take(int64_t bytes) {
  kept *k = bytes > KEPT_BYTES ? NULL : mine();
  if (k == NULL) return fresh(bytes);
  buffer *grow = NULL;
  for (int i = 0; i < KEPT; i++) {
    buffer *b = &k->b[i];
    if (b->held) continue;
    if (b->p != NULL && b->n >= bytes) {
      b->held = 1;
      return b->p;
    }
    if (grow == NULL || b->n < grow->n) grow = b;
  }
  if (grow == NULL) return fresh(bytes);
  free(grow->p);
  grow->p = fresh(bytes);
  grow->n = grow->p ? bytes : 0;
  grow->held = grow->p != NULL;
  return grow->p;
}

void nx_cpu_give(void *p) {
  kept *k = kept_here;
  for (int i = 0; k != NULL && i < KEPT; i++)
    if (k->b[i].p == p && k->b[i].held) {
      k->b[i].held = 0;
      return;
    }
  free(p);
}
