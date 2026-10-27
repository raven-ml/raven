/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Jobs: on how many threads a piece of work runs, and the scratch its
   threads keep.

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

   A job's cost sizes its threads; a call's price (cpu.h, Calls) decides
   whether the call releases the domain lock. The prices bound each kind's
   time on one core of the machines measured, an Apple M1 Max and an Intel
   Core Ultra 5 235; the jobs' costs are older estimates, many below it,
   so a costly kind runs on fewer threads than it would pay for.

   A job never releases the domain lock: the call it runs in releases it
   for its whole work at its entry, or holds it through work that costs
   less than NX_CPU_RELEASE_BYTES. A held call runs its jobs on the
   calling thread alone, and only a released call's jobs use the pool: the
   pool runs one job on several threads at a time, so a thread that held
   the lock while it waited for another domain's job would keep every
   domain's collections waiting as long. The flag is nx.cpu's own: rig
   runs its jobs on rig_pool directly, and the engine reaches nx.cpu
   through its externals, each of which sets the flag at its entry and
   clears it at its exit. A body may begin a job of its own, as a
   factorisation's GEMM updates do. */

#include <pthread.h>
#include <stdlib.h>

#include "cpu.h"

#define CACHE_BYTES (32 * 1024 * 1024)

_Thread_local int nx_cpu_held;
_Thread_local int nx_cpu_one;

/* Sets nx_cpu_one to [v] and answers its value before: one_thread's
   scope. */
value nx_cpu_one_thread(value v) {
  value was = Val_bool(nx_cpu_one);
  nx_cpu_one = Bool_val(v);
  return was;
}

int nx_cpu_threads(int64_t bytes, double cost) {
  return nx_cpu_threads_at(bytes, cost, NX_CPU_THREAD_BYTES);
}

int nx_cpu_threads_at(int64_t bytes, double cost, double thread_bytes) {
  if (nx_cpu_held || nx_cpu_one) return 1;
  double threads = cost / thread_bytes;
  int cores =
      bytes > CACHE_BYTES ? rig_pool_cores() : rig_pool_performance_cores();
  if (threads > cores) return cores;
  return threads < 1 ? 1 : (int)threads;
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
  if (threads <= 1) {
    body(0, total, 0, ctx);
    return;
  }
  rig_pool_run((int)threads, total, threads * NX_CPU_CHUNKS, body, ctx);
}

/* Scratch

   Each thread keeps KEPT buffers from call to call, each up to KEPT_BYTES:
   an allocation past 32 KiB goes to the system and back, 0.7 us on the
   M1, as long as packing a 128^3 product takes, and keeping the
   transforms' buffers saved 0.13 to 0.23 us a call of 64 to 1024 points
   (fft.c). KEPT is the most a kernel holds at once on one thread: eight
   for a complex product whose result is of another dtype than its sums
   and whose contracted runs are several (gemm.c), its sums R, a's and b's
   runs copied into C order, a's and the init's parts and b's blocks, and
   on worker 0 the packed b and its tile; five for a Moments scan of an
   aliased operand (reduce.c), four for a transform (fft.c). A thread's
   buffers are freed when it ends. */
#define KEPT 8
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
