/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Jobs for the pool's bench, run through nx_pool.h as a kernel runs them.

   The bench program has one domain, so the stubs keep the runtime during a
   job: releasing it would add its own cost to every row. */

#define _GNU_SOURCE

#include <caml/mlvalues.h>

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#include "nx_pool.h"
#include "pool_probe.h"

/* Empty jobs */

static void empty(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  (void)ctx;
}

/* [empty threads total chunks] runs a job whose bodies do nothing. */
value pool_bench_empty(value v_threads, value v_total, value v_chunks) {
  nx_pool_run(Int_val(v_threads), Long_val(v_total), Long_val(v_chunks), empty,
              NULL);
  return Val_unit;
}

/* [busy gap] keeps the calling thread busy for [gap] nanoseconds, as a
   caller between two jobs. */
value pool_bench_busy(value v_gap) {
  int64_t end = now_ns() + Long_val(v_gap);
  while (now_ns() < end) {
  }
  return Val_unit;
}

/* Compute-bound jobs

   A unit is a chain of dependent multiply-adds, which no core runs faster
   than its latency allows, so a job's time is its slowest thread's share.
   Each worker keeps its result a cache line away from the others'. */

enum { unit_steps = 64, line = 64 };

typedef struct {
  double x;
  char pad[line - sizeof(double)];
} sink;

typedef struct {
  int64_t total;
  int skewed;
  sink *sinks;
} work;

/* The steps of unit [u]: [unit_steps] each when balanced; when skewed, a
   ramp from 2 * unit_steps down to 0, the costliest first, with the same
   sum. */
static int64_t steps(const work *w, int64_t u) {
  if (!w->skewed) return unit_steps;
  return 2 * unit_steps * (w->total - u) / w->total;
}

static void compute(int64_t lo, int64_t hi, int worker, void *ctx) {
  work *w = ctx;
  double x = w->sinks[worker].x;
  for (int64_t u = lo; u < hi; u++)
    for (int64_t s = steps(w, u); s > 0; s--) x = x * 0.999999 + 1e-9;
  w->sinks[worker].x = x;
}

static sink *sinks(void) {
  static sink *s;
  if (s == NULL) {
    s = calloc((size_t)nx_pool_cores(), sizeof *s);
    if (s == NULL) abort();
  }
  return s;
}

/* [compute threads total chunks skewed] runs a compute-bound job. */
value pool_bench_compute(value v_threads, value v_total, value v_chunks,
                         value v_skewed) {
  work w = {Long_val(v_total), Bool_val(v_skewed), sinks()};
  nx_pool_run(Int_val(v_threads), w.total, Long_val(v_chunks), compute, &w);
  return Val_unit;
}

/* Floors

   A job without the pool: threads of the bench's own, each of which runs a
   fixed share of the chunks, one call each, once a generation word moves,
   then counts down to the caller, which runs share 0. Nothing is claimed,
   entered or parked. The threads wait as the pool's workers do, spinning,
   and yielding the core between reads after 2 us, so that a competing
   load does not starve them; they live for one row. */

static const int64_t floor_busy_ns = 2000;

static struct {
  _Alignas(128) _Atomic uint64_t generation;
  _Atomic int stop;
  _Alignas(128) _Atomic uint64_t left;
  int threads;
  pthread_t *ids;
  int64_t total, chunks;
  nx_pool_body body;
  void *ctx;
} floor_job;

static void relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

/* Share [i] of [threads]: chunks [i * c / t, (i + 1) * c / t), one call
   each, with the pool's chunk bounds. */
static void floor_share(int i) {
  int64_t t = floor_job.threads, c = floor_job.chunks, n = floor_job.total;
  for (int64_t k = i * c / t; k < (i + 1) * c / t; k++)
    floor_job.body(k * n / c, (k + 1) * n / c, i, floor_job.ctx);
}

/* Waits for [*word] to leave [value] (or [*stop] to be set), spinning,
   then yielding between reads, and returns the last value read. */
static uint64_t floor_wait(_Atomic uint64_t *word, uint64_t value) {
  uint64_t v;
  int64_t start = now_ns();
  while ((v = atomic_load_explicit(word, memory_order_acquire)) == value &&
         !atomic_load_explicit(&floor_job.stop, memory_order_relaxed)) {
    relax();
    if (now_ns() - start >= floor_busy_ns) sched_yield();
  }
  return v;
}

static void *floor_thread(void *arg) {
  int i = (int)(intptr_t)arg;
  uint64_t seen = 0;
  for (;;) {
    uint64_t g = floor_wait(&floor_job.generation, seen);
    if (g == seen) return NULL;
    seen = g;
    floor_share(i);
    atomic_fetch_sub_explicit(&floor_job.left, 1, memory_order_release);
  }
}

/* [floor_start threads] starts [threads] - 1 floor threads. They begin at
   generation 0, which a row's earlier threads may have moved past. */
value pool_bench_floor_start(value v_threads) {
  floor_job.threads = Int_val(v_threads);
  floor_job.ids = calloc((size_t)floor_job.threads, sizeof *floor_job.ids);
  if (floor_job.ids == NULL) abort();
  atomic_store(&floor_job.stop, 0);
  atomic_store(&floor_job.generation, 0);
  for (int i = 1; i < floor_job.threads; i++)
    if (pthread_create(&floor_job.ids[i], NULL, floor_thread,
                       (void *)(intptr_t)i) != 0)
      abort();
  return Val_unit;
}

/* [floor_stop ()] ends the threads [floor_start] started. */
value pool_bench_floor_stop(value unit) {
  (void)unit;
  atomic_store(&floor_job.stop, 1);
  for (int i = 1; i < floor_job.threads; i++)
    pthread_join(floor_job.ids[i], NULL);
  free(floor_job.ids);
  floor_job.ids = NULL;
  return Val_unit;
}

static void floor_run(int64_t total, int64_t chunks, nx_pool_body body,
                      void *ctx) {
  floor_job.total = total;
  floor_job.chunks = chunks;
  floor_job.body = body;
  floor_job.ctx = ctx;
  uint64_t left = (uint64_t)floor_job.threads - 1;
  atomic_store_explicit(&floor_job.left, left, memory_order_relaxed);
  atomic_fetch_add_explicit(&floor_job.generation, 1, memory_order_release);
  floor_share(0);
  for (uint64_t v = left; v != 0;) v = floor_wait(&floor_job.left, v);
}

/* [floor_empty total chunks] is [empty] on the floor threads. */
value pool_bench_floor_empty(value v_total, value v_chunks) {
  floor_run(Long_val(v_total), Long_val(v_chunks), empty, NULL);
  return Val_unit;
}

/* [floor_compute total chunks] is a balanced [compute] on the floor
   threads. */
value pool_bench_floor_compute(value v_total, value v_chunks) {
  work w = {Long_val(v_total), 0, sinks()};
  floor_run(w.total, Long_val(v_chunks), compute, &w);
  return Val_unit;
}
