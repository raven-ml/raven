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
#include <stdatomic.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include "nx_pool.h"

static uint64_t now_ns(void) {
#if defined(_WIN32)
  LARGE_INTEGER count, frequency;
  QueryPerformanceCounter(&count);
  QueryPerformanceFrequency(&frequency);
  uint64_t c = (uint64_t)count.QuadPart, f = (uint64_t)frequency.QuadPart;
  return c / f * 1000000000u + c % f * 1000000000u / f;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
#endif
}

value pool_bench_cores(value unit) {
  (void)unit;
  return Val_int(nx_pool_cores());
}

value pool_bench_performance_cores(value unit) {
  (void)unit;
  return Val_int(nx_pool_performance_cores());
}

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

/* [empty_after gap threads] keeps the calling thread busy for [gap]
   nanoseconds, as a caller between two jobs, then runs an empty job of one
   chunk per thread. */
value pool_bench_empty_after(value v_gap, value v_threads) {
  uint64_t end = now_ns() + (uint64_t)Long_val(v_gap);
  while (now_ns() < end) {
  }
  int threads = Int_val(v_threads);
  nx_pool_run(threads, threads, threads, empty, NULL);
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

/* [compute threads total chunks skewed] runs a compute-bound job. */
value pool_bench_compute(value v_threads, value v_total, value v_chunks,
                         value v_skewed) {
  static sink *sinks;
  if (sinks == NULL) {
    sinks = calloc((size_t)nx_pool_cores(), sizeof *sinks);
    if (sinks == NULL) abort();
  }
  work w = {Long_val(v_total), Bool_val(v_skewed), sinks};
  nx_pool_run(Int_val(v_threads), w.total, Long_val(v_chunks), compute, &w);
  return Val_unit;
}

/* Competing load

   Threads that spin until stopped, made the way another program makes its
   threads: at the default priority (on macOS, the default QoS class). */

static _Atomic int load_on;
static pthread_t *load_threads;
static int load_count;

static void *spin_load(void *arg) {
  volatile double x = 1.;
  (void)arg;
  while (atomic_load_explicit(&load_on, memory_order_relaxed))
    x = x * 0.999999 + 1e-9;
  return NULL;
}

/* [load_start n] starts [n] spinning threads. */
value pool_bench_load_start(value v_n) {
  int n = Int_val(v_n);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
#if defined(__APPLE__)
  pthread_attr_set_qos_class_np(&attr, QOS_CLASS_DEFAULT, 0);
#endif
  load_threads = calloc((size_t)n, sizeof *load_threads);
  if (load_threads == NULL) abort();
  atomic_store(&load_on, 1);
  for (load_count = 0; load_count < n; load_count++)
    if (pthread_create(&load_threads[load_count], &attr, spin_load, NULL) != 0)
      abort();
  pthread_attr_destroy(&attr);
  return Val_unit;
}

/* [load_stop ()] stops the threads [load_start] started. */
value pool_bench_load_stop(value unit) {
  (void)unit;
  atomic_store(&load_on, 0);
  for (int i = 0; i < load_count; i++) pthread_join(load_threads[i], NULL);
  free(load_threads);
  load_threads = NULL;
  load_count = 0;
  return Val_unit;
}
