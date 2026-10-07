/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/


/* Probes of nx_pool.h for the pool's suite: jobs whose bodies record what the
   pool did with them, called from OCaml as a consumer's stubs call the pool.
   They build on every system the pool does.

   Some bodies wait for another call, up to [patience] (pool_probe.h), to
   force a chunk onto a worker or to hold a job open. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sched.h>
#endif

#include "nx_pool.h"
#include "nx_pool_cgroup.h"
#include "pool_probe.h"

/* Time */

static void spin_ns(int64_t ns) {
  int64_t end = now_ns() + ns;
  while (now_ns() < end) {
  }
}

/* Recorded jobs */

typedef struct {
  int64_t lo, hi;
  int worker;
  pthread_t thread;
} call;

typedef struct {
  call *calls; /* in the order the calls began */
  int64_t cap;
  _Atomic int64_t n;
  _Atomic int *busy; /* per worker index: a call is running */
  int slots;
  _Atomic int overlaps;
} record_job;

static void record(int64_t lo, int64_t hi, int worker, void *ctx) {
  record_job *j = ctx;
  int tracked = worker >= 0 && worker < j->slots;
  if (tracked && atomic_exchange(&j->busy[worker], 1))
    atomic_fetch_add(&j->overlaps, 1);
  int64_t k = atomic_fetch_add(&j->n, 1);
  if (k < j->cap)
    j->calls[k] = (call){lo, hi, worker, pthread_self()};
  /* Long enough that several threads take part and an overlap shows. */
  spin_ns(1000);
  if (tracked)
    atomic_store(&j->busy[worker], 0);
}

static const int64_t max_recorded = INT64_C(1) << 20;

/* [probe_record threads total chunks] is (calls, count, overlaps): the calls
   as (lo, hi, worker, thread) in the order they began, [thread] numbering the
   threads in order of their first call, the caller's 0; the number of calls;
   the calls that began while another of their worker ran. */
value probe_record(value v_threads, value v_total, value v_chunks) {
  CAMLparam3(v_threads, v_total, v_chunks);
  CAMLlocal4(result, calls, entry, bound);
  int threads = Int_val(v_threads);
  int64_t total = Int64_val(v_total), chunks = Int64_val(v_chunks);
  int64_t cap = 0;
  if (total > 0) {
    cap = chunks < 1 ? 1 : chunks;
    if (cap > total)
      cap = total;
  }
  if (cap > max_recorded)
    caml_invalid_argument("probe_record: too many chunks");
  int slots = nx_pool_cores();
  record_job j = {calloc((size_t)cap + 1, sizeof(call)),      cap,   0,
                  calloc((size_t)slots, sizeof(_Atomic int)), slots, 0};
  pthread_t *seen = malloc(((size_t)cap + 1) * sizeof(pthread_t));
  if (j.calls == NULL || j.busy == NULL || seen == NULL) {
    free(j.calls);
    free((void *)j.busy);
    free(seen);
    caml_raise_out_of_memory();
  }
  seen[0] = pthread_self();
  caml_enter_blocking_section();
  nx_pool_run(threads, total, chunks, record, &j);
  caml_leave_blocking_section();
  int64_t n = atomic_load(&j.n), kept = n < cap ? n : cap;
  int nseen = 1;
  calls = caml_alloc((mlsize_t)kept, 0);
  for (int64_t k = 0; k < kept; k++) {
    call c = j.calls[k];
    int id = 0;
    while (id < nseen && !pthread_equal(seen[id], c.thread))
      id++;
    if (id == nseen)
      seen[nseen++] = c.thread;
    entry = caml_alloc_tuple(4);
    bound = caml_copy_int64(c.lo);
    Store_field(entry, 0, bound);
    bound = caml_copy_int64(c.hi);
    Store_field(entry, 1, bound);
    Store_field(entry, 2, Val_int(c.worker));
    Store_field(entry, 3, Val_int(id));
    Store_field(calls, k, entry);
  }
  free(j.calls);
  free((void *)j.busy);
  free(seen);
  result = caml_alloc_tuple(3);
  Store_field(result, 0, calls);
  Store_field(result, 1, Val_long(n));
  Store_field(result, 2, Val_int(atomic_load(&j.overlaps)));
  CAMLreturn(result);
}

/* Visibility */

#define COPY_UNITS 4096
#define COPY_CHUNKS 64

typedef struct {
  const int64_t *in;
  int64_t *out;
  int64_t job;
  _Atomic int64_t *misses;
} copy_job;

/* Reads the caller's [in] and writes [out], both plain memory. */
static void copy(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  copy_job *j = ctx;
  int64_t misses = 0;
  for (int64_t i = lo; i < hi; i++) {
    if (j->in[i] != j->job + i)
      misses++;
    j->out[i] = j->job + i + 1;
  }
  if (misses > 0)
    atomic_fetch_add(j->misses, misses);
}

/* [probe_visibility jobs threads] runs [jobs] jobs, each over values the
   caller writes just before it, and is (the values bodies read stale, the
   values the caller read stale after a job). */
value probe_visibility(value v_jobs, value v_threads) {
  CAMLparam2(v_jobs, v_threads);
  CAMLlocal1(result);
  int64_t jobs = Long_val(v_jobs);
  int threads = Int_val(v_threads);
  int64_t *in = malloc(COPY_UNITS * sizeof(int64_t));
  int64_t *out = malloc(COPY_UNITS * sizeof(int64_t));
  if (in == NULL || out == NULL) {
    free(in);
    free(out);
    caml_raise_out_of_memory();
  }
  _Atomic int64_t body_misses = 0;
  int64_t caller_misses = 0;
  caml_enter_blocking_section();
  for (int64_t k = 0; k < jobs; k++) {
    for (int64_t i = 0; i < COPY_UNITS; i++)
      in[i] = k + i;
    copy_job j = {in, out, k, &body_misses};
    nx_pool_run(threads, COPY_UNITS, COPY_CHUNKS, copy, &j);
    for (int64_t i = 0; i < COPY_UNITS; i++)
      if (out[i] != k + i + 1)
        caller_misses++;
  }
  caml_leave_blocking_section();
  free(in);
  free(out);
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_long(atomic_load(&body_misses)));
  Store_field(result, 1, Val_long(caller_misses));
  CAMLreturn(result);
}

/* Nested jobs */

#define MAX_INNER 64

typedef struct {
  pthread_t thread; /* the outer body's */
  _Atomic int *units;
  _Atomic int64_t calls, *misplaced;
} inner_job;

static void inner(int64_t lo, int64_t hi, int worker, void *ctx) {
  inner_job *j = ctx;
  atomic_fetch_add(&j->calls, 1);
  if (worker != 0 || !pthread_equal(pthread_self(), j->thread))
    atomic_fetch_add(j->misplaced, 1);
  for (int64_t i = lo; i < hi; i++)
    atomic_fetch_add(&j->units[i], 1);
}

typedef struct {
  int threads;
  int64_t inner_units;
  _Atomic int64_t outer_units, split, misplaced, unit_errors;
} outer_job;

/* Begins a job of one unit per chunk, and counts whether it ran in one call
   and its units not run once. */
static void outer(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  outer_job *j = ctx;
  atomic_fetch_add(&j->outer_units, hi - lo);
  _Atomic int units[MAX_INNER] = {0};
  inner_job ij = {pthread_self(), units, 0, &j->misplaced};
  nx_pool_run(j->threads, j->inner_units, j->inner_units, inner, &ij);
  if (atomic_load(&ij.calls) != 1)
    atomic_fetch_add(&j->split, 1);
  for (int64_t i = 0; i < j->inner_units; i++)
    if (atomic_load(&units[i]) != 1)
      atomic_fetch_add(&j->unit_errors, 1);
}

/* [probe_nested threads outer inner] runs a job of [outer] chunks of one unit
   on [threads] threads, whose every body begins a job of [inner] chunks of
   one unit on [threads] threads. It is (outer units run, inner jobs not run
   in one call, inner calls off their outer body's thread or not worker 0,
   inner units not run once). */
value probe_nested(value v_threads, value v_outer, value v_inner) {
  CAMLparam3(v_threads, v_outer, v_inner);
  CAMLlocal1(result);
  int64_t inner_units = Long_val(v_inner);
  if (inner_units < 0 || inner_units > MAX_INNER)
    caml_invalid_argument("probe_nested: inner");
  outer_job j = {Int_val(v_threads), inner_units, 0, 0, 0, 0};
  int64_t outer_units = Long_val(v_outer);
  caml_enter_blocking_section();
  nx_pool_run(j.threads, outer_units, outer_units, outer, &j);
  caml_leave_blocking_section();
  result = caml_alloc_tuple(4);
  Store_field(result, 0, Val_long(atomic_load(&j.outer_units)));
  Store_field(result, 1, Val_long(atomic_load(&j.split)));
  Store_field(result, 2, Val_long(atomic_load(&j.misplaced)));
  Store_field(result, 3, Val_long(atomic_load(&j.unit_errors)));
  CAMLreturn(result);
}

/* Balance */

typedef struct {
  int64_t chunks;
  _Atomic int64_t done;
  int balanced;
} balance_job;

/* The call that runs unit 0 lasts until every unit after its range has run:
   only a thread other than its own can run them. */
static void balance(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  balance_job *j = ctx;
  if (lo != 0) {
    atomic_fetch_add(&j->done, hi - lo);
    return;
  }
  int64_t rest = j->chunks - hi, deadline = now_ns() + patience;
  while (atomic_load(&j->done) < rest && now_ns() < deadline) {
  }
  j->balanced = rest > 0 && atomic_load(&j->done) == rest;
}

/* [probe_balance chunks] runs a job of [chunks] chunks of one unit on two
   threads, whose call that runs unit 0 lasts until the units after it have
   run, and is whether they ran while it lasted. */
value probe_balance(value v_chunks) {
  balance_job j = {Long_val(v_chunks), 0, 0};
  caml_enter_blocking_section();
  nx_pool_run(2, j.chunks, j.chunks, balance, &j);
  caml_leave_blocking_section();
  return Val_bool(j.balanced);
}

/* Held and counted jobs, observed from another domain */

static _Atomic int hold_arrived, hold_released;
static _Atomic int64_t counted_calls;

/* Every chunk waits for [probe_hold_release]; with [only_worker], the two
   chunks first meet, then the worker's alone waits. */
static void hold(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  int only_worker = *(int *)ctx;
  atomic_fetch_add(&hold_arrived, 1);
  int64_t deadline = now_ns() + patience;
  if (only_worker) {
    while (atomic_load(&hold_arrived) < 2 && now_ns() < deadline) {
    }
    if (worker == 0)
      return;
  }
  while (!atomic_load(&hold_released) && now_ns() < deadline) {
  }
}

value probe_reset(value unit) {
  (void)unit;
  atomic_store(&hold_arrived, 0);
  atomic_store(&hold_released, 0);
  atomic_store(&counted_calls, 0);
  return Val_unit;
}

/* [probe_hold only_worker] runs a job of two chunks on two threads that
   [hold]s, and is whether it was released before patience ran out. */
value probe_hold(value v_only_worker) {
  int only_worker = Bool_val(v_only_worker);
  caml_enter_blocking_section();
  nx_pool_run(2, 2, 2, hold, &only_worker);
  caml_leave_blocking_section();
  return Val_bool(atomic_load(&hold_released));
}

value probe_hold_arrived(value unit) {
  (void)unit;
  return Val_int(atomic_load(&hold_arrived));
}

value probe_hold_release(value unit) {
  (void)unit;
  atomic_store(&hold_released, 1);
  return Val_unit;
}

static void counted(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  (void)ctx;
  atomic_fetch_add(&counted_calls, 1);
}

value probe_counted(value v_threads, value v_total, value v_chunks) {
  int threads = Int_val(v_threads);
  int64_t total = Int64_val(v_total), chunks = Int64_val(v_chunks);
  caml_enter_blocking_section();
  nx_pool_run(threads, total, chunks, counted, NULL);
  caml_leave_blocking_section();
  return Val_unit;
}

value probe_counted_calls(value unit) {
  (void)unit;
  return Val_long(atomic_load(&counted_calls));
}

/* The host */

value probe_cores(value unit) {
  (void)unit;
  return Val_int(nx_pool_cores());
}

value probe_performance_cores(value unit) {
  (void)unit;
  return Val_int(nx_pool_performance_cores());
}

value probe_cgroup_cpus(value v_root) {
  return Val_long(nx_pool_cgroup_cpus(String_val(v_root)));
}

/* [probe_sysctl name] is the integer [name] reads, or -1. */
value probe_sysctl(value v_name) {
#if defined(__APPLE__)
  int n;
  size_t len = sizeof n;
  if (sysctlbyname(String_val(v_name), &n, &len, NULL, 0) == 0)
    return Val_int(n);
#else
  (void)v_name;
#endif
  return Val_int(-1);
}

/* [probe_active_processors ()] is the processors active in every group, or
   -1 off Windows. */
value probe_active_processors(value unit) {
  (void)unit;
#if defined(_WIN32)
  return Val_long((long)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
#else
  return Val_int(-1);
#endif
}

/* [probe_pinned_cores ()] pins the calling thread to one CPU of its affinity,
   reads nx_pool_cores, restores the affinity, reads it again: (first, second),
   or (-1, -1) where affinity is not Linux's. */
value probe_pinned_cores(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);
  int first = -1, second = -1;
#if defined(__linux__)
  cpu_set_t all, one;
  if (sched_getaffinity(0, sizeof all, &all) == 0) {
    CPU_ZERO(&one);
    for (int i = 0; i < CPU_SETSIZE; i++)
      if (CPU_ISSET(i, &all)) {
        CPU_SET(i, &one);
        break;
      }
    if (sched_setaffinity(0, sizeof one, &one) == 0) {
      first = nx_pool_cores();
      sched_setaffinity(0, sizeof all, &all);
      second = nx_pool_cores();
    }
  }
#endif
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(first));
  Store_field(result, 1, Val_int(second));
  CAMLreturn(result);
}
