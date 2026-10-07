/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A consumer of nx_pool.h: sums the units of a job from per-worker
   partials, optionally through a job each body begins for its chunk. */

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include <stdatomic.h>
#include <stdlib.h>

#include "nx_pool.h"

typedef struct {
  int64_t *partials; /* one per worker */
  _Atomic int64_t calls;
  int nested;
} sum_job;

static int64_t units(int64_t lo, int64_t hi) {
  int64_t s = 0;
  for (int64_t i = lo; i < hi; i++) s += i;
  return s;
}

/* A job begun from a body runs on the body's thread as worker 0. */
static void inner(int64_t lo, int64_t hi, int worker, void *ctx) {
  int64_t *s = ctx;
  s[worker] += units(lo, hi);
}

static void outer(int64_t lo, int64_t hi, int worker, void *ctx) {
  sum_job *j = ctx;
  atomic_fetch_add(&j->calls, 1);
  if (!j->nested) {
    j->partials[worker] += units(lo, hi);
    return;
  }
  /* The chunk's units, shifted to start at 0: lo is added back per unit. */
  int64_t s[1] = {0};
  nx_pool_run(4, hi - lo, 2, inner, s);
  j->partials[worker] += s[0] + (hi - lo) * lo;
}

/* [sum threads total chunks nested] is (the sum of the units, the chunks
   called). */
value pool_smoke_sum(value v_threads, value v_total, value v_chunks,
                     value v_nested) {
  CAMLparam4(v_threads, v_total, v_chunks, v_nested);
  CAMLlocal1(result);
  int threads = Int_val(v_threads);
  int slots = threads < 1 ? 1 : threads;
  sum_job j = {calloc((size_t)slots, sizeof(int64_t)), 0, Bool_val(v_nested)};
  if (j.partials == NULL) caml_raise_out_of_memory();
  int64_t total = Long_val(v_total), chunks = Long_val(v_chunks);
  caml_enter_blocking_section();
  nx_pool_run(threads, total, chunks, outer, &j);
  caml_leave_blocking_section();
  int64_t sum = 0;
  for (int w = 0; w < slots; w++) sum += j.partials[w];
  free(j.partials);
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_long(sum));
  Store_field(result, 1, Val_long(atomic_load(&j.calls)));
  CAMLreturn(result);
}

value pool_smoke_cores(value unit) {
  (void)unit;
  return Val_int(nx_pool_cores());
}

value pool_smoke_performance_cores(value unit) {
  (void)unit;
  return Val_int(nx_pool_performance_cores());
}
