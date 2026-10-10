/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdatomic.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

/* Every table, the base first and each later one's instructions a superset
   of the one before. */
static const nx_cpu_target *const tables[] = {
    &nx_cpu_base,
#if defined(__x86_64__)
    &nx_cpu_v3,
#endif
#if defined(__APPLE__) && defined(__aarch64__)
    &nx_cpu_amx,
#endif
};

#define TABLES (int)(sizeof tables / sizeof tables[0])

/* The table the program started with, the best the host runs. */
static const nx_cpu_target *best(void) {
  static const nx_cpu_target *b = NULL;
  if (b == NULL) b = nx_cpu_table;
  return b;
}

value nx_kernels_support_targets(value unit) {
  CAMLparam1(unit);
  CAMLlocal2(l, cell);
  const nx_cpu_target *top = best();
  int last = 0;
  while (tables[last] != top) last++;
  l = Val_emptylist;
  for (int i = last; i >= 0; i--) {
    cell = caml_alloc_small(2, Tag_cons);
    Field(cell, 0) = Val_unit;
    Field(cell, 1) = l;
    l = cell;
    Store_field(cell, 0, caml_copy_string(tables[i]->name));
  }
  CAMLreturn(l);
}

value nx_kernels_support_current(value unit) {
  (void)unit;
  best();
  return caml_copy_string(nx_cpu_table->name);
}

/* A table past the best one may use instructions the host lacks. */
value nx_kernels_support_use(value name) {
  const nx_cpu_target *top = best();
  for (int i = 0; i < TABLES; i++) {
    if (strcmp(tables[i]->name, String_val(name)) == 0) {
      nx_cpu_table = tables[i];
      return Val_unit;
    }
    if (tables[i] == top) break;
  }
  caml_invalid_argument(
      "Nx_kernels_support.with_target: not a target the host runs");
}

/* nx.cpu's reductions and scans on one thread. */
value nx_kernels_support_serial_reduce(value s, value dsts, value ops) {
  return nx_cpu_reduce_on(s, dsts, ops, 1);
}

value nx_kernels_support_serial_scan(value s, value dsts, value ops) {
  return nx_cpu_scan_on(s, dsts, ops, 1);
}

/* Jobs begun from jobs' bodies: [n] outer units of [cost] bytes, each
   adding 0 to n − 1 by a job of [n] units of that cost begun in its
   body. */
typedef struct {
  int64_t n, cost;
  _Atomic int64_t sum;
} nested;

static void nested_inner(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  nested *q = ctx;
  for (int64_t i = lo; i < hi; i++) atomic_fetch_add(&q->sum, i);
}

static void nested_outer(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  nested *q = ctx;
  for (int64_t u = lo; u < hi; u++)
    nx_cpu_job(q->n, 0, q->cost, nested_inner, q);
}

value nx_kernels_support_nested(value n, value cost) {
  nested q = {.n = Long_val(n), .cost = Long_val(cost)};
  atomic_init(&q.sum, 0);
  nx_cpu_job(q.n, 0, q.cost, nested_outer, &q);
  return Val_long(atomic_load(&q.sum));
}
