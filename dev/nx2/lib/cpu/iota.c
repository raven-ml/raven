/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Iota: each element is its index along axis [axis], an int64.

   A C-contiguous destination, as a fresh one is, runs as a job over its
   elements in C order: along it the index is constant for [inner]
   elements, the product of the extents past [axis], and counts up to the
   axis's extent and back to 0. Another layout runs on one thread, an
   odometer over its axes. */

#include <caml/mlvalues.h>

#include "cpu.h"

typedef struct {
  int64_t *base;
  int64_t numel, inner, ext;
} iota_job;

/* Elements [lo, hi) of a C-contiguous iota. */
static void iota_run(int64_t *d, int64_t lo, int64_t hi, int64_t inner,
                     int64_t ext) {
  int64_t i = lo, v = lo / inner % ext;
  if (inner == 1) {
    /* To the period's end, whole periods, then the rest. */
    int64_t head = (ext - v) % ext < hi - i ? (ext - v) % ext : hi - i;
    for (int64_t k = 0; k < head; k++) d[i + k] = v + k;
    i += head;
    for (; hi - i >= ext; i += ext)
      for (int64_t e = 0; e < ext; e++) d[i + e] = e;
    for (int64_t e = 0; i < hi; i++, e++) d[i] = e;
    return;
  }
  int64_t r = lo % inner;
  while (i < hi) {
    int64_t n = inner - r < hi - i ? inner - r : hi - i;
    for (int64_t k = 0; k < n; k++) d[i + k] = v;
    i += n;
    r = 0;
    if (++v == ext) v = 0;
  }
}

/* An iota of [a], of any layout, along [axis]. */
static void iota_odometer(const nx_array *a, int axis) {
  int r = a->rank;
  int64_t idx[NX_MAX_RANK] = {0}, n = a->dim[r - 1], s = a->dim[2 * r - 1];
  for (;;) {
    int64_t p = a->offset;
    for (int i = 0; i < r - 1; i++) p += idx[i] * a->dim[r + i];
    int64_t *d = (int64_t *)a->base + p;
    if (axis == r - 1)
      for (int64_t i = 0; i < n; i++) d[i * s] = i;
    else
      for (int64_t i = 0; i < n; i++) d[i * s] = idx[axis];
    int i = r - 2;
    while (i >= 0 && ++idx[i] == a->dim[i]) idx[i--] = 0;
    if (i < 0) return;
  }
}

/* Elements of an iota's unit of work. */
#define IOTA_UNIT (16 * 1024)

/* What an odometer element takes at most (cpu.h, Calls, scalar): the
   same loop at float32 took up to 124 on the M1. */
#define ODOMETER_COST 2048

/* Units [lo, hi) of the job: the last one ends at the last element. */
static void iota_body(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const iota_job *j = ctx;
  int64_t end = hi * IOTA_UNIT < j->numel ? hi * IOTA_UNIT : j->numel;
  iota_run(j->base, lo * IOTA_UNIT, end, j->inner, j->ext);
}

/* The external, into an int64 destination: nx_cpu.ml's map calls it for
   a program of one Coord. */
value nx_cpu_iota(value vaxis, value vd) {
  int axis = Int_val(vaxis);
  nx_operand in[1] = {{vd, NX_INT64, 1}};
  nx_array a[1];
  int e = nx_read(1, in, a);
  if (e) return Val_int(e);
  if (axis < 0 || axis >= a->rank) {
    nx_done(1, a);
    return Val_int(NX_SHAPE);
  }
  int64_t numel = 1, inner = 1;
  for (int i = 0; i < a->rank; i++) numel *= a->dim[i];
  for (int i = axis + 1; i < a->rank; i++) inner *= a->dim[i];
  int job = a->flags & NX_CONTIGUOUS;
  int released = nx_cpu_release((double)numel * (job ? 8 : ODOMETER_COST));
  if (numel > 0 && job) {
    iota_job j = {(int64_t *)a->base + a->offset, numel, inner, a->dim[axis]};
    int64_t units = (numel + IOTA_UNIT - 1) / IOTA_UNIT;
    nx_cpu_job(units, numel * 8, numel * 8, iota_body, &j);
  } else if (numel > 0)
    iota_odometer(a, axis);
  nx_cpu_reacquire(released);
  nx_done(1, a);
  return Val_int(NX_OK);
}
