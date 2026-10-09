/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Iota: each element is its index along axis [axis], cast to the dtype.

   A C-contiguous destination, as a fresh one is, runs as a job over its
   elements in C order: along it the index is constant for [inner]
   elements, the product of the extents past [axis], and counts up to the
   axis's extent and back to 0. Another layout runs on one thread, an
   odometer over its axes. */

#include <caml/mlvalues.h>

#include "cpu.h"

typedef struct {
  uint8_t *base;
  int64_t numel, inner, ext;
  void (*run)(void *d, int64_t lo, int64_t hi, int64_t inner, int64_t ext);
} iota_job;

#define IOTA(D, T)                                                           \
  static void iota_run_##D(void *d_, int64_t lo, int64_t hi, int64_t inner,  \
                           int64_t ext) {                                    \
    T *d = d_;                                                               \
    int64_t i = lo, v = lo / inner % ext;                                    \
    if (inner == 1) {                                                        \
      /* To the period's end, whole periods, then the rest. */               \
      int64_t head = (ext - v) % ext < hi - i ? (ext - v) % ext : hi - i;    \
      for (int64_t k = 0; k < head; k++) d[i + k] = (T)(v + k);              \
      i += head;                                                             \
      for (; hi - i >= ext; i += ext)                                        \
        for (int64_t e = 0; e < ext; e++) d[i + e] = (T)e;                   \
      for (int64_t e = 0; i < hi; i++, e++) d[i] = (T)e;                     \
      return;                                                                \
    }                                                                        \
    int64_t r = lo % inner;                                                  \
    while (i < hi) {                                                         \
      int64_t n = inner - r < hi - i ? inner - r : hi - i;                   \
      T x = (T)v;                                                            \
      for (int64_t k = 0; k < n; k++) d[i + k] = x;                          \
      i += n;                                                                \
      r = 0;                                                                 \
      if (++v == ext) v = 0;                                                 \
    }                                                                        \
  }                                                                          \
                                                                             \
  static void iota_##D(const nx_array *a, int axis) {                        \
    int r = a->rank;                                                         \
    int64_t idx[NX_MAX_RANK] = {0}, n = a->dim[r - 1], s = a->dim[2 * r - 1]; \
    for (;;) {                                                               \
      int64_t p = a->offset;                                                 \
      for (int i = 0; i < r - 1; i++) p += idx[i] * a->dim[r + i];           \
      T *d = (T *)a->base + p;                                               \
      if (axis == r - 1)                                                     \
        for (int64_t i = 0; i < n; i++) d[i * s] = (T)i;                     \
      else                                                                   \
        for (int64_t i = 0; i < n; i++) d[i * s] = (T)idx[axis];             \
      int i = r - 2;                                                         \
      while (i >= 0 && ++idx[i] == a->dim[i]) idx[i--] = 0;                  \
      if (i < 0) return;                                                     \
    }                                                                        \
  }

IOTA(f32, float)
IOTA(f64, double)
IOTA(i8, int8_t)
IOTA(i16, int16_t)
IOTA(i32, int32_t)
IOTA(i64, int64_t)
IOTA(u8, uint8_t)
IOTA(u16, uint16_t)
IOTA(u32, uint32_t)
IOTA(u64, uint64_t)

typedef struct {
  void (*run)(void *d, int64_t lo, int64_t hi, int64_t inner, int64_t ext);
  void (*odometer)(const nx_array *a, int axis);
} iota_fns;

#define IOTA_FNS(D) {iota_run_##D, iota_##D}

static const iota_fns iotas[NX_DTYPE_COUNT] = {
    [NX_FLOAT32] = IOTA_FNS(f32), [NX_FLOAT64] = IOTA_FNS(f64),
    [NX_INT8] = IOTA_FNS(i8),     [NX_INT16] = IOTA_FNS(i16),
    [NX_INT32] = IOTA_FNS(i32),   [NX_INT64] = IOTA_FNS(i64),
    [NX_UINT8] = IOTA_FNS(u8),    [NX_UINT16] = IOTA_FNS(u16),
    [NX_UINT32] = IOTA_FNS(u32),  [NX_UINT64] = IOTA_FNS(u64),
};

/* Elements of an iota's unit of work. */
#define IOTA_UNIT (16 * 1024)

/* Units [lo, hi) of the job: the last one ends at the last element. */
static void iota_body(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const iota_job *j = ctx;
  int64_t end = hi * IOTA_UNIT < j->numel ? hi * IOTA_UNIT : j->numel;
  j->run(j->base, lo * IOTA_UNIT, end, j->inner, j->ext);
}

value nx_cpu_iota(value vaxis, value vd) {
  int d = nx_array_dtype(vd), axis = Int_val(vaxis);
  const iota_fns *f = &iotas[d];
  if (f->run == NULL) return Val_int(NX_DECLINED);
  nx_operand in[1] = {{vd, d, 1}};
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
  if (numel > 0 && (a->flags & NX_CONTIGUOUS)) {
    int w = a->bits / 8;
    iota_job j = {a->base + a->offset * w, numel, inner, a->dim[axis], f->run};
    int64_t units = (numel + IOTA_UNIT - 1) / IOTA_UNIT;
    nx_cpu_job(units, numel * w, numel * w, iota_body, &j);
  } else if (numel > 0)
    f->odometer(a, axis);
  nx_done(1, a);
  return Val_int(NX_OK);
}
