/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Scales

   A scaled operand's scales are decoded into acc once a call, C-contiguous,
   and each of its groups of axes takes a table from an index to its block's
   position among them (scale): the scales' count in acc, and O(B + M + K)
   or O(B + N + K) positions. */

#include <string.h>

#include "gemm.h"

/* Whether axis [d] of side [o], 0 for a and 1 for b, is in a pair of
   [ps]. */
static int in_pairs(value ps, int o, int d) {
  for (mlsize_t k = 0; k < Wosize_val(ps); k++)
    if (Long_val(Field(Field(ps, k), o)) == d) return 1;
  return 0;
}

/* The axes of side [o], of rank [rank], in group [g] into [ax], and their
   count: the pairs' in pair order for the batch and the contraction, the
   free ones in axis order for the rows or the columns. */
static int group_axes(value vbatch, value vcon, int o, int rank, int g,
                      int *ax) {
  if (g == BATCH || g == CON) {
    value ps = g == BATCH ? vbatch : vcon;
    int n = (int)Wosize_val(ps);
    for (int k = 0; k < n; k++) ax[k] = (int)Long_val(Field(Field(ps, k), o));
    return n;
  }
  int n = 0;
  for (int d = 0; d < rank; d++)
    if (!in_pairs(vbatch, o, d) && !in_pairs(vcon, o, d)) ax[n++] = d;
  return n;
}

/* A decode of scales (decode): [sc] into [v] in [acc], [rows] rows of
   [n0] a plane, [per] rows a unit. */
typedef struct {
  const nx_array *sc;
  int exponent, acc;
  uint8_t *v;
  int64_t n0, rows, per, chunks;
} decoding;

/* The E8M0 scale of OCP's Microscaling Formats (MX) v1.0: code e is
   2^(e - E8M0_BIAS), and E8M0_NAN is NaN. */
#define E8M0_BIAS 127
#define E8M0_NAN 255

/* IEEE 754's quiet NaNs, binary32's and binary64's; binary32's 2^-127, a
   subnormal, the significand's top bit; and the exponents' biases and the
   significands' widths. */
#define F32_QNAN 0x7FC00000u
#define F64_QNAN 0x7FF8000000000000u
#define F32_TINY 0x00400000u
#define F32_BIAS 127
#define F32_MANT 23
#define F64_BIAS 1023
#define F64_MANT 52

/* The scale of code [e], 2^(e - 127), NaN for 255, by its bits: 0 is
   float32's subnormal 2^-127. */
static float code32(int e) {
  uint32_t b = e == E8M0_NAN ? F32_QNAN
               : e == 0      ? F32_TINY
                             : (uint32_t)(e - E8M0_BIAS + F32_BIAS) << F32_MANT;
  float f;
  memcpy(&f, &b, sizeof f);
  return f;
}

static double code64(int e) {
  uint64_t b = e == E8M0_NAN
                   ? F64_QNAN
                   : (uint64_t)(e + F64_BIAS - E8M0_BIAS) << F64_MANT;
  double f;
  memcpy(&f, &b, sizeof f);
  return f;
}

/* A unit is [per] rows of a plane, staged at once. */
static void decode_rows(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const decoding *f = ctx;
  const nx_array *sc = f->sc;
  int r = sc->rank, w = nx_cpu_width(f->acc);
  const int64_t *x = sc->dim, *st = sc->dim + r;
  int64_t s0 = r ? st[r - 1] : 0, s1 = r > 1 ? st[r - 2] : 0;
  for (int64_t u = lo; u < hi; u++) {
    int64_t plane = u / f->chunks, i0 = u % f->chunks * f->per;
    int64_t n1 = min64(f->per, f->rows - i0), at = sc->offset + i0 * s1;
    for (int64_t d = r - 3, q = plane; d >= 0; d--) {
      at += q % x[d] * st[d];
      q /= x[d];
    }
    int64_t first = (plane * f->rows + i0) * f->n0, n = n1 * f->n0;
    nx_cpu_stage_as(sc, at, s0, s1, f->n0, n1, f->acc, f->v + first * w,
                    f->n0);
    if (!f->exponent) continue;
    if (f->acc == NX_FLOAT32) {
      float *y = (float *)f->v + first;
      for (int64_t k = 0; k < n; k++) y[k] = code32((int)y[k]);
    } else {
      double *y = (double *)f->v + first;
      for (int64_t k = 0; k < n; k++) y[k] = code64((int)y[k]);
    }
  }
}

/* The scales [sc] into [v], C-contiguous in [acc]: a code e, where
   [exponent], as 2^(e - 127) and 255 as NaN; a value as a cast into acc
   gives it. Units of DECODE_ROW elements or more, each staged at once. */
#define DECODE_ROW 4096

static void decode(const nx_array *sc, int exponent, int acc, uint8_t *v) {
  int r = sc->rank;
  const int64_t *x = sc->dim;
  int64_t n0 = r ? x[r - 1] : 1, rows = r > 1 ? x[r - 2] : 1, planes = 1;
  for (int d = 0; d + 2 < r; d++) planes *= x[d];
  if (planes == 0 || rows == 0 || n0 == 0) return;
  int64_t per = min64(rows, ceil_div(DECODE_ROW, n0));
  decoding f = {sc, exponent, acc, v, n0, rows, per, ceil_div(rows, per)};
  int64_t bytes = planes * rows * n0 * nx_cpu_width(acc);
  nx_cpu_job(planes * f.chunks, bytes, nx_gemm_decode_cost(acc, sc),
             decode_rows, &f);
}

/* Fills [t] with a group's table over its [n] axes [ax]: each index of the
   group, in C order of the axes' extents [x], to the sum along each axis
   of the index's block, the index plus the axis's [start] divided by
   [ratio], times the scales' [step]. A group with an empty axis has no
   index and its table one entry, [0]: the axes before the empty one would
   write past it. */
static void table(int n, const int *ax, const int64_t *x, const int64_t *start,
                  const int64_t *ratio, const int64_t *step, int64_t *t) {
  int64_t len = 1;
  t[0] = 0;
  for (int a = 0; a < n; a++)
    if (x[ax[a]] == 0) return;
  for (int a = 0; a < n; a++) {
    int d = ax[a];
    int64_t e = x[d], s0 = start[d];
    for (int64_t q = len - 1; q >= 0; q--) {
      int64_t first = t[q];
      for (int64_t i = e - 1; i >= 0; i--)
        t[q * e + i] = first + (s0 + i) / ratio[d] * step[d];
    }
    len *= e;
  }
}

static const int groups[2][3] = {{BATCH, ROW, CON}, {BATCH, COL, CON}};

/* Side [o]'s [scale_axes], its scale [vs] an Nx_kernel.scale and its
   operand [vx]. Out of line, as nx_gemm_scales_fit. */
__attribute__((noinline)) void nx_gemm_axes_of(value vs, value vx,
                                               value vbatch, value vcon, int o,
                                               scale_axes *g) {
  int64_t dim[2 * NX_MAX_RANK], off;
  int rank = nx_array_layout(vx, dim, &off);
  g->exponent = Tag_val(vs) == 0;
  for (int t = 0; t < 3; t++)
    g->n[t] = group_axes(vbatch, vcon, o, rank, groups[o][t], g->ax[t]);
}

/* The scales [sc] of an operand of rank [r] and extents [x], its axes
   [g], decoded into [d]: 0 if its memory could not be allocated. */
int nx_gemm_decoded_of(int acc, int r, const int64_t *x, const nx_array *sc,
                       const scale_axes *g, decoded *d) {
  int64_t count = 1, positions = 0;
  for (int k = r - 1; k >= 0; k--) {
    d->step[k] = count;
    count *= sc->dim[k];
    d->ratio[k] = sc->dim[k] ? x[k] / sc->dim[k] : 1;
  }
  for (int t = 0; t < 3; t++) {
    int64_t len = 1;
    for (int k = 0; k < g->n[t]; k++) len *= x[g->ax[t][k]];
    positions += len > 0 ? len : 1;
  }
  int64_t values = ceil_div(count * nx_cpu_width(acc), 8);
  d->mem = alloc((values + positions) * 8);
  if (d->mem == NULL) return 0;
  d->t = d->mem + values;
  decode(sc, g->exponent, acc, (uint8_t *)d->mem);
  return 1;
}

/* [s], side [o]'s scale over the view of extents [x] whose index starts at
   [start] along each axis of the operand, from [d]: its tables at d->t.
   The scale is one along k over runs of the ratio along its last
   contracted axis, cut where the view starts or ends within a block. */
void nx_gemm_scale_at(const decoded *d, const scale_axes *g, int o,
                      const int64_t *x, const int64_t *start, scale *s) {
  *s = (scale){.v = (const uint8_t *)d->mem, .run = 1};
  int nk = g->n[2];
  if (nk) {
    int k = g->ax[2][nk - 1];
    s->run = gcd(gcd(d->ratio[k], start[k]), x[k]);
  }
  int64_t *t = d->t;
  for (int k = 0; k < 3; k++) {
    int64_t len = 1;
    for (int a = 0; a < g->n[k]; a++) len *= x[g->ax[k][a]];
    table(g->n[k], g->ax[k], x, start, d->ratio, d->step, t);
    s->t[groups[o][k]] = t;
    t += len > 0 ? len : 1;
  }
}

static const int64_t no_start[NX_MAX_RANK];

/* The scales [sc] of operand [x], side [o], its axes [g], into [s], its
   memory, to free, at [*mem]: 0 if it could not be allocated. Out of
   line, as nx_gemm_scales_fit. */
__attribute__((noinline)) int nx_gemm_scale_of(int acc, const nx_array *x,
                                               const nx_array *sc,
                                               const scale_axes *g, int o,
                                               scale *s, uint8_t **mem) {
  decoded d;
  if (!nx_gemm_decoded_of(acc, x->rank, x->dim, sc, g, &d)) return 0;
  *mem = (uint8_t *)d.mem;
  nx_gemm_scale_at(&d, g, o, x->dim, no_start, s);
  return 1;
}

/* Whether each scale of [vs], options of Nx_kernel.scale, fits its
   operand of [va] and [vb]: of its rank, each extent dividing the
   operand's, 0 only along an empty axis. Out of line: its layouts stay off
   the external's frame. */
__attribute__((noinline)) int nx_gemm_scales_fit(value va, value vb,
                                                 const value *vs) {
  int64_t x[2 * NX_MAX_RANK], dim[2 * NX_MAX_RANK], off;
  for (int o = A; o <= B; o++) {
    if (!Is_block(vs[o])) continue;
    /* Exponent and Value hold their array in field 0. */
    int r = nx_array_layout(o == A ? va : vb, x, &off);
    if (nx_array_layout(Field(Field(vs[o], 0), 0), dim, &off) != r) return 0;
    for (int d = 0; d < r; d++)
      if (dim[d] == 0 ? x[d] != 0 : x[d] % dim[d] != 0) return 0;
  }
  return 1;
}

/* The cost of decoding the scales [sc] into [acc] (decode), a strided
   copy: the call's part and its job's. */
double nx_gemm_decode_cost(int acc, const nx_array *sc) {
  return nx_cpu_elements(sc) * nx_cpu_width(acc) * NX_CPU_STRIDED_COST;
}
