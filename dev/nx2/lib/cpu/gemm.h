/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What gemm.c, the core contraction, shares with the derived
   contractions of their own files: complex numbers (gemm_complex.c),
   8-bit quads (gemm_quads.c), groups of several runs (gemm_runs.c),
   results of another dtype (gemm_rounded.c), scales (gemm_scales.c) and
   segments (gemm_segments.c). A file includes it after the C library's
   headers. */

#ifndef NX_CPU_GEMM_H
#define NX_CPU_GEMM_H

#include <stdatomic.h>
#include <stdlib.h>

#include <caml/mlvalues.h>

#include "cpu.h"

/* Lane order's dots take DOT_COLUMNS columns of b, as quads' dots do. */
#define DOT_COLUMNS 16

/* Units of a compute job per performance core, so that cores that run
   slower take fewer. */
#define UNITS 4

/* The cost of [flops] on the kernel [k]. Flops multiply the extents of
   several operands, so they and the costs are doubles; storage, within one
   operand's extents rounded to a tile, fits in an int64_t (NX_MAX_NUMEL). */
static inline double flop_cost(nx_cpu_micro k, double flops) {
  return flops / k.flops_per_byte;
}

/* The cost a thread of a job on [g]'s kernels brings to pay for the job's
   start (nx_cpu_job_at). */
static inline double thread_bytes(const nx_cpu_gemm *g) {
  return NX_CPU_THREAD_BYTES + g->start;
}

/* The units a compute job of [bytes] and [cost], one thread per [per] of
   it, splits into: UNITS a thread, or one where it runs on one thread and
   a split buys nothing. */
static inline int64_t units(int64_t bytes, double cost, double per) {
  int64_t threads = nx_cpu_threads_at(bytes, cost, per);
  return threads > 1 ? UNITS * threads : 1;
}

/* Operands and axes, as the view numbers them. */
enum { A = NX_VIEW_A, B = NX_VIEW_B, INIT = NX_VIEW_INIT, DST = NX_VIEW_DST };
enum {
  BATCH = NX_VIEW_BATCH,
  ROW = NX_VIEW_ROW,
  COL = NX_VIEW_COLUMN,
  CON = NX_VIEW_CONTRACTED
};

/* An operand's block scales: [v] holds them in acc, C-contiguous, and
   t[x][i] is the position in [v] of the block of index i along axis x, the
   sum over the axes of x's group of each index's block times the scales'
   step. t[x] is NULL along an axis the operand lacks, [v] NULL for an
   unscaled operand. The scale is one along k over each run of [run]
   indices from a multiple of it: the ratio along the operand's last
   contracted axis. No pass moves t[CON], so the runs stay aligned. */
typedef struct {
  const uint8_t *v;
  const int64_t *t[4];
  int64_t run;
} scale;

/* An operand's group of several runs, which its packs read (nx_gemm_pack_at):
   [n] runs in C order, run r of extent[r] indices stepping stride[r]. */
typedef struct {
  int n;
  int64_t extent[NX_MAX_RANK], stride[NX_MAX_RANK];
} spans;

/* a's and b's spans by group: of[o][x].n is at most 1 along a group of one
   run. */
typedef struct {
  spans of[2][4];
} reads;

/* An operand as the view lays it out: its element (e, i, j, k) lies at
   position first + e·st[BATCH] + i·st[ROW] + j·st[COL] + k·st[CON], and
   an axis it lacks has step 0; along an axis of several runs, sp[x] not
   NULL, the runs of index skip[x] + i step instead (along), so that lane
   order's passes move on by rows as [first] moves the others. */
typedef struct {
  const nx_array *x; /* NULL for an absent init */
  int64_t first;
  int64_t st[4];
  const spans *sp[4];
  int64_t skip[4];
  scale *s; /* NULL for an unscaled operand */
} operand;

typedef struct {
  operand op[4]; /* by view operand */
  int64_t ext[4];
  int acc, w;
  const nx_cpu_gemm *g;
  int init_is_dst;   /* init is dst itself: R starts as it is */
  int a_in, b_in;    /* chain order reads a, b in place */
  int keeps_a;       /* fewer columns than rows, a read in place, not
                        transposed (nx_cpu_contract_on) */
  int alone;         /* runs on the calling thread, in one state of
                        g's hardware (runs_alone) */
  atomic_int failed; /* an allocation failed */
} problem;

static inline int64_t ceil_div(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}
static inline int64_t gcd(int64_t a, int64_t b) {
  return b == 0 ? a : gcd(b, a % b);
}
static inline int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

/* The position from operand [q]'s first element of index [i] along its
   axis [x]. */
static inline int64_t along(const operand *q, int x, int64_t i) {
  const spans *s = q->sp[x];
  if (s == NULL) return i * q->st[x];
  int64_t at = 0;
  i += q->skip[x];
  for (int r = s->n - 1; r >= 0; r--) {
    at += i % s->extent[r] * s->stride[r];
    i /= s->extent[r];
  }
  return at;
}

/* The position of operand [o]'s element (e, i, j, k). */
static inline int64_t pos(const problem *p, int o, int64_t e, int64_t i,
                          int64_t j, int64_t k) {
  const operand *q = &p->op[o];
  return q->first + along(q, BATCH, e) + along(q, ROW, i) +
         along(q, COL, j) + along(q, CON, k);
}

/* Packed operands start on 128 bytes: the matrix unit loads 128 at once. */
#define ALIGN 128

static inline void *alloc(int64_t bytes) {
  return aligned_alloc(ALIGN,
                       (size_t)ceil_div(bytes > 0 ? bytes : 1, ALIGN) * ALIGN);
}

/* The calling worker's scratch of [bytes], given back by nx_cpu_give. */
static inline uint8_t *scratch(problem *p, int64_t bytes) {
  uint8_t *s = nx_cpu_take(bytes);
  if (s == NULL) atomic_store(&p->failed, 1);
  return s;
}

/* Dtypes */

/* The dtype of [acc]'s parts: float32 or float64 for a complex [acc],
   [acc] itself otherwise. */
static inline int part_of(int acc) {
  if (acc == NX_COMPLEX64) return NX_FLOAT32;
  return acc == NX_COMPLEX128 ? NX_FLOAT64 : acc;
}

/* The dtype a sum in [acc] is computed in: int8 and uint8 for int4 and
   uint4, whose sums modulo 2^8 are theirs modulo 2^4 once narrowed, as
   the int8 kernels' are; [acc] itself otherwise. */
static inline int summed_in(int acc) {
  if (acc == NX_INT4) return NX_INT8;
  return acc == NX_UINT4 ? NX_UINT8 : acc;
}

/* Stages operand [o]'s block of [n1] rows of [n0] elements, the first at
   position [at], stepping [s0] along a row and [s1] across rows, into [dst]
   in acc: row j's element i at dst + (j·pitch + i)·w. */
static inline void stage_acc(const problem *p, int o, int64_t at, int64_t s0,
                             int64_t s1, int64_t n0, int64_t n1, uint8_t *dst,
                             int64_t pitch) {
  nx_cpu_stage_as(p->op[o].x, at, s0, s1, n0, n1, p->acc, dst, pitch);
}

/* R's element (e, i, j): dst's, in acc. */
static inline uint8_t *at_r(const problem *p, int64_t e, int64_t i, int64_t j) {
  return p->op[DST].x->base + pos(p, DST, e, i, j, 0) * p->w;
}

/* Operand [q]'s step along axis [x] within a run: its last run's along
   several. */
static inline int64_t step_of(const operand *q, int x) {
  const spans *s = q->sp[x];
  return s ? s->stride[s->n - 1] : q->st[x];
}

/* The indices from [i] along operand [q]'s axis [x], at most [n], that one
   step reaches, the step into [*st]: the rest of its last run. */
static inline int64_t piece(const operand *q, int x, int64_t i, int64_t n,
                            int64_t *st) {
  const spans *s = q->sp[x];
  *st = step_of(q, x);
  if (s == NULL) return n;
  int64_t last = s->extent[s->n - 1];
  return min64(n, last - (q->skip[x] + i) % last);
}

/* Each product's set-up in a call's price (nx_gemm_call_cost, gemm.c). */
#define PRODUCT_COST (16 * 1024)

/* A scaled side's facts held in OCaml values, read before nx_read,
   which may run the collector: whether its scale is codes, and its groups'
   axes, batch, rows or columns, contraction (group_axes). */
typedef struct {
  int exponent, n[3], ax[3][NX_MAX_RANK];
} scale_axes;

/* A scaled operand's scales decoded into acc at [mem], C-contiguous, with
   room at [t] for its tables at its extents, and along each axis the ratio
   of the extents and the scales' step. */
typedef struct {
  int64_t *mem, *t;
  int64_t ratio[NX_MAX_RANK], step[NX_MAX_RANK];
} decoded;

/* gemm.c */

void nx_gemm_pack_at(const problem *p, nx_cpu_pack move, int o,
                     const int64_t *c, int x0, int x1, int64_t n0, int64_t n1,
                     uint8_t *d, int64_t pitch);
void nx_gemm_transpose(problem *p);
void nx_gemm_lay_out(problem *p, const nx_contract_view *v,
                     const nx_array *const *op);
int nx_gemm_contract_with(const nx_cpu_gemm *g, int acc, int order,
                          const nx_contract_view *v, const nx_array *const *op,
                          const scale *s, const reads *sp);
int nx_gemm_contract_view(int acc, int order, int da, int db,
                          const nx_contract_view *v, const nx_array *const *op,
                          const scale *s, const reads *sp);
int nx_gemm_spanned(int acc, int da, int db, const scale *s);
double nx_gemm_call_cost(int acc, int order, const double *x);
int nx_gemm_order_of(int acc, const double *x);

/* gemm_complex.c */

int nx_gemm_complex_on(int acc, int order, const nx_contract_view *v,
                       const nx_array *const *op, const reads *sp);

/* gemm_quads.c */

const nx_cpu_gemm *nx_gemm_quads_for(int acc, int da, int db);
int nx_gemm_quads_on(const nx_cpu_gemm *g, int acc, const nx_contract_view *v,
                     const nx_array *const *op);

/* gemm_runs.c */

int nx_gemm_stage_runs(nx_contract_runs *r, int o, const int *gs, int dt,
                       const nx_array *src, const scale *s, nx_array *x);
int nx_gemm_runs_on(int acc, int order, int da, int db, nx_contract_runs *r,
                    const nx_array *const *op, const scale *s);

/* gemm_rounded.c */

void nx_gemm_runs_of_view(nx_contract_runs *r, const nx_contract_view *v);
int nx_gemm_rounded_on(int acc, int sum, int order, nx_contract_runs *r,
                       const nx_array *const *op, const scale *s);

/* gemm_scales.c */

void nx_gemm_axes_of(value vs, value vx, value vbatch, value vcon, int o,
                     scale_axes *g);
int nx_gemm_decoded_of(int acc, int r, const int64_t *x, const nx_array *sc,
                       const scale_axes *g, decoded *d);
void nx_gemm_scale_at(const decoded *d, const scale_axes *g, int o,
                      const int64_t *x, const int64_t *start, scale *s);
int nx_gemm_scale_of(int acc, const nx_array *x, const nx_array *sc,
                     const scale_axes *g, int o, scale *s, uint8_t **mem);
int nx_gemm_scales_fit(value va, value vb, const value *vs);
double nx_gemm_decode_cost(int acc, const nx_array *sc);

/* gemm_segments.c */

value nx_gemm_contract_segmented(
    value vbatch, value vcontracting, value vacc, value vdst, value vi,
    value va, value vb, value vsa, value vsb, value vseg);

/* The contraction of the view [v], or of the runs [r] where [runs], over
   [op] in [acc], a of dtype dts[0] and b of dts[1], into dst of dtype
   [out], a and b scaled by [s] where it is not NULL: 0 if an allocation
   failed. */
static inline __attribute__((always_inline)) int computed(
    int acc, int order, int out, const int *dts, int runs,
    nx_contract_view *v, nx_contract_runs *r, const nx_array *const *op,
    const scale *s) {
  int sum = summed_in(acc);
  if (out != acc || sum != acc) {
    if (!runs) nx_gemm_runs_of_view(r, v);
    return nx_gemm_rounded_on(acc, sum, order, r, op, s);
  }
  if (runs) return nx_gemm_runs_on(acc, order, dts[0], dts[1], r, op, s);
  return nx_gemm_contract_view(acc, order, dts[0], dts[1], v, op, s, NULL);
}

#endif
