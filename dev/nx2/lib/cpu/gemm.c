/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contractions: y[b, i, j] = out (init[b, i, j] + Σ_k a[b, i, k] · b[b, k, j]),
   computed in acc.

   Nx_kernel.Spec.Contract_view has grouped the axes into one batch, row (M),
   column (N) and contracted (K) axis per operand. Every operand is read
   through the stage, which takes any strides and dtype into its carrier, and
   the target's conversion takes the carrier into acc: a dtype is computed
   only where acc holds each of its values, so operands convert exactly.

   The order of each output's sum is a function of M, N and K alone:

   - Chain order, M·N >= CHAIN_OUTPUTS: one fused multiply-add chain per
     output, in increasing k, from init or +0. The microkernels of every
     target add each product fused, and the blocking stores and reloads the
     sum in acc exactly, so tile shapes, block sizes and threads change no
     bit.
   - Lane order, M·N < CHAIN_OUTPUTS: the products fall into blocks of
     FOLD_BLOCK consecutive k and, within a block, into LANES lanes by k
     modulo LANES, each lane a fused chain from +0; a fixed balanced tree
     sums the lanes, the left-complete binary tree the blocks, then init is
     added. A single chain runs at one fused add per latency; CHAIN_OUTPUTS
     outputs or more keep every target's pipes full in chain order.

   Chain order follows BLIS's loops (Goto and van de Geijn, "Anatomy of
   High-Performance Matrix Multiplication", 2008): per NC columns of b and
   KC of the contraction, one job packs b's panel into NR-wide slivers,
   then a job computes MC-row blocks of a, each packed by its worker into
   MR-wide slivers, against ranges of b's slivers, MR × NR at a time:

     for each group of batch elements, NC panel, KC block:
       job 1: b[kc × nc] -> slivers kc × NR, shared
       job 2: per (element, MC block, range of slivers):
                a[mc × kc] -> slivers MR × kc, the worker's own
                for each sliver of b, each sliver of a: R tile += a · b

   A product of at most MC rows, as decoding a token is, packs a whole and
   then streams b: each unit packs a sliver of b and adds it in, so that b
   is read once and no panel passes between jobs. At most 4 rows run on a
   thin kernel of 1, 2 or 4 rows. One of at most MC columns, and fewer
   columns than rows, is computed as its transpose.

   R, the sum in acc, is dst itself: a dtype is computed with out = acc.
   Each tile starts as init, or +0, when its first KC block is computed. */

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/mlvalues.h>

#include "cpu.h"
#include "nx_spec.h"

/* Outputs per batch element from which chain order fills the pipes:
   NEON keeps 16 vectors of 4 chains in flight (4 pipes, 4 cycles), AVX2 8
   of 8. */
#define CHAIN_OUTPUTS 64

/* The bytes of a job's packed b: a group of batch elements shares one job
   while their panels fit, so that small batched products start few
   jobs. */
#define PANELS (8 * 1024 * 1024)

/* Units of a compute job per performance core, so that cores that run
   slower take fewer. */
#define UNITS 4

/* Operands and axes, as the view numbers them. */
enum { A = NX_VIEW_A, B = NX_VIEW_B, INIT = NX_VIEW_INIT, DST = NX_VIEW_DST };
enum {
  BATCH = NX_VIEW_BATCH,
  ROW = NX_VIEW_ROW,
  COL = NX_VIEW_COLUMN,
  CON = NX_VIEW_CONTRACTED
};

/* An operand as the view lays it out: its element (e, i, j, k) lies at
   position first + e·st[BATCH] + i·st[ROW] + j·st[COL] + k·st[CON], and
   an axis it lacks has step 0. */
typedef struct {
  const nx_array *x; /* NULL for an absent init */
  int64_t first;
  int64_t st[4];
} operand;

typedef struct {
  operand op[4]; /* by view operand */
  int64_t ext[4];
  int acc, w;
  const nx_cpu_gemm *g;
  int init_is_dst;   /* init is dst itself: R starts as it is */
  void **scratch;    /* per worker, allocated by the worker */
  atomic_int failed; /* an allocation failed */
} problem;

static int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
static int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

/* The position of operand [o]'s element (e, i, j, k). */
static int64_t pos(const problem *p, int o, int64_t e, int64_t i, int64_t j,
                   int64_t k) {
  const int64_t *s = p->op[o].st;
  return p->op[o].first + e * s[BATCH] + i * s[ROW] + j * s[COL] +
         k * s[CON];
}

static void *alloc(int64_t bytes) {
  return aligned_alloc(64, (size_t)ceil_div(bytes > 0 ? bytes : 1, 64) * 64);
}

/* The worker's scratch of [bytes], allocated at its first unit. */
static uint8_t *scratch(problem *p, int worker, int64_t bytes) {
  if (p->scratch[worker] == NULL) p->scratch[worker] = alloc(bytes);
  if (p->scratch[worker] == NULL) atomic_store(&p->failed, 1);
  return p->scratch[worker];
}

/* Dtypes */

/* Whether every value of [d] is one of [acc]'s, float32 or float64, as
   nx_cpu.mli lists them: the booleans, the floats of at most [acc]'s
   width, and the integers of at most half of it. */
static int holds(int acc, int d) {
  nx_dtype_row r = nx_dtype_row_of(d);
  int width = nx_dtype_row_of(acc).bits;
  switch (r.kind) {
    case NX_KIND_BOOLEAN: return 1;
    case NX_KIND_FLOAT: return r.bits <= width;
    case NX_KIND_SIGNED:
    case NX_KIND_UNSIGNED: return r.bits <= width / 2;
    default: return 0;
  }
}

/* Whether nx.cpu computes a contraction in [acc] into [out] of the [n]
   operands of dtypes [dts]. */
static int computes(int acc, int out, const int *dts, int n) {
  if (out != acc || nx_cpu_table->gemm[acc].kernel.f == NULL) return 0;
  for (int i = 0; i < n; i++)
    if (!holds(acc, dts[i])) return 0;
  return 1;
}

/* Staging */

/* Stages operand [o]'s block of [n1] rows of [n0] elements, the first at
   position [at], stepping [s0] along a row and [s1] across rows, into [dst]
   in acc: row j's element i at dst + (j·pitch + i)·w. */
static void stage_acc(const problem *p, int o, int64_t at, int64_t s0,
                      int64_t s1, int64_t n0, int64_t n1, uint8_t *dst,
                      int64_t pitch) {
  nx_cpu_stage_as(p->op[o].x, at, s0, s1, n0, n1, p->acc, dst, pitch);
}

/* R's element (e, i, j): dst's, in acc. */
static uint8_t *at_r(const problem *p, int64_t e, int64_t i, int64_t j) {
  return p->op[DST].x->base + pos(p, DST, e, i, j, 0) * p->w;
}

/* Sets R's rows [i0, i1) × columns [j0, j1) of element [e] to init or +0.
   An init that is dst itself is already there: copying it onto itself would
   be a memcpy whose source and destination overlap. */
static void start(const problem *p, int64_t e, int64_t i0, int64_t i1,
                  int64_t j0, int64_t j1) {
  const int64_t *s = p->op[DST].st;
  if (p->init_is_dst) return;
  for (int64_t i = i0; i < i1; i++) {
    if (p->op[INIT].x) {
      const int64_t *t = p->op[INIT].st;
      int64_t at = pos(p, INIT, e, i, j0, 0), n = j1 - j0;
      uint8_t *r = at_r(p, e, i, j0);
      if (s[COL] == 1 || n == 1)
        stage_acc(p, INIT, at, t[COL], 0, n, 1, r, 0);
      else
        stage_acc(p, INIT, at, 1, t[COL], 1, n, r, s[COL]);
    } else if (s[COL] == 1 || j1 - j0 == 1) {
      memset(at_r(p, e, i, j0), 0, (size_t)((j1 - j0) * p->w));
    } else {
      for (int64_t j = j0; j < j1; j++) memset(at_r(p, e, i, j), 0, p->w);
    }
  }
}

/* Chain order */

typedef struct {
  problem *p;
  int64_t e0, ne;      /* the group's batch elements */
  int64_t jc, nc, pc, kc;
  int64_t slivers;     /* of b's panel */
  int64_t mblocks, ranges, per_range;
  uint8_t *b;          /* the packed panel */
} panel;

/* Sliver [s] of element [e]'s panel, packed: kc steps of NR. */
static uint8_t *sliver(const panel *c, int64_t e, int64_t s) {
  int64_t nr = c->p->g->kernel.nr;
  return c->b + ((e - c->e0) * c->slivers + s) * nr * c->kc * c->p->w;
}

/* Packs [a]'s rows [i, i + m) of element [e], along k from
   [pc] for [kc], into [d]: kc steps of [lda] elements, zero past m. A
   block of rows is one stage: its rows transpose in square blocks, where
   a sliver as narrow as a microkernel's 6 rows would move element by
   element. */
static void pack_a(const problem *p, int64_t e, int64_t i, int64_t m,
                   int64_t lda, int64_t pc, int64_t kc, uint8_t *d) {
  const int64_t *s = p->op[A].st;
  for (int64_t q = 0; m < lda && q < kc; q++)
    memset(d + (q * lda + m) * p->w, 0, (size_t)((lda - m) * p->w));
  stage_acc(p, A, pos(p, A, e, i, 0, pc), s[ROW], s[CON], m, kc, d, lda);
}

/* Packs the sliver of [b]'s columns [j, j + n) likewise, kc steps of
   [nr]. */
static void pack_b(const problem *p, int64_t e, int64_t j, int64_t n, int nr,
                   int64_t pc, int64_t kc, uint8_t *d) {
  const int64_t *s = p->op[B].st;
  if (n < nr) memset(d, 0, (size_t)(nr * kc * p->w));
  stage_acc(p, B, pos(p, B, e, 0, j, pc), s[COL], s[CON], n, kc, d, nr);
}

static void pack_panel(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const panel *c = ctx;
  int nr = c->p->g->kernel.nr;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = c->e0 + u / c->slivers, v = u % c->slivers;
    int64_t j = c->jc + v * nr;
    pack_b(c->p, e, j, min64(nr, c->jc + c->nc - j), nr, c->pc, c->kc,
           sliver(c, e, v));
  }
}

/* Adds the products of the packed [a], its steps [lda] apart, and [b] to
   R's tile of [m] rows and [n] columns from (e, i, j) with [k]'s kernel: in
   place where its columns are adjacent and it is whole, else through a
   buffer. */
static void tile(const problem *p, nx_cpu_micro k, int64_t kc,
                 const uint8_t *a, int64_t lda, const uint8_t *b, int64_t e,
                 int64_t i, int64_t j, int64_t m, int64_t n) {
  const int64_t *s = p->op[DST].st;
  if (m == k.mr && n == k.nr && s[COL] == 1) {
    k.f(kc, a, lda, b, at_r(p, e, i, j), s[ROW]);
    return;
  }
  _Alignas(64) uint8_t t[NX_CPU_TILE];
  int w = p->w;
  memset(t, 0, (size_t)(k.mr * k.nr * w));
  for (int64_t r = 0; r < m; r++)
    if (s[COL] == 1)
      memcpy(t + r * k.nr * w, at_r(p, e, i + r, j), (size_t)(n * w));
    else
      for (int64_t q = 0; q < n; q++)
        memcpy(t + (r * k.nr + q) * w, at_r(p, e, i + r, j + q), w);
  k.f(kc, a, lda, b, t, k.nr);
  for (int64_t r = 0; r < m; r++)
    if (s[COL] == 1)
      memcpy(at_r(p, e, i + r, j), t + r * k.nr * w, (size_t)(n * w));
    else
      for (int64_t q = 0; q < n; q++)
        memcpy(at_r(p, e, i + r, j + q), t + (r * k.nr + q) * w, w);
}

static void compute(int64_t lo, int64_t hi, int worker, void *ctx) {
  const panel *c = ctx;
  problem *p = c->p;
  const nx_cpu_gemm *g = p->g;
  nx_cpu_micro k = g->kernel;
  int64_t m_all = p->ext[ROW], mr = k.mr, w = p->w;
  int64_t rows = min64(g->mc, ceil_div(m_all, mr) * mr);
  uint8_t *ap = scratch(p, worker, rows * min64(g->kc, p->ext[CON]) * w);
  if (ap == NULL) return;
  for (int64_t u = lo; u < hi; u++) {
    int64_t range = u % c->ranges, ib = u / c->ranges % c->mblocks;
    int64_t e = c->e0 + u / c->ranges / c->mblocks;
    int64_t i0 = ib * g->mc, mc = min64(g->mc, m_all - i0);
    int64_t v0 = range * c->per_range;
    int64_t v1 = min64(c->slivers, v0 + c->per_range);
    if (v0 >= v1) continue;
    int64_t j0 = c->jc + v0 * k.nr;
    int64_t j1 = min64(c->jc + c->nc, c->jc + v1 * k.nr);
    int64_t lda = ceil_div(mc, mr) * mr;
    pack_a(p, e, i0, mc, lda, c->pc, c->kc, ap);
    if (c->pc == 0) start(p, e, i0, i0 + mc, j0, j1);
    if (c->kc == 0) continue;
    for (int64_t v = v0; v < v1; v++) {
      int64_t j = c->jc + v * k.nr, n = min64(k.nr, c->jc + c->nc - j);
      const uint8_t *b = sliver(c, e, v);
      for (int64_t ir = 0; ir < mc; ir += mr)
        tile(p, k, c->kc, ap + ir * w, lda, b, e, i0 + ir, j,
             min64(mr, mc - ir), n);
    }
  }
}

/* Products of few rows: at most MC, as decoding is. One job packs a whole,
   each element's rows side by side over all of k; then a unit is a sliver of
   b's columns of one element, which it packs KC block by KC block and adds
   into R row sliver by row sliver. b, the large operand, is read once, by
   one thread, and no panel is shared between jobs. At most 4 rows run on a
   thin kernel. */
typedef struct {
  problem *p;
  nx_cpu_micro k;
  int64_t lda, kblocks, slivers;
  uint8_t *a; /* element e's step q at a + (e·K + q)·lda·w */
} few_rows_job;

static void few_rows_pack(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const few_rows_job *r = ctx;
  const problem *p = r->p;
  int64_t k = p->ext[CON], kc_most = p->g->kc;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = u / r->kblocks, pc = u % r->kblocks * kc_most;
    pack_a(p, e, 0, p->ext[ROW], r->lda, pc, min64(kc_most, k - pc),
           r->a + (e * k + pc) * r->lda * p->w);
  }
}

static void few_rows_unit(int64_t lo, int64_t hi, int worker, void *ctx) {
  const few_rows_job *r = ctx;
  problem *p = r->p;
  int64_t m = p->ext[ROW], n_all = p->ext[COL], k = p->ext[CON], w = p->w;
  int64_t kc_most = min64(p->g->kc, k), mr = r->k.mr, nr = r->k.nr;
  uint8_t *bp = scratch(p, worker, nr * kc_most * w);
  if (bp == NULL) return;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = u / r->slivers, j = u % r->slivers * nr;
    int64_t n = min64(nr, n_all - j), pc = 0;
    start(p, e, 0, m, j, j + n);
    while (pc < k) {
      int64_t kc = min64(kc_most, k - pc);
      const uint8_t *ap = r->a + (e * k + pc) * r->lda * w;
      pack_b(p, e, j, n, (int)nr, pc, kc, bp);
      for (int64_t ir = 0; ir < m; ir += mr)
        tile(p, r->k, kc, ap + ir * w, r->lda, bp, e, ir, j, min64(mr, m - ir),
             n);
      pc += kc;
    }
  }
}

/* The kernel for [m] rows: the thin kernel of fewest rows that holds
   them, else the main one. */
static nx_cpu_micro kernel_of(const nx_cpu_gemm *g, int64_t m) {
  for (int i = 0; i < 3; i++)
    if (g->thin[i].f && g->thin[i].mr >= m) return g->thin[i];
  return g->kernel;
}

/* [p] as the product of b's transpose by a's, which has the same outputs
   transposed and the same bits: fma(a, b, c) is fma(b, a, c). */
static void transpose(problem *p) {
  operand a = p->op[A];
  p->op[A] = p->op[B];
  p->op[B] = a;
  for (int o = 0; o < 4; o++) {
    int64_t r = p->op[o].st[ROW];
    p->op[o].st[ROW] = p->op[o].st[COL];
    p->op[o].st[COL] = r;
  }
  int64_t m = p->ext[ROW];
  p->ext[ROW] = p->ext[COL];
  p->ext[COL] = m;
}

static void few_rows(problem *p) {
  const nx_cpu_gemm *g = p->g;
  int64_t m = p->ext[ROW], n = p->ext[COL], k = p->ext[CON], w = p->w;
  int64_t batch = p->ext[BATCH];
  few_rows_job r = {p, kernel_of(g, m), 0, ceil_div(k, g->kc), 0, NULL};
  r.lda = ceil_div(m, r.k.mr) * r.k.mr;
  r.slivers = ceil_div(n, r.k.nr);
  int64_t packed = batch * k * r.lda * w;
  r.a = alloc(packed);
  if (r.a == NULL) {
    atomic_store(&p->failed, 1);
    return;
  }
  int64_t bytes = batch * n * k * w + packed;
  nx_cpu_job(batch * r.kblocks, packed, 2 * packed, few_rows_pack, &r);
  nx_cpu_job(batch * r.slivers, bytes, 2 * batch * m * n * k + bytes,
             few_rows_unit, &r);
  free(r.a);
}

static void chain(problem *p) {
  const nx_cpu_gemm *g = p->g;
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w;
  int64_t nc = min64(g->nc, n), nr = g->kernel.nr;
  int64_t slivers = ceil_div(nc, nr);
  int64_t panel_bytes = slivers * nr * min64(g->kc, k) * w;
  int64_t group = panel_bytes > 0 && PANELS / panel_bytes > 1
                      ? PANELS / panel_bytes
                      : 1;
  group = min64(group, batch);
  uint8_t *b = alloc(group * panel_bytes);
  if (b == NULL) {
    atomic_store(&p->failed, 1);
    return;
  }
  int64_t mblocks = ceil_div(m, g->mc);
  for (int64_t e0 = 0; e0 < batch; e0 += group) {
    int64_t ne = min64(group, batch - e0);
    for (int64_t jc = 0; jc < n; jc += g->nc) {
      panel c = {p, e0, ne, jc, min64(g->nc, n - jc), 0, 0, 0, mblocks, 1, 0, b};
      c.slivers = ceil_div(c.nc, nr);
      /* Enough units for the threads the compute job takes, splitting
         blocks of a into ranges of slivers only where there are too few:
         each range packs its block again. */
      int64_t kc = min64(g->kc, k);
      int64_t packed = ne * c.slivers * nr * kc * w;
      int64_t want =
          UNITS * nx_cpu_threads(packed, 2 * ne * m * c.nc * kc + packed);
      int64_t units = ne * mblocks;
      if (units < want) c.ranges = min64(c.slivers, ceil_div(want, units));
      c.per_range = ceil_div(c.slivers, c.ranges);
      int64_t pc = 0;
      do {
        c.pc = pc;
        c.kc = min64(g->kc, k - pc);
        int64_t flops = 2 * ne * m * c.nc * c.kc;
        packed = ne * c.slivers * nr * c.kc * w;
        if (c.kc > 0)
          nx_cpu_job(ne * c.slivers, packed, 2 * packed, pack_panel, &c);
        nx_cpu_job(ne * mblocks * c.ranges, packed, flops + packed, compute,
                   &c);
        if (atomic_load(&p->failed)) goto done;
        pc += g->kc;
      } while (pc < k);
    }
  }
done:
  free(b);
}

/* Lane order */

typedef struct {
  problem *p;
  int64_t blocks;
  uint8_t *sums; /* per (element, output, block), in acc, with blocks > 1 */
} lanes_job;

/* The left-complete binary tree's sum of the [n] >= 1 values at [s]. */
#define TREE(T, name)                                    \
  static T name(const T *s, int64_t n) {                 \
    if (n == 1) return s[0];                             \
    int64_t h = 1;                                       \
    while (2 * h < n) h *= 2;                            \
    return name(s, h) + name(s + h, n - h);              \
  }                                                      \
  static T name##_lanes(T *l) {                          \
    for (int w = NX_CPU_LANES / 2; w > 0; w /= 2)        \
      for (int i = 0; i < w; i++) l[i] = l[i] + l[i + w]; \
    return l[0];                                         \
  }
TREE(float, tree_f32)
TREE(double, tree_f64)

/* The sum of one block's lanes, at [d]. */
static void lane_sum(int acc, void *l, uint8_t *d) {
  if (acc == NX_FLOAT32) *(float *)d = tree_f32_lanes(l);
  else *(double *)d = tree_f64_lanes(l);
}

/* [d] = init (if any) + the tree of the [n] block sums at [s], +0 with no
   block. */
static void total(const problem *p, int64_t e, int64_t i, int64_t j,
                  const uint8_t *s, int64_t n, uint8_t *d) {
  _Alignas(16) uint8_t x[16] = {0};
  int init = p->op[INIT].x != NULL;
  if (init) stage_acc(p, INIT, pos(p, INIT, e, i, j, 0), 1, 0, 1, 1, x, 0);
  if (p->acc == NX_FLOAT32) {
    float y = n ? tree_f32((const float *)s, n) : 0.f;
    *(float *)d = init ? *(float *)x + y : y;
  } else {
    double y = n ? tree_f64((const double *)s, n) : 0.;
    *(double *)d = init ? *(double *)x + y : y;
  }
}

/* A unit is one block of one element: its rows of a and columns of b staged
   along k, then each output's lanes. With one block, each output's total. */
static void lanes_unit(int64_t lo, int64_t hi, int worker, void *ctx) {
  const lanes_job *f = ctx;
  problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL], k = p->ext[CON], w = p->w;
  const int64_t *sa = p->op[A].st, *sb = p->op[B].st;
  uint8_t *buf = scratch(p, worker, (m + n) * NX_CPU_FOLD_BLOCK * w);
  if (buf == NULL) return;
  uint8_t *bb = buf + m * NX_CPU_FOLD_BLOCK * w;
  nx_cpu_dot dot = nx_cpu_table->dot[p->acc];
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = u / f->blocks, x = u % f->blocks, k0 = x * NX_CPU_FOLD_BLOCK;
    int64_t len = min64(NX_CPU_FOLD_BLOCK, k - k0);
    stage_acc(p, A, pos(p, A, e, 0, 0, k0), sa[CON], sa[ROW], len, m, buf, len);
    stage_acc(p, B, pos(p, B, e, 0, 0, k0), sb[CON], sb[COL], len, n, bb, len);
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < n; j++) {
        _Alignas(64) uint8_t l[NX_CPU_LANES * 8] = {0};
        _Alignas(16) uint8_t s[16];
        dot(buf + i * len * w, bb + j * len * w, len, l);
        uint8_t *d = f->blocks == 1 ? s : f->sums + (((e * m + i) * n + j) * f->blocks + x) * w;
        lane_sum(p->acc, l, d);
        if (f->blocks == 1) total(p, e, i, j, s, 1, at_r(p, e, i, j));
      }
  }
}

/* A unit is one element: each output's total of its blocks. */
static void lanes_finish(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const lanes_job *f = ctx;
  const problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL];
  for (int64_t e = lo; e < hi; e++)
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < n; j++)
        total(p, e, i, j,
              f->blocks ? f->sums + ((e * m + i) * n + j) * f->blocks * p->w
                        : NULL,
              f->blocks, at_r(p, e, i, j));
}

static void lanes(problem *p) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w;
  lanes_job f = {p, ceil_div(k, NX_CPU_FOLD_BLOCK), NULL};
  int64_t bytes = batch * (m + n) * k * w;
  if (f.blocks == 1) {
    nx_cpu_job(batch, bytes, 2 * bytes, lanes_unit, &f);
    return;
  }
  if (f.blocks > 1) {
    f.sums = alloc(batch * m * n * f.blocks * w);
    if (f.sums == NULL) {
      atomic_store(&p->failed, 1);
      return;
    }
    nx_cpu_job(batch * f.blocks, bytes, 2 * bytes, lanes_unit, &f);
  }
  if (!atomic_load(&p->failed))
    nx_cpu_job(batch, batch * m * n * w, batch * m * n * (f.blocks + 1) * w,
               lanes_finish, &f);
  free(f.sums);
}

/* The entry */

/* Whether [p]'s init is its dst: the door lets a read operand be identical to
   the written one, every index at the same byte. */
static int init_is_dst(const problem *p) {
  const operand *i = &p->op[INIT], *d = &p->op[DST];
  if (i->x == NULL || i->x->base != d->x->base ||
      i->x->dtype != d->x->dtype || i->first != d->first)
    return 0;
  for (int x = BATCH; x <= COL; x++)
    if (p->ext[x] > 1 && i->st[x] != d->st[x]) return 0;
  return 1;
}

/* Stores into op[DST] the contraction in [acc] of op[A], op[B] and op[INIT]
   (or NULL), laid out as [v]. Answers 0 if an allocation failed. */
static int contract(int acc, const nx_contract_view *v,
                    const nx_array *const *op) {
  problem p = {.acc = acc, .w = nx_cpu_width(acc)};
  p.g = &nx_cpu_table->gemm[acc];
  for (int o = 0; o < 4; o++) {
    p.op[o].x = op[o];
    p.op[o].first = v->offset[o];
    for (int x = 0; x < 4; x++) p.op[o].st[x] = v->stride[o][x];
  }
  for (int x = 0; x < 4; x++) p.ext[x] = v->extent[x];
  if (p.ext[BATCH] == 0 || p.ext[ROW] == 0 || p.ext[COL] == 0) return 1;
  p.init_is_dst = init_is_dst(&p);
  int cores = rig_pool_cores();
  p.scratch = calloc((size_t)cores, sizeof(void *));
  if (p.scratch == NULL) return 0;
  atomic_init(&p.failed, 0);
  if (p.ext[ROW] * p.ext[COL] < CHAIN_OUTPUTS) lanes(&p);
  else {
    if (p.ext[COL] < p.ext[ROW] && p.ext[COL] <= p.g->mc) transpose(&p);
    if (p.ext[ROW] <= p.g->mc) few_rows(&p);
    else chain(&p);
  }
  for (int i = 0; i < cores; i++) free(p.scratch[i]);
  free(p.scratch);
  return !atomic_load(&p.failed);
}

/* The external: copies the view, declines what it does not compute, then
   reads the operands through the door. The spec's fields and the view are
   copied before nx_read, which may run OCaml code: a collection may move
   their bytes. */
value nx_cpu_contract(value vs, value vview, value vdst, value va, value vb,
                      value vi) {
  const nx_spec_contract *s = (const nx_spec_contract *)String_val(vs);
  int acc = s->acc, out = s->out, init = s->init;
  nx_contract_view v;
  memcpy(&v, Bytes_val(vview), sizeof v);
  int dts[3] = {nx_array_dtype(va), nx_array_dtype(vb), nx_array_dtype(vi)};
  if (!computes(acc, out, dts, 2 + init))
    return Val_int(NX_DECLINED);
  nx_operand in[4] = {{va, dts[0], 0},
                      {vb, dts[1], 0},
                      {vdst, nx_array_dtype(vdst), 1},
                      {vi, dts[2], 0}};
  nx_array a[4];
  int n = 3 + init;
  int e = nx_read(n, in, a);
  if (e) return Val_int(e);
  const nx_array *op[4] = {&a[0], &a[1], init ? &a[3] : NULL, &a[2]};
  int done = contract(acc, &v, op);
  nx_done(n, a);
  if (!done) caml_raise_out_of_memory();
  return Val_int(NX_OK);
}

value nx_cpu_contract_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_contract(argv[0], argv[1], argv[2], argv[3], argv[4],
                         argv[5]);
}
