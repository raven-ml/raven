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

   - Chain order, M·N >= FEW: one fused multiply-add chain per output, in
     increasing k, from init or +0. The microkernels of every target add
     each product fused, and the blocking stores and reloads the sum in acc
     exactly, so tile shapes, block sizes and threads change no bit.
   - Block order, M·N < FEW: the products fall into blocks of FOLD_BLOCK
     consecutive k and, within a block, into LANES lanes by k modulo LANES,
     each lane a fused chain from +0; a fixed balanced tree sums the lanes,
     the left-complete binary tree the blocks, then init is added. A single
     chain runs at one fused add per latency; FEW outputs or more keep
     every target's pipes full in chain order.

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

   A product of at most 4 rows, as decoding a token is, runs instead on a
   thin kernel of 1, 2 or 4 rows, each unit packing a sliver of b and adding
   it in, so that b is read once; one of at most 4 columns is computed as
   its transpose.

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
#define FEW 64

/* The bytes of a job's packed b: a group of batch elements shares one job
   while their panels fit, so that small batched products start few
   jobs. */
#define PANELS (8 * 1024 * 1024)

/* Units of a compute job per performance core, so that cores that run
   slower take fewer. */
#define UNITS 4

/* Operands, as the view numbers them, and axes. */
enum { A, B, INIT, DST };
enum { BATCH, ROW, COL, CON };

typedef struct {
  const nx_array *op[4]; /* by view operand; op[INIT] NULL without init */
  int64_t ext[4];
  int64_t first[4];
  int64_t st[4][4];
  int acc, w;
  const nx_cpu_gemm *g;
  void **scratch;    /* per worker, allocated by the worker */
  atomic_int failed; /* an allocation failed */
} problem;

static int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
static int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

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

static int is_int(int dt) {
  enum nx_kind k = nx_dtype_row_of(dt).kind;
  return k == NX_KIND_SIGNED || k == NX_KIND_UNSIGNED;
}

/* Whether every value of [d] is one of [acc]'s. */
static int holds(int acc, int d) {
  enum nx_kind k = nx_dtype_row_of(d).kind;
  int bits = nx_dtype_row_of(d).bits;
  if (d == acc || k == NX_KIND_BOOLEAN) return 1;
  switch (acc) {
    case NX_FLOAT32:
      return (k == NX_KIND_FLOAT && bits < 32) || (is_int(d) && bits <= 16);
    case NX_FLOAT64:
      return holds(NX_FLOAT32, d) || d == NX_FLOAT32 ||
             (is_int(d) && bits <= 32);
    case NX_COMPLEX64: return d == NX_FLOAT32 || holds(NX_FLOAT32, d);
    case NX_COMPLEX128:
      return d == NX_COMPLEX64 || d == NX_FLOAT64 || holds(NX_FLOAT64, d);
    default:
      return is_int(acc) && is_int(d) &&
             bits <= nx_dtype_row_of(acc).bits;
  }
}

/* Whether nx.cpu computes a contraction in [acc] into [out] of the [n]
   operands of dtypes [dts]. */
static int computes(int acc, int out, const int *dts, int n) {
  if (out != acc || nx_cpu_runs->gemm[acc].kernel == NULL) return 0;
  for (int i = 0; i < n; i++)
    if (!holds(acc, dts[i])) return 0;
  return 1;
}

/* Staging */

/* Stages operand [o]'s block of [n1] rows of [n0] elements, the first at
   position [at], stepping [s0] along a row and [s1] across rows, into [dst]
   in acc: row j's element i at dst + (j·pitch + i)·w. In pieces whose every
   form fits the stage's slot. */
static void stage(const problem *p, int o, int64_t at, int64_t s0, int64_t s1,
                  int64_t n0, int64_t n1, uint8_t *dst, int64_t pitch) {
  const nx_array *x = p->op[o];
  if (x->dtype == p->acc) {
    /* Elements of acc are copied in one block, through no buffer. */
    nx_cpu_block b = {.n0 = n0, .n1 = n1, .n2 = 1};
    b.at[0] = at;
    b.s0[0] = s0;
    b.s1[0] = s1;
    nx_cpu_stage(x, &b, 0, dst, pitch * p->w);
    return;
  }
  int c = nx_cpu_carrier(x->dtype), cw = nx_cpu_width(c);
  int widest = cw > p->w ? cw : p->w;
  if (x->bits / 8 > widest) widest = x->bits / 8;
  int64_t cols = min64(n0, NX_CPU_SLOT / widest);
  nx_cpu_run convert = nx_cpu_runs->convert[c][p->acc];
  _Alignas(64) uint8_t slot[NX_CPU_SLOT];
  for (int64_t i = 0; i < n0; i += cols) {
    int64_t n = min64(cols, n0 - i), rows = NX_CPU_SLOT / (n * widest);
    for (int64_t j = 0; j < n1; j += rows) {
      nx_cpu_block b = {.n0 = n, .n1 = min64(rows, n1 - j), .n2 = 1};
      b.at[0] = at + i * s0 + j * s1;
      b.s0[0] = s0;
      b.s1[0] = s1;
      uint8_t *d = dst + (j * pitch + i) * p->w;
      if (c == p->acc) {
        nx_cpu_stage(x, &b, 0, d, pitch * p->w);
        continue;
      }
      nx_cpu_stage(x, &b, 0, slot, n * cw);
      for (int64_t r = 0; r < b.n1; r++)
        convert(slot + r * n * cw, d + r * pitch * p->w, n);
    }
  }
}

/* R's element (e, i, j): dst's, in acc. */
static uint8_t *at_r(const problem *p, int64_t e, int64_t i, int64_t j) {
  const int64_t *s = p->st[DST];
  return p->op[DST]->base +
         (p->first[DST] + e * s[BATCH] + i * s[ROW] + j * s[COL]) * p->w;
}

/* Sets R's rows [i0, i1) × columns [j0, j1) of element [e] to init or +0. */
static void start(const problem *p, int64_t e, int64_t i0, int64_t i1,
                  int64_t j0, int64_t j1) {
  const int64_t *s = p->st[DST];
  for (int64_t i = i0; i < i1; i++) {
    if (p->op[INIT]) {
      const int64_t *t = p->st[INIT];
      int64_t at = p->first[INIT] + e * t[BATCH] + i * t[ROW] + j0 * t[COL];
      if (s[COL] == 1 || j1 - j0 == 1)
        stage(p, INIT, at, t[COL], 0, j1 - j0, 1, at_r(p, e, i, j0), 0);
      else
        stage(p, INIT, at, 1, t[COL], 1, j1 - j0, at_r(p, e, i, j0), s[COL]);
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

/* Sliver [s] of element [e]'s panel, packed: kc rows of NR. */
static uint8_t *sliver(const panel *c, int64_t e, int64_t s) {
  const nx_cpu_gemm *g = c->p->g;
  return c->b + ((e - c->e0) * c->slivers + s) * g->nr * c->kc * c->p->w;
}

/* Packs the sliver of [a]'s rows [i, i + m) of element [e], along k from
   [pc] for [kc], into [d]: kc steps of [mr] elements, zero past m. */
static void pack_a(const problem *p, int64_t e, int64_t i, int64_t m, int mr,
                   int64_t pc, int64_t kc, uint8_t *d) {
  const int64_t *s = p->st[A];
  if (m < mr) memset(d, 0, (size_t)(mr * kc * p->w));
  stage(p, A, p->first[A] + e * s[BATCH] + i * s[ROW] + pc * s[CON], s[ROW],
        s[CON], m, kc, d, mr);
}

/* Packs the sliver of [b]'s columns [j, j + n) likewise, [nr] wide. */
static void pack_b(const problem *p, int64_t e, int64_t j, int64_t n, int nr,
                   int64_t pc, int64_t kc, uint8_t *d) {
  const int64_t *s = p->st[B];
  if (n < nr) memset(d, 0, (size_t)(nr * kc * p->w));
  stage(p, B, p->first[B] + e * s[BATCH] + j * s[COL] + pc * s[CON], s[COL],
        s[CON], n, kc, d, nr);
}

static void pack_panel(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const panel *c = ctx;
  int nr = c->p->g->nr;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = c->e0 + u / c->slivers, v = u % c->slivers;
    int64_t j = c->jc + v * nr;
    pack_b(c->p, e, j, min64(nr, c->jc + c->nc - j), nr, c->pc, c->kc,
           sliver(c, e, v));
  }
}

/* A microkernel and its tile. */
typedef struct {
  nx_cpu_kernel kernel;
  int mr, nr;
} shape;

/* Adds the products of the packed slivers [a] and [b] to R's tile of [m]
   rows and [n] columns from (e, i, j) with [k]'s kernel: in place where
   its columns are adjacent and it is whole, else through a buffer. */
static void tile(const problem *p, shape k, int64_t kc, const uint8_t *a,
                 const uint8_t *b, int64_t e, int64_t i, int64_t j, int64_t m,
                 int64_t n) {
  const int64_t *s = p->st[DST];
  if (m == k.mr && n == k.nr && s[COL] == 1) {
    k.kernel(kc, a, b, at_r(p, e, i, j), s[ROW]);
    return;
  }
  _Alignas(64) uint8_t t[NX_CPU_TILE];
  int w = p->w;
  memset(t, 0, (size_t)(k.mr * k.nr * w));
  for (int64_t r = 0; r < m; r++)
    for (int64_t q = 0; q < n; q++)
      memcpy(t + (r * k.nr + q) * w, at_r(p, e, i + r, j + q), w);
  k.kernel(kc, a, b, t, k.nr);
  for (int64_t r = 0; r < m; r++)
    for (int64_t q = 0; q < n; q++)
      memcpy(at_r(p, e, i + r, j + q), t + (r * k.nr + q) * w, w);
}

static void compute(int64_t lo, int64_t hi, int worker, void *ctx) {
  const panel *c = ctx;
  problem *p = c->p;
  const nx_cpu_gemm *g = p->g;
  shape k = {g->kernel, g->mr, g->nr};
  int64_t m_all = p->ext[ROW], mr = g->mr, w = p->w;
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
    int64_t j0 = c->jc + v0 * g->nr;
    int64_t j1 = min64(c->jc + c->nc, c->jc + v1 * g->nr);
    for (int64_t ir = 0; ir < mc; ir += mr)
      pack_a(p, e, i0 + ir, min64(mr, mc - ir), g->mr, c->pc, c->kc,
             ap + ir * c->kc * w);
    if (c->pc == 0) start(p, e, i0, i0 + mc, j0, j1);
    if (c->kc == 0) continue;
    for (int64_t v = v0; v < v1; v++) {
      int64_t j = c->jc + v * g->nr, n = min64(g->nr, c->jc + c->nc - j);
      const uint8_t *b = sliver(c, e, v);
      for (int64_t ir = 0; ir < mc; ir += mr)
        tile(p, k, c->kc, ap + ir * c->kc * w, b, e, i0 + ir, j,
             min64(mr, mc - ir), n);
    }
  }
}

/* Thin products: at most 4 rows, the rows padded to a thin kernel's 1, 2 or
   4. A unit is a sliver of b's columns of one element, which it packs
   KC block by KC block with the rows of a and adds into R: b, the large
   operand, is read once, by one thread. */
typedef struct {
  problem *p;
  shape k;
  int64_t slivers;
} thin;

static void thin_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  const thin *t = ctx;
  problem *p = t->p;
  int64_t m = p->ext[ROW], n_all = p->ext[COL], k = p->ext[CON], w = p->w;
  int64_t kc_most = min64(p->g->kc, k);
  uint8_t *ap = scratch(p, worker, (t->k.mr + t->k.nr) * kc_most * w);
  if (ap == NULL) return;
  uint8_t *bp = ap + t->k.mr * kc_most * w;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = u / t->slivers, j = u % t->slivers * t->k.nr;
    int64_t n = min64(t->k.nr, n_all - j), pc = 0;
    start(p, e, 0, m, j, j + n);
    while (pc < k) {
      int64_t kc = min64(kc_most, k - pc);
      pack_a(p, e, 0, m, t->k.mr, pc, kc, ap);
      pack_b(p, e, j, n, t->k.nr, pc, kc, bp);
      tile(p, t->k, kc, ap, bp, e, 0, j, m, n);
      pc += kc;
    }
  }
}

/* The thin kernel for [m] rows, if the target has one. */
static const nx_cpu_thin *thin_of(const nx_cpu_gemm *g, int64_t m) {
  int i = m <= 1 ? 0 : m <= 2 ? 1 : m <= 4 ? 2 : 3;
  return i < 3 && g->thin[i].kernel ? &g->thin[i] : NULL;
}

/* [p] as the product of b's transpose by a's, which has the same outputs
   transposed and the same bits: fma(a, b, c) is fma(b, a, c). */
static void transpose(problem *p) {
  const nx_array *a = p->op[A];
  int64_t at = p->first[A], sa[4];
  memcpy(sa, p->st[A], sizeof sa);
  p->op[A] = p->op[B];
  p->first[A] = p->first[B];
  p->st[A][BATCH] = p->st[B][BATCH];
  p->st[A][ROW] = p->st[B][COL];
  p->st[A][CON] = p->st[B][CON];
  p->op[B] = a;
  p->first[B] = at;
  p->st[B][BATCH] = sa[BATCH];
  p->st[B][COL] = sa[ROW];
  p->st[B][CON] = sa[CON];
  for (int o = INIT; o <= DST; o++) {
    int64_t r = p->st[o][ROW];
    p->st[o][ROW] = p->st[o][COL];
    p->st[o][COL] = r;
  }
  int64_t m = p->ext[ROW];
  p->ext[ROW] = p->ext[COL];
  p->ext[COL] = m;
}

static void thin_product(problem *p, const nx_cpu_thin *t) {
  int64_t m = p->ext[ROW], n = p->ext[COL], k = p->ext[CON], w = p->w;
  int64_t batch = p->ext[BATCH];
  thin j = {p, {t->kernel, m <= 1 ? 1 : m <= 2 ? 2 : 4, t->nr}, 0};
  j.slivers = ceil_div(n, t->nr);
  int64_t bytes = batch * (m + n) * k * w;
  nx_cpu_job(batch * j.slivers, bytes, 2 * batch * m * n * k + bytes,
             thin_units, &j);
}

static void chain(problem *p) {
  const nx_cpu_gemm *g = p->g;
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w;
  int64_t nc = min64(g->nc, n), slivers = ceil_div(nc, g->nr);
  int64_t panel_bytes = slivers * g->nr * min64(g->kc, k) * w;
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
      c.slivers = ceil_div(c.nc, g->nr);
      /* Enough units for the threads the compute job takes, splitting
         blocks of a into ranges of slivers only where there are too few:
         each range packs its block again. */
      int64_t kc = min64(g->kc, k);
      int64_t packed = ne * c.slivers * g->nr * kc * w;
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
        packed = ne * c.slivers * g->nr * c.kc * w;
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

/* Block order */

typedef struct {
  problem *p;
  int64_t blocks;
  uint8_t *sums; /* per (element, output, block), in acc, with blocks > 1 */
} few;

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
static void lanes(int acc, void *l, uint8_t *d) {
  if (acc == NX_FLOAT32) *(float *)d = tree_f32_lanes(l);
  else *(double *)d = tree_f64_lanes(l);
}

/* [d] = init (if any) + the tree of the [n] block sums at [s], +0 with no
   block. */
static void total(const problem *p, int64_t e, int64_t i, int64_t j,
                  const uint8_t *s, int64_t n, uint8_t *d) {
  _Alignas(16) uint8_t x[16] = {0};
  if (p->op[INIT]) {
    const int64_t *t = p->st[INIT];
    stage(p, INIT, p->first[INIT] + e * t[BATCH] + i * t[ROW] + j * t[COL], 1,
          0, 1, 1, x, 0);
  }
  if (p->acc == NX_FLOAT32) {
    float y = n ? tree_f32((const float *)s, n) : 0.f;
    *(float *)d = p->op[INIT] ? *(float *)x + y : y;
  } else {
    double y = n ? tree_f64((const double *)s, n) : 0.;
    *(double *)d = p->op[INIT] ? *(double *)x + y : y;
  }
}

/* A unit is one block of one element: its rows of a and columns of b staged
   along k, then each output's lanes. With one block, each output's total. */
static void block(int64_t lo, int64_t hi, int worker, void *ctx) {
  const few *f = ctx;
  problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL], k = p->ext[CON], w = p->w;
  const int64_t *sa = p->st[A], *sb = p->st[B];
  uint8_t *buf = scratch(p, worker, (m + n) * NX_CPU_FOLD_BLOCK * w);
  if (buf == NULL) return;
  uint8_t *bb = buf + m * NX_CPU_FOLD_BLOCK * w;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = u / f->blocks, x = u % f->blocks, k0 = x * NX_CPU_FOLD_BLOCK;
    int64_t len = min64(NX_CPU_FOLD_BLOCK, k - k0);
    stage(p, A, p->first[A] + e * sa[BATCH] + k0 * sa[CON], sa[CON], sa[ROW],
          len, m, buf, len);
    stage(p, B, p->first[B] + e * sb[BATCH] + k0 * sb[CON], sb[CON], sb[COL],
          len, n, bb, len);
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < n; j++) {
        _Alignas(64) uint8_t l[NX_CPU_LANES * 8] = {0};
        _Alignas(16) uint8_t s[16];
        p->g->dot(buf + i * len * w, bb + j * len * w, len, l);
        uint8_t *d = f->blocks == 1 ? s : f->sums + (((e * m + i) * n + j) * f->blocks + x) * w;
        lanes(p->acc, l, d);
        if (f->blocks == 1) total(p, e, i, j, s, 1, at_r(p, e, i, j));
      }
  }
}

/* A unit is one element: each output's total of its blocks. */
static void finish(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const few *f = ctx;
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

static void blocks(problem *p) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w;
  few f = {p, ceil_div(k, NX_CPU_FOLD_BLOCK), NULL};
  int64_t bytes = batch * (m + n) * k * w;
  if (f.blocks == 1) {
    nx_cpu_job(batch, bytes, 2 * bytes, block, &f);
    return;
  }
  if (f.blocks > 1) {
    f.sums = alloc(batch * m * n * f.blocks * w);
    if (f.sums == NULL) {
      atomic_store(&p->failed, 1);
      return;
    }
    nx_cpu_job(batch * f.blocks, bytes, 2 * bytes, block, &f);
  }
  if (!atomic_load(&p->failed))
    nx_cpu_job(batch, batch * m * n * w, batch * m * n * (f.blocks + 1) * w,
               finish, &f);
  free(f.sums);
}

/* The entry */

/* A contraction's axes grouped as Nx_kernel.Spec.Contract_view groups them:
   extent[x] by axis, offset[o] and stride[o][x] by operand, in elements. */
typedef struct {
  int64_t extent[4];
  int64_t offset[4];
  int64_t stride[4][4];
} view;

/* Stores into op[DST] the contraction in [acc] of op[A], op[B] and op[INIT]
   (or NULL), laid out as [v]. Answers 0 if an allocation failed. */
static int run(int acc, const view *v, const nx_array *const *op) {
  problem p = {.acc = acc, .w = nx_cpu_width(acc)};
  p.g = &nx_cpu_runs->gemm[acc];
  for (int o = 0; o < 4; o++) {
    p.op[o] = op[o];
    p.first[o] = v->offset[o];
    for (int x = 0; x < 4; x++) p.st[o][x] = v->stride[o][x];
  }
  for (int x = 0; x < 4; x++) p.ext[x] = v->extent[x];
  if (p.ext[BATCH] == 0 || p.ext[ROW] == 0 || p.ext[COL] == 0) return 1;
  int cores = rig_pool_cores();
  p.scratch = calloc((size_t)cores, sizeof(void *));
  if (p.scratch == NULL) return 0;
  atomic_init(&p.failed, 0);
  if (p.ext[ROW] * p.ext[COL] < FEW) blocks(&p);
  else {
    if (p.ext[COL] < p.ext[ROW] && thin_of(p.g, p.ext[COL])) transpose(&p);
    const nx_cpu_thin *t = thin_of(p.g, p.ext[ROW]);
    if (t) thin_product(&p, t);
    else chain(&p);
  }
  for (int i = 0; i < cores; i++) free(p.scratch[i]);
  free(p.scratch);
  return !atomic_load(&p.failed);
}

/* The external: reads the view, declines what it does not compute, then
   reads the operands through the door. The view, an int array of the
   extents, offsets and strides (operand by operand) that
   Nx_kernel.Spec.Contract_view gives, and the spec's fields are copied
   before nx_read, which may run OCaml code: a collection may move them. */
value nx_cpu_contract(value vs, value vview, value vdst, value va, value vb,
                      value vi) {
  const nx_spec_contract *s = (const nx_spec_contract *)String_val(vs);
  int acc = s->acc, out = s->out, init = s->init;
  view v;
  for (int x = 0; x < 4; x++) {
    v.extent[x] = Long_val(Field(vview, x));
    v.offset[x] = Long_val(Field(vview, 4 + x));
    for (int y = 0; y < 4; y++)
      v.stride[x][y] = Long_val(Field(vview, 8 + 4 * x + y));
  }
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
  int done = run(acc, &v, op);
  nx_done(n, a);
  if (!done) caml_raise_out_of_memory();
  return Val_int(NX_OK);
}

value nx_cpu_contract_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_contract(argv[0], argv[1], argv[2], argv[3], argv[4],
                         argv[5]);
}
