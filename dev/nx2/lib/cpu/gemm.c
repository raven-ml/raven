/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contractions: y[b, i, j] = out (init[b, i, j] + Σ_k a[b, i, k] · b[b, k, j]),
   computed in acc.

   Nx_kernel.Spec.Contract_view has grouped the axes into one batch, row (M),
   column (N) and contracted (K) axis per operand. An operand is read
   through the stage, which takes any strides and dtype into its carrier, and
   the target's conversion takes the carrier into acc: a dtype is computed
   only where acc holds each of its values, so operands convert exactly.
   Lane order reads b in place where it is acc and adjacent along the axis
   it streams.

   The order of each output's sum is a function of M, N and K alone:

   - Chain order, M and N above LANE_MOST and M·N >= CHAIN_OUTPUTS: one
     fused multiply-add chain per output, in increasing k, from init or +0.
     The microkernels of every target add each product fused, and the
     blocking stores and reloads the sum in acc exactly, so tile shapes,
     block sizes and threads change no bit.
   - Lane order, otherwise: the products fall into blocks of FOLD_BLOCK
     consecutive k and, within a block, into LANES lanes by k modulo LANES,
     each lane a fused chain from +0; a fixed balanced tree sums the lanes,
     the left-complete binary tree the blocks, then init is added. A single
     chain runs at one fused add per latency; CHAIN_OUTPUTS outputs or more
     keep every target's pipes full in chain order. A product of LANE_MOST
     rows or fewer, as decoding a token is, reads b once in the order it
     lies, at the memory's speed, where chain order's tiles would pack it.

   Chain order follows BLIS's loops (Goto and van de Geijn, "Anatomy of
   High-Performance Matrix Multiplication", 2008): per NC columns of b and
   KC of the contraction, one job packs b's panel into NR-wide slivers and
   a's rows into MC-row blocks, then a job computes each block against
   ranges of b's slivers, MR × NR at a time:

     for each group of batch elements, NC panel, KC block:
       job 1: b[kc × nc] -> slivers kc × NR
              a[m × kc] -> blocks of MC rows, kc steps each
       job 2: per (element, MC block, range of slivers):
                for each sliver of b, each MR rows of the block:
                  R tile += a · b

   A product of at most MC rows packs a whole and then streams b: each unit
   packs a sliver of b and adds it in, so that b is read once and no panel
   passes between jobs. One of at most MC columns, and fewer columns than
   rows, is computed as its transpose. Where a table names other kernels,
   a product runs on those that take less time for its rows (fastest): on
   the M1, the cores' vector units take products of few rows from its
   matrix unit.

   R, the sum in acc, is dst itself: a dtype is computed with out = acc.
   Each tile starts as init, or +0, when its first KC block is computed. */

#include <pthread.h>
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

/* Rows or columns at most which a product runs in lane order. */
#define LANE_MOST 4

/* Lane order's units: dots take DOT_COLUMNS columns of b, rows ROW_BYTES
   of each row of b, a page; a lane has LANE_STEPS steps of a block. */
#define DOT_COLUMNS 16
#define ROW_BYTES 4096
#define LANE_STEPS (NX_CPU_FOLD_BLOCK / NX_CPU_LANES)

/* The bytes of a job's packed b: a group of batch elements shares one job
   while their panels fit, so that small batched products start few
   jobs. */
#define PANELS (8 * 1024 * 1024)

/* Units of a compute job per performance core, so that cores that run
   slower take fewer. */
#define UNITS 4

/* Job costs, in bytes of memcpy (job.c): a kernel's flops at its speed
   (cpu.h), and two a byte packed, a guess no measurement has refined. */
#define PACK_COST 2

/* The cost of [flops] on the kernel [k]. */
static int64_t flop_cost(nx_cpu_micro k, int64_t flops) {
  return flops / k.flops_per_byte;
}

/* The copies of [k]'s hardware the performance cores hold. */
static int64_t engines(nx_cpu_micro k) {
  int64_t n = rig_pool_performance_cores() / k.shared;
  return n > 1 ? n : 1;
}

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

/* Packed operands start on 128 bytes: the matrix unit loads 128 at once. */
#define ALIGN 128

static void *alloc(int64_t bytes) {
  return aligned_alloc(ALIGN,
                       (size_t)ceil_div(bytes > 0 ? bytes : 1, ALIGN) * ALIGN);
}

/* The calling thread keeps its packed operands' buffer for its next
   product, up to KEPT bytes: an allocation past 32 KiB goes to the system
   and back, 0.7 µs on the M1, as long as packing a 128³ product takes. */
#define KEPT (4 * 1024 * 1024)

typedef struct {
  uint8_t *p;
  int64_t n;
} kept;

static pthread_key_t kept_key;
static pthread_once_t kept_once = PTHREAD_ONCE_INIT;

static void kept_drop(void *v) {
  kept *k = v;
  free(k->p);
  free(k);
}

static void kept_init(void) { pthread_key_create(&kept_key, kept_drop); }

/* The calling thread's kept buffer, made at its first product; NULL if
   it cannot be. */
static kept *mine(void) {
  pthread_once(&kept_once, kept_init);
  kept *k = pthread_getspecific(kept_key);
  if (k != NULL) return k;
  k = calloc(1, sizeof *k);
  if (k != NULL && pthread_setspecific(kept_key, k) == 0) return k;
  free(k);
  return NULL;
}

/* [bytes] for the calling thread's packed operands, NULL if allocation
   fails; given back by [give]. */
static uint8_t *take(int64_t bytes) {
  kept *k = bytes > KEPT ? NULL : mine();
  if (k == NULL) return alloc(bytes);
  if (k->p == NULL || k->n < bytes) {
    free(k->p);
    k->p = alloc(bytes);
    k->n = k->p ? bytes : 0;
  }
  return k->p;
}

static void give(uint8_t *p) {
  kept *k = mine();
  if (k == NULL || k->p != p) free(p);
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

/* Sets R's rows [i0, i1) × columns [j0, j1) of element [e] to init, or
   to +0 where k is empty: with k and no init, the kernels start the first
   block from +0 themselves. An init that is dst itself is already there:
   copying it onto itself would be a memcpy whose source and destination
   overlap. */
static void start(const problem *p, int64_t e, int64_t i0, int64_t i1,
                  int64_t j0, int64_t j1) {
  const int64_t *s = p->op[DST].st;
  if (p->init_is_dst || (p->op[INIT].x == NULL && p->ext[CON] > 0)) return;
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

/* Where R's tiles start in the block of k from [pc]. */
static nx_cpu_from from(const problem *p, int64_t pc) {
  return pc == 0 && p->op[INIT].x == NULL ? NX_CPU_FROM_ZERO
                                          : NX_CPU_FROM_TILE;
}

/* Chain order */

/* A panel: batch elements [e0, e0 + ne), b's columns [jc, jc + nc), the
   contraction [pc, pc + kc), packed: b at [b] in slivers of NR columns, a
   at [a] in blocks of MC rows. */
typedef struct {
  problem *p;
  int64_t e0, ne;
  int64_t jc, nc, pc, kc;
  int64_t slivers;
  int64_t mblocks, rows;      /* blocks of a, each of at most [rows] */
  int64_t ranges, per_range;  /* ranges of slivers a compute unit takes */
  nx_cpu_pack move;           /* the pack job's pack, or NULL */
  uint8_t *b, *a;
} panel;

/* Splits the panel's slivers into ranges where its elements' blocks of a
   are fewer units than [want]: ranges of one block share its packing. */
static void split(panel *c, int64_t want) {
  int64_t units = c->ne * c->mblocks;
  c->ranges = units < want ? min64(c->slivers, ceil_div(want, units)) : 1;
  c->per_range = ceil_div(c->slivers, c->ranges);
}

/* Compute unit [u]'s work, units being (element, block, range) in C
   order: element [e], the block of rows from [i0], slivers [v0, v1). */
static void unit_of(const panel *c, int64_t u, int64_t *e, int64_t *i0,
                    int64_t *v0, int64_t *v1) {
  int64_t range = u % c->ranges, ib = u / c->ranges % c->mblocks;
  *e = c->e0 + u / c->ranges / c->mblocks;
  *i0 = ib * c->p->g->mc;
  *v0 = range * c->per_range;
  *v1 = min64(c->slivers, *v0 + c->per_range);
}

/* Sliver [s] of element [e]'s panel, packed: kc steps of NR. */
static uint8_t *sliver(const panel *c, int64_t e, int64_t s) {
  int64_t nr = c->p->g->kernel.nr;
  return c->b + ((e - c->e0) * c->slivers + s) * nr * c->kc * c->p->w;
}

/* Block [ib] of element [e]'s rows of a, packed: kc steps of [*lda]
   elements, a multiple of MR, for its [*m] rows. */
static uint8_t *block(const panel *c, int64_t e, int64_t ib, int64_t *m,
                      int64_t *lda) {
  const nx_cpu_gemm *g = c->p->g;
  *m = min64(g->mc, c->p->ext[ROW] - ib * g->mc);
  *lda = ceil_div(*m, g->kernel.mr) * g->kernel.mr;
  return c->a + ((e - c->e0) * c->mblocks + ib) * c->rows * c->kc * c->p->w;
}

/* The pack of a job of [total] units, [bytes] and [cost]: the target's
   where the job runs on at most one thread per copy of the kernel's
   hardware, else none. The matrix unit packs faster than a core, but its
   cluster's cores queue on it: on the M1 Max a 256³ product packs faster
   through it on two threads, a 512³ one through the stage on eight. */
static nx_cpu_pack mover(const problem *p, int64_t total, int64_t bytes,
                         int64_t cost) {
  int64_t threads = min64(total, nx_cpu_threads(bytes, cost));
  return threads <= engines(p->g->kernel) ? p->g->pack : NULL;
}

/* Moves operand [o]'s block as stage_acc does, through [move] where it is
   a pack, the operand is in acc and its rows or its columns are
   contiguous. */
static void pack(const problem *p, nx_cpu_pack move, int o, int64_t at,
                 int64_t s0, int64_t s1, int64_t n0, int64_t n1, uint8_t *d,
                 int64_t pitch) {
  const nx_array *x = p->op[o].x;
  if (move && x->dtype == p->acc && (s0 == 1 || s1 == 1)) {
    move(n0, n1, x->base + at * p->w, s0, s1, d, pitch);
    return;
  }
  stage_acc(p, o, at, s0, s1, n0, n1, d, pitch);
}

/* Packs [a]'s rows [i, i + m) of element [e], along k from
   [pc] for [kc], into [d]: kc steps of [lda] elements, zero past m. A
   block of rows is one pack: its rows transpose in square blocks, where a
   sliver as narrow as a microkernel's 6 rows would move element by
   element. */
static void pack_a(const problem *p, nx_cpu_pack move, int64_t e, int64_t i,
                   int64_t m, int64_t lda, int64_t pc, int64_t kc,
                   uint8_t *d) {
  const int64_t *s = p->op[A].st;
  for (int64_t q = 0; m < lda && q < kc; q++)
    memset(d + (q * lda + m) * p->w, 0, (size_t)((lda - m) * p->w));
  pack(p, move, A, pos(p, A, e, i, 0, pc), s[ROW], s[CON], m, kc, d, lda);
}

/* Packs the sliver of [b]'s columns [j, j + n) likewise, kc steps of
   [nr]. */
static void pack_b(const problem *p, nx_cpu_pack move, int64_t e, int64_t j,
                   int64_t n, int nr, int64_t pc, int64_t kc, uint8_t *d) {
  const int64_t *s = p->op[B].st;
  if (n < nr) memset(d, 0, (size_t)(nr * kc * p->w));
  pack(p, move, B, pos(p, B, e, 0, j, pc), s[COL], s[CON], n, kc, d, nr);
}

/* Units [0, ne·slivers) pack b's slivers, the next ne·mblocks a's blocks. */
static void pack_panel(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const panel *c = ctx;
  int nr = c->p->g->kernel.nr;
  int64_t bs = c->ne * c->slivers;
  for (int64_t u = lo; u < hi; u++) {
    if (u < bs) {
      int64_t e = c->e0 + u / c->slivers, v = u % c->slivers;
      int64_t j = c->jc + v * nr;
      pack_b(c->p, c->move, e, j, min64(nr, c->jc + c->nc - j), nr, c->pc,
             c->kc, sliver(c, e, v));
      continue;
    }
    int64_t e = c->e0 + (u - bs) / c->mblocks, ib = (u - bs) % c->mblocks;
    int64_t m, lda;
    uint8_t *d = block(c, e, ib, &m, &lda);
    pack_a(c->p, c->move, e, ib * c->p->g->mc, m, lda, c->pc, c->kc, d);
  }
}

/* Adds the products of the packed [a], its steps [lda] apart, and [b] to
   R's tile of [m] rows and [n] columns from (e, i, j) with [k]'s kernel,
   the block of k from [pc]: in place where its columns are adjacent and it
   is whole, else through a buffer. */
static void tile(const problem *p, nx_cpu_micro k, int64_t pc, int64_t kc,
                 const uint8_t *a, int64_t lda, const uint8_t *b, int64_t e,
                 int64_t i, int64_t j, int64_t m, int64_t n) {
  const int64_t *s = p->op[DST].st;
  nx_cpu_from f = from(p, pc);
  if (m == k.mr && n == k.nr && s[COL] == 1) {
    k.f(kc, a, lda, b, at_r(p, e, i, j), s[ROW], f);
    return;
  }
  _Alignas(ALIGN) uint8_t t[NX_CPU_TILE];
  int w = p->w;
  memset(t, 0, (size_t)(k.mr * k.nr * w));
  for (int64_t r = 0; f == NX_CPU_FROM_TILE && r < m; r++)
    if (s[COL] == 1)
      memcpy(t + r * k.nr * w, at_r(p, e, i + r, j), (size_t)(n * w));
    else
      for (int64_t q = 0; q < n; q++)
        memcpy(t + (r * k.nr + q) * w, at_r(p, e, i + r, j + q), w);
  k.f(kc, a, lda, b, t, k.nr, f);
  for (int64_t r = 0; r < m; r++)
    if (s[COL] == 1)
      memcpy(at_r(p, e, i + r, j), t + r * k.nr * w, (size_t)(n * w));
    else
      for (int64_t q = 0; q < n; q++)
        memcpy(at_r(p, e, i + r, j + q), t + (r * k.nr + q) * w, w);
}

static void compute(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const panel *c = ctx;
  problem *p = c->p;
  const nx_cpu_gemm *g = p->g;
  nx_cpu_micro k = g->kernel;
  int64_t mr = k.mr, w = p->w;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e, i0, v0, v1;
    unit_of(c, u, &e, &i0, &v0, &v1);
    if (v0 >= v1) continue;
    int64_t mc, lda;
    const uint8_t *ap = block(c, e, i0 / g->mc, &mc, &lda);
    int64_t j0 = c->jc + v0 * k.nr;
    int64_t j1 = min64(c->jc + c->nc, c->jc + v1 * k.nr);
    if (c->pc == 0) start(p, e, i0, i0 + mc, j0, j1);
    if (c->kc == 0) continue;
    for (int64_t v = v0; v < v1; v++) {
      int64_t j = c->jc + v * k.nr, n = min64(k.nr, c->jc + c->nc - j);
      const uint8_t *b = sliver(c, e, v);
      for (int64_t ir = 0; ir < mc; ir += mr)
        tile(p, k, c->pc, c->kc, ap + ir * w, lda, b, e, i0 + ir, j,
             min64(mr, mc - ir), n);
    }
  }
}

/* Products of few rows: at most MC. One job packs a whole, each element's
   rows side by side over all of k; then a unit is a sliver of b's columns
   of one element, which it packs KC block by KC block and adds into R row
   sliver by row sliver. b, the large operand, is read once, by one thread,
   and no panel is shared between jobs. */
typedef struct {
  problem *p;
  nx_cpu_micro k;
  nx_cpu_pack move_a, move_b; /* each job's pack, or NULL */
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
    pack_a(p, r->move_a, e, 0, p->ext[ROW], r->lda, pc,
           min64(kc_most, k - pc), r->a + (e * k + pc) * r->lda * p->w);
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
      pack_b(p, r->move_b, e, j, n, (int)nr, pc, kc, bp);
      for (int64_t ir = 0; ir < m; ir += mr)
        tile(p, r->k, pc, kc, ap + ir * w, r->lda, bp, e, ir, j,
             min64(mr, m - ir), n);
      pc += kc;
    }
  }
}

/* The rows of [m] that [k] computes: a multiple of its tile's. */
static int64_t padded(nx_cpu_micro k, int64_t m) {
  return ceil_div(m, k.mr) * k.mr;
}

/* [g] or its other kernels, whichever computes [m] rows in less time: the
   kernel few_rows or chain takes, its rows padded to its tile, at its
   speed on every copy of its hardware. On the M1 Max 8 rows are 8 on the
   vector units of 8 cores at speed 1, against 32 on 2 matrix units at 13:
   at most 8 rows run on the cores, more on the units. */
static const nx_cpu_gemm *fastest(const nx_cpu_gemm *g, int64_t m) {
  const nx_cpu_gemm *o = g->other;
  if (o == NULL) return g;
  nx_cpu_micro a = g->kernel, b = o->kernel;
  int64_t ta = padded(a, m) * b.flops_per_byte * engines(b);
  int64_t tb = padded(b, m) * a.flops_per_byte * engines(a);
  return tb < ta ? o : g;
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
  nx_cpu_micro km = g->kernel;
  few_rows_job r = {.p = p,
                    .k = km,
                    .lda = ceil_div(m, km.mr) * km.mr,
                    .kblocks = ceil_div(k, g->kc),
                    .slivers = ceil_div(n, km.nr)};
  /* CR: Check workspace counts before multiplying. Legal f32 broadcast
     views [8;2^58] and [2^58;8] each span four bytes, but this count
     overflows signed int64 before allocation. Keep storage sizes,
     alignment and work counts exact and checked; unrepresentable storage
     takes the existing failure/OOM path. Saturate only scheduling byte/flop
     estimates, with safe ceiling division and heuristic comparisons too.
     Cover chain's buffers and lanes' partial sums at the same boundary. */
  int64_t packed = batch * k * r.lda * w;
  r.a = take(packed);
  if (r.a == NULL) {
    atomic_store(&p->failed, 1);
    return;
  }
  int64_t bytes = batch * n * k * w + packed, flops = 2 * batch * m * n * k;
  int64_t cost = flop_cost(km, flops) + bytes;
  r.move_a = mover(p, batch * r.kblocks, packed, PACK_COST * packed);
  r.move_b = mover(p, batch * r.slivers, bytes, cost);
  nx_cpu_job(batch * r.kblocks, packed, PACK_COST * packed, few_rows_pack,
             &r);
  nx_cpu_job(batch * r.slivers, bytes, cost, few_rows_unit, &r);
  give(r.a);
}

static void chain(problem *p) {
  const nx_cpu_gemm *g = p->g;
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w;
  int64_t nc = min64(g->nc, n), nr = g->kernel.nr, kc_most = min64(g->kc, k);
  int64_t slivers = ceil_div(nc, nr), mblocks = ceil_div(m, g->mc);
  int64_t b_bytes = slivers * nr * kc_most * w;
  int64_t rows = min64(g->mc, ceil_div(m, g->kernel.mr) * g->kernel.mr);
  int64_t a_bytes = mblocks * rows * kc_most * w;
  int64_t per = b_bytes + a_bytes;
  int64_t group = per > 0 && PANELS / per > 1 ? PANELS / per : 1;
  group = min64(group, batch);
  int64_t b_all = ceil_div(group * b_bytes, ALIGN) * ALIGN;
  uint8_t *b = take(b_all + group * a_bytes);
  if (b == NULL) {
    atomic_store(&p->failed, 1);
    return;
  }
  uint8_t *a = b + b_all;
  for (int64_t e0 = 0; e0 < batch; e0 += group) {
    int64_t ne = min64(group, batch - e0);
    for (int64_t jc = 0; jc < n; jc += g->nc) {
      int64_t width = min64(g->nc, n - jc);
      panel c = {.p = p,
                 .e0 = e0,
                 .ne = ne,
                 .jc = jc,
                 .nc = width,
                 .slivers = ceil_div(width, nr),
                 .mblocks = mblocks,
                 .rows = rows,
                 .b = b,
                 .a = a};
      /* Enough units for the threads the compute job takes. */
      int64_t packed = ne * (c.slivers * nr + m) * kc_most * w;
      int64_t flops = 2 * ne * m * c.nc * kc_most;
      int64_t cost = flop_cost(g->kernel, flops) + packed;
      split(&c, UNITS * nx_cpu_threads(packed, cost));
      int64_t pc = 0;
      do {
        c.pc = pc;
        c.kc = min64(g->kc, k - pc);
        flops = 2 * ne * m * c.nc * c.kc;
        packed = ne * (c.slivers * nr + m) * c.kc * w;
        int64_t packs = ne * (c.slivers + mblocks);
        c.move = mover(p, packs, packed, PACK_COST * packed);
        if (c.kc > 0)
          nx_cpu_job(packs, packed, PACK_COST * packed, pack_panel, &c);
        nx_cpu_job(ne * mblocks * c.ranges, packed,
                   flop_cost(g->kernel, flops) + packed, compute, &c);
        if (atomic_load(&p->failed)) goto done;
        pc += g->kc;
      } while (pc < k);
    }
  }
done:
  give(b);
}

/* Lane order */

/* Rows are the fewer of the two (lanes transposes the product): at most 7,
   since M·N < CHAIN_OUTPUTS where LANE_MOST < M. b, the large operand, is
   read once, in the order it lies:

   - Dots, where b is adjacent along k, as a weight stored [n × k] is: a
     unit is (element, DOT_COLUMNS columns, range of blocks), and each
     output's block a dot along k, a column of b against every row of a. A
     product of few outputs over a long k, a dot's, splits its blocks into
     ranges, whose sums meet in a second job; otherwise a unit takes every
     block and totals its outputs itself.
   - Rows, otherwise, as for a weight stored [k × n]: a unit is (element,
     block, lane, ROW_BYTES of columns), and adds the lane's rows of b,
     k0 + l + 16t in increasing t, each times its row's element of a, into
     the lane's accumulators, one per output. A second job sums each
     block's lanes by the lanes' tree, column by column, then totals the
     outputs. */

/* The left-complete binary tree's sum of the [n] >= 1 values at [s]; the
   lanes' tree of [c] outputs, lane q of output j at l[q·ls + j], into lane
   0's. */
#define TREE(T, name)                                          \
  static T name(const T *s, int64_t n) {                       \
    if (n == 1) return s[0];                                   \
    int64_t h = 1;                                             \
    while (2 * h < n) h *= 2;                                  \
    return name(s, h) + name(s + h, n - h);                    \
  }                                                            \
  static void name##_lanes(T *l, int64_t ls, int64_t c) {      \
    for (int w = NX_CPU_LANES / 2; w > 0; w /= 2)              \
      for (int i = 0; i < w; i++)                              \
        for (int64_t j = 0; j < c; j++)                        \
          l[i * ls + j] = l[i * ls + j] + l[(i + w) * ls + j]; \
  }
TREE(float, tree_f32)
TREE(double, tree_f64)

static void lane_sums(int acc, uint8_t *l, int64_t ls, int64_t c) {
  if (acc == NX_FLOAT32) tree_f32_lanes((float *)l, ls, c);
  else tree_f64_lanes((double *)l, ls, c);
}

/* [d] = init (if any) + the tree of the [n] block sums at [s]; with no
   block, init as it is, or +0. */
static void total(const problem *p, int64_t e, int64_t i, int64_t j,
                  const uint8_t *s, int64_t n, uint8_t *d) {
  _Alignas(16) uint8_t x[16] = {0};
  int init = p->op[INIT].x != NULL;
  if (init) stage_acc(p, INIT, pos(p, INIT, e, i, j, 0), 1, 0, 1, 1, x, 0);
  if (init && n == 0) {
    memcpy(d, x, (size_t)p->w);
    return;
  }
  if (p->acc == NX_FLOAT32) {
    float y = n ? tree_f32((const float *)s, n) : 0.f;
    *(float *)d = init ? *(float *)x + y : y;
  } else {
    double y = n ? tree_f64((const double *)s, n) : 0.;
    *(double *)d = init ? *(double *)x + y : y;
  }
}

/* Whether operand [o] is read in place along [ax]: acc, adjacent there. */
static int adjacent(const problem *p, int o, int ax) {
  return p->op[o].x->dtype == p->acc && p->op[o].st[ax] == 1;
}

/* Each output with no products: init as it is, or +0. A unit is an
   element. */
static void empty(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const problem *p = ctx;
  for (int64_t e = lo; e < hi; e++)
    for (int64_t i = 0; i < p->ext[ROW]; i++)
      for (int64_t j = 0; j < p->ext[COL]; j++)
        total(p, e, i, j, NULL, 0, at_r(p, e, i, j));
}

/* Dots */

typedef struct {
  problem *p;
  int64_t blocks, groups;    /* of k, and of DOT_COLUMNS columns */
  int64_t ranges, per_range; /* ranges of blocks a unit takes */
  uint8_t *a;                /* a, adjacent along k */
  int64_t lda, a_batch;      /* its rows' and elements' steps */
  uint8_t *sums; /* per (element, output, block), with ranges > 1 */
} dots_job;

/* The block sums of output (i, j0 + j) of element [e], in a unit of [c]
   columns from [j0]: in the job's sums, or the unit's own at [mine]. */
static uint8_t *sums_of(const dots_job *f, uint8_t *mine, int64_t e,
                        int64_t i, int64_t j0, int64_t c, int64_t j) {
  const problem *p = f->p;
  if (f->ranges == 1) return mine + (i * c + j) * f->blocks * p->w;
  int64_t at = (e * p->ext[ROW] + i) * p->ext[COL] + j0 + j;
  return f->sums + at * f->blocks * p->w;
}

/* A unit is an element: its rows of a staged whole, adjacent along k. */
static void dots_a(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const dots_job *f = ctx;
  const problem *p = f->p;
  const int64_t *sa = p->op[A].st;
  int64_t m = p->ext[ROW], k = p->ext[CON];
  for (int64_t e = lo; e < hi; e++)
    stage_acc(p, A, pos(p, A, e, 0, 0, 0), sa[CON], sa[ROW], k, m,
              f->a + e * f->a_batch * p->w, k);
}

/* Columns [j0, j0 + c) of element [e], each over blocks [x0, x1) in turn,
   so that a column of b streams: b in place, or each block staged into
   [bp]. Each block's lanes are summed by the lanes' tree into its sum. */
static void dots_columns(const dots_job *f, uint8_t *bp, uint8_t *mine,
                         int64_t e, int64_t j0, int64_t c, int64_t x0,
                         int64_t x1) {
  const problem *p = f->p;
  int64_t m = p->ext[ROW], k = p->ext[CON], w = p->w;
  const uint8_t *a = f->a + e * f->a_batch * w;
  int in_place = adjacent(p, B, CON);
  nx_cpu_dot dot = nx_cpu_table->dot[p->acc];
  _Alignas(64) uint8_t l[NX_CPU_DOT_ROWS * NX_CPU_LANES * 8];
  for (int64_t j = 0; j < c; j++)
    for (int64_t x = x0; x < x1; x++) {
      int64_t k0 = x * NX_CPU_FOLD_BLOCK, len = min64(NX_CPU_FOLD_BLOCK, k - k0);
      int64_t at = pos(p, B, e, 0, j0 + j, k0);
      const uint8_t *b = bp;
      if (in_place) b = p->op[B].x->base + at * w;
      else stage_acc(p, B, at, 1, 0, len, 1, bp, len);
      for (int64_t i0 = 0; i0 < m; i0 += NX_CPU_DOT_ROWS) {
        int r = (int)min64(NX_CPU_DOT_ROWS, m - i0);
        memset(l, 0, sizeof l);
        dot(a + (i0 * f->lda + k0) * w, f->lda, r, b, len, l);
        for (int i = 0; i < r; i++) {
          uint8_t *li = l + i * NX_CPU_LANES * w;
          lane_sums(p->acc, li, 1, 1);
          memcpy(sums_of(f, mine, e, i0 + i, j0, c, j) + x * w, li,
                 (size_t)w);
        }
      }
    }
}

static void dots_unit(int64_t lo, int64_t hi, int worker, void *ctx) {
  const dots_job *f = ctx;
  problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL], w = p->w;
  int64_t b_bytes = NX_CPU_FOLD_BLOCK * w;
  int64_t own = f->ranges == 1 ? m * DOT_COLUMNS * f->blocks * w : 0;
  uint8_t *bp = scratch(p, worker, b_bytes + own);
  if (bp == NULL) return;
  for (int64_t u = lo; u < hi; u++) {
    int64_t r = u % f->ranges, g = u / f->ranges % f->groups;
    int64_t e = u / f->ranges / f->groups, j0 = g * DOT_COLUMNS;
    int64_t c = min64(DOT_COLUMNS, n - j0), x0 = r * f->per_range;
    uint8_t *mine = bp + b_bytes;
    dots_columns(f, bp, mine, e, j0, c, x0,
                 min64(f->blocks, x0 + f->per_range));
    if (f->ranges > 1) continue;
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < c; j++)
        total(p, e, i, j0 + j, sums_of(f, mine, e, i, j0, c, j), f->blocks,
              at_r(p, e, i, j0 + j));
  }
}

/* A unit is an element: each output's total of its blocks. */
static void dots_finish(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const dots_job *f = ctx;
  const problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL];
  for (int64_t e = lo; e < hi; e++)
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < n; j++)
        total(p, e, i, j, f->sums + ((e * m + i) * n + j) * f->blocks * p->w,
              f->blocks, at_r(p, e, i, j));
}

/* a is read in place where it is acc and adjacent along k, else staged
   whole first: it is the smaller operand. */
static void dots(problem *p, int64_t bytes, int64_t cost) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w, outputs = batch * m * n;
  dots_job f = {.p = p,
                .blocks = ceil_div(k, NX_CPU_FOLD_BLOCK),
                .groups = ceil_div(n, DOT_COLUMNS)};
  uint8_t *staged = NULL;
  if (adjacent(p, A, CON)) {
    f.a = p->op[A].x->base + p->op[A].first * w;
    f.lda = p->op[A].st[ROW];
    f.a_batch = p->op[A].st[BATCH];
  } else {
    staged = f.a = alloc(batch * m * k * w);
    if (staged == NULL) {
      atomic_store(&p->failed, 1);
      return;
    }
    f.lda = k;
    f.a_batch = m * k;
    nx_cpu_job(batch, batch * m * k * w, PACK_COST * batch * m * k * w,
               dots_a, &f);
  }
  int64_t want = UNITS * nx_cpu_threads(bytes, cost);
  int64_t outer = batch * f.groups;
  f.per_range = outer < want ? ceil_div(f.blocks, ceil_div(want, outer))
                             : f.blocks;
  f.ranges = ceil_div(f.blocks, f.per_range);
  if (f.ranges > 1) f.sums = alloc(outputs * f.blocks * w);
  if (f.ranges > 1 && f.sums == NULL) atomic_store(&p->failed, 1);
  else nx_cpu_job(outer * f.ranges, bytes, cost, dots_unit, &f);
  if (f.ranges > 1 && !atomic_load(&p->failed))
    nx_cpu_job(batch, outputs * f.blocks * w, outputs * (f.blocks + 1) * w,
               dots_finish, &f);
  free(f.sums);
  free(staged);
}

/* Rows */

typedef struct {
  problem *p;
  int64_t blocks, cols, groups; /* of k, and of [cols] columns */
  int64_t own;                  /* a worker's scratch, for both jobs */
  uint8_t *lanes; /* lane l of block x of output (i, j) of element e at
                     ((((e·blocks + x)·LANES + l)·M + i)·N + j)·w */
} rows_job;

static uint8_t *lane_at(const rows_job *f, int64_t e, int64_t x, int64_t l,
                        int64_t i, int64_t j) {
  const problem *p = f->p;
  int64_t at = ((e * f->blocks + x) * NX_CPU_LANES + l) * p->ext[ROW] + i;
  return f->lanes + (at * p->ext[COL] + j) * p->w;
}

/* A unit's lane from +0: a's elements and, unless in place, b's rows of
   the lane staged into the worker's scratch, then each row of b added
   times each row's element of a, prefetching the next row. */
static void rows_unit(int64_t lo, int64_t hi, int worker, void *ctx) {
  const rows_job *f = ctx;
  problem *p = f->p;
  const int64_t *sa = p->op[A].st, *sb = p->op[B].st;
  int64_t m = p->ext[ROW], n = p->ext[COL], k = p->ext[CON], w = p->w;
  int in_place = adjacent(p, B, COL);
  uint8_t *ap = scratch(p, worker, f->own);
  if (ap == NULL) return;
  uint8_t *bp = ap + m * LANE_STEPS * w;
  nx_cpu_axpy axpy = nx_cpu_table->axpy[p->acc];
  for (int64_t u = lo; u < hi; u++) {
    int64_t g = u % f->groups, l = u / f->groups % NX_CPU_LANES;
    int64_t x = u / f->groups / NX_CPU_LANES % f->blocks;
    int64_t e = u / f->groups / NX_CPU_LANES / f->blocks;
    int64_t j0 = g * f->cols, c = min64(f->cols, n - j0);
    int64_t k0 = x * NX_CPU_FOLD_BLOCK + l;
    int64_t end = min64(k, (x + 1) * NX_CPU_FOLD_BLOCK);
    int64_t steps = k0 < end ? ceil_div(end - k0, NX_CPU_LANES) : 0;
    uint8_t *y = lane_at(f, e, x, l, 0, j0);
    for (int64_t i = 0; i < m; i++) memset(y + i * n * w, 0, (size_t)(c * w));
    if (steps == 0) continue;
    stage_acc(p, A, pos(p, A, e, 0, 0, k0), sa[ROW], NX_CPU_LANES * sa[CON],
              m, steps, ap, m);
    const uint8_t *vb = bp;
    int64_t ldb = c, at = pos(p, B, e, 0, j0, k0);
    if (in_place) {
      vb = p->op[B].x->base + at * w;
      ldb = NX_CPU_LANES * sb[CON];
    } else {
      stage_acc(p, B, at, sb[COL], NX_CPU_LANES * sb[CON], c, steps, bp, c);
    }
    for (int64_t t = 0; t < steps; t++) {
      const uint8_t *row = vb + t * ldb * w;
      const uint8_t *next = vb + min64(t + 1, steps - 1) * ldb * w;
      for (int64_t i0 = 0; i0 < m; i0 += NX_CPU_DOT_ROWS)
        axpy(ap + (t * m + i0) * w, 1, (int)min64(NX_CPU_DOT_ROWS, m - i0),
             row, c, y + i0 * n * w, n, next);
    }
  }
}

/* A unit is (element, group of columns): each block's lanes summed by the
   lanes' tree into the worker's scratch, then each output's total. */
static void rows_finish(int64_t lo, int64_t hi, int worker, void *ctx) {
  const rows_job *f = ctx;
  problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL], w = p->w;
  uint8_t *s = scratch(p, worker, f->own);
  if (s == NULL) return;
  for (int64_t u = lo; u < hi; u++) {
    int64_t g = u % f->groups, e = u / f->groups;
    int64_t j0 = g * f->cols, c = min64(f->cols, n - j0);
    for (int64_t x = 0; x < f->blocks; x++)
      for (int64_t i = 0; i < m; i++) {
        uint8_t *l = lane_at(f, e, x, 0, i, j0);
        lane_sums(p->acc, l, m * n, c);
        for (int64_t j = 0; j < c; j++)
          memcpy(s + ((i * c + j) * f->blocks + x) * w, l + j * w, (size_t)w);
      }
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < c; j++)
        total(p, e, i, j0 + j, s + (i * c + j) * f->blocks * w, f->blocks,
              at_r(p, e, i, j0 + j));
  }
}

static void rows(problem *p, int64_t bytes, int64_t cost) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t w = p->w;
  rows_job f = {.p = p,
                .blocks = ceil_div(p->ext[CON], NX_CPU_FOLD_BLOCK),
                .cols = ROW_BYTES / w,
                .groups = ceil_div(n, ROW_BYTES / w)};
  /* A lane's steps of a and b, or an element's block sums. */
  f.own = (m + f.cols) * LANE_STEPS * w;
  if (m * f.cols * f.blocks * w > f.own) f.own = m * f.cols * f.blocks * w;
  int64_t lanes = batch * f.blocks * NX_CPU_LANES * m * n * w;
  f.lanes = alloc(lanes);
  if (f.lanes == NULL) {
    atomic_store(&p->failed, 1);
    return;
  }
  nx_cpu_job(batch * f.blocks * NX_CPU_LANES * f.groups, bytes, cost,
             rows_unit, &f);
  if (!atomic_load(&p->failed))
    nx_cpu_job(batch * f.groups, lanes, 2 * lanes, rows_finish, &f);
  free(f.lanes);
}

static void lanes(problem *p) {
  if (p->ext[COL] < p->ext[ROW]) transpose(p);
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w;
  if (k == 0) {
    nx_cpu_job(batch, batch * m * n * w, batch * m * n * w, empty, p);
    return;
  }
  /* The cores' kernels run a flop in the time memcpy moves a byte. */
  int64_t bytes = batch * (m + n) * k * w;
  int64_t cost = 2 * batch * m * n * k + bytes;
  if (p->op[B].st[CON] == 1) dots(p, bytes, cost);
  else rows(p, bytes, cost);
}

/* The entry */

/* Whether [p] adds in chain order: more than LANE_MOST rows and columns,
   and CHAIN_OUTPUTS outputs or more per element. */
static int chain_order(const problem *p) {
  int64_t m = p->ext[ROW], n = p->ext[COL];
  return m > LANE_MOST && n > LANE_MOST && m * n >= CHAIN_OUTPUTS;
}

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
  if (!chain_order(&p)) lanes(&p);
  else {
    if (p.ext[COL] < p.ext[ROW] && p.ext[COL] <= p.g->mc) transpose(&p);
    p.g = fastest(p.g, p.ext[ROW]);
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
