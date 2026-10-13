/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contractions: y[b, i, j] = init[b, i, j] + Σ_k a[b, i, k] · b[b, k, j],
   computed in acc and rounded once to y's dtype.

   nx_contract_view_fill groups the axes, as Nx_kernel.Contract_view does,
   into one batch, row (M), column (N) and contracted (K) axis per
   operand. An operand is read
   through the stage, which takes any strides and dtype into its carrier, and
   the target's conversion takes the carrier into acc: a dtype is computed
   only where acc holds each of its values, so operands convert exactly.
   Lane order reads b in place where it is acc and adjacent along the axis
   it streams.

   Each output's sum adds in the order the caller gives, at any M, N and
   K:

   - Chain order: one fused multiply-add chain per output, in increasing
     k, from init or +0. The microkernels of every target add each product
     fused, and the blocking stores and reloads the sum in acc exactly, so
     tile shapes, block sizes and threads change no bit. A single chain
     runs at one fused add per latency: many rows and columns keep every
     target's pipes full.
   - Lane order: the products fall into blocks of FOLD_BLOCK consecutive k
     and, within a block, into LANES lanes by k modulo LANES, each lane a
     fused chain from +0; a fixed balanced tree sums the lanes, the
     left-complete binary tree the blocks, then init is added. A product
     of few rows, as decoding a token is, reads b once in the order it
     lies, at the memory's speed, where chain order's tiles would pack it.

   Chain order follows BLIS's loops (Goto and van de Geijn, "Anatomy of
   High-Performance Matrix Multiplication", 2008): per NC columns of b and
   KC of the contraction, one job packs a's rows into blocks of at most MC
   rows, then a job computes each block against ranges of b's slivers, MR ×
   NR at a time, packing each sliver into the panel where it first needs
   it:

     for each group of batch elements, NC panel, KC block:
       job 1: a[m × kc] -> blocks of at most MC rows, kc steps each
       job 2: per (element, range of slivers, block):
                for each sliver of b, each MR rows of the block:
                  b[kc × NR] -> the panel's sliver, once
                  R tile += a · b

   The kernel reads both operands through their strides, and runs the
   block's MR-row tiles itself. An operand is read in place where its
   layout streams along the axis the kernel loads and it stays in the
   caches between its reads (in_place), as BLIS's path for small products
   does: it is not packed, and with both operands in place job 1 is gone.
   An edge sliver of b in place is packed where the kernel reads NR
   columns, so that it reads no column past b's.

   A product of few rows (streams) packs a whole and then streams b: each
   unit packs a sliver of b and adds it in along all of k, so that b is
   read once, in one job. One of fewer columns than rows is computed as its
   transpose where that streams. Where a table
   names other kernels, a product runs on those that take less time for
   its rows (fastest): on the M1, the cores' vector units take products of
   few rows from its matrix unit.

   R, the sum in acc, is dst itself: nx.cpu computes a dst of acc's dtype.
   Each tile starts as init, or +0, when its first KC block is computed. */

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/mlvalues.h>

#include "cpu.h"

/* Rows of a pass of lane order: a product of more runs in passes, each
   reading b again. */
#define LANE_ROWS 8

/* Lane order's units: dots take DOT_COLUMNS columns of b, rows ROW_BYTES
   of each row of b, a page. */
#define DOT_COLUMNS 16
#define ROW_BYTES 4096

/* The bytes of a's element up to which chain order reads a in place,
   where it allows: each tile streams its MR rows, which stay in the caches
   from one sliver of b to the next, and no job packs it. On the i9-9900K
   a 128³ float32 product takes 6.1 µs on 16 threads with a in place
   against 11.8 packed, a 256³ one 36.6 against 49.3; a 1024³ one the
   same either way. */
#define IN_PLACE_BYTES (2 * 1024 * 1024)

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

/* The cost of [flops] on the kernel [k]. Flops multiply the extents of
   several operands, so they and the costs are doubles; storage, within one
   operand's extents rounded to a tile, fits in an int64_t (NX_MAX_NUMEL). */
static double flop_cost(nx_cpu_micro k, double flops) {
  return flops / k.flops_per_byte;
}

/* The cost a thread of a job on [g]'s kernels brings to pay for the job's
   start (nx_cpu_job_at). */
static double thread_bytes(const nx_cpu_gemm *g) {
  return NX_CPU_THREAD_BYTES + g->start;
}

/* The units a compute job of [bytes] and [cost], one thread per [per] of
   it, splits into: UNITS a thread, or one where it runs on one thread and
   a split buys nothing. */
static int64_t units(int64_t bytes, double cost, double per) {
  int64_t threads = nx_cpu_threads_at(bytes, cost, per);
  return threads > 1 ? UNITS * threads : 1;
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
  int a_in, b_in;    /* chain order reads a, b in place */
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

/* The calling worker's scratch of [bytes], given back by nx_cpu_give. */
static uint8_t *scratch(problem *p, int64_t bytes) {
  uint8_t *s = nx_cpu_take(bytes);
  if (s == NULL) atomic_store(&p->failed, 1);
  return s;
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

/* Whether nx.cpu computes a contraction in [acc] into a dst of [out] of
   the [n] operands of dtypes [dts]. */
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

/* An operand's block as a kernel reads it: its first element at [at],
   [row] elements apart across a's rows and [k] along the contraction. */
typedef struct {
  const uint8_t *at;
  int64_t row, k;
} src;

/* Whether operand [o] is read in place along [ax]: acc, adjacent there. */
static int adjacent(const problem *p, int o, int ax) {
  return p->op[o].x->dtype == p->acc && p->op[o].st[ax] == 1;
}

/* a's rows from [i] of element [e], from the block of k at [pc], in
   place. */
static src a_at(const problem *p, int64_t e, int64_t i, int64_t pc) {
  const int64_t *s = p->op[A].st;
  return (src){p->op[A].x->base + pos(p, A, e, i, 0, pc) * p->w, s[ROW],
               s[CON]};
}

/* A panel: batch elements [e0, e0 + ne), b's columns [jc, jc + nc), the
   contraction [pc, pc + kc), packed: b at [b] in slivers of NR columns, a
   at [a] in blocks of [rows], each NULL where it is read in place. */
typedef struct {
  problem *p;
  int64_t e0, ne;
  int64_t jc, nc, pc, kc;
  int64_t slivers;
  int64_t mblocks, rows;      /* blocks of a, each of at most [rows] */
  int64_t ranges, per_range;  /* ranges of slivers a compute unit takes */
  nx_cpu_pack move_a, move_b; /* each job's pack, or NULL */
  uint8_t *b, *a;
  atomic_int *state;          /* per sliver of b: FREE, PACKING, PACKED */
} panel;

enum { FREE, PACKING, PACKED };

/* Splits the panel's slivers into ranges where its elements' blocks of a
   are fewer units than [want]: ranges of one block share its packing.
   Every range holds a sliver. */
static void split(panel *c, int64_t want) {
  int64_t units = c->ne * c->mblocks;
  c->ranges = units < want ? min64(c->slivers, ceil_div(want, units)) : 1;
  c->per_range = ceil_div(c->slivers, c->ranges);
  c->ranges = ceil_div(c->slivers, c->per_range);
}

/* Compute unit [u]'s work: element [e], slivers [v0, v1), the block of
   rows from [i0]. Threads claim units in turn. Where b is packed, units
   are (element, range, block) in C order, so that units that run together
   share their slivers and pack them between them (compute). Where b is
   read in place, they are (element, block, range), so that they share a's
   block and read b's columns apart: on the M1 Max a 256³ float32 product
   took 1.02 to 1.18 times as long in the other order. */
static void unit_of(const panel *c, int64_t u, int64_t *e, int64_t *i0,
                    int64_t *v0, int64_t *v1) {
  int64_t ib = u % c->mblocks, range = u / c->mblocks % c->ranges;
  if (c->b == NULL) {
    range = u % c->ranges;
    ib = u / c->ranges % c->mblocks;
  }
  *e = c->e0 + u / c->mblocks / c->ranges;
  *i0 = ib * c->rows;
  *v0 = range * c->per_range;
  *v1 = min64(c->slivers, *v0 + c->per_range);
}

/* Sliver [s] of element [e]'s panel, packed: kc steps of NR. */
static uint8_t *sliver(const panel *c, int64_t e, int64_t s) {
  int64_t nr = c->p->g->kernel.nr;
  return c->b + ((e - c->e0) * c->slivers + s) * nr * c->kc * c->p->w;
}

/* Block [ib] of element [e]'s rows of a, packed (packed_a) for its [*m]
   rows in [*lda], a multiple of MR. */
static uint8_t *block(const panel *c, int64_t e, int64_t ib, int64_t *m,
                      int64_t *lda) {
  const nx_cpu_gemm *g = c->p->g;
  *m = min64(c->rows, c->p->ext[ROW] - ib * c->rows);
  *lda = ceil_div(*m, g->kernel.mr) * g->kernel.mr;
  return c->a + ((e - c->e0) * c->mblocks + ib) * c->rows * c->kc * c->p->w;
}

/* The pack of a job of [total] units, [bytes] and [cost]: the target's
   where the job runs on at most one thread per copy of the kernel's
   hardware, else none. The matrix unit packs faster than a core, but its
   cluster's cores queue on it: on the M1 Max a 256³ product packs faster
   through it on two threads, a 512³ one through the stage on eight. */
static nx_cpu_pack mover(const problem *p, int64_t total, int64_t bytes,
                         double cost) {
  if (p->g->pack == NULL) return NULL;
  int64_t threads =
      min64(total, nx_cpu_threads_at(bytes, cost, thread_bytes(p->g)));
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

/* Packs [a]'s rows [i, i + m) of element [e], along k from [pc] for
   [kc], into [d], as the kernel reads them (packed_a). A block of rows is
   one pack: where it transposes, it does so in square blocks, where a
   sliver as narrow as a microkernel's 6 rows would move element by
   element. */
static void pack_a(const problem *p, nx_cpu_pack move, int64_t e, int64_t i,
                   int64_t m, int64_t lda, int64_t pc, int64_t kc,
                   uint8_t *d) {
  const int64_t *s = p->op[A].st;
  int64_t at = pos(p, A, e, i, 0, pc);
  if (p->g->kernel.a == NX_CPU_A_ELEMENTS) {
    pack(p, move, A, at, s[CON], s[ROW], kc, m, d, kc);
    return;
  }
  pack(p, move, A, at, s[ROW], s[CON], m, kc, d, lda);
}

/* A block of a packed at [d] for [kc] steps of [lda] elements: a kernel
   that reads MR elements of a step at once takes kc steps of lda, padding
   past its rows; one that reads an element at a time takes its rows, kc
   apart, so that a tile streams its MR rows. Read by steps of lda, a
   6-row tile uses 24 bytes of each line it loads and loads kc lines,
   which evict b's sliver from L1: on the i9-9900K (32 KiB) the AVX2
   kernel runs a 96-row block at 111 GFLOP/s so, 151 by rows. */
static src packed_a(const problem *p, const uint8_t *d, int64_t lda,
                    int64_t kc) {
  if (p->g->kernel.a == NX_CPU_A_ELEMENTS) return (src){d, kc, 1};
  return (src){d, 1, lda};
}

/* Packs the sliver of [b]'s columns [j, j + n) likewise, kc steps of
   [nr], padding past n. */
static void pack_b(const problem *p, nx_cpu_pack move, int64_t e, int64_t j,
                   int64_t n, int nr, int64_t pc, int64_t kc, uint8_t *d) {
  const int64_t *s = p->op[B].st;
  pack(p, move, B, pos(p, B, e, 0, j, pc), s[COL], s[CON], n, kc, d, nr);
}

/* b's sliver of columns [j, j + n) of element [e], the block of k from
   [pc]: in place where b is read so and the kernel reads its n columns,
   else packed into [d]. */
static src b_sliver(const problem *p, nx_cpu_pack move, int64_t e, int64_t j,
                    int64_t n, int64_t pc, int64_t kc, uint8_t *d) {
  int nr = p->g->kernel.nr;
  if (p->b_in && (n == nr || p->g->kernel.b == NX_CPU_B_COLUMNS))
    return (src){p->op[B].x->base + pos(p, B, e, 0, j, pc) * p->w, 0,
                 p->op[B].st[CON]};
  pack_b(p, move, e, j, n, nr, pc, kc, d);
  return (src){d, 0, nr};
}

/* A unit is a block of a's rows of one element. */
static void pack_blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const panel *c = ctx;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = c->e0 + u / c->mblocks, ib = u % c->mblocks, m, lda;
    uint8_t *d = block(c, e, ib, &m, &lda);
    pack_a(c->p, c->move_a, e, ib * c->rows, m, lda, c->pc, c->kc, d);
  }
}

/* Sliver [v] of element [e]'s panel, b's columns [j, j + n), into [b]:
   packed by the first unit that needs it. Where another unit is packing
   it, returns 0 without waiting, or, given [own], packs a copy of its own
   there, the same bytes. */
static int panel_sliver(const panel *c, int64_t e, int64_t v, int64_t j,
                        int64_t n, uint8_t *own, src *b) {
  int nr = c->p->g->kernel.nr;
  atomic_int *state = &c->state[(e - c->e0) * c->slivers + v];
  uint8_t *d = sliver(c, e, v);
  *b = (src){d, 0, nr};
  if (atomic_load_explicit(state, memory_order_acquire) == PACKED) return 1;
  int seen = FREE;
  int mine = atomic_compare_exchange_strong(state, &seen, PACKING);
  if (!mine && own == NULL) return 0;
  if (!mine) d = own;
  pack_b(c->p, c->move_b, e, j, n, nr, c->pc, c->kc, d);
  if (mine) atomic_store_explicit(state, PACKED, memory_order_release);
  *b = (src){d, 0, nr};
  return 1;
}

/* tile's path where R's columns are apart: MR rows at a time through a
   buffer, each way a box copy (nx_array.h), which transposes square blocks
   where R's rows are adjacent, as a transposed product's are. Apart from
   tile, so that tile's frame holds no buffer, which macOS would probe page
   by page at each call. */
static __attribute__((noinline)) void tile_apart(
    const problem *p, nx_cpu_micro k, nx_cpu_from f, int64_t kc, src a, src b,
    int64_t e, int64_t i, int64_t j, int64_t m, int64_t n) {
  _Alignas(ALIGN) uint8_t t[NX_CPU_TILE];
  const int64_t *s = p->op[DST].st;
  uint8_t *base = p->op[DST].x->base;
  int w = p->w;
  for (int64_t r0 = 0; r0 < m; r0 += k.mr) {
    int64_t rows = min64(k.mr, m - r0), at = pos(p, DST, e, i + r0, j, 0);
    nx_box in = {.extent = {1, rows, n},
                 .first = {0, at},
                 .step = {{0, k.nr, 1}, {0, s[ROW], s[COL]}}};
    nx_box out = {.extent = {1, rows, n},
                  .first = {at, 0},
                  .step = {{0, s[ROW], s[COL]}, {0, k.nr, 1}}};
    if (f == NX_CPU_FROM_TILE) nx_copy_box(t, base, &in, 8 * w);
    k.f(rows, n, kc, a.at + r0 * a.row * w, a.row, a.k, b.at, b.k, t, k.nr,
        f);
    nx_copy_box(base, t, &out, 8 * w);
  }
}

/* Adds the products of a's rows [i, i + m) and b's columns [j, j + n) of
   element [e], the kernel reading them from [a] and [b], over the block of
   k from [pc] to R with [k]'s kernel: in place where R's columns are
   adjacent, else through a buffer. */
static void tile(const problem *p, nx_cpu_micro k, int64_t pc, int64_t kc,
                 src a, src b, int64_t e, int64_t i, int64_t j, int64_t m,
                 int64_t n) {
  const int64_t *s = p->op[DST].st;
  nx_cpu_from f = from(p, pc);
  if (s[COL] == 1 || n == 1)
    k.f(m, n, kc, a.at, a.row, a.k, b.at, b.k, at_r(p, e, i, j), s[ROW], f);
  else tile_apart(p, k, f, kc, a, b, e, i, j, m, n);
}

/* A unit's work: each of its slivers of b against its block of a, the
   block's rows a kernel's call. b is the panel's (panel_sliver), or read
   in place (b_sliver), or packed into the worker's scratch.

   A sliver another unit is packing waits for the unit's end. Units that
   start together need the same slivers; were each to wait or copy, they
   would stay in step, one packing each sliver as the others wait or pack
   it too, and b's packing would run on one thread at a time: on the M1
   Max a 4096³ float32 product took 1.16 times as long as packing b in a
   job of its own. Deferring puts them out of step, each packing slivers
   the others then find packed. */
static void compute(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const panel *c = ctx;
  problem *p = c->p;
  const nx_cpu_gemm *g = p->g;
  nx_cpu_micro k = g->kernel;
  int64_t list = ceil_div(c->per_range * sizeof(int64_t), ALIGN) * ALIGN;
  uint8_t *s = scratch(p, list + k.nr * c->kc * p->w);
  if (s == NULL) return;
  int64_t *later = (int64_t *)s;
  uint8_t *bp = s + list;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e, i0, v0, v1;
    unit_of(c, u, &e, &i0, &v0, &v1);
    int64_t mc, lda;
    src a;
    if (c->a) {
      const uint8_t *d = block(c, e, i0 / c->rows, &mc, &lda);
      a = packed_a(p, d, lda, c->kc);
    } else {
      mc = min64(c->rows, p->ext[ROW] - i0);
      a = a_at(p, e, i0, c->pc);
    }
    int64_t j0 = c->jc + v0 * k.nr;
    int64_t j1 = min64(c->jc + c->nc, c->jc + v1 * k.nr);
    if (c->pc == 0) start(p, e, i0, i0 + mc, j0, j1);
    if (c->kc == 0) continue;
    int64_t nv = v1 - v0, deferred = 0;
    for (int64_t t = 0; t < nv + deferred; t++) {
      int64_t v = t < nv ? v0 + t : later[t - nv];
      int64_t j = c->jc + v * k.nr, n = min64(k.nr, c->jc + c->nc - j);
      src b;
      if (c->b == NULL) b = b_sliver(p, NULL, e, j, n, c->pc, c->kc, bp);
      else if (!panel_sliver(c, e, v, j, n, t < nv ? NULL : bp, &b)) {
        later[deferred++] = v;
        continue;
      }
      tile(p, k, c->pc, c->kc, a, b, e, i0, j, mc, n);
    }
  }
  nx_cpu_give(s);
}

/* Products of few rows (streams). One job packs a whole, each KC block as
   the kernel reads it (packed_a), unless a is read in place; then a unit
   is a sliver of b's columns of one element, which it reads in place or
   packs, KC block by KC block, and adds into R. b, the large operand, is
   read once, by one thread, along all of k. */
typedef struct {
  problem *p;
  nx_cpu_micro k;
  nx_cpu_pack move_a, move_b; /* each job's pack, or NULL */
  int64_t lda, kblocks, slivers;
  uint8_t *a; /* element e's step q at a + (e·K + q)·lda·w, or NULL */
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
  (void)worker;
  const few_rows_job *r = ctx;
  problem *p = r->p;
  int64_t m = p->ext[ROW], n_all = p->ext[COL], k = p->ext[CON], w = p->w;
  int64_t kc_most = min64(p->g->kc, k), nr = r->k.nr;
  uint8_t *bp = scratch(p, nr * kc_most * w);
  if (bp == NULL) return;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = u / r->slivers, j = u % r->slivers * nr;
    int64_t n = min64(nr, n_all - j), pc = 0;
    start(p, e, 0, m, j, j + n);
    while (pc < k) {
      int64_t kc = min64(kc_most, k - pc);
      src a = r->a ? packed_a(p, r->a + (e * k + pc) * r->lda * w,
                              r->lda, kc)
                   : a_at(p, e, 0, pc);
      src b = b_sliver(p, r->move_b, e, j, n, pc, kc, bp);
      tile(p, r->k, pc, kc, a, b, e, 0, j, m, n);
      pc += kc;
    }
  }
  nx_cpu_give(bp);
}

/* A job of the table's kernels, each of its bodies run between the table's
   enter and leave: on the M1, a body's calls share one live state of the
   matrix unit, where a state per call drained the unit at each end. */
typedef struct {
  const nx_cpu_gemm *g;
  rig_pool_body body;
  void *ctx;
} bracket;

static void bracketed(int64_t lo, int64_t hi, int worker, void *ctx) {
  const bracket *b = ctx;
  b->g->enter();
  b->body(lo, hi, worker, b->ctx);
  b->g->leave();
}

static void job(const problem *p, int64_t total, int64_t bytes, int64_t cost,
                rig_pool_body body, void *ctx) {
  double per = thread_bytes(p->g);
  if (p->g->enter == NULL) {
    nx_cpu_job_at(total, bytes, cost, per, body, ctx);
    return;
  }
  bracket b = {p->g, body, ctx};
  nx_cpu_job_at(total, bytes, cost, per, bracketed, &b);
}

/* The rows of [m] that [k] computes: a multiple of its tile's. */
static int64_t padded(nx_cpu_micro k, int64_t m) {
  return ceil_div(m, k.mr) * k.mr;
}

/* [g] or its other kernels, whichever computes [m] rows in less time: the
   kernel few_rows or chain takes, its rows padded to its tile, at its
   speed on every copy of its hardware. On the M1 Max 8 rows of float32
   are 8 on the vector units of 8 cores at speed 1, against 32 on 2 matrix
   units at 13: at most 8 rows run on the cores, more on the units. */
static const nx_cpu_gemm *fastest(const nx_cpu_gemm *g, int64_t m) {
  const nx_cpu_gemm *o = g->other;
  if (o == NULL) return g;
  nx_cpu_micro a = g->kernel, b = o->kernel;
  double ta = padded(a, m) * b.flops_per_byte * engines(b);
  double tb = padded(b, m) * a.flops_per_byte * engines(a);
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
  int64_t packed = p->a_in ? 0 : batch * k * r.lda * w;
  if (!p->a_in) r.a = nx_cpu_take(packed);
  if (!p->a_in && r.a == NULL) {
    atomic_store(&p->failed, 1);
    return;
  }
  int64_t bytes = batch * (n + m) * k * w;
  double cost = flop_cost(km, 2.0 * batch * m * n * k) + bytes;
  r.move_a = mover(p, batch * r.kblocks, packed, PACK_COST * packed);
  r.move_b = mover(p, batch * r.slivers, bytes, cost);
  if (!p->a_in)
    job(p, batch * r.kblocks, packed, PACK_COST * packed, few_rows_pack, &r);
  job(p, batch * r.slivers, bytes, cost, few_rows_unit, &r);
  if (!p->a_in) nx_cpu_give(r.a);
}

/* Blocks of a hold equal rows, a multiple of MR, so that a job's units
   take equal times: 100 rows are blocks of 54 and 46, not 96 and 4. */
static void chain(problem *p) {
  const nx_cpu_gemm *g = p->g;
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w;
  int64_t nc = min64(g->nc, n), nr = g->kernel.nr, kc_most = min64(g->kc, k);
  int64_t slivers = ceil_div(nc, nr);
  int64_t rows = padded(g->kernel, ceil_div(m, ceil_div(m, g->mc)));
  int64_t mblocks = ceil_div(m, rows);
  int64_t b_bytes = p->b_in ? 0 : slivers * nr * kc_most * w;
  int64_t a_bytes = p->a_in ? 0 : mblocks * rows * kc_most * w;
  int64_t per = b_bytes + a_bytes;
  int64_t group = per > 0 ? PANELS / per : batch;
  group = group < 1 ? 1 : min64(group, batch);
  int64_t b_all = ceil_div(group * b_bytes, ALIGN) * ALIGN;
  int64_t a_all = ceil_div(group * a_bytes, ALIGN) * ALIGN;
  int64_t states = p->b_in ? 0 : group * slivers;
  uint8_t *b = nx_cpu_take(b_all + a_all + states * sizeof(atomic_int));
  if (b == NULL) {
    atomic_store(&p->failed, 1);
    return;
  }
  uint8_t *a = b + b_all;
  atomic_int *state = (atomic_int *)(a + a_all);
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
                 .b = p->b_in ? NULL : b,
                 .a = p->a_in ? NULL : a,
                 .state = state};
      /* Enough units for the threads the compute job takes. */
      int64_t packed = ne * (c.slivers * nr + m) * kc_most * w;
      double cost =
          flop_cost(g->kernel, 2.0 * ne * m * c.nc * kc_most) + packed;
      split(&c, units(packed, cost, thread_bytes(g)));
      int64_t total = ne * c.ranges * mblocks, pc = 0;
      do {
        c.pc = pc;
        c.kc = min64(g->kc, k - pc);
        double flops = 2.0 * ne * m * c.nc * c.kc;
        int64_t a_packed = c.a ? ne * m * c.kc * w : 0;
        packed = ne * c.slivers * nr * c.kc * w + a_packed;
        cost = flop_cost(g->kernel, flops) + packed;
        c.move_a = mover(p, ne * mblocks, a_packed, PACK_COST * a_packed);
        c.move_b = mover(p, total, packed, cost);
        if (c.kc > 0 && c.a)
          job(p, ne * mblocks, a_packed, PACK_COST * a_packed, pack_blocks,
              &c);
        if (c.b)
          for (int64_t v = 0; v < ne * c.slivers; v++)
            atomic_store_explicit(&state[v], FREE, memory_order_relaxed);
        job(p, total, packed, cost, compute, &c);
        if (atomic_load(&p->failed)) goto done;
        pc += g->kc;
      } while (pc < k);
    }
  }
done:
  nx_cpu_give(b);
}

/* Lane order */

/* Rows are the fewer of the two (lanes transposes the product), at most
   LANE_ROWS a pass. b, the large operand, is read once a pass, in the
   order it lies:

   - Dots, where b is adjacent along k, as a weight stored [n × k] is: a
     unit is (element, DOT_COLUMNS columns, range of blocks), and each
     output's block a dot along k, a column of b against every row of a. A
     product of few outputs over a long k, a dot's, splits its blocks into
     ranges, whose sums meet in a second job; otherwise a unit takes every
     block and totals its outputs itself.
   - Rows, otherwise, as for a weight stored [k × n]: a block of ROW_BYTES
     of columns adds the block's rows of b in turn, row k0 + t times its
     row's element of a into lane t modulo 16's accumulators, one per
     output. Where an element's groups of columns are enough units for the
     threads, a unit takes a group whole, sums each block's lanes by the
     lanes' tree and totals its outputs; otherwise a unit is a span of
     the lanes of one block of one group, and a second job sums and totals
     them. */

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
      int64_t k0 = x * NX_CPU_LANE_BLOCK;
      int64_t len = min64(NX_CPU_LANE_BLOCK, k - k0);
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
  (void)worker;
  const dots_job *f = ctx;
  problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL], w = p->w;
  int64_t b_bytes = NX_CPU_LANE_BLOCK * w;
  int64_t own = f->ranges == 1 ? m * DOT_COLUMNS * f->blocks * w : 0;
  uint8_t *bp = scratch(p, b_bytes + own);
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
  nx_cpu_give(bp);
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
static void dots(problem *p, int64_t bytes, double cost) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w, outputs = batch * m * n;
  dots_job f = {.p = p,
                .blocks = ceil_div(k, NX_CPU_LANE_BLOCK),
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
  int64_t want = units(bytes, cost, NX_CPU_THREAD_BYTES);
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
  int64_t spans;                /* of a block's lanes, each a unit's */
  int64_t staged;               /* a unit's staging of a and b */
  int64_t own;                  /* a worker's scratch */
  uint8_t *lanes; /* split: lane l of block x of output (i, j) of element e
                     at ((((e·blocks + x)·LANES + l)·M + i)·N + j)·w */
} rows_job;

static uint8_t *lane_at(const rows_job *f, int64_t e, int64_t x, int64_t l,
                        int64_t i, int64_t j) {
  const problem *p = f->p;
  int64_t at = ((e * f->blocks + x) * NX_CPU_LANES + l) * p->ext[ROW] + i;
  return f->lanes + (at * p->ext[COL] + j) * p->w;
}

/* Lanes [q0, q1) of block [x] of element [e]'s columns [j0, j0 + c),
   from +0, lane q's row i at l + (q·ls + i·ldy)·w: the block's elements
   of a staged into [ap], then the lanes' rows of b in turn, in place or
   staged NX_CPU_LANES rows at a time, each added times its row's element
   of a into its lane, prefetching the next row. b's rows are read in the
   order they lie: read a lane at a time, every 16th row, b came at 35 GB/s
   on the i9-9900K's 16 threads, whose memory gives 42. */
static void rows_block(const rows_job *f, uint8_t *ap, int64_t e, int64_t x,
                       int64_t q0, int64_t q1, int64_t j0, int64_t c,
                       uint8_t *l, int64_t ls, int64_t ldy) {
  const problem *p = f->p;
  const int64_t *sa = p->op[A].st, *sb = p->op[B].st;
  int64_t m = p->ext[ROW], w = p->w, k0 = x * NX_CPU_LANE_BLOCK;
  int64_t len = min64(NX_CPU_LANE_BLOCK, p->ext[CON] - k0);
  for (int64_t q = q0; q < q1; q++)
    for (int64_t i = 0; i < m; i++)
      memset(l + (q * ls + i * ldy) * w, 0, (size_t)(c * w));
  stage_acc(p, A, pos(p, A, e, 0, 0, k0), sa[ROW], sa[CON], m, len, ap, m);
  uint8_t *bp = ap + m * NX_CPU_LANE_BLOCK * w;
  nx_cpu_axpy axpy = nx_cpu_table->axpy[p->acc];
  for (int64_t t = 0; t + q0 < len; t += NX_CPU_LANES) {
    int64_t r = min64(q1, len - t), at = pos(p, B, e, 0, j0, k0 + t + q0);
    /* Rows from q0: in place, the next row to prefetch after the last
       lane's is the next group's first, within the block; staged, the
       group's last row is the last the buffer holds. */
    const uint8_t *vb = bp;
    int64_t ldb = c, wrap = r - 1 - q0, last = r - 1 - q0;
    if (adjacent(p, B, COL)) {
      vb = p->op[B].x->base + at * w;
      ldb = sb[CON];
      wrap = NX_CPU_LANES;
      last = len - t - q0 - 1;
    } else {
      stage_acc(p, B, at, sb[COL], sb[CON], c, r - q0, bp, c);
    }
    for (int64_t q = q0; q < r; q++) {
      const uint8_t *row = vb + (q - q0) * ldb * w, *aq = ap + (t + q) * m * w;
      int64_t up = q + 1 < r ? q + 1 - q0 : wrap;
      const uint8_t *next = vb + min64(up, last) * ldb * w;
      for (int64_t i0 = 0; i0 < m; i0 += NX_CPU_DOT_ROWS)
        axpy(aq + i0 * w, 1, (int)min64(NX_CPU_DOT_ROWS, m - i0), row, c,
             l + (q * ls + i0 * ldy) * w, ldy, next);
    }
  }
}

/* Block [x]'s lanes at [l], lane q's row i at l + (q·ls + i·ldy)·w, summed
   by the lanes' tree into lane 0's, and each output's sum copied to [s] at
   ((i·c + j)·blocks + x)·w. */
static void block_sums(const rows_job *f, uint8_t *l, int64_t ls,
                       int64_t ldy, int64_t c, int64_t x, uint8_t *s) {
  const problem *p = f->p;
  int64_t w = p->w;
  for (int64_t i = 0; i < p->ext[ROW]; i++) {
    uint8_t *li = l + i * ldy * w;
    lane_sums(p->acc, li, ls, c);
    for (int64_t j = 0; j < c; j++)
      memcpy(s + ((i * c + j) * f->blocks + x) * w, li + j * w, (size_t)w);
  }
}

/* Each output of element [e]'s columns [j0, j0 + c): the total of its
   block sums at [s]. */
static void totals(const rows_job *f, int64_t e, int64_t j0, int64_t c,
                   const uint8_t *s) {
  const problem *p = f->p;
  for (int64_t i = 0; i < p->ext[ROW]; i++)
    for (int64_t j = 0; j < c; j++)
      total(p, e, i, j0 + j, s + (i * c + j) * f->blocks * p->w, f->blocks,
            at_r(p, e, i, j0 + j));
}

/* A unit is (element, group of columns), every block: the lanes in the
   worker's scratch after its staging, and the block sums after them. */
static void rows_whole(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const rows_job *f = ctx;
  problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL], w = p->w;
  uint8_t *ap = scratch(p, f->own);
  if (ap == NULL) return;
  uint8_t *l = ap + f->staged, *s = l + NX_CPU_LANES * m * f->cols * w;
  for (int64_t u = lo; u < hi; u++) {
    int64_t g = u % f->groups, e = u / f->groups;
    int64_t j0 = g * f->cols, c = min64(f->cols, n - j0);
    for (int64_t x = 0; x < f->blocks; x++) {
      rows_block(f, ap, e, x, 0, NX_CPU_LANES, j0, c, l, m * c, c);
      block_sums(f, l, m * c, c, c, x, s);
    }
    totals(f, e, j0, c, s);
  }
  nx_cpu_give(ap);
}

/* A unit is (element, block, group of columns): its lanes into the job's
   lanes. */
static void rows_unit(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const rows_job *f = ctx;
  problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL];
  uint8_t *ap = scratch(p, f->own);
  if (ap == NULL) return;
  int64_t lanes = NX_CPU_LANES / f->spans;
  for (int64_t u = lo; u < hi; u++) {
    int64_t q0 = u % f->spans * lanes, v = u / f->spans;
    int64_t g = v % f->groups, x = v / f->groups % f->blocks;
    int64_t e = v / f->groups / f->blocks;
    int64_t j0 = g * f->cols, c = min64(f->cols, n - j0);
    rows_block(f, ap, e, x, q0, q0 + lanes, j0, c, lane_at(f, e, x, 0, 0, j0),
               m * n, n);
  }
  nx_cpu_give(ap);
}

/* A unit is (element, group of columns): each block's lanes summed into
   the worker's scratch, then each output's total. */
static void rows_finish(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const rows_job *f = ctx;
  problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL];
  uint8_t *s = scratch(p, f->own);
  if (s == NULL) return;
  for (int64_t u = lo; u < hi; u++) {
    int64_t g = u % f->groups, e = u / f->groups;
    int64_t j0 = g * f->cols, c = min64(f->cols, n - j0);
    for (int64_t x = 0; x < f->blocks; x++)
      block_sums(f, lane_at(f, e, x, 0, 0, j0), m * n, n, c, x, s);
    totals(f, e, j0, c, s);
  }
  nx_cpu_give(s);
}

/* A unit takes a group of columns whole where the groups are enough for
   the threads; otherwise each span of lanes of each block of each group
   is a unit, and a second job sums them. */
static void rows(problem *p, int64_t bytes, double cost) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t w = p->w, cols = min64(n, ROW_BYTES / w);
  rows_job f = {.p = p,
                .blocks = ceil_div(p->ext[CON], NX_CPU_LANE_BLOCK),
                .cols = cols,
                .groups = ceil_div(n, cols),
                .staged = (m * NX_CPU_LANE_BLOCK + NX_CPU_LANES * cols) * w};
  int64_t sums = m * cols * f.blocks * w;
  if (batch * f.groups >= units(bytes, cost, NX_CPU_THREAD_BYTES)) {
    f.own = f.staged + NX_CPU_LANES * m * cols * w + sums;
    nx_cpu_job(batch * f.groups, bytes, cost, rows_whole, &f);
    return;
  }
  f.own = f.staged > sums ? f.staged : sums;
  int64_t lanes = batch * f.blocks * NX_CPU_LANES * m * n * w;
  f.lanes = alloc(lanes);
  if (f.lanes == NULL) {
    atomic_store(&p->failed, 1);
    return;
  }
  /* Lanes split into spans, a power of two, until the units are enough
     for cores that run at unequal speeds to share; each lane's rows and
     the sum's bits stay those of a whole block. */
  int64_t blocks = batch * f.blocks * f.groups;
  int64_t want = units(bytes, cost, NX_CPU_THREAD_BYTES);
  for (f.spans = 1; f.spans < NX_CPU_LANES && blocks * f.spans < want;)
    f.spans *= 2;
  nx_cpu_job(blocks * f.spans, bytes, cost, rows_unit, &f);
  if (!atomic_load(&p->failed))
    nx_cpu_job(batch * f.groups, lanes, 2 * lanes, rows_finish, &f);
  free(f.lanes);
}

static void lane_pass(problem *p) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w;
  if (k == 0) {
    nx_cpu_job(batch, batch * m * n * w, batch * m * n * w, empty, p);
    return;
  }
  /* The cores' kernels run a flop in the time memcpy moves a byte. */
  int64_t bytes = batch * (m + n) * k * w;
  double cost = 2.0 * batch * m * n * k + bytes;
  if (p->op[B].st[CON] == 1) dots(p, bytes, cost);
  else rows(p, bytes, cost);
}

/* Passes of LANE_ROWS rows: each moves a, init and dst on by its rows. */
static void lanes(problem *p) {
  if (p->ext[COL] < p->ext[ROW]) transpose(p);
  int64_t m = p->ext[ROW];
  for (int64_t i = 0; i < m && !atomic_load(&p->failed); i += LANE_ROWS) {
    p->ext[ROW] = min64(LANE_ROWS, m - i);
    lane_pass(p);
    for (int o = 0; o < 4; o++)
      p->op[o].first += p->ext[ROW] * p->op[o].st[ROW];
  }
}

/* The entry */

/* Whether chain order reads operand [o] of [p] in place: in acc, adjacent
   along the axis the kernel streams (b along its columns, a along k where
   the kernel reads it an element at a time), and within its bound: a,
   IN_PLACE_BYTES an element; b, half a core's L2. b's sliver, kc rows a
   row of b apart, is read again for each tile of a's rows, and past the
   bound its rows fall out between tiles. On the i9-9900K (256 KiB) a 256³
   float32 product takes 406 µs on one core with b in place against 244
   packed, a 512³ one 683 against 322 µs on 16 threads; a 128³ one, within
   the bound, 33.2 against 33.9 µs on one core. On kimchi (3 MiB) a 4096³
   one took twice as long in place (205 against 102 ms). */
static int in_place(const problem *p, int o) {
  int64_t n = o == A ? p->ext[ROW] : p->ext[COL];
  int64_t most = o == A ? IN_PLACE_BYTES : nx_cpu_table->l2 / 2;
  if (p->ext[CON] > most / p->w / n) return 0;
  if (o == B) return adjacent(p, B, COL);
  return p->g->kernel.a == NX_CPU_A_ELEMENTS && adjacent(p, A, CON);
}

/* Whether [p] streams b past all of a (few_rows) rather than run in chain
   order: where a is one block of rows, or where b, read along k, is larger
   than a core's L2 and a stays within the table's few rows. Chain order's
   jobs, one per KC block, read such a b in pieces KC long; the streaming
   job reads each sliver along all of k. Where b is smaller, chain order's
   units, more than b's slivers, spread over the threads: on fudge's six
   performance cores a 128³ float32 product took 8.3 µs streamed, 6.7 in
   chain order. */
static int streams(const problem *p) {
  int64_t m = p->ext[ROW], b_bytes = p->ext[CON] * p->ext[COL] * p->w;
  if (m <= p->g->mc) return 1;
  return m <= p->g->few && adjacent(p, B, CON) && b_bytes > nx_cpu_table->l2;
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

int nx_cpu_contract_on(int acc, int order, const nx_contract_view *v,
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
  atomic_init(&p.failed, 0);
  if (order == NX_CPU_LANE) lanes(&p);
  else {
    if (p.ext[COL] < p.ext[ROW]) {
      transpose(&p);
      if (!streams(&p)) transpose(&p);
    }
    p.g = fastest(p.g, p.ext[ROW]);
    p.a_in = in_place(&p, A);
    p.b_in = in_place(&p, B);
    if (streams(&p)) few_rows(&p);
    else chain(&p);
  }
  return !atomic_load(&p.failed);
}

double nx_cpu_core_speed(int acc) {
  const nx_cpu_gemm *g = &nx_cpu_table->gemm[acc];
  if (g->kernel.shared > 1 && g->other != NULL) g = g->other;
  return g->kernel.flops_per_byte;
}

/* The external: groups the view of the pairs [vbatch] and [vcontracting]
   on the stack, answering what the fill answers where it does not group,
   declines what it does not compute, then reads the operands through the
   door. [vorder] is the order, [vacc] acc's dtype code and [vi] the init,
   an option. */
value nx_cpu_contract(value vorder, value vbatch, value vcontracting,
                      value vacc, value vdst, value vi, value va, value vb) {
  int acc = Int_val(vacc), init = Is_block(vi);
  value vinit = init ? Field(vi, 0) : va;
  nx_contract_view v;
  int fit = nx_contract_view_fill(&v, vbatch, vcontracting, vdst, va, vb, vi);
  if (fit != NX_OK) return Val_int(fit);
  int dts[3] = {nx_array_dtype(va), nx_array_dtype(vb),
                nx_array_dtype(vinit)};
  if (acc < 0 || acc >= NX_DTYPE_COUNT ||
      !computes(acc, nx_array_dtype(vdst), dts, 2 + init))
    return Val_int(NX_DECLINED);
  nx_operand in[4] = {{va, dts[0], 0},
                      {vb, dts[1], 0},
                      {vdst, acc, 1},
                      {vinit, dts[2], 0}};
  nx_array a[4];
  int n = 3 + init;
  int e = nx_read(n, in, a);
  if (e) return Val_int(e);
  const nx_array *op[4] = {&a[0], &a[1], init ? &a[3] : NULL, &a[2]};
  int done = nx_cpu_contract_on(acc, Int_val(vorder), &v, op);
  nx_done(n, a);
  if (!done) caml_raise_out_of_memory();
  return Val_int(NX_OK);
}

value nx_cpu_contract_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_contract(argv[0], argv[1], argv[2], argv[3], argv[4],
                         argv[5], argv[6], argv[7]);
}
