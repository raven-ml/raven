/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contractions: y[b, i, j] = init[b, i, j] + Σ_k a[b, i, k] · b[b, k, j],
   computed in acc and rounded once to y's dtype.

   nx_contract_view_fill groups the axes, as Nx_kernel.Contract_view does,
   into one batch, row (M), column (N) and contracted (K) axis per
   operand; a group that makes several runs takes an outer loop or a copy
   (Groups of several runs). An operand is read
   through the stage, which takes any strides and dtype into its carrier, and
   the target's conversion takes the carrier into acc: each element as a
   cast to acc gives it, exact where acc holds its values. A scaled
   operand is staged wherever it is read, each element then multiplied by
   its block's scale in acc (stage_at, Scales). An integer acc
   computes on chain order's paths whatever the order asked. Lane order
   reads b in place where it is acc and adjacent along the axis it streams.

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
   transpose where that streams, unless a is read in place: then a is read
   where it lies, as BLIS's small products read it (bli_gemmsup), in chain
   order's blocks of rows, which give the threads units where b's few
   slivers would not. Where a table names other kernels, a product runs on
   those that take less time for its rows, or on one thread where it runs
   alone (fastest): on the M1, the cores' vector units take products of
   few rows, and small ones, from its matrix unit.

   R, the sum in acc, is dst itself where dst is of acc's dtype, and
   otherwise scratch in acc that a cast walk rounds into dst once every
   product is in (Results of another dtype). Each tile starts as init, or
   +0, when its first KC block is computed. */

#include <math.h>
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

/* An operand as the view lays it out: its element (e, i, j, k) lies at
   position first + e·st[BATCH] + i·st[ROW] + j·st[COL] + k·st[CON], and
   an axis it lacks has step 0. */
typedef struct {
  const nx_array *x; /* NULL for an absent init */
  int64_t first;
  int64_t st[4];
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

static int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
static int64_t gcd(int64_t a, int64_t b) { return b == 0 ? a : gcd(b, a % b); }
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

/* The dtype of [acc]'s parts: float32 or float64 for a complex [acc],
   [acc] itself otherwise. */
static int part_of(int acc) {
  if (acc == NX_COMPLEX64) return NX_FLOAT32;
  return acc == NX_COMPLEX128 ? NX_FLOAT64 : acc;
}

static int is_integer(int dt) {
  int k = nx_dtype_row_of(dt).kind;
  return k == NX_KIND_SIGNED || k == NX_KIND_UNSIGNED;
}

/* The dtype a sum in [acc] is computed in: int8 and uint8 for int4 and
   uint4, whose sums modulo 2^8 are theirs modulo 2^4 once narrowed, as
   the int8 kernels' are; [acc] itself otherwise. */
static int summed_in(int acc) {
  if (acc == NX_INT4) return NX_INT8;
  return acc == NX_UINT4 ? NX_UINT8 : acc;
}

/* Whether nx.cpu computes a contraction in [acc]: one its kernels sum
   in, of operands of any dtype, into a dst of any dtype. */
static int computes(int acc) {
  return nx_cpu_table->gemm[part_of(summed_in(acc))].kernel.f != NULL;
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

/* The codes along k that MXFP4 scales as one block: a run of them is
   staged through one table. */
#define F4_RUN 32

/* stage_f4's lines in acc's type [T]: each of [lines] lines of [len]
   codes, [sl] apart from [first] and [sk] apart along k, into dst [du]
   apart along k and [dl] apart across lines, each run of F4_RUN codes
   through its 16 values times the run's scale. */
#define F4_LINES(T)                                                        \
  do {                                                                     \
    const T *v = (const T *)s->v;                                          \
    T *d = (T *)dst, base[16], tab[16];                                    \
    for (int u = 0; u < 16; u++)                                           \
      base[u] = (T)nx_e2m1fn_to_float((uint8_t)u);                         \
    for (int64_t l = 0; l < lines; l++) {                                  \
      int64_t at = 0, line = first + l * sl;                               \
      for (int x = BATCH; x <= COL; x++)                                   \
        if (s->t[x]) at += s->t[x][c[x] + (x == across ? l : 0)];          \
      for (int64_t u0 = 0; u0 < len; u0 += F4_RUN) {                       \
        T sv = v[at + tk[u0]];                                             \
        for (int u = 0; u < 16; u++) tab[u] = base[u] * sv;                \
        int64_t un = min64(F4_RUN, len - u0);                              \
        if (pairs && un == F4_RUN) {                                       \
          const uint8_t *b = codes + ((line + u0) >> 1);                   \
          T *y = d + u0 * du + l * dl;                                     \
          for (int h = 0; du == 1 && h < F4_RUN / 2; h++) {                \
            y[2 * h] = tab[b[h] & 15];                                     \
            y[2 * h + 1] = tab[b[h] >> 4];                                 \
          }                                                                \
          for (int h = 0; du != 1 && h < F4_RUN / 2; h++) {                \
            y[2 * h * du] = tab[b[h] & 15];                                \
            y[(2 * h + 1) * du] = tab[b[h] >> 4];                          \
          }                                                                \
          continue;                                                        \
        }                                                                  \
        for (int64_t u = u0; u < u0 + un; u++) {                           \
          int64_t xi = line + u * sk;                                      \
          d[u * du + l * dl] = tab[(codes[xi >> 1] >> ((xi & 1) * 4)) & 15]; \
        }                                                                  \
      }                                                                    \
    }                                                                      \
  } while (0)

/* Stages operand [o]'s block as stage_at does where it is float4 and its
   block along k starts a run of its scale's, [run] a multiple of
   F4_RUN, so each run of F4_RUN codes takes one scale: 1 if it did. The
   bits are stage_at's: each code's value times its scale, rounded once. */
static int stage_f4(const problem *p, int o, const int64_t *c, int x0,
                    int x1, int64_t n0, int64_t n1, uint8_t *dst,
                    int64_t pitch) {
  const operand *q = &p->op[o];
  const scale *s = q->s;
  if (q->x->dtype != NX_FLOAT4_E2M1FN || (x0 != CON && x1 != CON) ||
      s->run % F4_RUN || c[CON] % F4_RUN)
    return 0;
  int along = x0 == CON, across = along ? x1 : x0;
  int64_t len = along ? n0 : n1, lines = along ? n1 : n0;
  int64_t du = along ? 1 : pitch, dl = along ? pitch : 1;
  int64_t first = pos(p, o, c[BATCH], c[ROW], c[COL], c[CON]);
  int64_t sk = q->st[CON], sl = q->st[across];
  const uint8_t *codes = q->x->base;
  const int64_t *tk = s->t[CON] + c[CON];
  /* Codes adjacent along k, each line from a byte's first: a byte's two
     codes at once. */
  int pairs = sk == 1 && first % 2 == 0 && sl % 2 == 0;
  if (p->acc == NX_FLOAT32) F4_LINES(float);
  else F4_LINES(double);
  return 1;
}

/* Stages operand [o]'s block as stage_acc does, from its element
   (e, i, j, k), [n0] along axis [x0] in a row and [n1] rows along [x1];
   then, where it is scaled, multiplies each element by its block's scale,
   rounded once in acc, a float32 or float64. */
static void stage_at(const problem *p, int o, int64_t e, int64_t i,
                     int64_t j, int64_t k, int x0, int x1, int64_t n0,
                     int64_t n1, uint8_t *dst, int64_t pitch) {
  const operand *q = &p->op[o];
  const int64_t c[4] = {e, i, j, k};
  if (q->s && stage_f4(p, o, c, x0, x1, n0, n1, dst, pitch)) return;
  stage_acc(p, o, pos(p, o, e, i, j, k), q->st[x0], q->st[x1], n0, n1, dst,
            pitch);
  if (q->s == NULL) return;
  const scale *s = q->s;
  const int64_t *t0 = s->t[x0] + c[x0];
  for (int64_t r = 0; r < n1; r++) {
    int64_t at = 0;
    for (int x = BATCH; x <= CON; x++)
      if (x != x0 && s->t[x]) at += s->t[x][c[x] + (x == x1 ? r : 0)];
    if (p->acc == NX_FLOAT32) {
      float *d = (float *)dst + r * pitch;
      const float *v = (const float *)s->v + at;
      for (int64_t u = 0; u < n0; u++) d[u] *= v[t0[u]];
    } else {
      double *d = (double *)dst + r * pitch;
      const double *v = (const double *)s->v + at;
      for (int64_t u = 0; u < n0; u++) d[u] *= v[t0[u]];
    }
  }
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

/* Whether operand [o] may be read in place: in acc and unscaled. */
static int raw(const problem *p, int o) {
  return p->op[o].x->dtype == p->acc && p->op[o].s == NULL;
}

/* Whether operand [o] is read in place along [ax]: raw, adjacent there. */
static int adjacent(const problem *p, int o, int ax) {
  return raw(p, o) && p->op[o].st[ax] == 1;
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

/* Moves operand [o]'s block as stage_acc does. A block in acc moves through
   [move] where it is a pack and the block's rows or columns are
   contiguous, else by a memcpy a row where its rows are contiguous: for a
   small block the stage's setup costs more than the copy. */
static void pack(const problem *p, nx_cpu_pack move, int o, int64_t at,
                 int64_t s0, int64_t s1, int64_t n0, int64_t n1, uint8_t *d,
                 int64_t pitch) {
  const nx_array *x = p->op[o].x;
  int w = p->w;
  if (x->dtype != p->acc) {
    stage_acc(p, o, at, s0, s1, n0, n1, d, pitch);
    return;
  }
  if (move && (s0 == 1 || s1 == 1)) {
    move(n0, n1, x->base + at * w, s0, s1, d, pitch);
    return;
  }
  if (s0 == 1) {
    for (int64_t j = 0; j < n1; j++)
      memcpy(d + j * pitch * w, x->base + (at + j * s1) * w, (size_t)(n0 * w));
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
  int elements = p->g->kernel.a == NX_CPU_A_ELEMENTS;
  if (p->op[A].s && elements)
    stage_at(p, A, e, i, 0, pc, CON, ROW, kc, m, d, kc);
  else if (p->op[A].s) stage_at(p, A, e, i, 0, pc, ROW, CON, m, kc, d, lda);
  else if (elements) pack(p, move, A, at, s[CON], s[ROW], kc, m, d, kc);
  else pack(p, move, A, at, s[ROW], s[CON], m, kc, d, lda);
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
  if (p->op[B].s) stage_at(p, B, e, 0, j, pc, COL, CON, n, kc, d, nr);
  else pack(p, move, B, pos(p, B, e, 0, j, pc), s[COL], s[CON], n, kc, d, nr);
}

/* Whether a sliver of [n] columns of b is packed: unless b is read in
   place and the kernel reads its n columns. */
static int packs_b(const problem *p, int64_t n) {
  int nr = p->g->kernel.nr;
  nx_cpu_b_read r = p->g->kernel.b;
  return !(p->b_in && (n == nr || r == NX_CPU_B_COLUMNS ||
                       (r == NX_CPU_B_HALVES && 2 * n >= nr)));
}

/* b's sliver of columns [j, j + n) of element [e], the block of k from
   [pc]: in place, or packed into [d] (packs_b). */
static src b_sliver(const problem *p, nx_cpu_pack move, int64_t e, int64_t j,
                    int64_t n, int64_t pc, int64_t kc, uint8_t *d) {
  int nr = p->g->kernel.nr;
  if (!packs_b(p, n))
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
   buffer, each way a box copy (nx_copy_box), which transposes square blocks
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
   is a sliver of b's columns of one element and a block of a's rows,
   which it reads in place or packs, KC block by KC block, and adds into
   R. b, the large operand, is read once along all of k, by one thread.
   Where b is read in place, within the caches, and its slivers do not
   share evenly among the job's threads, the rows split into the fewest
   blocks that make the units a multiple of the threads, each reading the
   sliver again; a block holds at least two of the kernel's tiles of rows,
   and where that leaves fewer blocks, the floor wins and the units stay
   uneven. Every table's kernels take this path, the M1's NEON integers
   and AMX float64 among them. On fudge's six performance cores a 24 × 24 × 512 float32 product took
   3.40 µs on b's 2 slivers, 2.29 in blocks; 64³ float64 2.63 on 8
   slivers, 2.13; 8 × 512 × 64 float32 2.26 in one-tile blocks against
   2.13 unsplit, and int8 128 rows by a 4096 × 4096 weight, b from
   memory, 1.61 ms in two blocks against 1.42. */
typedef struct {
  problem *p;
  nx_cpu_micro k;
  nx_cpu_pack move_a, move_b; /* each job's pack, or NULL */
  int64_t lda, kblocks, slivers;
  int64_t rows;    /* a block's rows of a, a multiple of the kernel's */
  int64_t mblocks; /* the blocks of a's rows a sliver of b is split into */
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
  /* Scratch where a sliver is packed: the edge, or every one. */
  int packs = packs_b(p, nr) || (n_all % nr && packs_b(p, n_all % nr));
  uint8_t *bp = packs ? scratch(p, nr * kc_most * w) : NULL;
  if (packs && bp == NULL) return;
  for (int64_t u = lo; u < hi; u++) {
    /* One division a unit where the rows do not split. */
    int64_t v = u, i = 0;
    if (r->mblocks > 1) i = v % r->mblocks * r->rows, v /= r->mblocks;
    int64_t e = v / r->slivers, j = v % r->slivers * nr;
    int64_t n = min64(nr, n_all - j), mc = min64(r->rows, m - i), pc = 0;
    start(p, e, i, i + mc, j, j + n);
    while (pc < k) {
      int64_t kc = min64(kc_most, k - pc);
      src a = a_at(p, e, i, pc);
      if (r->a) {
        a = packed_a(p, r->a + (e * k + pc) * r->lda * w, r->lda, kc);
        a.at += i * a.row * w;
      }
      src b = b_sliver(p, r->move_b, e, j, n, pc, kc, bp);
      tile(p, r->k, pc, kc, a, b, e, i, j, mc, n);
      pc += kc;
    }
  }
  if (packs) nx_cpu_give(bp);
}

/* A job of the table's kernels, each of its bodies run between the table's
   enter and leave: on the M1, a body's calls share one live state of the
   matrix unit, where a state per call drained the unit at each end. A
   product that runs alone runs each job's body itself, in the state its
   call holds (runs_alone). */
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
  if (p->alone) {
    if (total > 0) body(0, total, 0, ctx);
    return;
  }
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

/* [p]'s cost on [g]'s kernels in job.c's terms, its operands' [bytes]
   set: its flops at the kernels' speed and its bytes. */
static double alone_cost(const problem *p, const nx_cpu_gemm *g,
                         double *bytes) {
  double e = (double)p->ext[BATCH], m = (double)p->ext[ROW];
  double n = (double)p->ext[COL], k = (double)p->ext[CON];
  *bytes = e * (m * k + k * n + m * n) * p->w;
  return flop_cost(g->kernel, 2 * e * m * n * k) + *bytes;
}

/* Whether [p] runs alone on [g]'s kernels: on the calling thread, where
   its whole cost buys no second thread, as BLIS's small products take one
   thread below their thresholds (bli_gemmsup). Its jobs then are calls,
   which run in one state of g's hardware, made once: on the M1 Max a 64³
   float32 product took 0.94 µs as two jobs, each in its own state of the
   matrix unit, 0.91 alone. */
static int runs_alone(const problem *p, const nx_cpu_gemm *g) {
  double bytes, cost = alone_cost(p, g, &bytes);
  return nx_cpu_threads_at((int64_t)bytes, cost, thread_bytes(g)) <= 1;
}

/* Sets [p]'s kernels to the table's or their other ones, whichever
   computes it in less time, and whether it runs alone, decided once. By
   rows: the kernel few_rows or chain takes, its rows padded to its tile,
   at its speed on every copy of its hardware. On the M1 Max 8 rows of
   float32 are 8 on the vector units of 8 cores at speed 1, against 32 on
   2 matrix units at 13: at most 8 rows run on the cores, more on the
   units. A product that runs alone on those then takes the ones that
   compute it in less time on one thread, with what a call on them adds
   (nx_cpu_gemm's call), and runs alone on them: on the M1 Max float32
   products up to 22³ and float64 ones up to 14³ run on a core, and 8 ×
   64 × 64 float32 runs on the unit, in 0.50 µs against 1.25 on the
   cores. */
static void fastest(problem *p) {
  const nx_cpu_gemm *g = p->g, *o = g->other;
  int64_t m = p->ext[ROW];
  if (o != NULL) {
    double ta = padded(g->kernel, m) * o->kernel.flops_per_byte *
                engines(o->kernel);
    double tb = padded(o->kernel, m) * g->kernel.flops_per_byte *
                engines(g->kernel);
    if (tb < ta) p->g = o;
  }
  p->alone = runs_alone(p, p->g);
  if (o == NULL || !p->alone) return;
  double f = 2.0 * p->ext[BATCH] * p->ext[COL] * p->ext[CON];
  double ta = flop_cost(g->kernel, f * padded(g->kernel, m)) + g->call;
  double tb = flop_cost(o->kernel, f * padded(o->kernel, m)) + o->call;
  p->g = tb < ta ? o : g;
}

/* [p] as the product of b's transpose by a's, which has the same outputs
   transposed and the same bits: fma(a, b, c) is fma(b, a, c). */
static void transpose(problem *p) {
  operand a = p->op[A];
  p->op[A] = p->op[B];
  p->op[B] = a;
  for (int o = 0; o < 4; o++) {
    operand *q = &p->op[o];
    int64_t r = q->st[ROW];
    q->st[ROW] = q->st[COL];
    q->st[COL] = r;
    if (q->s == NULL) continue;
    const int64_t *t = q->s->t[ROW];
    q->s->t[ROW] = q->s->t[COL];
    q->s->t[COL] = t;
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
  int64_t threads = nx_cpu_threads_at(bytes, cost, thread_bytes(g));
  int64_t blocks = 1;
  if (p->b_in && threads > 1)
    blocks = min64(threads / gcd(batch * r.slivers, threads),
                   ceil_div(m, km.mr) / 2);
  r.rows = m, r.mblocks = 1;
  if (blocks > 1) {
    r.rows = padded(km, ceil_div(m, blocks));
    r.mblocks = ceil_div(m, r.rows);
  }
  int64_t total = batch * r.slivers * r.mblocks;
  r.move_a = mover(p, batch * r.kblocks, packed, PACK_COST * packed);
  r.move_b = mover(p, total, bytes, cost);
  if (!p->a_in)
    job(p, batch * r.kblocks, packed, PACK_COST * packed, few_rows_pack, &r);
  job(p, total, bytes, cost, few_rows_unit, &r);
  if (!p->a_in) nx_cpu_give(r.a);
}

/* The rows of a block of a, the job's units being [want] blocks and
   ranges of b's [slivers] together: equal blocks of at most MC rows, a
   multiple of MR, so that a job's units take equal times (100 rows are
   blocks of 54 and 46, not 96 and 4). A product that keeps a (keeps_a),
   as BLIS's small products read it, takes as many blocks as make a unit's
   tiles of rows about as many as its slivers (bli_thread_partition_2x2):
   ranges of b's few slivers would each write a piece of every row of R.
   On the i9-9900K's 16 threads a 1024 × 16 × 16 float64 product takes 3.0
   µs in blocks of 24 rows against 20 in blocks of 96; a 256³ float32 one
   45 µs in such blocks against 37 in MC's. */
static int64_t block_rows(const problem *p, int64_t want, int64_t slivers) {
  int64_t m = p->ext[ROW], tiles = ceil_div(m, p->g->kernel.mr);
  int64_t blocks = ceil_div(m, p->g->mc);
  if (p->keeps_a) {
    double per = (double)ceil_div(want, p->ext[BATCH]);
    int64_t most = min64((int64_t)sqrt(per * tiles / slivers), tiles);
    blocks = blocks < most ? most : blocks;
  }
  return padded(p->g->kernel, ceil_div(m, blocks));
}

static void chain(problem *p) {
  const nx_cpu_gemm *g = p->g;
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON], w = p->w;
  int64_t nc = min64(g->nc, n), nr = g->kernel.nr, kc_most = min64(g->kc, k);
  int64_t slivers = ceil_div(nc, nr);
  int64_t b_bytes = p->b_in ? 0 : slivers * nr * kc_most * w;
  int64_t packed = batch * (slivers * nr + m) * kc_most * w;
  double cost = flop_cost(g->kernel, 2.0 * batch * m * nc * kc_most) + packed;
  int64_t rows =
      block_rows(p, units(packed, cost, thread_bytes(g)), slivers);
  int64_t mblocks = ceil_div(m, rows);
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
      packed = ne * (c.slivers * nr + m) * kc_most * w;
      cost = flop_cost(g->kernel, 2.0 * ne * m * c.nc * kc_most) + packed;
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

/* A product of one block of k that runs on one thread, a in acc and b
   adjacent along its columns, runs in tiles: a row of a against
   NX_CPU_LANES columns of b at a time, summed in registers, each lane a
   vector of columns. Otherwise rows are the fewer of the two (lanes
   transposes the product), at most LANE_ROWS a pass. b, the large
   operand, is read once a pass, in the order it lies:

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
   block, init as it is, or +0. An init in acc is read in place: the stage
   costs more than the sum of a small product's output. */
static void total(const problem *p, int64_t e, int64_t i, int64_t j,
                  const uint8_t *s, int64_t n, uint8_t *d) {
  _Alignas(16) uint8_t x[16] = {0};
  const nx_array *init = p->op[INIT].x;
  int64_t at = init ? pos(p, INIT, e, i, j, 0) : 0;
  if (init && init->dtype == p->acc)
    memcpy(x, init->base + at * p->w, (size_t)p->w);
  else if (init) stage_acc(p, INIT, at, 1, 0, 1, 1, x, 0);
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
  int64_t m = p->ext[ROW], k = p->ext[CON];
  for (int64_t e = lo; e < hi; e++)
    stage_at(p, A, e, 0, 0, 0, CON, ROW, k, m, f->a + e * f->a_batch * p->w,
             k);
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
      else stage_at(p, B, e, 0, j0 + j, k0, CON, COL, len, 1, bp, len);
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
  const int64_t *sb = p->op[B].st;
  int64_t m = p->ext[ROW], w = p->w, k0 = x * NX_CPU_LANE_BLOCK;
  int64_t len = min64(NX_CPU_LANE_BLOCK, p->ext[CON] - k0);
  for (int64_t q = q0; q < q1; q++)
    for (int64_t i = 0; i < m; i++)
      memset(l + (q * ls + i * ldy) * w, 0, (size_t)(c * w));
  stage_at(p, A, e, 0, 0, k0, ROW, CON, m, len, ap, m);
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
      stage_at(p, B, e, 0, j0, k0 + t + q0, COL, CON, c, r - q0, bp, c);
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

/* [p]'s cost in lane order, its operands' [bytes] set: the cores'
   kernels run a flop in the time memcpy moves a byte. */
static double lane_cost(const problem *p, int64_t *bytes) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON];
  *bytes = batch * (m + n) * k * p->w;
  return 2.0 * batch * m * n * k + *bytes;
}

static void lane_pass(problem *p) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t w = p->w, bytes;
  if (p->ext[CON] == 0) {
    nx_cpu_job(batch, batch * m * n * w, batch * m * n * w, empty, p);
    return;
  }
  double cost = lane_cost(p, &bytes);
  if (p->op[B].st[CON] == 1) dots(p, bytes, cost);
  else rows(p, bytes, cost);
}

/* Whether [p] runs in tiles: one block of k, on one thread, a in acc and b
   adjacent along its columns. */
static int tiled(const problem *p) {
  int64_t k = p->ext[CON], bytes;
  double cost = lane_cost(p, &bytes);
  return k > 0 && k <= NX_CPU_LANE_BLOCK && raw(p, A) && adjacent(p, B, COL) &&
         nx_cpu_threads(bytes, cost) <= 1;
}

/* Each row's outputs NX_CPU_LANES columns at a time by the target's tile,
   which sums them in registers, then init. On the M1 Max a 4³ float32
   product took 0.58 µs through rows, whose lanes and sums move through
   memory, 0.14 in tiles; 4 × 64 × 64 took 4.4 µs and 1.6. */
static void tiles(const problem *p) {
  nx_cpu_lane_tile tile = nx_cpu_table->tile[p->acc];
  const int64_t *sa = p->op[A].st, *sb = p->op[B].st;
  int64_t m = p->ext[ROW], n = p->ext[COL], w = p->w;
  int copies = p->op[INIT].x == NULL && p->op[DST].st[COL] == 1;
  _Alignas(16) double sums[NX_CPU_LANES];
  uint8_t *y = (uint8_t *)sums;
  for (int64_t e = 0; e < p->ext[BATCH]; e++)
    for (int64_t i = 0; i < m; i++)
      for (int64_t j0 = 0; j0 < n; j0 += NX_CPU_LANES) {
        int64_t c = min64(NX_CPU_LANES, n - j0);
        tile(p->op[A].x->base + pos(p, A, e, i, 0, 0) * w, sa[CON],
             p->op[B].x->base + pos(p, B, e, 0, j0, 0) * w, sb[CON],
             p->ext[CON], c, y);
        if (copies) {
          memcpy(at_r(p, e, i, j0), y, (size_t)(c * w));
          continue;
        }
        for (int64_t j = 0; j < c; j++)
          total(p, e, i, j0 + j, y + j * w, 1, at_r(p, e, i, j0 + j));
      }
}

/* Passes of LANE_ROWS rows: each moves a, init and dst on by its rows.
   The call's entry prices the whole product (call_cost), so that it
   releases the domain lock though each pass's jobs cost less: on one
   core a 2000 × 2000 × 16 float32 product held it for 45 ms, in 250
   passes of 0.6 MiB each. */
static void lanes(problem *p) {
  if (tiled(p)) {
    tiles(p);
    return;
  }
  if (p->ext[COL] < p->ext[ROW]) transpose(p);
  int64_t m = p->ext[ROW];
  for (int64_t i = 0; i < m && !atomic_load(&p->failed); i += LANE_ROWS) {
    p->ext[ROW] = min64(LANE_ROWS, m - i);
    lane_pass(p);
    for (int o = 0; o < 4; o++) {
      operand *q = &p->op[o];
      q->first += p->ext[ROW] * q->st[ROW];
      if (q->s && q->s->t[ROW]) q->s->t[ROW] += p->ext[ROW];
    }
  }
}

/* The entry */

/* Whether chain order reads operand [o] of [p] in place: in acc, adjacent
   along the axis the kernel streams (b along its columns, a along k where
   the kernel reads it an element at a time), and within its bound: a,
   IN_PLACE_BYTES an element; b, half a core's L2, or, where the product
   keeps a (keeps_a), the span of a sliver's kc rows, a row of b apart,
   within half its L1. b's sliver is read again for each tile of a's
   rows, and past the bound its rows fall out between tiles. On the
   i9-9900K (256 KiB) a 256³ float32 product takes 406 µs on one core with
   b in place against 244 packed, a 512³ one 683 against 322 µs on 16
   threads; a 128³ one, within the bound, 33.2 against 33.9 µs on one
   core. A 1024 × 1024 triangular solve's 512 × 64 × 512 float64 products,
   which keep a, its rows 8 KiB apart, take 2.8 ms each on 16 threads with
   b in place, its sliver's span half of L2, against 1.3 packed. On kimchi (3 MiB) a
   4096³ one took twice as long in place (205 against 102 ms). */
static int in_place(const problem *p, int o) {
  if (o == B && p->keeps_a) {
    int64_t kc = min64(p->g->kc, p->ext[CON]);
    return adjacent(p, B, COL) &&
           kc * p->op[B].st[CON] * p->w <= nx_cpu_table->l1 / 2;
  }
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

/* Chain order's products: streamed past all of a, or in chain's blocks. */
static void products(problem *p) {
  if (streams(p) && !p->keeps_a) few_rows(p);
  else chain(p);
}

/* A product that runs alone, in one state of its kernels' hardware. */
static void alone(problem *p) {
  if (p->g->enter) p->g->enter();
  products(p);
  if (p->g->leave) p->g->leave();
}

/* [p]'s operands and extents: [v]'s of [op]. */
static void lay_out(problem *p, const nx_contract_view *v,
                    const nx_array *const *op) {
  for (int o = 0; o < 4; o++) {
    p->op[o].x = op[o];
    p->op[o].first = v->offset[o];
    for (int x = 0; x < 4; x++) p->op[o].st[x] = v->stride[o][x];
  }
  for (int x = 0; x < 4; x++) p->ext[x] = v->extent[x];
}

/* nx_cpu_contract_on on the kernels [g], a and b scaled by [s] where it is
   not NULL. */
static int contract_with(const nx_cpu_gemm *g, int acc, int order,
                         const nx_contract_view *v, const nx_array *const *op,
                         const scale *s) {
  problem p = {.acc = acc, .w = nx_cpu_width(acc), .g = g};
  lay_out(&p, v, op);
  scale own[2]; /* the problem's, which its passes move on */
  for (int o = A; s && o <= B; o++)
    if (s[o].v) {
      own[o] = s[o];
      p.op[o].s = &own[o];
    }
  if (p.ext[BATCH] == 0 || p.ext[ROW] == 0 || p.ext[COL] == 0) return 1;
  p.init_is_dst = init_is_dst(&p);
  atomic_init(&p.failed, 0);
  if (order == NX_CPU_LANE) lanes(&p);
  else {
    /* Kept in place, whether or not a stays in the caches: on the
       i9-9900K a 300 × 16 × 600 float64 product takes 91 µs on one core
       so against 166 transposed, a 256 × 64 × 256 one 24 µs on 16 threads
       against 33. */
    int narrow = p.ext[COL] < p.ext[ROW];
    p.keeps_a = narrow && in_place(&p, A);
    if (narrow && !p.keeps_a) {
      transpose(&p);
      if (!streams(&p)) transpose(&p);
    }
    fastest(&p);
    p.a_in = in_place(&p, A);
    p.b_in = in_place(&p, B);
    if (p.alone) alone(&p);
    else products(&p);
  }
  return !atomic_load(&p.failed);
}

int nx_cpu_contract_on(int acc, int order, const nx_contract_view *v,
                       const nx_array *const *op) {
  return contract_with(&nx_cpu_table->gemm[acc], acc, order, v, op, NULL);
}

/* Complex */

/* A complex contraction is one in its parts' dtype, of twice the columns
   and twice the products (the 1m method, Van Zee, "Implementing
   high-performance complex matrix multiplication via the 1m method", SIAM
   J. Sci. Comput. 2020, its variant for row-stored c). a's K complex
   numbers along the contraction are 2K reals, real and imaginary parts in
   turn, read in place where its numbers lie adjacent along k; b's element
   (k, j) = x + iy is copied as the block

     row 2k:      … x   y …
     row 2k + 1:  … −y  x …

   so that column 2j sums re·x then im·(−y) for each k, output j's real
   part, and column 2j + 1 re·y then im·x, its imaginary part. Each part is
   a float contraction of 2K terms, whose bits the real contraction keeps
   at any layout and thread count. Where dst's or init's numbers lie apart
   along the columns, each part is a contraction of its own over b's
   columns of that part. */

/* Rows 2k and 2k + 1 of the block of [n] complex numbers at [x], [sx]
   parts apart: (x, y) into [even] and (−y, x) into [odd]. Adjacent
   numbers take a loop of their own, which vectorises. */
#define BLOCK_ROWS(T, name)                                                \
  static void name(const T *x, int64_t sx, T *restrict even,               \
                   T *restrict odd, int64_t n) {                           \
    if (sx == 2) {                                                         \
      for (int64_t j = 0; j < 2 * n; j += 2) {                             \
        T re = x[j], im = x[j + 1];                                        \
        even[j] = re, even[j + 1] = im, odd[j] = -im, odd[j + 1] = re;     \
      }                                                                    \
      return;                                                              \
    }                                                                      \
    for (int64_t j = 0; j < n; j++) {                                      \
      T re = x[j * sx], im = x[j * sx + 1];                                \
      even[2 * j] = re, even[2 * j + 1] = im;                              \
      odd[2 * j] = -im, odd[2 * j + 1] = re;                               \
    }                                                                      \
  }
BLOCK_ROWS(float, block_rows_f32)
BLOCK_ROWS(double, block_rows_f64)

/* b's blocks: batch element e's rows 2k and 2k + 1 at
   d + (e·2K + 2k)·2N·w, w the parts' width. A unit is a row k of b, read
   in place where b is acc, staged into the odd row first otherwise. */
typedef struct {
  const nx_array *b;
  const nx_contract_view *v;
  int acc, w;
  uint8_t *d;
} blocks_job;

static void blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const blocks_job *j = ctx;
  const int64_t *s = j->v->stride[B];
  int64_t n = j->v->extent[COL], k = j->v->extent[CON], row = 2 * n * j->w;
  for (int64_t u = lo; u < hi; u++) {
    int64_t at = j->v->offset[B] + u / k * s[BATCH] + u % k * s[CON];
    uint8_t *even = j->d + 2 * u * row, *odd = even + row;
    const uint8_t *x = j->b->base + at * 2 * j->w;
    int64_t sx = 2 * s[COL];
    if (j->b->dtype != j->acc) {
      nx_cpu_stage_as(j->b, at, s[COL], 0, n, 1, j->acc, odd, 0);
      x = memcpy(even, odd, (size_t)row);
      sx = 2;
    }
    if (j->w == 4)
      block_rows_f32((const float *)x, sx, (float *)even, (float *)odd, n);
    else
      block_rows_f64((const double *)x, sx, (double *)even, (double *)odd,
                     n);
  }
}

/* [x]'s numbers read as their parts, of the dtype [r]. */
static nx_array as_parts(const nx_array *x, int r) {
  nx_array y = *x;
  y.dtype = r;
  y.bits = x->bits / 2;
  return y;
}

/* Operand [o] of [v], rows along the axis [x1] of numbers along [x0], as
   [u]'s operand [y] of parts: in place, or under [copy] copied into [acc]
   in scratch (nx_cpu_take), each batch element's rows C-contiguous, one
   element where [o] steps 0 along the batch. Answers the copy, NULL in
   place or if allocation fails. */
static uint8_t *parts_of(const nx_contract_view *v, const nx_array *const *op,
                         int o, int x0, int x1, int acc, int copy,
                         nx_contract_view *u, nx_array *y) {
  const int64_t *s = v->stride[o];
  int r = part_of(acc), w = nx_cpu_width(r);
  int64_t n0 = v->extent[x0], n1 = v->extent[x1];
  if (!copy) {
    *y = as_parts(op[o], r);
    u->offset[o] = 2 * v->offset[o];
    for (int x = BATCH; x <= CON; x++) u->stride[o][x] = 2 * s[x];
    return NULL;
  }
  int64_t ne = s[BATCH] == 0 ? 1 : v->extent[BATCH];
  uint8_t *d = nx_cpu_take(ne * n1 * n0 * 2 * w);
  if (d == NULL) return NULL;
  for (int64_t e = 0; e < ne; e++)
    nx_cpu_stage_as(op[o], v->offset[o] + e * s[BATCH], s[x0], s[x1], n0, n1,
                    acc, d + e * n1 * n0 * 2 * w, n0);
  *y = (nx_array){.base = d, .dtype = r, .bits = 8 * w};
  u->offset[o] = 0;
  u->stride[o][BATCH] = s[BATCH] == 0 ? 0 : 2 * n1 * n0;
  u->stride[o][x1] = 2 * n0;
  u->stride[o][x0] = 2;
  return d;
}

static int complex_on(int acc, int order, const nx_contract_view *v,
                      const nx_array *const *op) {
  int r = part_of(acc), w = nx_cpu_width(r);
  int64_t batch = v->extent[BATCH], m = v->extent[ROW], n = v->extent[COL];
  int64_t k = v->extent[CON];
  if (batch == 0 || m == 0 || n == 0) return 1;
  nx_contract_view u = {.extent = {batch, m, 2 * n, 2 * k}};
  nx_array x[4];
  const nx_array *ops[4] = {&x[A], &x[B], op[INIT] ? &x[INIT] : NULL, &x[DST]};
  int copy_a = op[A]->dtype != acc || (v->stride[A][CON] != 1 && k > 1);
  int64_t eb = v->stride[B][BATCH] == 0 ? 1 : batch;
  uint8_t *ab = parts_of(v, op, A, CON, ROW, acc, copy_a, &u, &x[A]);
  uint8_t *bb = nx_cpu_take(eb * 4 * k * n * w), *ib = NULL;
  /* An init of another dtype than acc, a result's of another dtype, is
     copied into acc's parts. */
  int copy_i = op[INIT] && op[INIT]->dtype != acc;
  if (op[INIT])
    ib = parts_of(v, op, INIT, COL, ROW, acc, copy_i, &u, &x[INIT]);
  parts_of(v, op, DST, COL, ROW, acc, 0, &u, &x[DST]);
  int done = 0;
  if ((copy_a && ab == NULL) || (copy_i && ib == NULL) || bb == NULL)
    goto out;
  u.stride[A][CON] = 1;
  blocks_job j = {.b = op[B], .v = v, .acc = acc, .w = w, .d = bb};
  /* Each number of b read, two parts, and its block written, four. */
  int64_t bytes = eb * k * n * 6 * w;
  nx_cpu_job(eb * k, bytes, bytes, blocks, &j);
  x[B] = (nx_array){.base = bb, .dtype = r, .bits = 8 * w};
  u.stride[B][BATCH] = eb == 1 ? 0 : 4 * k * n;
  u.stride[B][CON] = 2 * n;
  /* Each part's columns, in turn where dst's or init's numbers lie apart. */
  int apart = n > 1 && (u.stride[DST][COL] != 2 ||
                        (op[INIT] && u.stride[INIT][COL] != 2));
  if (!apart) {
    u.stride[B][COL] = 1;
    u.stride[DST][COL] = 1;
    if (op[INIT]) u.stride[INIT][COL] = 1;
    done = nx_cpu_contract_on(r, order, &u, ops);
    goto out;
  }
  u.extent[COL] = n;
  u.stride[B][COL] = 2;
  done = 1;
  for (int part = 0; part < 2 && done; part++) {
    nx_contract_view up = u;
    for (int o = B; o <= DST; o++) up.offset[o] += part;
    done = nx_cpu_contract_on(r, order, &up, ops);
  }
out:
  nx_cpu_give(ab);
  nx_cpu_give(ib);
  nx_cpu_give(bb);
  return done;
}

/* Quads */

/* An 8-bit a by a b of the same dtype into a 32-bit acc runs on the
   table's quads kernel where it has one: four consecutive elements along
   k are one 32-bit word, a quad, and the driver contracts words of K/4
   products each, which the kernel adds four products at a time. A word of
   a holds a row's four elements, a word of b a column's. Each operand is
   read in place where it lies adjacent along k with each word on a word's
   boundary, and copied into words otherwise, k padded with zeros to a
   multiple of 4: a as rows of words, b as rows of each quad's words. The
   sum is exact modulo 2^32, as acc's: an integer's bits do not depend on
   how its products are grouped. */

/* b's words, copied: column i's word q at d + (i + q·pitch)·4. A unit is
   a column of a batch element where b is adjacent along k, else a quad of
   each of its columns, so that the reads run along the memory. */
typedef struct {
  const nx_array *b;
  const nx_contract_view *v;
  int64_t quads, pitch;
  uint8_t *d;
} quads_job;

/* Four rows of [n] bytes, each adjacent, interleaved into words: row t's
   byte i to d[4i + t]. */
static void interleave(const uint8_t *r0, const uint8_t *r1,
                       const uint8_t *r2, const uint8_t *r3, uint8_t *d,
                       int64_t n) {
  for (int64_t i = 0; i < n; i++) {
    d[4 * i] = r0[i];
    d[4 * i + 1] = r1[i];
    d[4 * i + 2] = r2[i];
    d[4 * i + 3] = r3[i];
  }
}

/* A full quad of b's rows, each adjacent, interleaves; any other element
   moves alone. */
static void quad_words(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const quads_job *j = ctx;
  const int64_t *s = j->v->stride[B];
  int64_t n = j->v->extent[COL], k = j->v->extent[CON];
  int by_column = s[CON] == 1;
  int64_t per = by_column ? n : j->quads;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = u / per, i0 = 0, i1 = n, p0 = 0, p1 = 4 * j->quads;
    if (by_column) i0 = u % per, i1 = i0 + 1;
    else p0 = 4 * (u % per), p1 = p0 + 4;
    const uint8_t *x = j->b->base + j->v->offset[B] + e * s[BATCH];
    uint8_t *d = j->d + e * j->pitch * j->quads * 4;
    if (!by_column && s[COL] == 1 && p1 <= k) {
      const uint8_t *r = x + p0 * s[CON];
      interleave(r, r + s[CON], r + 2 * s[CON], r + 3 * s[CON],
                 d + p0 * j->pitch, n);
      continue;
    }
    for (int64_t i = i0; i < i1; i++)
      for (int64_t p = p0; p < p1; p++)
        d[(i + p / 4 * j->pitch) * 4 + p % 4] =
            p < k ? x[p * s[CON] + i * s[COL]] : 0;
  }
}

/* Whether operand [o] of [v] is read in place as words: adjacent along k,
   each word's first element on a word's boundary. */
static int words_in_place(const nx_contract_view *v, int o) {
  const int64_t *s = v->stride[o];
  int64_t k = v->extent[CON];
  if (s[CON] != 1 && k > 1) return 0;
  if (k % 4 != 0 || v->offset[o] % 4 != 0) return 0;
  for (int x = BATCH; x < CON; x++)
    if (s[x] % 4 != 0) return 0;
  return 1;
}

/* The words between the rows of a copy into words, words_of's or
   zip_rows': a row's [words], or a line more where that is a multiple of
   4 KiB. Rows a multiple of 4 KiB apart share their low 12 address bits,
   and with them the L1 sets of a sliver's steps. On one fudge core,
   contract-i8-i32-1024, whose b is copied into rows of 1024 words, took
   3.77 ms, and 3.68 with its rows a line further apart; the kernel alone,
   in cache, ran 273 G products a second, and 294. */
static int64_t pitch_of(int64_t words) {
  return words > 0 && words * 4 % 4096 == 0 ? words + 16 : words;
}

/* Operand [o] of [v] as [u]'s operand [y] of words of [acc]: in place, or
   under [copy] copied into memory of its own, a's rows of words and b's as
   quads_job lays them out, rows pitch_of apart, one batch element where
   [o] steps 0 along the batch. Answers the copy, NULL in place or if
   allocation fails. */
static uint8_t *words_of(const nx_contract_view *v, const nx_array *const *op,
                         int o, int acc, int copy, nx_contract_view *u,
                         nx_array *y) {
  const int64_t *s = v->stride[o];
  int64_t k = v->extent[CON], quads = ceil_div(k, 4);
  int64_t n = v->extent[o == A ? ROW : COL];
  *y = *op[o];
  y->dtype = acc;
  y->bits = 32;
  if (!copy) {
    u->offset[o] = v->offset[o] / 4;
    for (int x = BATCH; x < CON; x++) u->stride[o][x] = s[x] / 4;
    u->stride[o][CON] = 1;
    return NULL;
  }
  /* a's rows are of quads words, b's of n. */
  int64_t pitch = pitch_of(o == A ? quads : n);
  int64_t rows = o == A ? n : quads;
  int64_t ne = s[BATCH] == 0 ? 1 : v->extent[BATCH], words = rows * pitch;
  uint8_t *d = alloc(ne * words * 4);
  if (d == NULL) return NULL;
  y->base = d;
  u->offset[o] = 0;
  u->stride[o][BATCH] = s[BATCH] == 0 ? 0 : words;
  if (o == A) {
    for (int64_t e = 0; e < ne; e++)
      nx_cpu_stage_as(op[A], v->offset[A] + e * s[BATCH], s[CON], s[ROW], k,
                      n, op[A]->dtype, d + e * words * 4, 4 * pitch);
    for (int64_t r = 0; r < ne * n; r++)
      memset(d + r * pitch * 4 + k, 0, (size_t)(pitch * 4 - k));
    u->stride[A][ROW] = pitch;
    u->stride[A][CON] = 1;
    return d;
  }
  quads_job j = {.b = op[B], .v = v, .quads = quads, .pitch = pitch, .d = d};
  int64_t units = ne * (s[CON] == 1 ? n : quads);
  nx_cpu_job(units, ne * words * 8, ne * words * 8, quad_words, &j);
  u->stride[B][COL] = 1;
  u->stride[B][CON] = pitch;
  return d;
}

/* A product of few rows whose b's words lie adjacent along k: each
   output the dot of a's row and b's column, which reads b once, in the
   order it lies, where chain order's tiles would pad a's rows to MR and
   pack b's slivers. A unit is DOT_COLUMNS columns of a batch element,
   which stay in the L1 cache from one pair of rows to the next while k is
   at most 8192. The dots do not split k, as lane order's do: a product of
   at most DOT_COLUMNS rows and columns runs on one core whatever k. On
   the M1 Max, int8 rows by a 4096 × 4096 weight stored [n × k] take 50 µs
   at 1 row against chain order's 255, 595 at 32 against 710, and the
   same 1.2 ms at 64. */
#define QUAD_ROWS 32

typedef struct {
  const problem *p; /* a and b of words */
  nx_cpu_quads_dot dot;
  int64_t groups; /* of DOT_COLUMNS columns */
} quad_dots_job;

/* Operand [o]'s words from (e, i, j, 0). */
static const uint32_t *words_at(const problem *p, int o, int64_t e,
                                int64_t i, int64_t j) {
  return (const uint32_t *)p->op[o].x->base + pos(p, o, e, i, j, 0);
}

/* A product of few rows whose b lies adjacent along its columns, as a
   weight stored [k × n] does: each row of a times b's rows, read in place
   in the order they lie, where chain order would interleave all of b into
   words and pack it. A unit is a group of a batch element's columns, in
   multiples of AXPY_TILE, UNITS of them a thread, so that a busy core
   takes fewer: with one core busy elsewhere, one unit a thread took 1 row
   from memory in 250-400 µs against 225-275, 8 rows in 660-900 against
   435-585. Its rows' sums, at most AXPY_SUMS bytes, stay in L1.
   Below the target's quads_zip_rows, the axpy zips rows of b in
   registers for the rows of a, AXPY_STEP rows of b at a time; from there,
   AXPY_ZIPS bytes of b's rows are zipped into words once, by
   interleave(), and the quads kernel runs all rows over them, as chain
   order would without padding a's rows. A unit's rows lie a row of b
   apart, a stride no x86 hardware prefetcher follows (the L1's follow at
   most 2 KiB, the L2's stay within a 4 KiB page; Intel's optimization
   manual), so each axpy is passed the next step's rows to prefetch.

   On the M1 Max, int8 rows by a 4096 × 4096 weight stored [k × n] take,
   against chain order (from the system cache unless said): 1 row 142-149
   µs against 825-831, and 192-227 from memory against 880-920 (148 to
   read the weight); 8 rows 268-287 against 810-877; 32 rows 694-735
   against 1.31-1.35 ms; 64 1.25 ms against 2.0; 128 2.6 against 3.1.
   Steps of 16 rows ran
   1 row in 140 µs where 64 and 256 swung from 120 to 320. Splitting k
   across units as well read a cached weight 1.4 times faster but one
   from memory 1.15 times slower, and memory is where a model's weights
   lie. */
#define AXPY_ROWS 128
/* Past AXPY_FEW rows, the axpys take a product only where b's columns
   alone give each thread a unit and k is at least AXPY_DEEP, and past
   AXPY_MANY rows only where b is more than 1 / AXPY_HELD of the largest cache
   the host reports: chain order splits rows and copies b once for all of
   them, which few columns, a short k or a b that stays in cache repay.
   Through AXPY_FEW rows, units split rows too where the columns are too
   few. On the M1 Max (12 MiB of L2 a cluster), rows by b of n columns and
   k rows against chain order: 64 by n = k = 64 took 3.7 µs in bands
   against 5.2, and 6.5 by columns alone; 32 by n 4096, k 256 86 against
   55; 128 by n = k = 1024 181 against 158 and 64 by n = k = 1024 96
   against 87, b in cache; 128 by n = k = 2048 661 against 692. 16 by n =
   k = 64 took 3.08 µs in bands against 2.86: 65K products, which the
   bands' threads cost more than they save. 32 by n = k = 512 took 15.7
   µs at best against 22.3, but lost 7 of 8 rounds with a busy core: 8
   groups of columns give the 8 threads a unit each, no slack to
   rebalance; 32 by n = k = 1024 61 against 65. On the i9-9900K at 16
   threads (16 MiB of L3), with AVX2's axpys of several rows, by the
   axpys against chain order: 16 by n = k = 256 6.9 µs in bands against
   11.2, by n 64, k 4096 23.7 against 23.4; 32 by n = k = 1024 73 against
   97; 64 by n = k = 4096 2.10 ms against 3.22. Of those chain order
   takes: 32 by n 64, k 4096 43 µs against 32 and 64 by n = k = 64 4.4
   against 4.0, as the bounds mean; 32 by n 4096, k 256 91 against 114
   and 64 by n = k = 1024 136 against 152, which the axpys would win here
   and lose on the M1. */
#define AXPY_FEW 16
#define AXPY_MANY 32
/* The share of the largest cache that b passes past AXPY_MANY rows: on
   the M1 Max (12 MiB) 128 rows by b of 1 MiB lost to chain order and by
   4 MiB won. */
#define AXPY_HELD 8
#define AXPY_DEEP 512
#define AXPY_STEP 16
/* A unit's zipped rows and sums, 96 KiB, within the M1's 128 KiB L1. */
#define AXPY_ZIPS (64 * 1024)
#define AXPY_SUMS (32 * 1024)
#define AXPY_TILE 64 /* the axpy's columns at once */
_Static_assert(AXPY_SUMS / 4 / AXPY_ROWS >= AXPY_TILE,
               "a unit of the most rows holds a tile of columns");

/* Row [i] of element [e]'s outputs [j0, j0 + c): init, or 0, plus [y]. */
static void quad_row(const problem *p, int64_t e, int64_t i, int64_t j0,
                     int64_t c, const uint32_t *y) {
  uint32_t *d = (uint32_t *)at_r(p, e, i, j0), init[DOT_COLUMNS];
  int64_t sy = p->op[DST].st[COL];
  for (int64_t q0 = 0; q0 < c; q0 += DOT_COLUMNS) {
    int64_t cq = min64(DOT_COLUMNS, c - q0);
    memset(init, 0, sizeof init);
    if (p->op[INIT].x)
      stage_acc(p, INIT, pos(p, INIT, e, i, j0 + q0, 0), p->op[INIT].st[COL],
                0, cq, 1, (uint8_t *)init, cq);
    for (int64_t q = 0; q < cq; q++)
      d[(q0 + q) * sy] = init[q] + y[q0 + q];
  }
}

static void quad_dots(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const quad_dots_job *j = ctx;
  const problem *p = j->p;
  int64_t m = p->ext[ROW], n = p->ext[COL], k = p->ext[CON];
  int64_t lda = p->op[A].st[ROW], ldb = p->op[B].st[COL];
  _Alignas(16) uint32_t y[2][DOT_COLUMNS + 4];
  for (int64_t t = lo; t < hi; t++) {
    int64_t e = t / j->groups, j0 = t % j->groups * DOT_COLUMNS;
    int64_t c = min64(DOT_COLUMNS, n - j0), q;
    const uint32_t *b = words_at(p, B, e, 0, j0);
    for (int64_t i = 0; i < m; i += 2) {
      const uint32_t *a = words_at(p, A, e, i, 0);
      /* A last row alone is its own pair, a last column its own four. */
      int64_t pair = i + 1 < m ? lda : 0, ldy = DOT_COLUMNS + 4;
      for (q = 0; q + 4 <= c; q += 4)
        j->dot(a, pair, b + q * ldb, ldb, k, y[0] + q, ldy);
      for (; q < c; q++) j->dot(a, pair, b + q * ldb, 0, k, y[0] + q, ldy);
      quad_row(p, e, i, j0, c, y[0]);
      if (i + 1 < m) quad_row(p, e, i + 1, j0, c, y[1]);
    }
  }
}

/* [u]'s product of the words [op] in [acc] by quads' dots, the fewer of
   rows and columns as rows: a's and b's words both lie adjacent along k,
   so each output is the same dot with a and b swapped. */
static void quad_dots_on(nx_cpu_quads_dot dot, const nx_cpu_gemm *g, int acc,
                         const nx_contract_view *u,
                         const nx_array *const *op) {
  problem p = {.acc = acc, .w = nx_cpu_width(acc), .g = g};
  lay_out(&p, u, op);
  if (p.ext[COL] < p.ext[ROW]) transpose(&p);
  int64_t batch = p.ext[BATCH], m = p.ext[ROW], n = p.ext[COL];
  int64_t k = p.ext[CON], groups = ceil_div(n, DOT_COLUMNS);
  quad_dots_job j = {.p = &p, .dot = dot, .groups = groups};
  int64_t bytes = batch * (m + n) * k * p.w;
  double cost = flop_cost(g->kernel, 2.0 * batch * m * n * k) + bytes;
  nx_cpu_job(batch * groups, bytes, cost, quad_dots, &j);
}

typedef struct {
  problem *p; /* a of words, b of its own elements */
  nx_cpu_quads_axpy axpy;
  int64_t k;            /* in elements */
  int64_t cols, groups; /* columns a group, groups */
  int64_t rows, bands;  /* rows a band, bands */
} quad_axpys_job;

/* The rows ahead of a zip that it prefetches, a line at a time, for the
   stride above AXPY_ROWS: on the i9-9900K 8 rows by a 4096 × 4096 weight
   took 579 µs at 16 rows ahead, 591 at 32 and 737 without; 128 rows 4.61
   ms against 4.81. On the M1 Max neither moved. */
#define ZIP_AHEAD 16
#define LINE_BYTES 64 /* x86's cache line, half the M1's */

/* Rows [q, q + kb) of b from column [j0] of element [e], zipped as
   interleave() does into rows of words of [c] columns at [d], [pitch]
   words apart, a row past k repeating the last, prefetching ZIP_AHEAD
   rows on. */
static void zip_rows(const problem *p, int64_t e, int64_t j0, int64_t c,
                     int64_t pitch, int64_t q, int64_t kb, uint32_t *d) {
  const uint8_t *b = p->op[B].x->base + pos(p, B, e, 0, j0, q);
  int64_t ldb = p->op[B].st[CON];
  for (int64_t w = 0; w < ceil_div(kb, 4); w++) {
    const uint8_t *r0 = b + 4 * w * ldb;
    int64_t ahead = 4 * w + ZIP_AHEAD, ahead_end = min64(ahead + 4, kb);
    for (int64_t r = ahead; r < ahead_end; r++)
      for (int64_t x = 0; x < c; x += LINE_BYTES)
        __builtin_prefetch(b + r * ldb + x);
    const uint8_t *r1 = 4 * w + 1 < kb ? r0 + ldb : r0;
    const uint8_t *r2 = 4 * w + 2 < kb ? r1 + ldb : r1;
    const uint8_t *r3 = 4 * w + 3 < kb ? r2 + ldb : r2;
    interleave(r0, r1, r2, r3, (uint8_t *)(d + w * pitch), c);
  }
}

/* A unit's sums [s] of [m] rows from [i0] and [c] columns from [j0] of
   element [e]: by the axpy below the target's quads_zip_rows, else by the
   quads kernel over b's rows zipped into [z], AXPY_ZIPS bytes. */
static void quad_unit(const quad_axpys_job *j, int64_t e, int64_t i0,
                      int64_t m, int64_t j0, int64_t c, uint32_t *s,
                      uint32_t *z) {
  const problem *p = j->p;
  int64_t ldb = p->op[B].st[CON];
  nx_cpu_micro km = p->g->kernel;
  if (m < nx_cpu_table->quads_zip_rows) {
    const uint8_t *a = (const uint8_t *)words_at(p, A, e, i0, 0);
    int64_t lda = 4 * p->op[A].st[ROW];
    for (int64_t q = 0; q < j->k; q += AXPY_STEP) {
      const uint8_t *b = p->op[B].x->base + pos(p, B, e, 0, j0, q);
      /* The next step's rows, or b's last AXPY_STEP, or all of b. */
      int64_t ahead = min64(q + AXPY_STEP, j->k - AXPY_STEP);
      if (ahead < 0) ahead = 0;
      const uint8_t *next = b + (ahead - q) * ldb;
      j->axpy(a + q, lda, m, b, ldb, min64(AXPY_STEP, j->k - q), c, s, c,
              next);
    }
    return;
  }
  int64_t zp = pitch_of(c), step = AXPY_ZIPS / zp / 16 * 16;
  for (int64_t q = 0; q < j->k; q += step) {
    int64_t kb = min64(step, j->k - q), words = ceil_div(kb, 4);
    zip_rows(p, e, j0, c, zp, q, kb, z);
    const uint32_t *a = words_at(p, A, e, i0, 0) + q / 4;
    for (int64_t jj = 0; jj < c; jj += km.nr)
      km.f(m, min64(km.nr, c - jj), words, a, p->op[A].st[ROW], 1, z + jj,
           zp, s + jj, c, NX_CPU_FROM_TILE);
  }
}

static void quad_axpys(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const quad_axpys_job *j = ctx;
  problem *p = j->p;
  int64_t m = p->ext[ROW], n = p->ext[COL];
  /* The sums in scratch, as a thread's stack should not hold 32 KiB. */
  uint32_t *s = (uint32_t *)scratch(p, AXPY_SUMS);
  if (s == NULL) return;
  /* A tile reads past its last column: a sliver's width beyond the zips. */
  uint32_t *z = NULL;
  if (j->rows >= nx_cpu_table->quads_zip_rows) {
    z = (uint32_t *)scratch(p, AXPY_ZIPS + 4 * p->g->kernel.nr);
    if (z == NULL) {
      nx_cpu_give(s);
      return;
    }
  }
  for (int64_t t = lo; t < hi; t++) {
    int64_t e = t / (j->groups * j->bands), g = t % (j->groups * j->bands);
    int64_t j0 = g / j->bands * j->cols, i0 = g % j->bands * j->rows;
    int64_t c = min64(j->cols, n - j0), r = min64(j->rows, m - i0);
    memset(s, 0, (size_t)(r * c * 4));
    quad_unit(j, e, i0, r, j0, c, s, z);
    for (int64_t i = 0; i < r; i++) quad_row(p, e, i0 + i, j0, c, s + i * c);
  }
  nx_cpu_give(z);
  nx_cpu_give(s);
}

/* The cost of quads' axpys over [v]'s extents, [k] its elements along k,
   and its bytes: each element's product counts as a word's does in the
   dots, the zips spending about three instructions on each SDOT. */
static double axpys_cost(const nx_cpu_gemm *g, const nx_contract_view *v,
                         int64_t k, int64_t *bytes) {
  int64_t batch = v->extent[BATCH], m = v->extent[ROW], n = v->extent[COL];
  *bytes = batch * (m + n) * k;
  return flop_cost(g->kernel, 2.0 * batch * m * n * k) + *bytes;
}

/* Whether quads' axpys take [v]'s product, [k] its elements along k
   (AXPY_FEW says when). */
static int axpys_fit(const nx_cpu_gemm *g, const nx_contract_view *v,
                     int64_t k) {
  int64_t m = v->extent[ROW], bytes;
  if (m > AXPY_ROWS) return 0;
  if (m <= AXPY_FEW) return 1;
  int64_t batch = v->extent[BATCH], n = v->extent[COL];
  if (k < AXPY_DEEP) return 0;
  if (m > AXPY_MANY && batch * n * k <= nx_cpu_table->llc / AXPY_HELD)
    return 0;
  double cost = axpys_cost(g, v, k, &bytes);
  int64_t threads = nx_cpu_threads_at(bytes, cost, NX_CPU_THREAD_BYTES);
  return batch * ceil_div(n, AXPY_TILE) >= threads;
}

/* [u]'s product in [acc] of a's words and b's [k] elements along k by
   quads' axpys: 0 if a worker's scratch could not be allocated. Where the
   groups of columns are fewer than the threads, units split rows too.
   Where the product zips b's rows (the target's quads_zip_rows), bands
   keep the quads kernel's MR rows, which zip as the whole would, unless
   whole tiles leave more than half the threads idle: on the M1 Max 2
   bands of 8 rows took 16 rows by n 64, k 4096 29.7 µs against 15 in 8
   bands of 2. */
static int quad_axpys_on(nx_cpu_quads_axpy axpy, const nx_cpu_gemm *g,
                         int acc, int64_t k, const nx_contract_view *u,
                         const nx_array *const *op) {
  problem p = {.acc = acc, .w = nx_cpu_width(acc), .g = g};
  lay_out(&p, u, op);
  atomic_init(&p.failed, 0);
  int64_t batch = p.ext[BATCH], m = p.ext[ROW], n = p.ext[COL], bytes;
  double cost = axpys_cost(g, u, k, &bytes);
  int64_t share = ceil_div(n, units(bytes, cost, NX_CPU_THREAD_BYTES));
  int64_t cols = min64(ceil_div(share, AXPY_TILE),
                       AXPY_SUMS / 4 / m / AXPY_TILE) * AXPY_TILE;
  int64_t groups = ceil_div(n, cols), mr = g->kernel.mr;
  int64_t threads = nx_cpu_threads_at(bytes, cost, NX_CPU_THREAD_BYTES);
  int64_t bands = min64(m, ceil_div(threads, batch * groups));
  int64_t whole = min64(bands, m / mr);
  int zips = m >= nx_cpu_table->quads_zip_rows;
  if (zips && 2 * batch * groups * whole >= threads) bands = whole;
  int64_t rows = ceil_div(m, bands);
  quad_axpys_job j = {.p = &p, .axpy = axpy, .k = k, .cols = cols,
                      .groups = groups, .rows = rows,
                      .bands = ceil_div(m, rows)};
  nx_cpu_job(batch * groups * j.bands, bytes, cost, quad_axpys, &j);
  return !atomic_load(&p.failed);
}

/* The table's quads kernels for a contraction in [acc] of operands of
   [da] and [db], or NULL. */
static const nx_cpu_gemm *quads_for(int acc, int da, int db) {
  if (acc != NX_INT32 && acc != NX_UINT32) return NULL;
  if (da != db || (da != NX_INT8 && da != NX_UINT8)) return NULL;
  const nx_cpu_gemm *g = &nx_cpu_table->quads[da];
  return g->kernel.f ? g : NULL;
}

static int quads_on(const nx_cpu_gemm *g, int acc, const nx_contract_view *v,
                    const nx_array *const *op) {
  if (v->extent[BATCH] == 0 || v->extent[ROW] == 0 || v->extent[COL] == 0)
    return 1;
  nx_contract_view u = *v;
  u.extent[CON] = ceil_div(v->extent[CON], 4);
  int64_t m = v->extent[ROW], n = v->extent[COL];
  nx_cpu_quads_axpy axpy = nx_cpu_table->quads_axpy[op[A]->dtype];
  int by_rows = axpy && v->stride[B][CON] != 1 && v->stride[B][COL] == 1 &&
                axpys_fit(g, v, v->extent[CON]);
  nx_array x[2] = {*op[A], *op[B]};
  int copy_a = !words_in_place(v, A);
  int copy_b = !by_rows && !words_in_place(v, B);
  uint8_t *ab = words_of(v, op, A, acc, copy_a, &u, &x[A]), *bb = NULL;
  if (!by_rows) bb = words_of(v, op, B, acc, copy_b, &u, &x[B]);
  int done = 0;
  if ((copy_a && ab == NULL) || (copy_b && bb == NULL)) goto out;
  const nx_array *ops[4] = {&x[A], &x[B], op[INIT], op[DST]};
  if (by_rows) {
    done = quad_axpys_on(axpy, g, acc, v->extent[CON], &u, ops);
    goto out;
  }
  /* a's words lie adjacent along k; the dots need b's to as well. */
  nx_cpu_quads_dot dot = nx_cpu_table->quads_dot[op[A]->dtype];
  if (dot && u.stride[B][CON] == 1 && min64(m, n) <= QUAD_ROWS) {
    quad_dots_on(dot, g, acc, &u, ops);
    done = 1;
  } else done = contract_with(g, acc, NX_CPU_CHAIN, &u, ops, NULL);
out:
  free(ab);
  free(bb);
  return done;
}

double nx_cpu_core_speed(int acc) {
  const nx_cpu_gemm *g = &nx_cpu_table->gemm[acc];
  if (g->kernel.shared > 1 && g->other != NULL) g = g->other;
  return g->kernel.flops_per_byte;
}

/* The contraction of the view [v] of [op], in [acc] and [order], [da] and
   [db] a's and b's dtypes: 0 if an allocation failed. */
/* A scaled contraction, [s] not NULL, is a float one: it runs on the float
   kernels. */
static int contract_view(int acc, int order, int da, int db,
                         const nx_contract_view *v, const nx_array *const *op,
                         const scale *s) {
  if (s) return contract_with(&nx_cpu_table->gemm[acc], acc, order, v, op, s);
  const nx_cpu_gemm *quads = quads_for(acc, da, db);
  if (quads) return quads_on(quads, acc, v, op);
  if (part_of(acc) != acc) return complex_on(acc, order, v, op);
  return nx_cpu_contract_on(acc, order, v, op);
}

/* Groups of several runs

   A group whose axes do not coalesce into one, over every operand that
   has it, is several runs (nx_contract_runs). The last run of each group
   is the view's axis and the batch's, rows' and columns' other runs an
   outer loop, each of whose positions is a contraction of the view: an
   output's sum is the same whichever contraction computes it. Attention's
   heads, a [b; s; h; d] array read as [b; h; s; d], are so a batch of h
   with b outside, read in place.

   The contracted runs instead make one: a and b are copied, every element
   of them, into buffers where their runs lie in C order, the contracted
   ones last, and those then coalesce. The copy keeps the products'
   numbering, so every order gives its bits; a loop over the outer
   contracted runs, each from the last's sums, would keep chain order's
   bits alone. */

typedef struct {
  int acc, order, da, db;
  nx_contract_view v;      /* the view at the outer loop's first position */
  int n;                   /* outer runs */
  int64_t extent[NX_MAX_RANK], stride[4][NX_MAX_RANK];
  int group[NX_MAX_RANK];  /* each outer run's */
  int64_t weight[NX_MAX_RANK]; /* the indices of its group one of its
                                  steps moves on */
  const nx_array *const *op;
  const scale *s;          /* a and b's at the first position, or NULL */
  atomic_int failed;
} outer;

/* The outer loop's positions [lo, hi), each a contraction of its view. */
static void outer_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  outer *q = ctx;
  for (int64_t u = lo; u < hi; u++) {
    nx_contract_view v = q->v;
    scale s[2];
    if (q->s) memcpy(s, q->s, sizeof s);
    for (int64_t r = q->n - 1, x = u; r >= 0; r--) {
      int64_t i = x % q->extent[r];
      x /= q->extent[r];
      for (int o = A; o <= DST; o++) v.offset[o] += i * q->stride[o][r];
      for (int o = A; q->s && o <= B; o++)
        if (s[o].t[q->group[r]]) s[o].t[q->group[r]] += i * q->weight[r];
    }
    if (!contract_view(q->acc, q->order, q->da, q->db, &v, q->op,
                       q->s ? s : NULL))
      atomic_store(&q->failed, 1);
  }
}

/* Multiplies the copy at [y] of a scaled operand, of acc [dt], laid out as
   stage_runs lays it, its groups [gs] whole, by its scales [s]. */
static void scale_runs(const nx_contract_runs *r, const int *gs, int dt,
                       const scale *s, uint8_t *y) {
  int64_t n[3] = {1, 1, 1};
  for (int t = 0; t < 3; t++)
    for (int k = 0; k < r->runs[gs[t]]; k++) n[t] *= r->extent[gs[t]][k];
  const int64_t *t0 = s->t[gs[0]], *t1 = s->t[gs[1]], *t2 = s->t[gs[2]];
  for (int64_t e = 0; e < n[0]; e++)
    for (int64_t i = 0; i < n[1]; i++) {
      int64_t at = t0[e] + t1[i], row = (e * n[1] + i) * n[2];
      if (dt == NX_FLOAT32) {
        float *d = (float *)y + row;
        const float *v = (const float *)s->v + at;
        for (int64_t k = 0; k < n[2]; k++) d[k] *= v[t2[k]];
      } else {
        double *d = (double *)y + row;
        const double *v = (const double *)s->v + at;
        for (int64_t k = 0; k < n[2]; k++) d[k] *= v[t2[k]];
      }
    }
}

/* Copies operand [o] of [r] into scratch (nx_cpu_take) of the dtype [dt],
   cast where it is not [src]'s, where its runs lie in C order, groups [gs]
   in turn, skipping runs it steps 0 along but the contracted ones, and
   lays [r] out over it in [x]: 0 if the scratch could not be allocated.
   The copies hold the operand's elements, O(a + b) for a and b. A scaled
   operand, [s] not NULL, is copied into acc, [dt], with each element times
   its scale (stage_at), skipping no run: its scales may differ where its
   elements do not. */
static int stage_runs(nx_contract_runs *r, int o, const int *gs, int dt,
                      const nx_array *src, const scale *s, nx_array *x) {
  nx_loop l = {.first = {0, r->offset[o]}};
  int64_t *at[NX_MAX_RANK], n = 1;
  int rank = 0;
  for (int t = 0; t < 3; t++) {
    int g = gs[t];
    for (int k = 0; k < r->runs[g]; k++) {
      if (r->stride[o][g][k] == 0 && g != CON && s == NULL) continue;
      l.extent[rank] = r->extent[g][k];
      l.step[1][rank] = r->stride[o][g][k];
      at[rank++] = &r->stride[o][g][k];
    }
  }
  for (int d = rank - 1; d >= 0; d--) {
    l.step[0][d] = n;
    *at[d] = n;
    n *= l.extent[d];
  }
  int bits = nx_dtype_row_of(dt).bits;
  uint8_t *base = nx_cpu_take((n * bits + 7) / 8);
  if (base == NULL) return 0;
  *x = *src;
  x->base = base;
  x->dtype = dt;
  x->bits = bits;
  x->offset = 0;
  x->alias = 0;
  l.rank = nx_coalesce_dims(2, rank, l.extent, l.step);
  if (dt == src->dtype) nx_cpu_copy_loop(base, src->base, bits, &l);
  else nx_cpu_cast_loop(x, src, &l);
  r->offset[o] = 0;
  if (s) scale_runs(r, gs, dt, s, base);
  return 1;
}

/* Whether [r]'s contracted runs coalesce into one over a and b, which
   then holds it. */
static int merges(nx_contract_runs *r) {
  int64_t ext[NX_MAX_RANK], st[2][NX_MAX_RANK];
  for (int k = 0; k < r->runs[CON]; k++) {
    ext[k] = r->extent[CON][k];
    st[0][k] = r->stride[A][CON][k];
    st[1][k] = r->stride[B][CON][k];
  }
  if (nx_coalesce_dims(2, r->runs[CON], ext, st) != 1) return 0;
  r->runs[CON] = 1;
  r->extent[CON][0] = ext[0];
  r->stride[A][CON][0] = st[0][0];
  r->stride[B][CON][0] = st[1][0];
  return 1;
}

/* The contraction of [r]'s groups over [op], as contract_view's, a and b
   scaled by [s] where it is not NULL: 0 if an allocation failed. A scaled
   operand copied (stage_runs) is copied scaled, and read unscaled. */
static int runs_on(int acc, int order, int da, int db, nx_contract_runs *r,
                   const nx_array *const *op, const scale *s) {
  for (int g = BATCH; g <= COL; g++)
    for (int k = 0; k < r->runs[g]; k++)
      if (r->extent[g][k] == 0) return 1;
  nx_array x[2];
  const nx_array *ops[4] = {op[A], op[B], op[INIT], op[DST]};
  scale kept[2] = {{0}};
  if (s) memcpy(kept, s, sizeof kept);
  int staged = r->runs[CON] > 1 && !merges(r), done = 0;
  if (staged) {
    static const int gs[2][3] = {{BATCH, ROW, CON}, {BATCH, COL, CON}};
    for (int o = A; o <= B; o++) {
      const scale *so = kept[o].v ? &kept[o] : NULL;
      int dt = so ? acc : op[o]->dtype;
      if (!stage_runs(r, o, gs[o], dt, op[o], so, &x[o])) {
        if (o == B) nx_cpu_give(x[A].base);
        return 0;
      }
      ops[o] = &x[o];
      kept[o] = (scale){0};
    }
    da = ops[A]->dtype;
    db = ops[B]->dtype;
    merges(r);
  }
  int scaled = kept[A].v || kept[B].v;
  outer q = {.acc = acc, .order = order, .da = da, .db = db, .op = ops,
             .s = scaled ? kept : NULL};
  atomic_init(&q.failed, 0);
  int64_t units = 1;
  for (int g = BATCH; g <= CON; g++) {
    int last = r->runs[g] - 1;
    q.v.extent[g] = r->extent[g][last];
    for (int o = A; o <= DST; o++) q.v.stride[o][g] = r->stride[o][g][last];
    int64_t weight = r->extent[g][last];
    for (int k = last - 1; k >= 0; k--) {
      q.group[q.n + k] = g;
      q.weight[q.n + k] = weight;
      weight *= r->extent[g][k];
    }
    for (int k = 0; k < last; k++) {
      q.extent[q.n] = r->extent[g][k];
      for (int o = A; o <= DST; o++) q.stride[o][q.n] = r->stride[o][g][k];
      units *= r->extent[g][k];
      q.n++;
    }
  }
  for (int o = A; o <= DST; o++) q.v.offset[o] = r->offset[o];
  /* As a factorisation's matrices (linalg.c): a position a unit where they
     are as many as the threads the whole takes, each then on its own
     thread, else in turn, each on the threads it takes. */
  const nx_cpu_gemm *g = &nx_cpu_table->gemm[part_of(acc)];
  double e = (double)units * (double)q.v.extent[BATCH];
  double m = (double)q.v.extent[ROW], n = (double)q.v.extent[COL];
  double k = (double)q.v.extent[CON];
  double bytes = e * (m * k + k * n + m * n) * nx_cpu_width(acc);
  double cost = flop_cost(g->kernel, 2 * e * m * n * k) + bytes;
  int threads = nx_cpu_threads_at((int64_t)bytes, cost, thread_bytes(g));
  if (threads > 1 && units >= threads)
    nx_cpu_job_at(units, (int64_t)bytes, cost, thread_bytes(g), outer_units,
                  &q);
  else outer_units(0, units, 0, &q);
  done = !atomic_load(&q.failed);
  if (staged) {
    nx_cpu_give(x[A].base);
    nx_cpu_give(x[B].base);
  }
  return done;
}

/* Results of another dtype

   A dst of another dtype than acc takes the sum in acc rounded once, as a
   cast of it would round it: R is scratch in acc, laid out C-contiguous
   over the batch's, rows' and columns' runs, and a cast walk takes it
   into dst once every product is in. A 4-bit acc sums in 8 bits
   (summed_in): R is then narrowed to acc, by a cast into dst where dst
   is of acc, else into scratch of acc cast into dst in turn. */

/* [r]'s view as runs: one a group. */
static void runs_of_view(nx_contract_runs *r, const nx_contract_view *v) {
  for (int g = BATCH; g <= CON; g++) {
    r->runs[g] = 1;
    r->extent[g][0] = v->extent[g];
    for (int o = A; o <= DST; o++) r->stride[o][g][0] = v->stride[o][g];
  }
  for (int o = A; o <= DST; o++) r->offset[o] = v->offset[o];
}

/* R's sums [y] into [dst], over the loop of [rank] axes of [extent], dst
   from [first] stepping [step], R C-contiguous: narrowed to [acc] first,
   in scratch of its own laid out as R, where [narrows]. 0 if an
   allocation failed. Apart, so that its two loops, 5 KiB each, are off
   the frame the product's jobs run under. */
static __attribute__((noinline)) int store_rounded(
    const nx_array *dst, const nx_array *y, int acc, int narrows,
    int64_t first, int rank, const int64_t *extent, const int64_t *step) {
  nx_loop l = {.first = {first, 0}};
  int64_t n = 1;
  for (int d = rank - 1; d >= 0; d--) {
    l.extent[d] = extent[d];
    l.step[0][d] = step[d];
    l.step[1][d] = n;
    n *= extent[d];
  }
  nx_array t = *y;
  t.base = NULL;
  int done = 1;
  if (narrows) {
    t.dtype = acc;
    t.bits = nx_dtype_row_of(acc).bits;
    t.base = nx_cpu_take((n * t.bits + 7) / 8);
    done = t.base != NULL;
    if (done) {
      nx_loop lt = l;
      for (int d = 0; d < rank; d++) lt.step[0][d] = l.step[1][d];
      lt.first[0] = 0;
      lt.rank = nx_coalesce_dims(2, rank, lt.extent, lt.step);
      nx_cpu_cast_loop(&t, y, &lt);
    }
  }
  if (done) {
    l.rank = nx_coalesce_dims(2, rank, l.extent, l.step);
    nx_cpu_cast_loop(dst, narrows ? &t : y, &l);
  }
  if (t.base) nx_cpu_give(t.base);
  return done;
}

/* The contraction of [r]'s groups over [op] in [acc], summed in [sum],
   into op[DST] of another dtype than [sum], through scratch R: 0 if an
   allocation failed. */
static int rounded_on(int acc, int sum, int order, nx_contract_runs *r,
                      const nx_array *const *op, const scale *s) {
  /* The loop the cast walks, over the runs of more than one element:
     their extents and dst's steps; R's steps C-contiguous over them. */
  int64_t extent[NX_MAX_RANK], step[NX_MAX_RANK], first = r->offset[DST];
  int64_t *at[NX_MAX_RANK], n = 1;
  int rank = 0;
  for (int g = BATCH; g <= COL; g++)
    for (int k = 0; k < r->runs[g]; k++) {
      int64_t e = r->extent[g][k], s = r->stride[DST][g][k];
      if (e == 0) return 1;
      r->stride[DST][g][k] = 0;
      if (e == 1) continue;
      extent[rank] = e;
      step[rank] = s;
      at[rank++] = &r->stride[DST][g][k];
    }
  for (int d = rank - 1; d >= 0; d--) {
    *at[d] = n;
    n *= extent[d];
  }
  r->offset[DST] = 0;
  nx_array y = *op[DST];
  y.dtype = sum;
  y.bits = nx_dtype_row_of(sum).bits;
  y.offset = 0;
  y.alias = 0;
  y.base = nx_cpu_take(n * nx_cpu_width(sum));
  if (y.base == NULL) return 0;
  const nx_array *ops[4] = {op[A], op[B], op[INIT], &y};
  /* A 4-bit acc's float and complex operands saturate into its range, as
     a cast into it does, and an 8-bit sum's would into the 8 bits': they
     are cast into acc first, and the sum reads integers, which convert
     modulo 2^8. Where the contracted runs are several, a and b are both
     cast so, contracted last, so that the runs merge and runs_on copies
     neither again. */
  static const int gs[3][3] = {[A] = {BATCH, ROW, CON},
                               [B] = {BATCH, COL, CON},
                               [INIT] = {BATCH, ROW, COL}};
  nx_array z[3];
  int done = 1;
  for (int o = A; o <= INIT && done; o++) {
    int kind = op[o] ? nx_dtype_row_of(op[o]->dtype).kind : NX_KIND_BOOLEAN;
    int split = o != INIT && r->runs[CON] > 1;
    if (acc == sum ||
        (!split && kind != NX_KIND_FLOAT && kind != NX_KIND_COMPLEX))
      continue;
    done = stage_runs(r, o, gs[o], acc, op[o], NULL, &z[o]);
    if (done) ops[o] = &z[o];
  }
  if (done)
    done = runs_on(sum, order, ops[A]->dtype, ops[B]->dtype, r, ops, s);
  for (int o = A; o <= INIT; o++)
    if (ops[o] == &z[o]) nx_cpu_give(z[o].base);
  int narrows = acc != sum && op[DST]->dtype != acc;
  if (done)
    done = store_rounded(op[DST], &y, acc, narrows, first, rank, extent, step);
  nx_cpu_give(y.base);
  return done;
}

/* Scales

   A scaled operand's scales are decoded into acc once a call, C-contiguous,
   and each of its groups of axes takes a table from an index to its block's
   position among them (scale): the scales' count in acc, and O(B + M + K)
   or O(B + N + K) positions. */

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

/* The scale of code [e], 2^(e - 127), NaN for 255, by its bits: 0 is
   float32's subnormal 2^-127. */
static float code32(int e) {
  uint32_t b = e == 255 ? 0x7FC00000u : e == 0 ? 0x00400000u : (uint32_t)e << 23;
  float f;
  memcpy(&f, &b, sizeof f);
  return f;
}

static double code64(int e) {
  uint64_t b = e == 255 ? 0x7FF8000000000000u : (uint64_t)(e + 1023 - 127) << 52;
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
  nx_cpu_job(planes * f.chunks, bytes, bytes * NX_CPU_STRIDED_COST,
             decode_rows, &f);
}

/* Fills [t] with a group's table over its [n] axes [ax]: each index of the
   group, in C order of the axes' extents [x], to the sum along each axis
   of the index's block, its index divided by [ratio], times the scales'
   [step]. */
static void table(int n, const int *ax, const int64_t *x, const int64_t *ratio,
                  const int64_t *step, int64_t *t) {
  int64_t len = 1;
  t[0] = 0;
  for (int a = 0; a < n; a++) {
    int d = ax[a];
    int64_t e = x[d];
    for (int64_t q = len - 1; q >= 0; q--) {
      int64_t first = t[q];
      for (int64_t i = e - 1; i >= 0; i--)
        t[q * e + i] = first + i / ratio[d] * step[d];
    }
    len *= e;
  }
}

/* A scaled side's facts held in OCaml values, read before nx_read,
   which may run the collector: whether its scale is codes, and its groups'
   axes, batch, rows or columns, contraction (group_axes). */
typedef struct {
  int exponent, n[3], ax[3][NX_MAX_RANK];
} scale_axes;

static const int groups[2][3] = {{BATCH, ROW, CON}, {BATCH, COL, CON}};

/* Side [o]'s [scale_axes], its scale [vs] an Nx_kernel.scale and its
   operand [vx]. Out of line, as scales_fit. */
static __attribute__((noinline)) void axes_of(value vs, value vx,
                                              value vbatch, value vcon, int o,
                                              scale_axes *g) {
  int64_t dim[2 * NX_MAX_RANK], off;
  int rank = nx_array_layout(vx, dim, &off);
  g->exponent = Tag_val(vs) == 0;
  for (int t = 0; t < 3; t++)
    g->n[t] = group_axes(vbatch, vcon, o, rank, groups[o][t], g->ax[t]);
}

/* The scales [sc] of operand [x], side [o], its axes [g], into [s], its
   memory, to free, at [*mem]: 0 if it could not be allocated. Out of
   line, as scales_fit. */
static __attribute__((noinline)) int scale_of(int acc, const nx_array *x,
                                              const nx_array *sc,
                                              const scale_axes *g, int o,
                                              scale *s, uint8_t **mem) {
  int r = x->rank, w = nx_cpu_width(acc);
  int64_t count = 1, step[NX_MAX_RANK], ratio[NX_MAX_RANK], len[3];
  for (int d = r - 1; d >= 0; d--) {
    step[d] = count;
    count *= sc->dim[d];
    ratio[d] = sc->dim[d] ? x->dim[d] / sc->dim[d] : 1;
  }
  int64_t values = ceil_div(count * w, 8), positions = 0;
  for (int t = 0; t < 3; t++) {
    len[t] = 1;
    for (int k = 0; k < g->n[t]; k++) len[t] *= x->dim[g->ax[t][k]];
    positions += len[t] > 0 ? len[t] : 1;
  }
  int64_t *m = alloc((values + positions) * 8);
  if (m == NULL) return 0;
  *mem = (uint8_t *)m;
  *s = (scale){.v = (const uint8_t *)m};
  decode(sc, g->exponent, acc, (uint8_t *)m);
  int nk = g->n[2];
  s->run = nk ? ratio[g->ax[2][nk - 1]] : 1;
  int64_t *t = m + values;
  for (int k = 0; k < 3; k++) {
    table(g->n[k], g->ax[k], x->dim, ratio, step, t);
    s->t[groups[o][k]] = t;
    t += len[k] > 0 ? len[k] : 1;
  }
  return 1;
}

/* Whether each scale of [vs], options of Nx_kernel.scale, fits its
   operand of [va] and [vb]: of its rank, each extent dividing the
   operand's, 0 only along an empty axis. Out of line: its layouts stay off
   the external's frame. */
static __attribute__((noinline)) int scales_fit(value va, value vb,
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

/* The price (cpu.h, Calls) of a contraction in [acc] in [order] whose
   groups have the extents [x]: scalar, its flops at a core's speed
   (nx_cpu_core_speed) in the dtype it sums in, lane order's at a flop a
   byte (lane_cost), a complex flop at four of its parts', and each of its
   batch's products' set-up, 7.7 KiB in a batch of 8192 1 × 1 products
   (M1); streaming, its operands' bytes in that dtype, each a strided
   copy, which it stages inline where they are not in that dtype or not
   in place (parts_of, words_of, stage_runs, rounded_on). */
#define PRODUCT_COST (16 * 1024)

static double call_cost(int acc, int order, const double *x) {
  int sum = summed_in(acc), r = part_of(sum);
  double e = x[BATCH], m = x[ROW], n = x[COL], k = x[CON];
  double flops = 2 * e * m * n * k * (r == sum ? 1 : 4);
  double bytes = e * (m * k + k * n + 2 * m * n) * nx_cpu_width(sum);
  bytes = bytes * NX_CPU_STRIDED_COST + e * PRODUCT_COST;
  if (order == NX_CPU_LANE) return flops + bytes;
  /* A call held at the slowest kernels' speed skips the lookup of its
     own: gemm_generic.c's C library fma, 0.01 flops a byte. */
  if (flops * 100 + bytes < NX_CPU_RELEASE_BYTES) return flops * 100 + bytes;
  return flops / nx_cpu_core_speed(r) + bytes;
}

/* The price of decoding the scales [sc] into [acc] (decode), a strided
   copy. */
static double decode_cost(int acc, const nx_array *sc) {
  return nx_cpu_elements(sc) * nx_cpu_width(acc) * NX_CPU_STRIDED_COST;
}

/* The external: groups the view of the pairs [vbatch] and [vcontracting]
   on the stack, or their runs where a group makes several, answering what
   the fill answers where it does not fit, declines what it does not
   compute, then reads the operands through the door. [vorder] is the
   order, [vacc] acc's dtype code, [vi] the init and [vsa] and [vsb] a's
   and b's scales, options of Nx_kernel.scale. Its body is inlined into
   two copies, [scaled] a constant in each, so that an unscaled call runs
   none of the scales' setup. */
static inline __attribute__((always_inline)) value contract_body(
    value vorder, value vbatch, value vcontracting, value vacc, value vdst,
    value vi, value va, value vb, value vsa, value vsb, int scaled) {
  int acc = Int_val(vacc), init = Is_block(vi);
  value vinit = init ? Field(vi, 0) : va;
  nx_contract_view v;
  nx_contract_runs r;
  int fit = nx_contract_fill(&v, &r, vbatch, vcontracting, vdst, va, vb, vi);
  int runs = fit == NX_DECLINED;
  if (fit != NX_OK && !runs) return Val_int(fit);
  int dts[3] = {nx_array_dtype(va), nx_array_dtype(vb),
                nx_array_dtype(vinit)};
  int out = nx_array_dtype(vdst);
  if (acc < 0 || acc >= NX_DTYPE_COUNT || !computes(acc))
    return Val_int(NX_DECLINED);
  value vs[2] = {vsa, vsb};
  if (scaled && acc != NX_FLOAT32 && acc != NX_FLOAT64)
    return Val_int(NX_DECLINED);
  if (scaled && !scales_fit(va, vb, vs)) return Val_int(NX_SHAPE);
  /* Read before nx_read, which may run the collector. */
  scale_axes sx[2];
  for (int o = A; scaled && o <= B; o++)
    if (Is_block(vs[o]))
      axes_of(Field(vs[o], 0), o == A ? va : vb, vbatch, vcontracting, o,
              &sx[o]);
  nx_operand in[6];
  in[0] = (nx_operand){va, dts[0], 0};
  in[1] = (nx_operand){vb, dts[1], 0};
  in[2] = (nx_operand){vdst, out, 1};
  in[3] = (nx_operand){vinit, dts[2], 0};
  nx_array a[6];
  int n = 3 + init, at[2] = {-1, -1};
  for (int o = A; scaled && o <= B; o++)
    if (Is_block(vs[o])) {
      value sv = Field(Field(vs[o], 0), 0);
      at[o] = n;
      in[n++] = (nx_operand){sv, nx_array_dtype(sv), 0};
    }
  int e = nx_read(n, in, a);
  if (e) return Val_int(e);
  const nx_array *op[4] = {&a[0], &a[1], init ? &a[3] : NULL, &a[2]};
  /* Only chain order has integer kernels, and an integer sum is exact
     modulo acc's width in any order, so chain order computes it whatever
     the order asked. */
  int order = is_integer(acc) ? NX_CPU_CHAIN : Int_val(vorder), done;
  double x[4];
  for (int g = BATCH; g <= CON; g++) {
    x[g] = runs ? 1 : (double)v.extent[g];
    for (int k = 0; runs && k < r.runs[g]; k++)
      x[g] *= (double)r.extent[g][k];
  }
  double cost = call_cost(acc, order, x);
  for (int o = A; scaled && o <= B; o++)
    if (at[o] >= 0) cost += decode_cost(acc, &a[at[o]]);
  int released = nx_cpu_release(cost);
  int sum = summed_in(acc);
  scale s[2];
  uint8_t *mem[2] = {NULL, NULL};
  done = 1;
  if (scaled) memset(s, 0, sizeof s);
  for (int o = A; scaled && o <= B && done; o++)
    if (at[o] >= 0)
      done = scale_of(acc, op[o], &a[at[o]], &sx[o], o, &s[o], &mem[o]);
  const scale *sp = scaled ? s : NULL;
  if (done && (out != acc || sum != acc)) {
    if (!runs) runs_of_view(&r, &v);
    done = rounded_on(acc, sum, order, &r, op, sp);
  } else if (done && runs)
    done = runs_on(acc, order, dts[0], dts[1], &r, op, sp);
  else if (done) done = contract_view(acc, order, dts[0], dts[1], &v, op, sp);
  if (scaled) {
    free(mem[A]);
    free(mem[B]);
  }
  nx_cpu_reacquire(released);
  nx_done(n, a);
  if (!done) caml_raise_out_of_memory();
  return Val_int(NX_OK);
}

static __attribute__((noinline)) value contract_scaled(
    value vorder, value vbatch, value vcontracting, value vacc, value vdst,
    value vi, value va, value vb, value vsa, value vsb) {
  return contract_body(vorder, vbatch, vcontracting, vacc, vdst, vi, va, vb,
                       vsa, vsb, 1);
}

value nx_cpu_contract(value vorder, value vbatch, value vcontracting,
                      value vacc, value vdst, value vi, value va, value vb,
                      value vsa, value vsb) {
  if (Is_block(vsa) || Is_block(vsb))
    return contract_scaled(vorder, vbatch, vcontracting, vacc, vdst, vi, va,
                           vb, vsa, vsb);
  return contract_body(vorder, vbatch, vcontracting, vacc, vdst, vi, va, vb,
                       vsa, vsb, 0);
}

value nx_cpu_contract_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_contract(argv[0], argv[1], argv[2], argv[3], argv[4],
                         argv[5], argv[6], argv[7], argv[8], argv[9]);
}
