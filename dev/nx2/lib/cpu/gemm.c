/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contractions: y[b, i, j] = init[b, i, j] + Σ_k a[b, i, k] · b[b, k, j],
   computed in acc and rounded once to y's dtype.

   nx_contract_fill (gemm_view.c) groups the axes, as cpu.h states, into one
   batch, row (M), column (N) and contracted (K) axis per operand; a group
   that makes several runs takes an outer loop, a's or b's packs, or a copy
   (gemm_runs.c). An operand is read through the stage, which takes any
   strides and dtype into its carrier, and the target's conversion takes
   the carrier into acc: each element as a cast to acc gives it, exact
   where acc holds its values. A scaled
   operand is staged wherever it is read, each element then multiplied by
   its block's scale in acc (stage_at, gemm_scales.c). An integer acc
   computes on chain order's paths. Lane order reads b in place where it
   is acc and adjacent along the axis it streams.

   Each output's sum adds in one of two orders, which nx_gemm_order_of chooses
   from a call's M, N and K, a segmented call's once from the whole
   call's, so that its offsets never change a bit:

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
   product is in (gemm_rounded.c). Each tile starts as init, or +0, when
   its first KC block is computed.

   The derived contractions have a file each: complex numbers
   (gemm_complex.c), 8-bit quads (gemm_quads.c), groups of several runs
   (gemm_runs.c), results of another dtype (gemm_rounded.c), scales
   (gemm_scales.c) and segments (gemm_segments.c). gemm.h holds what they
   share with this file. */

#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/mlvalues.h>

#include "gemm.h"

/* Rows of a pass of lane order: a product of more runs in passes, each
   reading b again. */
#define LANE_ROWS 8

/* Lane order's units: dots take DOT_COLUMNS columns of b (gemm.h), rows
   ROW_BYTES of each row of b, a page. */
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

/* Job costs, in bytes of memcpy (job.c): a kernel's flops at its speed
   (cpu.h), and two a byte packed, a guess no measurement has refined. */
#define PACK_COST 2

/* The copies of [k]'s hardware the performance cores hold. */
static int64_t engines(nx_cpu_micro k) {
  int64_t n = rig_pool_performance_cores() / k.shared;
  return n > 1 ? n : 1;
}

/* Staging */

/* The lines and the scales a call of the target's float4 stage takes at
   most. */
#define F4_LINES 16
#define F4_SCALES 256

/* The position among [s]'s scales of line [l]'s block, the lines along
   [across] from the element [c]. */
static int64_t line_at(const scale *s, const int64_t *c, int across,
                       int64_t l) {
  int64_t at = 0;
  for (int x = BATCH; x <= COL; x++)
    if (s->t[x]) at += s->t[x][c[x] + (x == across ? l : 0)];
  return at;
}

/* stage_f4's lines in acc's type [T], each code from [from] along k:
   each of [lines] lines of [len] codes, [sl] apart from [first] and [sk]
   apart along k, into dst [du] apart along k and [dl] apart across
   lines, each run of NX_CPU_F4_RUN codes through its 16 values times
   the run's scale. */
#define F4_EACH(T)                                                         \
  do {                                                                     \
    const T *v = (const T *)s->v;                                          \
    T *d = (T *)dst, base[16], tab[16];                                    \
    for (int u = 0; u < 16; u++)                                           \
      base[u] = (T)nx_e2m1fn_to_float((uint8_t)u);                         \
    for (int64_t l = 0; l < lines; l++) {                                  \
      int64_t at = line_at(s, c, across, l), line = first + l * sl;        \
      for (int64_t u0 = from; u0 < len; u0 += NX_CPU_F4_RUN) {             \
        T sv = v[at + tk[u0]];                                             \
        for (int u = 0; u < 16; u++) tab[u] = base[u] * sv;                \
        for (int64_t u = u0; u < u0 + min64(NX_CPU_F4_RUN, len - u0);      \
             u++) {                                                        \
          int64_t xi = line + u * sk;                                      \
          d[u * du + l * dl] = tab[(codes[xi >> 1] >> ((xi & 1) * 4)) & 15]; \
        }                                                                  \
      }                                                                    \
    }                                                                      \
  } while (0)

/* The scales of runs [r0, r0 + nr) of lines [l0, l0 + nl) into [sv], run
   r of line l at sv[r·nl + l], as the target's float4 stage reads them. */
#define F4_GATHER(T)                                                       \
  do {                                                                     \
    const T *v = (const T *)s->v;                                          \
    T *g = (T *)sv;                                                        \
    for (int64_t l = 0; l < nl; l++) {                                     \
      int64_t at = line_at(s, c, across, l0 + l);                          \
      for (int64_t r = 0; r < nr; r++)                                     \
        g[r * nl + l] = v[at + tk[(r0 + r) * NX_CPU_F4_RUN]];              \
    }                                                                      \
  } while (0)

/* Stages operand [o]'s block as stage_at does where it is float4 and its
   block along k starts a run of its scale's, [run] a multiple of
   NX_CPU_F4_RUN, so each run of NX_CPU_F4_RUN codes takes one scale: 1
   if it did. The bits are stage_at's: each code's value times its scale,
   rounded once. Codes adjacent along k, each line from a byte's first, go
   a whole run at a time through the target's float4 stage; others, and a
   last run of fewer codes, one at a time. */
static int stage_f4(const problem *p, int o, const int64_t *c, int x0,
                    int x1, int64_t n0, int64_t n1, uint8_t *dst,
                    int64_t pitch) {
  const operand *q = &p->op[o];
  const scale *s = q->s;
  if (q->x->dtype != NX_FLOAT4_E2M1FN || (x0 != CON && x1 != CON) ||
      s->run % NX_CPU_F4_RUN || c[CON] % NX_CPU_F4_RUN)
    return 0;
  int along = x0 == CON, across = along ? x1 : x0;
  int64_t len = along ? n0 : n1, lines = along ? n1 : n0;
  int64_t du = along ? 1 : pitch, dl = along ? pitch : 1;
  int64_t first = pos(p, o, c[BATCH], c[ROW], c[COL], c[CON]);
  int64_t sk = q->st[CON], sl = q->st[across], w = p->w, from = 0;
  const uint8_t *codes = q->x->base;
  const int64_t *tk = s->t[CON] + c[CON];
  if (sk == 1 && first % 2 == 0 && sl % 2 == 0) {
    nx_cpu_f4 f4 = nx_cpu_table->f4[p->acc];
    int64_t runs = len / NX_CPU_F4_RUN;
    _Alignas(32) uint8_t sv[F4_SCALES * sizeof(double)];
    for (int64_t l0 = 0, nl; l0 < lines; l0 += nl) {
      nl = min64(F4_LINES, lines - l0);
      for (int64_t r0 = 0, nr; r0 < runs; r0 += nr) {
        nr = min64(F4_SCALES / nl, runs - r0);
        if (p->acc == NX_FLOAT32) F4_GATHER(float);
        else F4_GATHER(double);
        f4(codes + (first + l0 * sl) / 2 + r0 * NX_CPU_F4_RUN / 2, sl / 2,
           nl, nr, sv, dst + (r0 * NX_CPU_F4_RUN * du + l0 * dl) * w, du,
           dl);
      }
    }
    from = runs * NX_CPU_F4_RUN;
  }
  if (from == len) return 1;
  if (p->acc == NX_FLOAT32) F4_EACH(float);
  else F4_EACH(double);
  return 1;
}

/* Stages operand [o]'s block as stage_acc does, from its element
   (e, i, j, k), [n0] along axis [x0] in a row and [n1] rows along [x1]:
   along an axis of several runs, piece by piece as nx_gemm_pack_at reads it, so
   that the operand is read where it lies; otherwise in one stage, each
   element of a scaled operand, whose axes are one run each, times its
   block's scale, rounded once in acc, a float32 or float64. */
static void stage_at(const problem *p, int o, int64_t e, int64_t i,
                     int64_t j, int64_t k, int x0, int x1, int64_t n0,
                     int64_t n1, uint8_t *dst, int64_t pitch) {
  const operand *q = &p->op[o];
  const int64_t c[4] = {e, i, j, k};
  if (q->sp[x0] || q->sp[x1]) {
    nx_gemm_pack_at(p, NULL, o, c, x0, x1, n0, n1, dst, pitch);
    return;
  }
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

/* Whether operand [o] may be read in place: in acc, unscaled, of one run
   along each axis. */
static int raw(const problem *p, int o) {
  const operand *q = &p->op[o];
  return q->x->dtype == p->acc && q->s == NULL && q->sp[ROW] == NULL &&
         q->sp[COL] == NULL && q->sp[CON] == NULL;
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

/* Packs operand [o]'s block from its element [c], (e, i, j, k), as pack
   does: [n0] along axis [x0] in a row and [n1] rows along [x1], into [d]
   [pitch] apart. Along an axis of several runs, as a window's taps make,
   a pack per piece of the block that lies in one run of each axis, so
   that the operand is read where it lies, never copied whole: a
   convolution's a packs a block of its windows as an explicit copy of
   them would hold it, the products numbered as they are there.

   [move] may write past a piece's elements up to the next multiple of
   the kernel's tile (cpu.h). Within the piece's row, the pieces after it
   overwrite those writes; past the row's [pitch] they would land on the
   next row's first elements, packed before them. A piece whose writes
   would reach past the row packs without [move]: in a block of 96 rows of
   windows 20 apart, the piece from row 80 would otherwise give the next
   step's rows 0 to 15 values past the piece's 16 rows. */
void nx_gemm_pack_at(const problem *p, nx_cpu_pack move, int o,
                     const int64_t *c, int x0, int x1, int64_t n0, int64_t n1,
                     uint8_t *d, int64_t pitch) {
  const operand *q = &p->op[o];
  int64_t tile = move == NULL ? 1 : o == A ? p->g->kernel.mr : p->g->kernel.nr;
  for (int64_t u1 = 0, m1; u1 < n1; u1 += m1) {
    int64_t s0, s1;
    m1 = piece(q, x1, c[x1] + u1, n1 - u1, &s1);
    for (int64_t u0 = 0, m0; u0 < n0; u0 += m0) {
      m0 = piece(q, x0, c[x0] + u0, n0 - u0, &s0);
      int fits = u0 + ceil_div(m0, tile) * tile <= pitch;
      int64_t t[4] = {c[0], c[1], c[2], c[3]};
      t[x0] += u0;
      t[x1] += u1;
      pack(p, fits ? move : NULL, o,
           pos(p, o, t[BATCH], t[ROW], t[COL], t[CON]), s0, s1, m0, m1,
           d + (u1 * pitch + u0) * p->w, pitch);
    }
  }
}

/* Packs [a]'s rows [i, i + m) of element [e], along k from [pc] for
   [kc], into [d], as the kernel reads them (packed_a). A block of rows is
   one pack: where it transposes, it does so in square blocks, where a
   sliver as narrow as a microkernel's 6 rows would move element by
   element. */
static void pack_a(const problem *p, nx_cpu_pack move, int64_t e, int64_t i,
                   int64_t m, int64_t lda, int64_t pc, int64_t kc,
                   uint8_t *d) {
  const int64_t c[4] = {e, i, 0, pc};
  int elements = p->g->kernel.a == NX_CPU_A_ELEMENTS;
  if (p->op[A].s && elements)
    stage_at(p, A, e, i, 0, pc, CON, ROW, kc, m, d, kc);
  else if (p->op[A].s) stage_at(p, A, e, i, 0, pc, ROW, CON, m, kc, d, lda);
  else if (elements) nx_gemm_pack_at(p, move, A, c, CON, ROW, kc, m, d, kc);
  else nx_gemm_pack_at(p, move, A, c, ROW, CON, m, kc, d, lda);
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
  const int64_t c[4] = {e, 0, j, pc};
  if (p->op[B].s) stage_at(p, B, e, 0, j, pc, COL, CON, n, kc, d, nr);
  else nx_gemm_pack_at(p, move, B, c, COL, CON, n, kc, d, nr);
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
void nx_gemm_transpose(problem *p) {
  operand a = p->op[A];
  p->op[A] = p->op[B];
  p->op[B] = a;
  for (int o = 0; o < 4; o++) {
    operand *q = &p->op[o];
    int64_t r = q->st[ROW];
    q->st[ROW] = q->st[COL];
    q->st[COL] = r;
    const spans *sp = q->sp[ROW];
    q->sp[ROW] = q->sp[COL];
    q->sp[COL] = sp;
    r = q->skip[ROW];
    q->skip[ROW] = q->skip[COL];
    q->skip[COL] = r;
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

   - Dots, where b is adjacent along k or along each run of it, as a
     weight stored [n × k] is, or a convolution's windows: a unit is
     (element, DOT_COLUMNS columns, range of blocks), and each output's
     block a dot along k, a column of b against every row of a. A product
     of few outputs over a long k, a dot's, splits its blocks into ranges,
     whose sums meet in a second job; otherwise a unit takes every block
     and totals its outputs itself.
   - Rows, otherwise, as for a weight stored [k × n]: a block of ROW_BYTES
     of columns adds the block's rows of b in turn, row k0 + t times its
     row's element of a into lane t modulo 16's accumulators, one per
     output. Where an element's groups of columns are enough units for the
     threads, a unit takes a group whole, sums each block's lanes by the
     lanes' tree and totals its outputs; otherwise a unit is a span of
     the lanes of one block of one group, and a second job sums and totals
     them. */

/* The blocks' tree's sum of the [n] >= 1 values at [s]; the lanes' tree
   of [c] outputs, lane q of output j at l[q·ls + j], into lane 0's. */
#define TREE(T, name)                                                        \
  static T name(const T *s, int64_t n) {                                     \
    T v[NX_CPU_TREE_DEPTH];                                                  \
    int top = 0;                                                             \
    for (int64_t b = 0; b < n; b++) {                                        \
      v[top++] = s[b];                                                       \
      NX_CPU_TREE_STEP(b + 1, top, v[top - 2] = v[top - 2] + v[top - 1]);    \
    }                                                                        \
    NX_CPU_TREE_END(top, v[top - 2] = v[top - 2] + v[top - 1]);              \
    return v[0];                                                             \
  }                                                                          \
  static void name##_lanes(T *l, int64_t ls, int64_t c) {                    \
    NX_CPU_LANE_TREE(i, q,                                                   \
                     for (int64_t j = 0; j < c; j++)                         \
                         l[i * ls + j] += l[q * ls + j]);                    \
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
   so that a column of b streams. In place, a column of b is the dot's
   vector, against NX_CPU_DOT_ROWS rows of a at a time. Staged, the block
   of NX_CPU_DOT_ROWS columns at once into [bp] is the dot's rows,
   against each row of a, so that a product of one row keeps as many
   chains in flight: a product's bits do not depend on which factor is
   which. Each block's lanes are summed by the lanes' tree into its sum. */
static void dots_columns(const dots_job *f, uint8_t *bp, uint8_t *mine,
                         int64_t e, int64_t j0, int64_t c, int64_t x0,
                         int64_t x1) {
  const problem *p = f->p;
  int64_t m = p->ext[ROW], k = p->ext[CON], w = p->w;
  const uint8_t *a = f->a + e * f->a_batch * w;
  int in_place = adjacent(p, B, CON);
  int64_t cs = in_place ? 1 : NX_CPU_DOT_ROWS; /* columns at once */
  int64_t rs = in_place ? NX_CPU_DOT_ROWS : 1; /* rows at once */
  nx_cpu_dot dot = nx_cpu_table->dot[p->acc];
  _Alignas(64) uint8_t l[NX_CPU_DOT_ROWS * NX_CPU_LANES * 8];
  for (int64_t j = 0; j < c; j += cs)
    for (int64_t x = x0; x < x1; x++) {
      int64_t k0 = x * NX_CPU_LANE_BLOCK;
      int64_t len = min64(NX_CPU_LANE_BLOCK, k - k0), nc = min64(cs, c - j);
      const uint8_t *b = bp;
      if (in_place) b = p->op[B].x->base + pos(p, B, e, 0, j0 + j, k0) * w;
      else stage_at(p, B, e, 0, j0 + j, k0, CON, COL, len, nc, bp, len);
      for (int64_t i0 = 0; i0 < m; i0 += rs) {
        int64_t nr = min64(rs, m - i0);
        const uint8_t *ai = a + (i0 * f->lda + k0) * w;
        memset(l, 0, sizeof l);
        if (in_place) dot(ai, f->lda, (int)nr, b, len, l);
        else dot(b, len, (int)nc, ai, len, l);
        /* The dot's row i + q: output (i0 + i, j + q), one of i and q 0. */
        for (int64_t i = 0; i < nr; i++)
          for (int64_t q = 0; q < nc; q++) {
            uint8_t *lq = l + (i + q) * NX_CPU_LANES * w;
            lane_sums(p->acc, lq, 1, 1);
            memcpy(sums_of(f, mine, e, i0 + i, j0, c, j + q) + x * w, lq,
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
  int64_t b_bytes = NX_CPU_DOT_ROWS * NX_CPU_LANE_BLOCK * w;
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

/* Whether [p] takes dots: b steps one along k, within runs of at least
   NX_CPU_LANES where k is several runs. Dots stage a run of b a column
   at a time, rows a row of k across the columns: on fudge a 3 × 3
   convolution of an [8; 112; 112; 3] image into 3 channels, runs of 9,
   took twice as long in dots as in rows, 0.79 ms against 0.40; one of
   [8; 56; 56; 64], runs of 192, 0.35 ms in dots against 1.26 in rows. */
static int dotted(const problem *p) {
  const spans *s = p->op[B].sp[CON];
  if (step_of(&p->op[B], CON) != 1) return 0;
  return s == NULL || s->extent[s->n - 1] >= NX_CPU_LANES;
}

static void lane_pass(problem *p) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t w = p->w, bytes;
  if (p->ext[CON] == 0) {
    nx_cpu_job(batch, batch * m * n * w, batch * m * n * w, empty, p);
    return;
  }
  double cost = lane_cost(p, &bytes);
  if (dotted(p)) dots(p, bytes, cost);
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
   The call's entry prices the whole product (nx_gemm_call_cost), so that it
   releases the domain lock though each pass's jobs cost less: on one
   core a 2000 × 2000 × 16 float32 product held it for 45 ms, in 250
   passes of 0.6 MiB each. */
static void lanes(problem *p) {
  if (tiled(p)) {
    tiles(p);
    return;
  }
  if (p->ext[COL] < p->ext[ROW]) nx_gemm_transpose(p);
  int64_t m = p->ext[ROW];
  for (int64_t i = 0; i < m && !atomic_load(&p->failed); i += LANE_ROWS) {
    p->ext[ROW] = min64(LANE_ROWS, m - i);
    lane_pass(p);
    for (int o = 0; o < 4; o++) {
      operand *q = &p->op[o];
      q->first += p->ext[ROW] * q->st[ROW];
      if (q->sp[ROW]) q->skip[ROW] += p->ext[ROW];
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
           kc * p->op[B].st[CON] * p->w <= nx_cpu_l1 / 2;
  }
  int64_t n = o == A ? p->ext[ROW] : p->ext[COL];
  int64_t most = o == A ? IN_PLACE_BYTES : nx_cpu_l2 / 2;
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
  return m <= p->g->few && adjacent(p, B, CON) && b_bytes > nx_cpu_l2;
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
void nx_gemm_lay_out(problem *p, const nx_contract_view *v,
                     const nx_array *const *op) {
  for (int o = 0; o < 4; o++) {
    p->op[o].x = op[o];
    p->op[o].first = v->offset[o];
    for (int x = 0; x < 4; x++) p->op[o].st[x] = v->stride[o][x];
  }
  for (int x = 0; x < 4; x++) p->ext[x] = v->extent[x];
}

/* nx_cpu_contract_on on the kernels [g], a and b scaled by [s] where it is
   not NULL, and read along a group of several runs where [sp], a's and
   b's by group, is not NULL and the group's spans hold more than one. */
int nx_gemm_contract_with(const nx_cpu_gemm *g, int acc, int order,
                          const nx_contract_view *v, const nx_array *const *op,
                          const scale *s, const reads *sp) {
  problem p = {.acc = acc, .w = nx_cpu_width(acc), .g = g};
  nx_gemm_lay_out(&p, v, op);
  for (int o = A; sp && o <= B; o++)
    for (int x = ROW; x <= CON; x++)
      if (sp->of[o][x].n > 1) p.op[o].sp[x] = &sp->of[o][x];
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
      nx_gemm_transpose(&p);
      if (!streams(&p)) nx_gemm_transpose(&p);
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
  return nx_gemm_contract_with(&nx_cpu_table->gemm[acc], acc, order, v, op,
                               NULL, NULL);
}

double nx_cpu_core_speed(int acc) {
  const nx_cpu_gemm *g = &nx_cpu_table->gemm[acc];
  if (g->kernel.shared > 1 && g->other != NULL) g = g->other;
  return g->kernel.flops_per_byte;
}

/* The contraction of the view [v] of [op], in [acc] and [order], [da] and
   [db] a's and b's dtypes: 0 if an allocation failed. A scaled
   contraction, [s] not NULL, is a float one: it runs on the float
   kernels. a and b are read along their spans [sp] where it is not NULL,
   which only the paths nx_gemm_spanned names are given. */
int nx_gemm_contract_view(int acc, int order, int da, int db,
                          const nx_contract_view *v, const nx_array *const *op,
                          const scale *s, const reads *sp) {
  const nx_cpu_gemm *g = &nx_cpu_table->gemm[acc];
  if (s) return nx_gemm_contract_with(g, acc, order, v, op, s, NULL);
  const nx_cpu_gemm *quads = nx_gemm_quads_for(acc, da, db);
  if (quads) return nx_gemm_quads_on(quads, acc, v, op);
  if (part_of(acc) != acc) return nx_gemm_complex_on(acc, order, v, op, sp);
  return nx_gemm_contract_with(g, acc, order, v, op, NULL, sp);
}

/* Whether nx_gemm_contract_view reads a and b along spans: chain order's packs
   and lane order's stages do, and a complex contraction's parts of a and
   blocks of b, where it is unscaled and no quads take it. */
int nx_gemm_spanned(int acc, int da, int db, const scale *s) {
  return s == NULL && nx_gemm_quads_for(acc, da, db) == NULL;
}

/* The price (cpu.h, Calls) of a contraction in [acc] in [order] whose
   groups have the extents [x]: scalar, its flops at a core's speed
   (nx_cpu_core_speed) in the dtype it sums in, lane order's at a flop a
   byte (lane_cost), a complex flop at four of its parts', and each of its
   batch's products' set-up, 7.7 KiB in a batch of 8192 1 × 1 products
   (M1); streaming, its operands' bytes in that dtype, each a strided
   copy, which it stages inline where they are not in that dtype or not
   in place (gemm_complex.c's parts_of, gemm_quads.c's words_of,
   nx_gemm_stage_runs, nx_gemm_rounded_on). */
double nx_gemm_call_cost(int acc, int order, const double *x) {
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

/* The order of a contraction in [acc] whose groups have the extents [x].
   Integers take chain order, the one with integer kernels: an integer sum
   is exact modulo acc's width in any order.

   Chain order where a contraction has the outputs to keep every target's
   chains in flight, CHAIN_OUTPUTS (NEON keeps 16 vectors of 4 chains, 4
   pipes of 4 cycles; AVX2 8 of 8); lane order where its rows or columns
   are at most LANE_MOST, as decoding a token's are, so that b streams once
   in the order it lies. Lane order also where the depth is at least
   LANE_DEPTH products per output, as a weight gradient's few outputs over
   many tokens are: chain order's threads take tiles of outputs, too few
   to share the depth, where lane order's take ranges of its blocks. On
   the Core Ultra 5 235, 8 × 8 outputs over 16384 products took 84-125 µs
   in chain order against 42-57 in lane, 16 × 16 over 65536 663-873
   against 272-293; 64 × 64 over 262144 14.1 ms against 15.0-16.5, and
   8 × 8 over 4096 21-25 µs against 44-46. */
#define CHAIN_OUTPUTS 64
#define LANE_MOST 4
#define LANE_DEPTH 128

static int is_integer(int dt) {
  int k = nx_dtype_row_of(dt).kind;
  return k == NX_KIND_SIGNED || k == NX_KIND_UNSIGNED;
}

int nx_gemm_order_of(int acc, const double *x) {
  double m = x[ROW], n = x[COL];
  if (is_integer(acc)) return NX_CPU_CHAIN;
  if (m > LANE_MOST && n > LANE_MOST && m * n >= CHAIN_OUTPUTS &&
      x[CON] < LANE_DEPTH * m * n)
    return NX_CPU_CHAIN;
  return NX_CPU_LANE;
}

/* The external: groups the view of the pairs [vbatch] and [vcontracting]
   on the stack, or their runs where a group makes several, answering what
   the fill answers where it does not fit, then reads the operands through
   the door. [vacc] is acc's dtype code, one
   Nx_kernel.accumulates takes, float where a scale is given, [vi] the
   init and [vsa] and [vsb] a's and b's scales, options of
   Nx_kernel.scale. Its body is inlined into two copies, [scaled] a
   constant in each, so that an unscaled call runs none of the scales'
   setup. */
static inline __attribute__((always_inline)) value contract_body(
    value vbatch, value vcontracting, value vacc, value vdst, value vi,
    value va, value vb, value vsa, value vsb, int scaled) {
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
  value vs[2] = {vsa, vsb};
  if (scaled && !nx_gemm_scales_fit(va, vb, vs)) return Val_int(NX_SHAPE);
  /* Read before nx_read, which may run the collector. */
  scale_axes sx[2];
  for (int o = A; scaled && o <= B; o++)
    if (Is_block(vs[o]))
      nx_gemm_axes_of(Field(vs[o], 0), o == A ? va : vb, vbatch, vcontracting,
                      o, &sx[o]);
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
  double x[4];
  for (int g = BATCH; g <= CON; g++) {
    x[g] = runs ? 1 : (double)v.extent[g];
    for (int k = 0; runs && k < r.runs[g]; k++)
      x[g] *= (double)r.extent[g][k];
  }
  int order = nx_gemm_order_of(acc, x), done;
  double cost = nx_gemm_call_cost(acc, order, x);
  for (int o = A; scaled && o <= B; o++)
    if (at[o] >= 0) cost += nx_gemm_decode_cost(acc, &a[at[o]]);
  int released = nx_cpu_release(cost);
  scale s[2];
  uint8_t *mem[2] = {NULL, NULL};
  done = 1;
  if (scaled) memset(s, 0, sizeof s);
  for (int o = A; scaled && o <= B && done; o++)
    if (at[o] >= 0)
      done = nx_gemm_scale_of(acc, op[o], &a[at[o]], &sx[o], o, &s[o], &mem[o]);
  if (done)
    done = computed(acc, order, out, dts, runs, &v, &r, op, scaled ? s : NULL);
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
    value vbatch, value vcontracting, value vacc, value vdst, value vi,
    value va, value vb, value vsa, value vsb) {
  return contract_body(vbatch, vcontracting, vacc, vdst, vi, va, vb, vsa,
                       vsb, 1);
}

value nx_cpu_contract(value vbatch, value vcontracting, value vacc,
                      value vdst, value vi, value va, value vb, value vsa,
                      value vsb, value vseg) {
  if (Is_block(vseg))
    return nx_gemm_contract_segmented(vbatch, vcontracting, vacc, vdst, vi,
                                      va, vb, vsa, vsb, vseg);
  if (Is_block(vsa) || Is_block(vsb))
    return contract_scaled(vbatch, vcontracting, vacc, vdst, vi, va, vb, vsa,
                           vsb);
  return contract_body(vbatch, vcontracting, vacc, vdst, vi, va, vb, vsa, vsb,
                       0);
}

value nx_cpu_contract_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_contract(argv[0], argv[1], argv[2], argv[3], argv[4],
                         argv[5], argv[6], argv[7], argv[8], argv[9]);
}
