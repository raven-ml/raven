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

   - Chain order: the products fall into blocks of NX_CPU_CHAIN_TERMS
     consecutive k (cpu.h), each block a fused multiply-add chain from +0
     in increasing k; the first block's sum starts R, each later one's is
     added into R in increasing k, and then init is added. The microkernels
     of every target add each product fused, a call of one a block, and
     the blocking stores and reloads R in acc exactly, so tile shapes,
     block sizes and threads change no bit, and every target sums in one
     association. An integer sum, exact in any order, chains along all of
     k from init or +0. A chain runs at one fused add per latency: many
     rows and columns keep every target's pipes full.
   - Lane order: the products fall into blocks of FOLD_BLOCK consecutive k
     and, within a block, into LANES lanes by k modulo LANES, each lane a
     fused chain from +0; a fixed balanced tree sums the lanes, the
     left-complete binary tree the blocks, then init is added. A product
     of few rows, as decoding a token is, reads each element of b once,
     where chain order's tiles would pack it.

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
   transpose where that streams, unless a is read in place, or the
   transpose would write R's tiles apart (writes_apart), as BLIS fixes a
   product's orientation by C's storage. Where a is read in place, it is
   read where it lies, as BLIS's small products read it (bli_gemmsup), in
   chain order's blocks of rows, which give the threads units where b's
   few slivers would not. Where a table names other kernels, a product runs on
   those that take less time for its rows, or on one thread where it runs
   alone (fastest): on the M1, the cores' vector units take products of
   few rows, and small ones, from its matrix unit.

   R, the sum in acc, is dst itself where dst is of acc's dtype, and
   otherwise scratch in acc that a cast walk rounds into dst once every
   product is in (gemm_rounded.c). A float tile's first block writes it
   from +0, and its last adds init. An init that is dst itself is R's
   start, each block added into it: one addition of init where the
   products are one block, and a contraction of more is computed into
   scratch (gemm.h's computed).

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

/* Lane order's kernels' speed (nx_cpu_micro): a flop in the time memcpy
   moves a byte, the speed of the cores' float32 vector kernels on a
   performance core of fudge (gemm_avx2.c, 2026-10-13), taken for lane
   order's own kernels in every dtype since 2026-10-14 and not measured
   for them. */
#define LANE_SPEED 1.0

/* The slowest kernels' speed: gemm_generic.c's, each fused multiply-add a
   call of the C library's fma, on a performance core of fudge
   (2026-10-13). */
#define SLOWEST_SPEED 0.01

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

/* Row [r] of a staged block in acc's type [T], [n0] elements at dst,
   each times its scale: the scales at s->v from [at], element u's at
   t0[u], or a run's, along k from c[CON], where [runs]. */
#define SCALE_ROW(T)                                                      \
  do {                                                                    \
    T *d = (T *)dst + r * pitch;                                          \
    const T *v = (const T *)s->v + at;                                    \
    if (!runs) {                                                          \
      for (int64_t u = 0; u < n0; u++) d[u] *= v[t0[u]];                  \
      break;                                                              \
    }                                                                     \
    for (int64_t u = 0, m; u < n0; u += m) {                              \
      m = min64(n0 - u, s->run - (c[CON] + u) % s->run);                  \
      T sv = v[t0[u]];                                                    \
      for (int64_t l = 0; l < m; l++) d[u + l] *= sv;                     \
    }                                                                     \
  } while (0)

/* Stages operand [o]'s block as stage_acc does, from its element
   (e, i, j, k), [n0] along axis [x0] in a row and [n1] rows along [x1]:
   along an axis of several runs, piece by piece as nx_gemm_pack_at reads it, so
   that the operand is read where it lies; otherwise in one stage. Each
   element of a scaled operand is then times its block's scale, rounded
   once in acc, a float32 or float64: the scales' tables index the
   operand's elements whatever runs its axes make, so a Q8_0 weight, its
   blocks' quants between their scales, is read where it lies. */
static void stage_at(const problem *p, int o, int64_t e, int64_t i,
                     int64_t j, int64_t k, int x0, int x1, int64_t n0,
                     int64_t n1, uint8_t *dst, int64_t pitch) {
  const operand *q = &p->op[o];
  const int64_t c[4] = {e, i, j, k};
  if (q->sp[x0] || q->sp[x1])
    nx_gemm_pack_at(p, NULL, o, c, x0, x1, n0, n1, dst, pitch);
  else if (q->s && stage_f4(p, o, c, x0, x1, n0, n1, dst, pitch))
    return;
  else
    stage_acc(p, o, pos(p, o, e, i, j, k), q->st[x0], q->st[x1], n0, n1,
              dst, pitch);
  if (q->s == NULL) return;
  const scale *s = q->s;
  const int64_t *t0 = s->t[x0] + c[x0];
  /* Along k, a run of s->run elements takes one scale: each run's is
     read once, and its elements multiplied by it in a loop the compiler
     vectorises, where a table lookup an element costs more than the
     stage. */
  int runs = x0 == CON && s->run > 1;
  for (int64_t r = 0; r < n1; r++) {
    int64_t at = 0;
    for (int x = BATCH; x <= CON; x++)
      if (x != x0 && s->t[x]) at += s->t[x][c[x] + (x == x1 ? r : 0)];
    if (p->acc == NX_FLOAT32) SCALE_ROW(float);
    else SCALE_ROW(double);
  }
}

/* Whether [p] sums in chain order's blocks: a float acc. */
static int blocked(const problem *p) {
  return p->acc == NX_FLOAT32 || p->acc == NX_FLOAT64;
}

/* Sets R's rows [i0, i1) × columns [j0, j1) of element [e] to init, or
   to +0 where k is empty: with k, the kernels start the first block from
   +0 themselves, and a float sum adds init after the last (finish). An
   init that is dst itself is already there: copying it onto itself would
   be a memcpy whose source and destination overlap. */
static void start(const problem *p, int64_t e, int64_t i0, int64_t i1,
                  int64_t j0, int64_t j1) {
  const int64_t *s = p->op[DST].st;
  if (p->init_is_dst ||
      (p->ext[CON] > 0 && (blocked(p) || p->op[INIT].x == NULL)))
    return;
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

/* Where an integer sum's tiles start in the block of k from [pc]. */
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

/* The kernel's sums of a's rows [i, i + m) and b's columns [j, j + n) of
   element [e] over [kc] steps of k from [a] and [b], started as [f] says:
   in place where R's columns are adjacent, else through a buffer. */
static void put(const problem *p, nx_cpu_micro k, nx_cpu_from f, int64_t kc,
                src a, src b, int64_t e, int64_t i, int64_t j, int64_t m,
                int64_t n) {
  const int64_t *s = p->op[DST].st;
  if (s[COL] == 1 || n == 1)
    k.f(m, n, kc, a.at, a.row, a.k, b.at, b.k, at_r(p, e, i, j), s[ROW], f);
  else tile_apart(p, k, f, kc, a, b, e, i, j, m, n);
}

/* Adds the [n] sums at [t] into R's outputs at [r], [step] elements
   apart. */
static void add_row(const problem *p, uint8_t *r, int64_t step,
                    const uint8_t *t, int64_t n) {
  if (p->acc == NX_FLOAT32) {
    float *y = (float *)r;
    const float *x = (const float *)t;
    for (int64_t j = 0; j < n; j++) y[j * step] += x[j];
  } else {
    double *y = (double *)r;
    const double *x = (const double *)t;
    for (int64_t j = 0; j < n; j++) y[j * step] += x[j];
  }
}

/* Adds a block's sums into R: the kernel computes them from +0 into a
   buffer, MR rows at a time, each row then added into R's. Apart from
   tile, as tile_apart is, for its buffer. */
static __attribute__((noinline)) void joined(const problem *p,
                                             nx_cpu_micro k, int64_t kc,
                                             src a, src b, int64_t e,
                                             int64_t i, int64_t j, int64_t m,
                                             int64_t n) {
  _Alignas(ALIGN) uint8_t t[NX_CPU_TILE];
  int w = p->w;
  for (int64_t r0 = 0; r0 < m; r0 += k.mr) {
    int64_t rows = min64(k.mr, m - r0);
    k.f(rows, n, kc, a.at + r0 * a.row * w, a.row, a.k, b.at, b.k, t, k.nr,
        NX_CPU_FROM_ZERO);
    for (int64_t r = 0; r < rows; r++)
      add_row(p, at_r(p, e, i + r0 + r, j), p->op[DST].st[COL],
              t + r * k.nr * w, n);
  }
}

/* Adds init into R's rows [i, i + m) × columns [j, j + n) of element [e],
   n at most a kernel's NR, each row staged into acc. */
static __attribute__((noinline)) void finish(const problem *p, int64_t e,
                                             int64_t i, int64_t j, int64_t m,
                                             int64_t n) {
  _Alignas(ALIGN) uint8_t t[NX_CPU_TILE];
  const int64_t *s = p->op[INIT].st;
  for (int64_t r = i; r < i + m; r++) {
    stage_acc(p, INIT, pos(p, INIT, e, r, j, 0), s[COL], 0, n, 1, t, 0);
    add_row(p, at_r(p, e, r, j), p->op[DST].st[COL], t, n);
  }
}

/* Adds the products of a's rows [i, i + m) and b's columns [j, j + n) of
   element [e], the kernel reading them from [a] and [b], over the block of
   k from [pc] to R with [k]'s kernel. A float sum takes each block of
   NX_CPU_CHAIN_TERMS (cpu.h) in a call of its own from +0, the first
   writing R and each later one added into it, then adds init after the
   last block of k. */
static void tile(const problem *p, nx_cpu_micro k, int64_t pc, int64_t kc,
                 src a, src b, int64_t e, int64_t i, int64_t j, int64_t m,
                 int64_t n) {
  if (!blocked(p)) {
    put(p, k, from(p, pc), kc, a, b, e, i, j, m, n);
    return;
  }
  int w = p->w;
  for (int64_t q = 0; q < kc; q += NX_CPU_CHAIN_TERMS) {
    int64_t len = min64(NX_CPU_CHAIN_TERMS, kc - q);
    src aq = {a.at + q * a.k * w, a.row, a.k};
    src bq = {b.at + q * b.k * w, b.row, b.k};
    if (pc + q == 0 && !p->init_is_dst)
      put(p, k, NX_CPU_FROM_ZERO, len, aq, bq, e, i, j, m, n);
    else joined(p, k, len, aq, bq, e, i, j, m, n);
  }
  if (pc + kc == p->ext[CON] && p->op[INIT].x && !p->init_is_dst)
    finish(p, e, i, j, m, n);
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

static void job(const problem *p, int64_t total, int64_t bytes, double cost,
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

/* [p]'s cost on [g]'s kernels (nx_gemm_cost), its rows padded to their
   tile, and the bytes its operands touch into [bytes]. */
static double problem_cost(const problem *p, const nx_cpu_gemm *g,
                           double *bytes) {
  double e = (double)p->ext[BATCH], m = (double)p->ext[ROW];
  double n = (double)p->ext[COL], k = (double)p->ext[CON];
  *bytes = e * (m * k + k * n + m * n) * p->w;
  return nx_gemm_cost(g->kernel.flops_per_byte, e,
                      (double)padded(g->kernel, p->ext[ROW]), n, k, p->w);
}

/* Whether [p] runs alone on [g]'s kernels: on the calling thread, where
   its whole cost buys no second thread, as BLIS's small products take one
   thread below their thresholds (bli_gemmsup). Its jobs then are calls,
   which run in one state of g's hardware, made once: on the M1 Max a 64³
   float32 product took 0.94 µs as two jobs, each in its own state of the
   matrix unit, 0.91 alone. */
static int runs_alone(const problem *p, const nx_cpu_gemm *g) {
  double bytes, cost = problem_cost(p, g, &bytes);
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
  double bytes;
  double ta = problem_cost(p, g, &bytes) + g->call;
  double tb = problem_cost(p, o, &bytes) + o->call;
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
  double cost = nx_gemm_cost(km.flops_per_byte, batch, m, n, k, w);
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
  double pack_cost = nx_gemm_part_cost(batch, m, k, w);
  r.move_a = mover(p, batch * r.kblocks, packed, pack_cost);
  r.move_b = mover(p, total, bytes, cost);
  if (!p->a_in)
    job(p, batch * r.kblocks, packed, pack_cost, few_rows_pack, &r);
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
  double speed = g->kernel.flops_per_byte;
  double cost = nx_gemm_cost(speed, batch, m, nc, kc_most, w);
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
      cost = nx_gemm_cost(speed, ne, m, c.nc, kc_most, w);
      split(&c, units(packed, cost, thread_bytes(g)));
      int64_t total = ne * c.ranges * mblocks, pc = 0;
      do {
        c.pc = pc;
        c.kc = min64(g->kc, k - pc);
        int64_t a_packed = c.a ? ne * m * c.kc * w : 0;
        double a_cost = c.a ? nx_gemm_part_cost(ne, m, c.kc, w) : 0;
        packed = ne * c.slivers * nr * c.kc * w + a_packed;
        /* The products' set-up counts in their first panel's first block
           of k. */
        cost = nx_gemm_work_cost(speed, ne, m, c.nc, c.kc, w) +
               (jc == 0 && pc == 0 ? ne * PRODUCT_COST : 0);
        c.move_a = mover(p, ne * mblocks, a_packed, a_cost);
        c.move_b = mover(p, total, packed, cost);
        if (c.kc > 0 && c.a)
          job(p, ne * mblocks, a_packed, a_cost, pack_blocks, &c);
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
   LANE_ROWS a pass. b, the large operand, is read once a pass. A unit is
   (element, groups of LANE_COLUMNS columns, range of blocks), and each
   block's sums of the unit's outputs come from one of two kinds:

   - Dots, where b is adjacent along k or along each run of it, as a
     weight stored [n × k] is, attention's keys, or a convolution's
     windows: each output's block a dot along k, up to DOT_COLUMNS columns
     of b at once against a row of a, summed in registers. On fudge one
     thread took a float32 token's scores against 4096 keys of 64 in 5.4
     instructions a fused vector, where a column of b against each row of
     a, its lanes stored and summed one at a time, took 75.
   - Tiles, otherwise, as for a weight stored [k × n] or attention's
     values: each block of the pass's rows against the unit's columns a
     lane tile, which holds each lane of a vector of columns of each row
     in registers across the block's rows of b. b is read in place where
     it is acc and adjacent along its columns, else staged TILE_STAGE
     bytes at a time. On fudge one thread took a float32 token's product
     with 4096 values of 64, b's rows added into 16 lanes of accumulators
     in memory, in 25 instructions a fused vector, MKL's gemv in 3.7; by
     tiles in 1.8.

   A product of few outputs over a long k splits its blocks into ranges,
   whose sums meet in a second job; otherwise a unit takes every block and
   totals its outputs itself, or, of one block and no init, writes them
   to R. */

/* The blocks' tree of [c] outputs at once, into the first of the [n]
   rows of their block sums at [s], [ls] apart. */
#define TREE(T, name)                                                        \
  static void name(T *s, int64_t n, int64_t ls, int64_t c) {                 \
    if (n < 2) return;                                                       \
    int64_t h = 1;                                                           \
    while (2 * h < n) h *= 2;                                                \
    name(s, h, ls, c);                                                       \
    name(s + h * ls, n - h, ls, c);                                          \
    for (int64_t j = 0; j < c; j++) s[j] += s[h * ls + j];                   \
  }
TREE(float, tree_f32)
TREE(double, tree_f64)

/* [d] = init (if any) + the sum at [s]; with no sum, [s] NULL, init as it
   is, or +0. An init in acc is read in place: the stage costs more than
   the sum of a small product's output. */
static void total(const problem *p, int64_t e, int64_t i, int64_t j,
                  const uint8_t *s, uint8_t *d) {
  _Alignas(16) uint8_t x[16] = {0};
  const nx_array *init = p->op[INIT].x;
  int64_t at = init ? pos(p, INIT, e, i, j, 0) : 0;
  if (init && init->dtype == p->acc)
    memcpy(x, init->base + at * p->w, (size_t)p->w);
  else if (init) stage_acc(p, INIT, at, 1, 0, 1, 1, x, 0);
  if (s == NULL || init == NULL) {
    memcpy(d, s ? s : x, (size_t)p->w);
    return;
  }
  if (p->acc == NX_FLOAT32) *(float *)d = *(float *)x + *(const float *)s;
  else *(double *)d = *(double *)x + *(const double *)s;
}

/* Each output with no products: init as it is, or +0. A unit is an
   element. */
static void empty(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const problem *p = ctx;
  for (int64_t e = lo; e < hi; e++)
    for (int64_t i = 0; i < p->ext[ROW]; i++)
      for (int64_t j = 0; j < p->ext[COL]; j++)
        total(p, e, i, j, NULL, at_r(p, e, i, j));
}

/* a, read in place or staged whole: element e's row i, element t at
   a + (e·batch + i·row + t·k)·w. */
typedef struct {
  problem *p;
  uint8_t *a, *staged;
  int64_t batch, row, k;
} rows_of_a;

/* A unit is an element: its rows of a staged whole, adjacent along k. */
static void stage_a(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const rows_of_a *r = ctx;
  const problem *p = r->p;
  int64_t m = p->ext[ROW], k = p->ext[CON];
  for (int64_t e = lo; e < hi; e++)
    stage_at(p, A, e, 0, 0, 0, CON, ROW, k, m, r->a + e * r->batch * p->w,
             k);
}

/* [p]'s a in place where [in_place], else staged whole: 0 if the stage
   could not be allocated. */
static int rows_a(problem *p, int in_place, rows_of_a *r) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], k = p->ext[CON], w = p->w;
  *r = (rows_of_a){.p = p};
  if (in_place) {
    const int64_t *s = p->op[A].st;
    r->a = p->op[A].x->base + p->op[A].first * w;
    r->batch = s[BATCH], r->row = s[ROW], r->k = s[CON];
    return 1;
  }
  r->staged = r->a = alloc(batch * m * k * w);
  if (r->a == NULL) {
    atomic_store(&p->failed, 1);
    return 0;
  }
  r->batch = m * k, r->row = k, r->k = 1;
  nx_cpu_job(batch, batch * m * k * w, nx_gemm_part_cost(batch, m, k, w),
             stage_a, r);
  return 1;
}

/* Units */

/* A unit takes groups of LANE_COLUMNS columns and a range of blocks; its
   own block sums hold at most LANE_SUMS bytes; a tile's staged block of
   b at most TILE_STAGE, rows of 512 bytes, which the tile does not
   prefetch (gemm_lanes.h). */
#define LANE_COLUMNS 16
#define LANE_SUMS (64 * 1024)
#define TILE_STAGE (512 * 1024)

typedef struct lane_job lane_job;

/* A kind of block sums: element [e]'s columns [j0, j0 + c), blocks [x0,
   x1), block x's output (i, j) to y + (x·ys + i·ldy + j·ldc)·w, its
   stage of b at [bp]. */
typedef void (*lane_sums)(const lane_job *f, uint8_t *bp, int64_t e,
                          int64_t j0, int64_t c, int64_t x0, int64_t x1,
                          uint8_t *y, int64_t ldy, int64_t ldc, int64_t ys);

struct lane_job {
  problem *p;
  int64_t blocks, groups;    /* of k, and of LANE_COLUMNS columns */
  int64_t span, units;       /* groups a unit takes, units an element */
  int64_t ranges, per_range; /* ranges of blocks a unit takes */
  int64_t cols;              /* columns of b a dot or a stage takes */
  int64_t own;               /* columns a unit's sums hold */
  int64_t stage;             /* bytes of a worker's stage of b */
  lane_sums sums_of;
  nx_cpu_dot dot;            /* acc's */
  nx_cpu_lane_tile tile;     /* acc's */
  int in_place;              /* b read where it lies */
  int whole;                 /* an output's one block is its sum */
  rows_of_a a;
  uint8_t *sums; /* block x of output (i, j) of element e at
                    (((e·blocks + x)·M + i)·N + j)·w, with ranges > 1 */
};

/* Dots: f->cols columns of b at a time, each block's dot against each
   row of a: a product of one row keeps as many chains in flight as one
   of many, and loads each step of a's row once for the columns, which
   stay in L1 across a's rows. A product's bits do not depend on which
   factor is which. In place, a dot's columns are ones a step apart;
   staged, they are staged into [bp]. */
static void dots_of(const lane_job *f, uint8_t *bp, int64_t e, int64_t j0,
                    int64_t c, int64_t x0, int64_t x1, uint8_t *y,
                    int64_t ldy, int64_t ldc, int64_t ys) {
  const problem *p = f->p;
  const operand *ob = &p->op[B];
  int64_t m = p->ext[ROW], k = p->ext[CON], w = p->w;
  const uint8_t *a = f->a.a + e * f->a.batch * w;
  int64_t nc;
  for (int64_t j = 0; j < c; j += nc) {
    int64_t ldb = 0;
    if (f->in_place) nc = piece(ob, COL, j0 + j, min64(f->cols, c - j), &ldb);
    else nc = min64(f->cols, c - j);
    for (int64_t x = x0; x < x1; x++) {
      int64_t k0 = x * NX_CPU_LANE_BLOCK;
      int64_t len = min64(NX_CPU_LANE_BLOCK, k - k0);
      const uint8_t *b = bp;
      if (f->in_place) b = ob->x->base + pos(p, B, e, 0, j0 + j, k0) * w;
      else {
        stage_at(p, B, e, 0, j0 + j, k0, CON, COL, len, nc, bp, len);
        ldb = len;
      }
      for (int64_t i = 0; i < m; i++)
        f->dot(b, ldb, (int)nc, a + (i * f->a.row + k0) * w, len,
               y + (x * ys + i * ldy + j * ldc) * w, ldc);
    }
  }
}

/* Tiles: each block of the pass's rows against the columns, b in place,
   or staged f->cols columns at a time into [bp]. In place, every block
   of the columns in turn: a block's rows of a weight lie a page apart,
   and a column of tiles stays within one block's pages. */
static void tiles_of(const lane_job *f, uint8_t *bp, int64_t e, int64_t j0,
                     int64_t c, int64_t x0, int64_t x1, uint8_t *y,
                     int64_t ldy, int64_t ldc, int64_t ys) {
  const problem *p = f->p;
  int64_t m = p->ext[ROW], k = p->ext[CON], w = p->w;
  const uint8_t *a = f->a.a + e * f->a.batch * w;
  (void)ldc;
  int64_t nc;
  for (int64_t j = 0; j < c; j += nc) {
    nc = f->in_place ? c - j : min64(f->cols, c - j);
    for (int64_t x = x0; x < x1; x++) {
      int64_t k0 = x * NX_CPU_LANE_BLOCK;
      int64_t len = min64(NX_CPU_LANE_BLOCK, k - k0);
      const uint8_t *b = bp;
      int64_t ldb = nc;
      if (f->in_place) {
        b = p->op[B].x->base + pos(p, B, e, 0, j0 + j, k0) * w;
        ldb = p->op[B].st[CON];
      } else {
        stage_at(p, B, e, 0, j0 + j, k0, COL, CON, nc, len, bp, nc);
      }
      f->tile(a + k0 * f->a.k * w, f->a.row, f->a.k, m, b, ldb, len, nc,
              y + (x * ys + j) * w, ldy);
    }
  }
}

/* Element [e]'s outputs of the pass's rows and columns [j0, j0 + c)
   from their block sums at [s], output (i, j0 + j)'s block x at s +
   (x·ys + i·ldy + j)·w: the blocks' tree a row at a time, then init. */
static void totals(const lane_job *f, int64_t e, int64_t j0, int64_t c,
                   uint8_t *s, int64_t ldy, int64_t ys) {
  const problem *p = f->p;
  const operand *d = &p->op[DST];
  int64_t w = p->w;
  int copies = p->op[INIT].x == NULL && d->st[COL] == 1 && d->sp[COL] == NULL;
  for (int64_t i = 0; i < p->ext[ROW]; i++) {
    uint8_t *r = s + i * ldy * w;
    if (p->acc == NX_FLOAT32) tree_f32((float *)r, f->blocks, ys, c);
    else tree_f64((double *)r, f->blocks, ys, c);
    if (copies) {
      memcpy(at_r(p, e, i, j0), r, (size_t)(c * w));
      continue;
    }
    for (int64_t j = 0; j < c; j++)
      total(p, e, i, j0 + j, r + j * w, at_r(p, e, i, j0 + j));
  }
}

/* A unit is f->span groups of an element's columns and a range of
   blocks. An output of one block and no init is that block's sum: the
   unit writes R where its columns are one step apart (adjacent, for
   tiles), and its rows too. Otherwise the block sums go to the job's, or
   to the unit's own f->own columns at a time, then to their totals. */
static void lane_unit(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const lane_job *f = ctx;
  problem *p = f->p;
  const operand *d = &p->op[DST];
  int64_t m = p->ext[ROW], n = p->ext[COL], w = p->w;
  int64_t own = f->ranges == 1 ? f->blocks * m * f->own * w : 0;
  uint8_t *bp = scratch(p, f->stage + own);
  if (bp == NULL) return;
  uint8_t *mine = bp + f->stage;
  for (int64_t u = lo; u < hi; u++) {
    int64_t r = u % f->ranges, v = u / f->ranges, e = v / f->units;
    int64_t j0 = v % f->units * f->span * LANE_COLUMNS;
    int64_t c = min64(f->span * LANE_COLUMNS, n - j0);
    int64_t x0 = r * f->per_range, x1 = min64(f->blocks, x0 + f->per_range);
    int64_t ldr, steps = piece(d, COL, j0, c, &ldr);
    if (f->whole && steps == c && d->sp[ROW] == NULL &&
        (f->dot || ldr == 1)) {
      f->sums_of(f, bp, e, j0, c, 0, 1, at_r(p, e, 0, j0), d->st[ROW], ldr,
                 0);
      continue;
    }
    if (f->ranges > 1) {
      uint8_t *y = f->sums + (e * f->blocks * m * n + j0) * w;
      f->sums_of(f, bp, e, j0, c, x0, x1, y, n, 1, m * n);
      continue;
    }
    for (int64_t j = j0; j < j0 + c; j += f->own) {
      int64_t cj = min64(f->own, j0 + c - j);
      f->sums_of(f, bp, e, j, cj, 0, f->blocks, mine, cj, 1, m * cj);
      totals(f, e, j, cj, mine, cj, m * cj);
    }
  }
  nx_cpu_give(bp);
}

/* A unit is an element: each output's total of its blocks. */
static void lane_finish(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const lane_job *f = ctx;
  const problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL];
  for (int64_t e = lo; e < hi; e++)
    totals(f, e, 0, n, f->sums + e * f->blocks * m * n * p->w, n, m * n);
}

/* [f]'s units, then a second job that totals their block sums where the
   blocks split into ranges: groups of columns first, then ranges of
   blocks, until each thread has its units. */
static void lane_job_run(lane_job *f, int64_t bytes, double cost) {
  problem *p = f->p;
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t w = p->w, outputs = batch * m * n;
  f->blocks = ceil_div(p->ext[CON], NX_CPU_LANE_BLOCK);
  f->groups = ceil_div(n, LANE_COLUMNS);
  f->whole = f->blocks == 1 && p->op[INIT].x == NULL;
  int64_t want = units(bytes, cost, NX_CPU_THREAD_BYTES);
  int64_t per = ceil_div(want, batch);
  f->span = f->groups > per ? f->groups / per : 1;
  f->units = ceil_div(f->groups, f->span);
  int64_t outer = batch * f->units;
  f->per_range = outer < want ? ceil_div(f->blocks, ceil_div(want, outer))
                              : f->blocks;
  f->ranges = ceil_div(f->blocks, f->per_range);
  int64_t most = min64(f->span * LANE_COLUMNS,
                       LANE_SUMS / (f->blocks * m * w));
  f->own = most < LANE_COLUMNS ? LANE_COLUMNS
                               : most / LANE_COLUMNS * LANE_COLUMNS;
  if (f->ranges > 1) f->sums = alloc(outputs * f->blocks * w);
  if (f->ranges > 1 && f->sums == NULL) atomic_store(&p->failed, 1);
  else nx_cpu_job(outer * f->ranges, bytes, cost, lane_unit, f);
  if (f->ranges > 1 && !atomic_load(&p->failed))
    nx_cpu_job(batch, outputs * f->blocks * w,
               nx_gemm_work_cost(LANE_SPEED, batch, m, n, f->blocks, w),
               lane_finish, f);
  nx_cpu_aligned_free(f->sums);
  nx_cpu_aligned_free(f->a.staged);
}

/* a is read in place where it is acc and adjacent along k, else staged
   whole first: it is the smaller operand. A dot takes the columns of
   NX_CPU_DOT_ROWS whole blocks' bytes, up to DOT_COLUMNS: a short block
   more, so that a call's and a stage's set-up are shared, as a token's
   scores against 64 elements of each key take 16 columns a call. */
static void dots(problem *p, int64_t bytes, double cost) {
  int64_t len = min64(p->ext[CON], NX_CPU_LANE_BLOCK);
  int64_t fit = NX_CPU_DOT_ROWS * (NX_CPU_LANE_BLOCK / len);
  lane_job f = {.p = p,
                .cols = min64(DOT_COLUMNS, fit),
                .stage = NX_CPU_DOT_ROWS * NX_CPU_LANE_BLOCK * p->w,
                .sums_of = dots_of,
                .dot = nx_cpu_table->dot[p->acc],
                .in_place = adjacent(p, B, CON)};
  if (rows_a(p, adjacent(p, A, CON), &f.a)) lane_job_run(&f, bytes, cost);
}

/* a is read in place where it is raw, else staged whole first. */
static void tiles(problem *p, int64_t bytes, double cost) {
  int64_t w = p->w;
  lane_job f = {.p = p,
                .tile = nx_cpu_table->tile[p->acc],
                .sums_of = tiles_of,
                .in_place = adjacent(p, B, COL)};
  if (!f.in_place) {
    f.cols = TILE_STAGE / (NX_CPU_LANE_BLOCK * w);
    f.stage = NX_CPU_LANE_BLOCK * f.cols * w;
  }
  if (rows_a(p, raw(p, A), &f.a)) lane_job_run(&f, bytes, cost);
}

/* [p]'s cost in lane order (nx_gemm_cost), its operands' [bytes] set. */
static double lane_cost(const problem *p, int64_t *bytes) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t k = p->ext[CON];
  *bytes = batch * (m + n) * k * p->w;
  return nx_gemm_cost(LANE_SPEED, batch, m, n, k, p->w);
}

/* Whether [p] takes dots: b steps one along k, within runs of at least
   NX_CPU_LANES where k is several runs. Dots stage a run of b a column
   at a time, tiles a block of k across the columns: on one thread of
   fudge, Nx.conv's 3 × 3 convolution of an [8; 112; 112; 3] image into 3
   channels, runs of 9, took 46.4M instructions in dots against 38.3M in
   tiles; of an [8; 56; 56; 64] one, runs of 192, 35.2M in dots against
   48.3M in tiles. */
static int dotted(const problem *p) {
  const spans *s = p->op[B].sp[CON];
  if (step_of(&p->op[B], CON) != 1) return 0;
  return s == NULL || s->extent[s->n - 1] >= NX_CPU_LANES;
}

static void lane_pass(problem *p) {
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL];
  int64_t w = p->w, bytes;
  if (p->ext[CON] == 0) {
    nx_cpu_job(batch, batch * m * n * w,
               nx_gemm_cost(LANE_SPEED, batch, m, n, 0, w), empty, p);
    return;
  }
  double cost = lane_cost(p, &bytes);
  if (dotted(p)) dots(p, bytes, cost);
  else tiles(p, bytes, cost);
}

/* Passes of LANE_ROWS rows: each moves a, init and dst on by its rows.
   The call's entry prices the whole product (nx_gemm_call_cost), so that it
   releases the domain lock though each pass's jobs cost less: on one
   core a 2000 × 2000 × 16 float32 product held it for 45 ms, in 250
   passes of 0.6 MiB each. */
static void lanes(problem *p) {
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

/* Depth one */

/* A product of a depth of one and no init in chain order: R's rows, each
   the axpy of its row's element of a and b's row, fused from +0, along R's
   adjacent axis (the product transposed where that is its rows). a and b
   are read in place where they are raw and, b, adjacent, else staged a
   band and a row at a time; a unit is a band of an element's rows. On
   fudge one thread took 8 heads' products of 4096 by 64 outputs, as
   attention's backward through a decoded token's keys and values makes
   them, through chain order's kernel with R's columns apart in 15
   instructions an output, MKL's in 1.1; by axpys in 0.9. */
typedef struct {
  problem *p;
  int64_t rows, bands; /* a band's rows, an element's bands */
  nx_cpu_axpy axpy;
} outer_job;

static void outer_unit(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const outer_job *f = ctx;
  problem *p = f->p;
  int64_t m = p->ext[ROW], n = p->ext[COL], w = p->w;
  const operand *oa = &p->op[A], *ob = &p->op[B], *od = &p->op[DST];
  int a_in = raw(p, A), b_in = adjacent(p, B, COL);
  uint8_t *s = scratch(p, (b_in ? 0 : n) * w + (a_in ? 0 : f->rows) * w);
  if (s == NULL) return;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = u / f->bands, i0 = u % f->bands * f->rows;
    int64_t r = min64(f->rows, m - i0);
    const uint8_t *a = s + (b_in ? 0 : n) * w, *b = s;
    int64_t lda = 1;
    if (b_in) b = ob->x->base + pos(p, B, e, 0, 0, 0) * w;
    else stage_at(p, B, e, 0, 0, 0, COL, CON, n, 1, s, n);
    if (a_in) a = oa->x->base + pos(p, A, e, i0, 0, 0) * w, lda = oa->st[ROW];
    else stage_at(p, A, e, i0, 0, 0, ROW, CON, r, 1, (uint8_t *)a, r);
    if (od->sp[ROW] == NULL)
      f->axpy(a, lda, r, b, n, at_r(p, e, i0, 0), od->st[ROW],
              NX_CPU_FROM_ZERO);
    else
      for (int64_t i = 0; i < r; i++)
        f->axpy(a + i * lda * w, 1, 1, b, n, at_r(p, e, i0 + i, 0), 0,
                NX_CPU_FROM_ZERO);
  }
  nx_cpu_give(s);
}

/* Whether [p] runs as axpys: a float of a depth of one and no init, R
   adjacent along its columns or its rows, which it then transposes to its
   columns. With init, chain order's tiles add it after the product. */
static int outer(problem *p) {
  nx_cpu_axpy axpy = nx_cpu_table->axpy[p->acc];
  const operand *d = &p->op[DST];
  if (p->ext[CON] != 1 || axpy == NULL || p->op[INIT].x) return 0;
  int cols = d->st[COL] == 1 && d->sp[COL] == NULL;
  int rows = d->st[ROW] == 1 && d->sp[ROW] == NULL;
  if (!cols && !rows) return 0;
  if (!cols) nx_gemm_transpose(p);
  int64_t batch = p->ext[BATCH], m = p->ext[ROW], n = p->ext[COL], w = p->w;
  int64_t bytes = batch * (m + n + m * n) * w;
  double cost = nx_gemm_cost(LANE_SPEED, batch, m, n, 1, w);
  int64_t want = units(bytes, cost, NX_CPU_THREAD_BYTES);
  int64_t bands = min64(ceil_div(want, batch), m);
  outer_job f = {.p = p, .rows = ceil_div(m, bands), .axpy = axpy};
  f.bands = ceil_div(m, f.rows);
  nx_cpu_job(batch * f.bands, bytes, cost, outer_unit, &f);
  return 1;
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

/* Whether [p]'s tiles write R through a buffer (tile_apart): where R's
   columns are apart. A narrow product's R, C-contiguous, has them
   adjacent; computed as its transpose, it would have them apart, and each
   tile of each KC block would be copied out of the buffer, and back in
   after the first: in the gradient of attention at 2048 keys, its
   products 64 columns wide took 343M instructions in those copies against
   3,658M in the kernels. */
static int writes_apart(const problem *p) {
  return p->op[DST].st[COL] != 1 && p->ext[COL] > 1;
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
  else if (!outer(&p)) {
    /* Kept in place, whether or not a stays in the caches: on the
       i9-9900K a 300 × 16 × 600 float64 product takes 91 µs on one core
       so against 166 transposed, a 256 × 64 × 256 one 24 µs on 16 threads
       against 33. */
    int narrow = p.ext[COL] < p.ext[ROW];
    p.keeps_a = narrow && in_place(&p, A);
    if (narrow && !p.keeps_a) {
      int apart = writes_apart(&p);
      nx_gemm_transpose(&p);
      if (!streams(&p) || writes_apart(&p) > apart) nx_gemm_transpose(&p);
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
  if (s) return nx_gemm_contract_with(g, acc, order, v, op, s, sp);
  const nx_cpu_gemm *quads = nx_gemm_quads_for(acc, da, db);
  if (quads) return nx_gemm_quads_on(quads, acc, v, op);
  if (part_of(acc) != acc) return nx_gemm_complex_on(acc, order, v, op, sp);
  return nx_gemm_contract_with(g, acc, order, v, op, NULL, sp);
}

/* Whether nx_gemm_contract_view reads a and b along spans: chain order's packs
   and lane order's stages do, scaled or not, and a complex contraction's
   parts of a and blocks of b, where no quads take it. A scaled contraction
   takes no quads. */
int nx_gemm_spanned(int acc, int da, int db, const scale *s) {
  return s != NULL || nx_gemm_quads_for(acc, da, db) == NULL;
}

/* The cost of [e] products of the extents [m], [n] and [k] in chain order
   on [r]'s kernels, of [w] bytes an element and [parts] flops a flop: the
   most of the table's kernels and their other ones, between which each
   problem chooses after the release (fastest), each kernel's rows padded
   to its tile, either way round, as a product may run as its
   transpose. */
static double chain_cost(int r, double parts, double e, double m, double n,
                         double k, double w) {
  const nx_cpu_gemm *g = &nx_cpu_table->gemm[r];
  const nx_cpu_gemm *kernels[2] = {g, g->other};
  double most = 0;
  for (int i = 0; i < 2 && kernels[i] != NULL; i++) {
    nx_cpu_micro km = kernels[i]->kernel;
    double speed = km.flops_per_byte / parts;
    double rows = (double)padded(km, (int64_t)m);
    double cols = (double)padded(km, (int64_t)n);
    double c = fmax(nx_gemm_cost(speed, e, rows, n, k, w),
                    nx_gemm_cost(speed, e, m, cols, k, w));
    most = fmax(most, c);
  }
  return most;
}

/* The cost (cpu.h, Calls and jobs) of a contraction in [acc] in [order]
   whose groups have the extents [x]: nx_gemm_cost in the dtype it sums in,
   at the slower kernels it may run on: lane order's (LANE_SPEED), or
   chain order's chain_cost; a complex flop at four of its parts'. Its
   operands' stages are gemm_complex.c's parts_of, gemm_quads.c's
   words_of, nx_gemm_stage_runs and nx_gemm_rounded_on. */
double nx_gemm_call_cost(int acc, int order, const double *x) {
  int sum = summed_in(acc), r = part_of(sum);
  double parts = r == sum ? 1 : 4, w = nx_cpu_width(sum);
  double e = x[BATCH], m = x[ROW], n = x[COL], k = x[CON];
  if (order == NX_CPU_LANE)
    return nx_gemm_cost(LANE_SPEED / parts, e, m, n, k, w);
  /* A call held at the slowest kernels' speed skips the lookup of its
     own. */
  double held = nx_gemm_cost(SLOWEST_SPEED / parts, e, m, n, k, w);
  if (held < NX_CPU_RELEASE_BYTES) return held;
  return chain_cost(r, parts, e, m, n, k, w);
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
    nx_cpu_aligned_free(mem[A]);
    nx_cpu_aligned_free(mem[B]);
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
