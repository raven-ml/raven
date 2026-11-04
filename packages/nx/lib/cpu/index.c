/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Gathers and scatters.

   A gather is a loop over its positions' shape of three operands: the
   destination, the positions, and the source read with step 0 along the
   gathered axis, whose stride is kept apart. Each element reads its
   position p, then the source at its own index plus p strides, or zero
   bits for a p outside the axis: one unsigned compare. Where the positions
   step 0 along a block's row, as a row take's or an embedding's do, the
   row is one copy from the source's row p.

   A scatter copies [into] to the destination, unless the two are one
   array, then applies the updates; a sorted scatter copies each bin's
   targets just before its updates. Update j's target has j's indices off
   the axis, so those indices cut the targets into disjoint slices, which
   any thread may take; within a slice the updates apply in increasing
   index along the axis, which is C order of the updates for each target.
   A unit is a run of at most UNIT_ROW slices along the innermost axis off
   the scattered one, of one where a narrow float sums positions that
   differ along it (Narrow sums), and walks the scattered axis once, the
   whole run at each step: where the positions step 0 along the run, as an
   embedding gradient's do, the step is one row of the kind. Where runs are
   fewer than the job's threads and the axis long, as a 1-d scatter's is,
   each run's updates are sorted into bins of their targets (Sorted
   scatters below), or each thread also takes a part of the targets along
   the axis and walks every update of its run, applying those that land in
   its part: the order per target stays C order. A float's Add of such
   runs, each one slice, sums each on one thread where its targets fit the
   cache (Sums in row order below). A combine whose result is the same in
   any order, an integer's or a real float's Max and Min, folds each
   thread's share of the updates into targets of its own instead, where
   they are few (Owned targets below).

   Add adds left to right from the target's element, x + u1 + ... + un,
   with no +0 before it, so that a -0 target and -0 updates sum to -0, as
   IEEE 754 sums them. A narrow float sums in float32 and rounds once
   (Narrow sums below), in the scratch nx_cpu.mli states: a few words per
   update along the axis and a bin's sums on each thread, or every target's
   sums where they carry from one sorted block to the next. Max and Min are
   nx_kinds.h's, so a NaN target or update makes the target a NaN. On
   narrow floats and complex numbers they select an operand by the order of
   prog.mli's Domains section, so that the result is an operand's bits. */

#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"
#include "nx_kinds.h"

/* Elements of a gather's block: a copy's. */
#define GATHER_MOST (64 * 1024)

/* Slices of a scatter's unit: a run of 1024 float32 targets is a 4 KiB
   row, and a unit walks the whole axis of updates per run. */
#define UNIT_ROW 1024

/* Updates along the axis from which threads split the targets into parts,
   each reading every update, or sort the updates into bins. One count
   serves every target size: float32 adds at random positions on fudge (Core
   Ultra 5 235, six P-cores, gcc 14, median of six interleaved rounds, load
   1.6-3.5), sorted against parts, in us:

     updates     into 1K targets     into 1M targets
     64K          42.6 /   44.4       59.6 /   51.5
     128K         81.4 /   88.9       96.9 /   88.7
     192K          121 /    132        136 /    127
     256K          154 /    176        174 /    168
     512K          300 /    365        339 /    327
     1M            637 /    726        689 /    656
     2M           1280 /   1515       1525 /   2100
     4M           2745 /   4620       3080 /   5000

   The sort wins into 1K targets at every count and into 1M from 2M
   updates; from 64K to 1M updates into 1M targets the parts run 3-13%
   faster. A second bound on the targets would win that back by a rule
   that follows each machine's caches. */
#define PART_UPDATES (64 * 1024)

/* Updates a sorted scatter sorts at once: about as many as the axis has
   targets, so that a target meets more than a few of them while its bin
   is in the cache, within these bounds. */
#define FEWEST_BLOCK (512 * 1024)
#define MOST_BLOCK (4 * 1024 * 1024)

/* At least two bins per thread, so that threads finish together; at most
   MOST_BINS. */
#define BINS_PER_THREAD 2
#define MOST_BINS 1024

/* Updates a chunk gathers per bin before it writes them out: writing each
   to its bin's run straight away ran several times slower, the bins' runs
   being more than the cache keeps open. */
#define GATHER 16

typedef struct {
  uint64_t lo, hi;
} u128;

/* Gathers */

typedef struct {
  const nx_array *a; /* the destination, the positions, the source */
  int64_t sa;        /* the source's stride along the axis */
  uint64_t dim;      /* its extent there */
  int w;             /* bytes per element, 0 for a sub-byte dtype */
} gather;

/* A row of [n] elements of the type [T]: element i of [d], stepping [sd],
   is the source [x] at i·sx plus its position's [sa] strides, its
   position at i·si of [idx]. */
#define GATHER_ROW(NAME, T)                                                  \
  static void NAME(uint8_t *d_, const int64_t *idx, const uint8_t *x_,      \
                   int64_t n, int64_t sd, int64_t si, int64_t sx,           \
                   int64_t sa, uint64_t dim) {                              \
    T *d = (T *)d_;                                                          \
    const T *x = (const T *)x_;                                              \
    const T zero = {0};                                                      \
    if (si == 0) {                                                           \
      uint64_t p = (uint64_t)idx[0];                                         \
      if (p >= dim) {                                                        \
        for (int64_t i = 0; i < n; i++) d[i * sd] = zero;                    \
        return;                                                              \
      }                                                                      \
      const T *r = x + (int64_t)p * sa;                                      \
      if (sd == 1 && sx == 1) memcpy(d, r, (size_t)n * sizeof(T));           \
      else                                                                   \
        for (int64_t i = 0; i < n; i++) d[i * sd] = r[i * sx];               \
      return;                                                                \
    }                                                                        \
    for (int64_t i = 0; i < n; i++) {                                        \
      uint64_t p = (uint64_t)idx[i * si];                                    \
      d[i * sd] = p < dim ? x[i * sx + (int64_t)p * sa] : zero;              \
    }                                                                        \
  }

GATHER_ROW(gather_1, uint8_t)
GATHER_ROW(gather_2, uint16_t)
GATHER_ROW(gather_4, uint32_t)
GATHER_ROW(gather_8, uint64_t)
GATHER_ROW(gather_16, u128)

/* Copies [m] rows of [n] bytes, the destination's from [d] stepping [sd]
   bytes, each from the source's row p at x + p sa, p read from [idx]
   stepping [si], [x] stepping [sx] bytes; zeros for a p outside [dim]. A
   constant [n] makes a row one load and one store of its size. */
static inline __attribute__((always_inline)) void copy_rows(
    uint8_t *d, const int64_t *idx, const uint8_t *x, int64_t m, int64_t sd,
    int64_t si, int64_t sx, int64_t sa, uint64_t dim, int64_t n) {
  for (int64_t j = 0; j < m; j++, d += sd, idx += si, x += sx) {
    uint64_t p = (uint64_t)*idx;
    if (p < dim) memcpy(d, x + (int64_t)p * sa, (size_t)n);
    else memset(d, 0, (size_t)n);
  }
}

/* A block of a row take, whose positions step 0 along its rows of
   contiguous elements in the destination and the source: a copy a row.
   Ragged rows packed into words take windows of 8 to 64 bytes, whose copy
   is a few loads and stores, where a call a row and its memcpy cost several
   times them. The row's size picks its loop once a run of rows: picked a
   row, with offsets recomputed a row, a row of 8 bytes took 28
   instructions, and takes 12 (fudge, 2026-11-04). */
static void take_rows(const nx_cpu_block *b, const gather *g, int across) {
  const nx_array *a = g->a;
  const int64_t *idx = (const int64_t *)a[1].base;
  /* A row lies along the block's inner axis, or, [across], along its rows:
     the walk puts an axis shorter than its SHORT outside a longer one, as
     the rows of a take of two int64 bounds a row lie. */
  const int64_t *between = across ? b->s0 : b->s1;
  int64_t w = g->w, n = (across ? b->n1 : b->n0) * w, sa = g->sa * w;
  int64_t sd = between[0] * w, si = between[1], sx = between[2] * w;
  int64_t m = across ? b->n0 : b->n1;
  for (int64_t q = 0; q < b->n2; q++) {
    uint8_t *d = a[0].base + (b->at[0] + q * b->s2[0]) * w;
    const int64_t *i = idx + b->at[1] + q * b->s2[1];
    const uint8_t *x = a[2].base + (b->at[2] + q * b->s2[2]) * w;
    switch (n) {
      case 4: copy_rows(d, i, x, m, sd, si, sx, sa, g->dim, 4); break;
      case 8: copy_rows(d, i, x, m, sd, si, sx, sa, g->dim, 8); break;
      case 16: copy_rows(d, i, x, m, sd, si, sx, sa, g->dim, 16); break;
      case 32: copy_rows(d, i, x, m, sd, si, sx, sa, g->dim, 32); break;
      case 64: copy_rows(d, i, x, m, sd, si, sx, sa, g->dim, 64); break;
      default: copy_rows(d, i, x, m, sd, si, sx, sa, g->dim, n);
    }
  }
}

static void gather_block(const nx_cpu_block *b, void *ctx) {
  const gather *g = ctx;
  const nx_array *a = g->a;
  const int64_t *idx = (const int64_t *)a[1].base;
  if (g->w > 0 && b->s0[1] == 0 && b->s0[0] == 1 && b->s0[2] == 1) {
    take_rows(b, g, 0);
    return;
  }
  if (g->w > 0 && b->n1 > 1 && b->s1[1] == 0 && b->s1[0] == 1 &&
      b->s1[2] == 1) {
    take_rows(b, g, 1);
    return;
  }
  for (int64_t q = 0; q < b->n2; q++)
    for (int64_t j = 0; j < b->n1; j++) {
      int64_t pd = b->at[0] + q * b->s2[0] + j * b->s1[0];
      int64_t pi = b->at[1] + q * b->s2[1] + j * b->s1[1];
      int64_t px = b->at[2] + q * b->s2[2] + j * b->s1[2];
      int64_t sd = b->s0[0], si = b->s0[1], sx = b->s0[2];
      if (g->w == 0) {
        int bits = a[0].bits;
        for (int64_t i = 0; i < b->n0; i++) {
          uint64_t p = (uint64_t)idx[pi + i * si];
          uint32_t v = p < g->dim ? nx_sub_load(a[2].base, bits,
                                                px + i * sx + (int64_t)p * g->sa)
                                  : 0;
          nx_sub_store(a[0].base, bits, pd + i * sd, v);
        }
        continue;
      }
      uint8_t *d = a[0].base + pd * g->w;
      const uint8_t *x = a[2].base + px * g->w;
      void (*row)(uint8_t *, const int64_t *, const uint8_t *, int64_t,
                  int64_t, int64_t, int64_t, int64_t, uint64_t) =
          g->w == 1   ? gather_1
          : g->w == 2 ? gather_2
          : g->w == 4 ? gather_4
          : g->w == 8 ? gather_8
                      : gather_16;
      row(d, idx + pi, x, b->n0, sd, si, sx, g->sa, g->dim);
    }
}

/* The shape of [v] into [s], its rank answered. */
static int shape_of(value v, int64_t *s) {
  int64_t dim[2 * NX_MAX_RANK], off;
  int r = nx_array_layout(v, dim, &off);
  for (int i = 0; i < r; i++) s[i] = dim[i];
  return r;
}

/* Whether [x] of rank [r] and [y] of rank [ry] have one rank, above [axis],
   and one extent along every other axis. */
static int fits_off_axis(const int64_t *x, int r, const int64_t *y, int ry,
                         intnat axis) {
  if (r != ry || axis < 0 || axis >= r) return 0;
  for (int i = 0; i < r; i++)
    if (i != axis && x[i] != y[i]) return 0;
  return 1;
}

static int same_shape(const int64_t *x, int r, const int64_t *y, int ry) {
  if (r != ry) return 0;
  for (int i = 0; i < r; i++)
    if (x[i] != y[i]) return 0;
  return 1;
}

/* What a call takes (cpu.h, Calls and jobs, scalar; M1 Max and fudge,
   2026-10-14): a gathered element, 51 read at random over 2^24 elements
   (M1); a scatter's update, 325 (a float16 Max, fudge). */
#define GATHER_COST 128
#define UPDATE_COST 512

/* What a scatter's copy of a target from into takes an element of [dt]
   (cpu.h, Calls and jobs, streaming): twice its bytes, read and written,
   the walk of the copy's strided bound NX_CPU_STRIDED_COST times that. A
   formula, not measured (2026-10-27). */
static double target_cost(int dt) { return 2.0 * nx_cpu_width(dt); }

/* What a scatter's [n] updates take, its call's and each job's share. */
static double updates_cost(int64_t n) { return (double)n * UPDATE_COST; }

value nx_cpu_gather(value vaxis, value vd, value vi, value vx) {
  CAMLparam4(vaxis, vd, vi, vx);
  intnat axis = Long_val(vaxis);
  int dt = nx_array_dtype(vx);
  int64_t ys[NX_MAX_RANK], is[NX_MAX_RANK], xs[NX_MAX_RANK];
  int ry = shape_of(vd, ys), ri = shape_of(vi, is), rx = shape_of(vx, xs);
  if (!fits_off_axis(is, ri, xs, rx, axis) || !same_shape(ys, ry, is, ri))
    CAMLreturn(Val_int(NX_SHAPE));
  nx_operand in[3] = {{vd, dt, 1}, {vi, NX_INT64, 0}, {vx, dt, 0}};
  nx_array a[3];
  int e = nx_read(3, in, a);
  if (e) CAMLreturn(Val_int(e));
  int r = a[0].rank;
  int bits = nx_dtype_row_of(dt).bits;
  gather g = {.a = a,
              .sa = a[2].dim[r + axis],
              .dim = (uint64_t)a[2].dim[axis],
              .w = bits < 8 ? 0 : bits / 8};
  nx_loop l = {.rank = r, .first = {a[0].offset, a[1].offset, a[2].offset}};
  for (int i = 0; i < r; i++) {
    l.extent[i] = a[1].dim[i];
    l.step[0][i] = a[0].dim[r + i];
    l.step[1][i] = a[1].dim[r + i];
    l.step[2][i] = i == axis ? 0 : a[2].dim[r + i];
  }
  l.rank = nx_coalesce_dims(3, r, l.extent, l.step);
  int released = nx_cpu_release(nx_cpu_loop_cost(3, a, &l, GATHER_COST));
  nx_cpu_walk(3, a, &l, GATHER_MOST, GATHER_COST, gather_block, &g);
  nx_cpu_reacquire(released);
  nx_done(3, a);
  CAMLreturn(Val_int(NX_OK));
}

/* Range gathers

   A range gather writes its destination one row at a time, a row being
   one position p along the ranged axis at one lead index, with its cells.
   Row p belongs to run j, the first whose end exceeds p: the runs that
   hold rows hold them in order, each from where the ends before it reached,
   so one forward walk over the ends finds every row's run. Run j's rows
   read x from position w = starts[j] + p - ends[j - 1], modulo 2^64; the
   rows whose position lies in [0, n) are one span, copied as one memcpy
   where x's rows lie one after the other, else as a box (nx_copy_box), and
   the rows before and after it are zeros, as are the rows past every end.

   Threads split the rows into units that start on whole cache lines of the
   destination, so that no two threads write one byte, sub-byte dtypes'
   included. A serial pass over the ends finds the run each unit starts
   in. */

/* The most units a range gather's job cuts its rows into. */
#define RANGE_UNITS 512

/* What a run costs on top of its bytes (Calls and jobs, scalar): finding
   it, its position's clip and its copy. A run of one byte took 2.0 ns,
   one core's copy of about 140 bytes (fudge alone, quiet, 2026-11-04). */
#define RUN_COST 256

/* A unit's first row: its lead index, its run, its position. */
typedef struct {
  int64_t lead, j, p;
} range_at;

typedef struct {
  const nx_array *a;    /* the destination, the starts, the ends, the source */
  int lr;               /* the lead axes, before the ranged one */
  int64_t lead, k, n;   /* lead indices, runs, the source's positions */
  int64_t size, cells;  /* the destination's rows per lead index, cells a row */
  int64_t sa;           /* the source's stride along the ranged axis */
  int nc;               /* the source's cell axes, coalesced */
  int64_t ce[NX_MAX_RANK], cs[NX_MAX_RANK]; /* their extents and strides */
  int line;             /* the source's rows are one run of cells */
  int64_t units;
  int64_t first[RANGE_UNITS + 1]; /* each unit's first row, then the end */
  range_at at[RANGE_UNITS];
} ranges;

/* The positions of lead index [lead] in the starts, the ends and the
   source, into [o]. */
static void lead_at(const ranges *g, int64_t lead, int64_t *o) {
  const nx_array *a = g->a;
  o[0] = a[1].offset, o[1] = a[2].offset, o[2] = a[3].offset;
  for (int i = g->lr - 1; i >= 0; i--) {
    int64_t d = a[1].dim[i], q = lead % d;
    lead /= d;
    o[0] += q * a[1].dim[a[1].rank + i];
    o[1] += q * a[2].dim[a[2].rank + i];
    o[2] += q * a[3].dim[a[3].rank + i];
  }
}

/* Zeros at the destination's [m] elements from element [d]. */
static void range_zeros(const ranges *g, int64_t d, int64_t m) {
  const nx_array *y = &g->a[0];
  if (y->bits >= 8)
    memset(y->base + d * (y->bits / 8), 0, (size_t)(m * (y->bits / 8)));
  else nx_sub_fill_run(y->base, y->bits, d, m, 0);
}

/* Copies [m] rows of the source from its element [s] to the destination's
   element [d]. */
static void range_copy(const ranges *g, int64_t d, int64_t s, int64_t m) {
  const nx_array *a = g->a;
  int bits = a[0].bits;
  if (g->line || (m == 1 && g->nc == 0)) {
    int64_t e = m * g->cells;
    if (bits >= 8) {
      int64_t w = bits / 8;
      memcpy(a[0].base + d * w, a[3].base + s * w, (size_t)(e * w));
    }
    else nx_sub_copy_run(a[0].base, d, a[3].base, s, 1, e, bits);
    return;
  }
  /* A box of the rows and the last two cell axes, for each index of the
     cell axes before those. */
  int nc = g->nc;
  nx_box b = {.first = {d, s}};
  if (nc == 0) {
    b.extent[0] = 1, b.extent[1] = 1, b.extent[2] = m;
    b.step[0][2] = 1, b.step[1][2] = g->sa;
    nx_copy_box(a[0].base, a[3].base, &b, bits);
    return;
  }
  if (nc == 1) {
    b.extent[0] = 1, b.extent[1] = m, b.extent[2] = g->ce[0];
    b.step[0][1] = g->cells, b.step[0][2] = 1;
    b.step[1][1] = g->sa, b.step[1][2] = g->cs[0];
    nx_copy_box(a[0].base, a[3].base, &b, bits);
    return;
  }
  int64_t c1 = g->ce[nc - 2], c2 = g->ce[nc - 1], plane = c1 * c2;
  b.extent[0] = m, b.extent[1] = c1, b.extent[2] = c2;
  b.step[0][0] = g->cells, b.step[0][1] = c2, b.step[0][2] = 1;
  b.step[1][0] = g->sa, b.step[1][1] = g->cs[nc - 2];
  b.step[1][2] = g->cs[nc - 1];
  for (int64_t o = 0; o < g->cells / plane; o++) {
    int64_t at = s, q = o;
    for (int i = nc - 3; i >= 0; i--) {
      at += (q % g->ce[i]) * g->cs[i];
      q /= g->ce[i];
    }
    b.first[0] = d + o * plane, b.first[1] = at;
    nx_copy_box(a[0].base, a[3].base, &b, bits);
  }
}

/* The [m] rows from row [p] of lead index [lead] that read the source from
   position [w] on, its lead index's elements from [xo]: zeros, the span
   inside [0, n), zeros. */
static void range_piece(const ranges *g, int64_t lead, int64_t p, int64_t m,
                        uint64_t w, int64_t xo) {
  uint64_t n = (uint64_t)g->n, lo, hi;
  /* The offsets i in the run of w + i inside [0, n), modulo 2^64: from 0
     where w is inside, else from where w + i wraps past 2^64. */
  if (w < n) lo = 0, hi = n - w;
  else lo = -w, hi = lo + n;
  if (lo > (uint64_t)m) lo = (uint64_t)m;
  if (hi > (uint64_t)m) hi = (uint64_t)m;
  int64_t d = g->a[0].offset + (lead * g->size + p) * g->cells;
  int64_t c = g->cells;
  if (lo > 0) range_zeros(g, d, (int64_t)lo * c);
  if (hi > lo)
    range_copy(g, d + (int64_t)lo * c, xo + (int64_t)(w + lo) * g->sa,
               (int64_t)(hi - lo));
  if ((uint64_t)m > hi)
    range_zeros(g, d + (int64_t)hi * c, (m - (int64_t)hi) * c);
}

/* Copies [n] bytes from [s] to [d]: up to 32 as two loads and two stores
   that overlap, so that a short row, as a text's, costs a few
   instructions where memcpy's call costs several times them. */
static inline __attribute__((always_inline)) void copy_bytes(
    uint8_t *d, const uint8_t *s, size_t n) {
  if (n > 32) {
    memcpy(d, s, n);
    return;
  }
  if (n > 16) {
    uint64_t x[2], y[2];
    memcpy(x, s, 16), memcpy(y, s + n - 16, 16);
    memcpy(d, x, 16), memcpy(d + n - 16, y, 16);
  }
  else if (n >= 8) {
    uint64_t x, y;
    memcpy(&x, s, 8), memcpy(&y, s + n - 8, 8);
    memcpy(d, &x, 8), memcpy(d + n - 8, &y, 8);
  }
  else if (n >= 4) {
    uint32_t x, y;
    memcpy(&x, s, 4), memcpy(&y, s + n - 4, 4);
    memcpy(d, &x, 4), memcpy(d + n - 4, &y, 4);
  }
  else if (n > 0) {
    uint8_t x = s[0], y = s[n / 2], z = s[n - 1];
    d[0] = x, d[n / 2] = y, d[n - 1] = z;
  }
}

/* A run's [m] rows at [d], [row] bytes each, that read the source's rows
   at [src] from position [w] on, modulo 2^64, past the source's [n]: zeros,
   the span inside [0, n), zeros. Out of line: a run inside the axis, as
   every run of a take is, never reaches it. */
static __attribute__((noinline)) void lines_clipped(uint8_t *d,
                                                    const uint8_t *src,
                                                    uint64_t w, uint64_t m,
                                                    uint64_t n, int64_t row) {
  uint64_t lo, hi;
  /* The offsets i of the run whose w + i lies inside [0, n): from 0 where
     w does, else from where w + i wraps past 2^64. */
  if (w < n) lo = 0, hi = n - w;
  else lo = -w, hi = lo + n;
  if (lo > m) lo = m;
  if (hi > m) hi = m;
  size_t r = (size_t)row;
  if (lo > 0) memset(d, 0, lo * r);
  if (hi > lo) memcpy(d + lo * r, src + (w + lo) * r, (hi - lo) * r);
  if (m > hi) memset(d + hi * r, 0, (m - hi) * r);
}

/* The rows [p, stop) of one lead index from run [j], its rows at [dst]
   and the source's at [src], the runs' starts at [st] and ends at [en]
   stepping [ss] and [se], where the source's rows lie one after the
   other, of whole bytes: each run's span is one copy. It keeps few values
   live, in locals, so that the compiler holds them in registers. Answers
   the run it stops in. */
static int64_t lead_lines(const ranges *g, uint8_t *dst, const uint8_t *src,
                          const int64_t *st, int64_t ss, const int64_t *en,
                          int64_t se, int64_t j, int64_t p, int64_t stop) {
  const int64_t k = g->k, row = g->cells * (g->a[0].bits / 8);
  const uint64_t n = (uint64_t)g->n;
  int64_t before = j > 0 ? en[(j - 1) * se] : 0;
  const int64_t *sp = st + j * ss, *ep = en + j * se;
  uint8_t *d = dst + p * row;
  /* Run j holds rows [p, e) where its end e exceeds p, and none else; a run
     the unit's stop cuts is the next unit's first. */
  for (; j < k && p < stop; j++, ep += se, sp += ss) {
    int64_t e = *ep;
    if (e > p) {
      int64_t q = e < stop ? e : stop;
      uint64_t m = (uint64_t)(q - p);
      uint64_t w = (uint64_t)*sp + (uint64_t)p - (uint64_t)before;
      if (w < n && m <= n - w)
        copy_bytes(d, src + w * row, m * (uint64_t)row);
      else lines_clipped(d, src, w, m, n, row);
      d += m * row;
      p = q;
      if (q < e) return j;
    }
    before = e;
  }
  if (p < stop) memset(d, 0, (size_t)((stop - p) * row));
  return j;
}

/* lead_lines for any source and dtype: each run's span through
   range_piece. */
static int64_t lead_pieces(const ranges *g, int64_t lead, const int64_t *o,
                           int64_t j, int64_t p, int64_t stop) {
  const nx_array *a = g->a;
  int64_t k = g->k;
  int64_t ss = a[1].dim[a[1].rank + g->lr], se = a[2].dim[a[2].rank + g->lr];
  const int64_t *st = (const int64_t *)a[1].base + o[0];
  const int64_t *en = (const int64_t *)a[2].base + o[1];
  while (p < stop) {
    while (j < k && en[j * se] <= p) j++;
    if (j == k) {
      range_zeros(g, a[0].offset + (lead * g->size + p) * g->cells,
                  (stop - p) * g->cells);
      return j;
    }
    int64_t e = en[j * se], q = e < stop ? e : stop;
    uint64_t before = j > 0 ? (uint64_t)en[(j - 1) * se] : 0;
    uint64_t w = (uint64_t)st[j * ss] + (uint64_t)p - before;
    range_piece(g, lead, p, q - p, w, o[2]);
    p = q;
  }
  return j;
}

/* Unit [u]'s rows, from its first to the next unit's. */
static void range_unit(const ranges *g, int64_t u) {
  const nx_array *a = g->a;
  int64_t lead = g->at[u].lead, j = g->at[u].j, p = g->at[u].p;
  int64_t end = g->first[u + 1], size = g->size;
  int lines = g->line && a[0].bits >= 8;
  int64_t w = a[0].bits / 8;
  int64_t ss = a[1].dim[a[1].rank + g->lr], se = a[2].dim[a[2].rank + g->lr];
  while (lead * size + p < end) {
    int64_t o[3];
    lead_at(g, lead, o);
    int64_t stop = end - lead * size;
    if (stop > size) stop = size;
    if (lines)
      j = lead_lines(g,
                     a[0].base + (a[0].offset + lead * size * g->cells) * w,
                     a[3].base + o[2] * w, (const int64_t *)a[1].base + o[0],
                     ss, (const int64_t *)a[2].base + o[1], se, j, p, stop);
    else j = lead_pieces(g, lead, o, j, p, stop);
    p = stop;
    if (p < size) break;
    lead++, j = 0, p = 0;
  }
}

static void range_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  for (int64_t u = lo; u < hi; u++) range_unit(ctx, u);
}

static int64_t gcd64(int64_t a, int64_t b) {
  while (b) {
    int64_t t = a % b;
    a = b, b = t;
  }
  return a;
}

/* Cuts the rows into [units] units of about as many rows each, each first
   row one whose first bit begins a cache line of the destination, and
   finds the run each starts in: the first whose end exceeds its row. */
static void range_plan(ranges *g, int64_t units) {
  const nx_array *a = g->a;
  int64_t rows = g->lead * g->size;
  int64_t row_bits = g->cells * a[0].bits;
  int64_t grain = 512 / gcd64(512, row_bits);
  int64_t u = 0;
  for (int64_t i = 0; i < units; i++) {
    int64_t r = (int64_t)((double)rows * (double)i / (double)units);
    r = (r + grain - 1) / grain * grain;
    if (r > rows) r = rows;
    if (u > 0 && r <= g->first[u - 1]) continue;
    if (r == rows) break;
    g->first[u++] = r;
  }
  g->first[u] = rows;
  g->units = u;
  int64_t se = a[2].dim[a[2].rank + g->lr];
  const int64_t *en = (const int64_t *)a[2].base;
  int64_t j = 0, lead = -1, o[3] = {0, 0, 0};
  for (int64_t i = 0; i < u; i++) {
    int64_t l = g->first[i] / g->size, p = g->first[i] % g->size;
    if (l != lead) lead = l, j = 0, lead_at(g, lead, o);
    while (j < g->k && en[o[1] + j * se] <= p) j++;
    g->at[i] = (range_at){lead, j, p};
  }
}

value nx_cpu_gather_ranges(value vd, value vs, value ve, value vx) {
  CAMLparam4(vd, vs, ve, vx);
  int dt = nx_array_dtype(vx);
  int64_t ys[NX_MAX_RANK], ss[NX_MAX_RANK], es[NX_MAX_RANK], xs[NX_MAX_RANK];
  int ry = shape_of(vd, ys), rs = shape_of(vs, ss), re = shape_of(ve, es),
      rx = shape_of(vx, xs);
  int ax = rs - 1;
  if (rs < 1 || !same_shape(ss, rs, es, re) || rx <= ax ||
      !fits_off_axis(ys, ry, xs, rx, ax))
    CAMLreturn(Val_int(NX_SHAPE));
  for (int i = 0; i < ax; i++)
    if (ss[i] != xs[i]) CAMLreturn(Val_int(NX_SHAPE));
  nx_operand in[4] = {
      {vd, dt, 1}, {vs, NX_INT64, 0}, {ve, NX_INT64, 0}, {vx, dt, 0}};
  nx_array a[4];
  int e = nx_read(4, in, a);
  if (e) CAMLreturn(Val_int(e));
  ranges plan, *g = &plan;
  g->a = a, g->lr = ax;
  g->lead = 1;
  for (int i = 0; i < ax; i++) g->lead *= ys[i];
  g->k = ss[ax], g->n = xs[ax], g->size = ys[ax];
  g->sa = a[3].dim[rx + ax];
  /* The cell axes, those of one element dropped, each merged into the one
     before it where the source steps through both as one. */
  g->nc = 0, g->cells = 1;
  for (int i = ax + 1; i < rx; i++) {
    int64_t d = xs[i], s = a[3].dim[rx + i];
    g->cells *= d;
    if (d == 1) continue;
    if (g->nc > 0 && g->cs[g->nc - 1] == s * d) {
      g->ce[g->nc - 1] *= d, g->cs[g->nc - 1] = s;
      continue;
    }
    g->ce[g->nc] = d, g->cs[g->nc] = s, g->nc++;
  }
  g->line = (g->nc == 0 && g->sa == 1) ||
            (g->nc == 1 && g->cs[0] == 1 && g->sa == g->ce[0]);
  int64_t rows = g->lead * g->size;
  if (rows > 0 && g->cells > 0) {
    double bytes = (double)rows * g->cells * nx_cpu_width(dt);
    double cost = (double)g->lead * g->k * RUN_COST +
                  bytes * (g->line ? 1 : NX_CPU_STRIDED_COST);
    int released = nx_cpu_release(cost);
    int threads = nx_cpu_threads((int64_t)bytes, cost);
    int64_t units = threads > 1 ? (int64_t)threads * NX_CPU_CHUNKS : 1;
    if (units > RANGE_UNITS) units = RANGE_UNITS;
    range_plan(g, units);
    nx_cpu_job(g->units, (int64_t)bytes, cost, range_units, g);
    nx_cpu_reacquire(released);
  }
  nx_done(4, a);
  CAMLreturn(Val_int(NX_OK));
}

/* Scatters */

typedef struct scatter scatter;

/* Applies the updates of a run of [n] slices from the positions [pd], [pi]
   and [pu] of the destination, the positions and the updates, those whose
   position lies in [lo, lo + span). */
typedef void (*scatter_run)(const scatter *c, int64_t pd, int64_t pi,
                            int64_t pu, int64_t n, uint64_t lo,
                            uint64_t span);

struct scatter {
  const nx_array *a; /* the destination, into, the positions, the updates */
  scatter_run run;
  int dt, combine;
  /* The slices: a loop of the destination, the positions, the updates and
     into off the axis; [sd], [si], [su] and [st] their steps along its last
     axis, whose [len] slices are a row of slices. */
  nx_loop l;
  int64_t sd, si, su, st, len;
  int64_t run_len; /* slices per run */
  int64_t pieces;  /* runs per row of slices */
  int64_t parts;  /* parts of the axis's targets, each a unit's */
  /* Along the axis: the updates' extent, their steps through the positions
     and the updates, the destination's and into's strides, the
     destination's extent. */
  int64_t m, ia, ua, da, ta;
  uint64_t dim;
  /* Whether this is a narrow float's Add, and its scratch (Narrow sums):
     [each] bytes a worker at [sums]. */
  int narrow;
  uint8_t *sums;
  int64_t each;
  /* Whether each unit sums its targets in row order on its thread (Sums in
     row order). */
  int in_order;
  /* A sorted scatter (Sorted scatters): the run's [n] slices, at [pd],
     [pi], [pu] and [pt] in the destination, the positions, the updates and
     into; the block's first step [b0], its [steps] and [chunks], of at most
     [block] steps; the [bins] of 2^[shift] targets. Chunk k's updates of
     bin q end at seg[k·bins + q] in [pos] and [vals], whose entries are
     the updates, of [w] bytes, or their rows' steps (entry_bytes), and
     start at the end of bin q - 1, or at the chunk's first. Each chunk
     gathers [gather_bytes] at [gather]. */
  int64_t pd, pi, pu, pt, n, b0, steps, chunks, bins, block, w, gather_bytes;
  /* A row's sorted positions and steps (narrow_rows): [order] holds them,
     their sort's scratch and where each target's run starts. Where the
     targets are parted (narrow_run), unit u's sort takes [order]'s words
     from 4·at[u] to 4·at[u + 1]. */
  int64_t *order, *at;
  const uint64_t *rp;
  const int64_t *rr, *runs;
  int shift;
  int64_t *pos, *seg;
  uint8_t *vals, *gather;
};

/* Sums by Prog.Add, part by part for complex numbers. */
static inline nx_c64 add_c64(nx_c64 a, nx_c64 b) {
  return nx_c64_of(nx_add_f32(a.re, b.re), nx_add_f32(a.im, b.im));
}

static inline nx_c128 add_c128(nx_c128 a, nx_c128 b) {
  return nx_c128_of(nx_add_f64(a.re, b.re), nx_add_f64(a.im, b.im));
}

/* Typed runs: [F] combines the target's element with the update. Where the
   positions step 0 along the run, each step of the axis is one row. */
#define SCATTER_RUN(NAME, T, F)                                              \
  static void NAME(const scatter *c, int64_t pd, int64_t pi, int64_t pu,    \
                   int64_t n, uint64_t lo, uint64_t span) {                  \
    T *d = (T *)c->a[0].base;                                                \
    const int64_t *x = (const int64_t *)c->a[2].base;                        \
    const T *u = (const T *)c->a[3].base;                                    \
    int64_t sd = c->sd, si = c->si, su = c->su, da = c->da;                  \
    int64_t m = c->m, ia = c->ia, ua = c->ua;                                \
    if (n == 1) {                                                            \
      d += pd;                                                               \
      for (int64_t t = 0; t < m; t++, pi += ia, pu += ua) {                  \
        uint64_t p = (uint64_t)x[pi];                                        \
        if (p - lo >= span) continue;                                        \
        T *e = d + (int64_t)p * da;                                          \
        *e = F(*e, u[pu]);                                                   \
      }                                                                      \
      return;                                                                \
    }                                                                        \
    for (int64_t t = 0; t < m; t++, pi += ia, pu += ua) {                    \
      if (si == 0) {                                                         \
        uint64_t p = (uint64_t)x[pi];                                        \
        if (p - lo >= span) continue;                                        \
        T *e = d + pd + (int64_t)p * da;                                     \
        const T *v = u + pu;                                                 \
        if (sd == 1 && su == 1)                                              \
          for (int64_t i = 0; i < n; i++) e[i] = F(e[i], v[i]);              \
        else                                                                 \
          for (int64_t i = 0; i < n; i++)                                    \
            e[i * sd] = F(e[i * sd], v[i * su]);                             \
        continue;                                                            \
      }                                                                      \
      for (int64_t i = 0; i < n; i++) {                                      \
        uint64_t p = (uint64_t)x[pi + i * si];                               \
        if (p - lo >= span) continue;                                        \
        T *e = d + pd + (int64_t)p * da + i * sd;                            \
        *e = F(*e, u[pu + i * su]);                                          \
      }                                                                      \
    }                                                                        \
  }

#define SET(a, b) (b)
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((b) < (a) ? (b) : (a))
/* A boolean is true where its byte is not zero; its extremes are Or and
   And of truths, stored as 0 or 1. */
#define OR_B(a, b) ((uint8_t)(((a) | (b)) != 0))
#define AND_B(a, b) ((uint8_t)((a) != 0 && (b) != 0))
#define ADD_U8(a, b) ((uint8_t)((a) + (b)))
#define ADD_U16(a, b) ((uint16_t)((a) + (b)))
#define ADD_U32(a, b) ((uint32_t)((a) + (b)))
#define ADD_U64(a, b) ((uint64_t)((a) + (b)))

SCATTER_RUN(set_1, uint8_t, SET)
SCATTER_RUN(set_2, uint16_t, SET)
SCATTER_RUN(set_4, uint32_t, SET)
SCATTER_RUN(set_8, uint64_t, SET)
SCATTER_RUN(set_16, u128, SET)
SCATTER_RUN(add_f32_run, float, nx_add_f32)
SCATTER_RUN(add_f64_run, double, nx_add_f64)
SCATTER_RUN(add_8, uint8_t, ADD_U8)
SCATTER_RUN(add_16, uint16_t, ADD_U16)
SCATTER_RUN(add_32, uint32_t, ADD_U32)
SCATTER_RUN(add_64, uint64_t, ADD_U64)
SCATTER_RUN(max_f32, float, nx_maximum_f32)
SCATTER_RUN(min_f32, float, nx_minimum_f32)
SCATTER_RUN(max_f64, double, nx_maximum_f64)
SCATTER_RUN(min_f64, double, nx_minimum_f64)
SCATTER_RUN(max_i8, int8_t, MAX)
SCATTER_RUN(min_i8, int8_t, MIN)
SCATTER_RUN(max_u8, uint8_t, MAX)
SCATTER_RUN(min_u8, uint8_t, MIN)
SCATTER_RUN(max_i16, int16_t, MAX)
SCATTER_RUN(min_i16, int16_t, MIN)
SCATTER_RUN(max_u16, uint16_t, MAX)
SCATTER_RUN(min_u16, uint16_t, MIN)
SCATTER_RUN(max_i32, int32_t, MAX)
SCATTER_RUN(min_i32, int32_t, MIN)
SCATTER_RUN(max_u32, uint32_t, MAX)
SCATTER_RUN(min_u32, uint32_t, MIN)
SCATTER_RUN(max_i64, int64_t, MAX)
SCATTER_RUN(min_i64, int64_t, MIN)
SCATTER_RUN(max_u64, uint64_t, MAX)
SCATTER_RUN(min_u64, uint64_t, MIN)
SCATTER_RUN(max_b, uint8_t, OR_B)
SCATTER_RUN(min_b, uint8_t, AND_B)
SCATTER_RUN(add_c64_run, nx_c64, add_c64)
SCATTER_RUN(max_c64, nx_c64, nx_maximum_c64)
SCATTER_RUN(min_c64, nx_c64, nx_minimum_c64)
SCATTER_RUN(add_c128_run, nx_c128, add_c128)
SCATTER_RUN(max_c128, nx_c128, nx_maximum_c128)
SCATTER_RUN(min_c128, nx_c128, nx_minimum_c128)

/* Elements one at a time: sub-byte dtypes and narrow floats. */

/* The code of a narrow float's element nearest [f]. */
static uint32_t narrow_of_float(int dt, float f) {
  switch (dt) {
    case NX_FLOAT16: return nx_float_to_f16(f);
    case NX_BFLOAT16: return nx_float_to_bf16(f);
    case NX_FLOAT8_E4M3FN: return nx_float_to_e4m3fn(f);
    case NX_FLOAT8_E5M2: return nx_float_to_e5m2(f);
    default: return nx_float_to_e2m1fn(f);
  }
}

static int is_narrow(int dt) {
  return dt == NX_FLOAT16 || dt == NX_BFLOAT16 || dt == NX_FLOAT8_E4M3FN ||
         dt == NX_FLOAT8_E5M2 || dt == NX_FLOAT4_E2M1FN;
}

/* The code at position [p] of [a]: an element of at most 16 bits. */
static inline uint32_t code_at(const nx_array *a, int64_t p) {
  switch (a->bits) {
    case 1:
    case 4: return nx_sub_load(a->base, a->bits, p);
    case 8: return a->base[p];
    default: return ((const uint16_t *)a->base)[p];
  }
}

static inline void code_put(const nx_array *a, int64_t p, uint32_t v) {
  switch (a->bits) {
    case 1:
    case 4: nx_sub_store(a->base, a->bits, p, v); return;
    case 8: a->base[p] = (uint8_t)v; return;
    default: ((uint16_t *)a->base)[p] = (uint16_t)v;
  }
}

/* A sub-byte integer's value: int4's sign-extended. */
static inline int32_t sub_value(int dt, uint32_t c) {
  return dt == NX_INT4 ? (int32_t)((c ^ 8u) - 8u) : (int32_t)c;
}

/* Combines the update at position [pu] into the destination's element at
   [pd]. */
static void combine_one(const scatter *c, int64_t pd, int64_t pu) {
  const nx_array *d = &c->a[0], *u = &c->a[3];
  int dt = c->dt;
  uint32_t a = code_at(d, pd), b = code_at(u, pu);
  if (c->combine == NX_COMBINE_SET) {
    code_put(d, pd, b);
    return;
  }
  if (is_narrow(dt)) {
    /* The update's code where the extreme is not the element's own value,
       else the element's; Add sums apart (Narrow sums). */
    float x = nx_bits_to_float(dt, a), y = nx_bits_to_float(dt, b);
    float g = c->combine == NX_COMBINE_MAX ? nx_maximum_f32(x, y)
                                        : nx_minimum_f32(x, y);
    uint32_t r = nx_float_bits(g) != nx_float_bits(x) ? b : a;
    if (r != a) code_put(d, pd, r);
    return;
  }
  /* int4, uint4 and bit; Add wraps modulo 16. */
  if (c->combine == NX_COMBINE_ADD) {
    code_put(d, pd, a + b);
    return;
  }
  int32_t x = sub_value(dt, a), y = sub_value(dt, b);
  int keep = c->combine == NX_COMBINE_MAX ? x >= y : x <= y;
  if (!keep) code_put(d, pd, b);
}

static void element_run(const scatter *c, int64_t pd, int64_t pi, int64_t pu,
                        int64_t n, uint64_t lo, uint64_t span) {
  const int64_t *x = (const int64_t *)c->a[2].base;
  for (int64_t t = 0; t < c->m; t++, pi += c->ia, pu += c->ua)
    for (int64_t i = 0; i < n; i++) {
      uint64_t p = (uint64_t)x[pi + i * c->si];
      if (p - lo >= span) continue;
      combine_one(c, pd + (int64_t)p * c->da + i * c->sd, pu + i * c->su);
    }
}

/* The typed run of [combine] at [dt], or element_run. */
static scatter_run run_of(int combine, int dt) {
  int bits = nx_dtype_row_of(dt).bits;
  if (combine == NX_COMBINE_SET) switch (bits) {
      case 8: return set_1;
      case 16: return set_2;
      case 32: return set_4;
      case 64: return set_8;
      case 128: return set_16;
      default: return element_run;
    }
  int add = combine == NX_COMBINE_ADD, max = combine == NX_COMBINE_MAX;
#define PICK(A, MX, MN) return add ? A : max ? MX : MN
  switch (dt) {
    case NX_FLOAT32: PICK(add_f32_run, max_f32, min_f32);
    case NX_FLOAT64: PICK(add_f64_run, max_f64, min_f64);
    case NX_INT8: PICK(add_8, max_i8, min_i8);
    case NX_UINT8: PICK(add_8, max_u8, min_u8);
    case NX_BOOL: PICK(add_8, max_b, min_b);
    case NX_INT16: PICK(add_16, max_i16, min_i16);
    case NX_UINT16: PICK(add_16, max_u16, min_u16);
    case NX_INT32: PICK(add_32, max_i32, min_i32);
    case NX_UINT32: PICK(add_32, max_u32, min_u32);
    case NX_INT64: PICK(add_64, max_i64, min_i64);
    case NX_UINT64: PICK(add_64, max_u64, min_u64);
    case NX_COMPLEX64: PICK(add_c64_run, max_c64, min_c64);
    case NX_COMPLEX128: PICK(add_c128_run, max_c128, min_c128);
    default: return element_run;
  }
#undef PICK
}

/* Run [v] of slices: its positions in the destination, the positions, the
   updates and into, into [p]; its slices answered. */
static int64_t run_at(const scatter *c, int64_t v, int64_t *p) {
  const nx_loop *l = &c->l;
  int r = l->rank;
  int64_t len = l->extent[r - 1];
  int64_t i0 = v % c->pieces * c->run_len;
  v /= c->pieces;
  for (int k = 0; k < 4; k++) p[k] = l->first[k] + i0 * l->step[k][r - 1];
  for (int i = r - 2; i >= 0; i--) {
    int64_t x = v % l->extent[i];
    v /= l->extent[i];
    for (int k = 0; k < 4; k++) p[k] += x * l->step[k][i];
  }
  return len - i0 < c->run_len ? len - i0 : c->run_len;
}

/* Narrow sums

   A narrow float's Add sums each target's updates in float32 from its
   element, left to right in C order, and rounds the sum once into the
   destination. Its updates are grouped by target first, by one of three
   ways that each keep scratch of at most the order of the updates:

   - A row of slices whose positions step 0 along it, an embedding
     gradient's, few such rows (narrow_rows): its positions are sorted by
     target, stably, once, and a job sums the targets' runs, each cut into
     pieces of RUN_PIECE slices. A piece's sums are a row of floats: the
     targets' row read into float32, each update's row added in turn, the
     row encoded. On fudge (Core Ultra 5 235) the embedding gradient's row
     of 8192 tokens of 4096 into 128256 rows of bfloat16 took 3.6 ms so,
     against 7 ms with each unit of UNIT_ROW slices summing its own runs,
     whose updates' rows, read out of order, are short.
   - A unit of one slice, few such units, as a 1-d scatter's or a take
     along axis 0's gradient over fewer columns than threads, whose
     positions differ along its rows: the sorted scatter's bins
     (narrow_bins) keep their targets' sums in float32, a flag beside each
     that an update reached, then encode the reached targets once. Where
     the axis has fewer targets than updates, the sums carry from block to
     block in every bin's scratch, of the order of the targets; otherwise
     the axis is one block, so that each sum spans its target's updates.
     On fudge 16M float16 updates into 1M targets took 25 ms as one block,
     14 ms in blocks of 1M, whose sorted entries stay in the cache; a
     bfloat16 [1M, 3] take along axis 0's gradient 5.6 ms, against 21 ms
     with 5 parts of each column's targets sorting their positions, a unit
     each (narrow_run).
   - Otherwise a unit (narrow_run), of a part of the targets where units
     are fewer than the threads, sorts the positions its run applies by
     target, stably, and sums one target after another, as a row's pieces
     do, in four words for each position it sorts: its worker's for every
     update along the axis, or where the targets are parted, its own for
     its part's positions, counted first (part_order), so that the parts
     take four words per update in all: a float4_e2m1fn [16M, 2] take
     along axis 0's gradient, whose elements share bytes across bins, 7
     parts of each slice on 14 threads, holds 1 GiB so, against 7 GiB with
     each worker's sized by the axis.
     A run whose positions differ along it is one slice, so that one sort
     serves each run and the threads share the slices: [1M, 64] bfloat16
     updates with a position each took 3.8 s on one thread, each run of 64
     slices sorting them a slice at a time. */

/* The bytes of a bin's flags, padded to a line. */
static int64_t flag_bytes(int shift) { return (((int64_t)1 << shift) + 63) & -64; }

/* Slices of a piece of a row's run (narrow_rows): a run's updates on
   2 KiB of float32 sums at a time, enough pieces for the threads to share
   the runs evenly where a few targets take most updates. */
#define RUN_PIECE 512

/* Positions below which a unit's sort is by insertion. */
#define FEW_STEPS 64

/* Row [n] elements of [a] from [at], stepping [s], into float32 at [f]. */
static void load_row(int dt, const nx_array *a, int64_t at, int64_t s,
                     int64_t n, float *f) {
  if (s == 1 && n > 1 && a->bits >= 8) {
    nx_cpu_convert(dt, NX_FLOAT32)(a->base + at * (a->bits / 8), f, n);
    return;
  }
  for (int64_t i = 0; i < n; i++)
    f[i] = nx_bits_to_float(dt, code_at(a, at + i * s));
}

/* Stores the [n] float32 at [f] into the row of load_row, each rounded
   once. */
static void store_row(int dt, const nx_array *a, int64_t at, int64_t s,
                      int64_t n, const float *f) {
  if (s == 1 && n > 1 && a->bits >= 8) {
    nx_cpu_convert(NX_FLOAT32, dt)(f, a->base + at * (a->bits / 8), n);
    return;
  }
  for (int64_t i = 0; i < n; i++)
    code_put(a, at + i * s, narrow_of_float(dt, f[i]));
}

/* [y] is [x] plus [u], [n] sums by Prog.Add. */
static void add_sums(const float *x, const float *u, float *y, int64_t n) {
  for (int64_t i = 0; i < n; i++) y[i] = nx_add_f32(x[i], u[i]);
}

/* Sorts the [k] positions [p] and their steps [r] by position, stably,
   through [p2] and [r2] of k entries: by insertion below FEW_STEPS, else
   by radix over the bytes their span takes of each position less the
   least. Answers whether the sorted entries ended in p2 and r2. */
static int sort_steps(uint64_t *p, int64_t *r, uint64_t *p2, int64_t *r2,
                      int64_t k) {
  if (k < FEW_STEPS) {
    nx_cpu_insertion(p, r, k);
    return 0;
  }
  uint64_t lo = p[0], hi = p[0];
  for (int64_t j = 1; j < k; j++) {
    lo = p[j] < lo ? p[j] : lo;
    hi = p[j] > hi ? p[j] : hi;
  }
  return nx_cpu_radix(p, r, p2, r2, k, nx_cpu_span_bytes(hi - lo), lo);
}

/* Sums the [k] updates at steps [r] along the axis, sorted by their
   positions [p], into their targets, rows of [n] slices from [pd] and
   [pu], through [sum] and [row] of n floats. */
static void sum_sorted(const scatter *c, int64_t pd, int64_t pu, int64_t n,
                       const uint64_t *p, const int64_t *r, int64_t k,
                       float *sum, float *row, float *next) {
  const nx_array *d = &c->a[0], *u = &c->a[3];
  int64_t w = d->bits / 8;
  int64_t db = c->sd == 1 ? n * w : 0, ub = c->su == 1 ? n * w : 0;
  for (int64_t j = 0, e; j < k; j = e) {
    int64_t at = pd + (int64_t)p[j] * c->da;
    load_row(c->dt, d, at, c->sd, n, sum);
    for (e = j; e < k && p[e] == p[j]; e++) {
      if (e + 1 < k) {
        const uint8_t *nu = u->base + (pu + r[e + 1] * c->ua) * w;
        const uint8_t *nd = d->base + (pd + (int64_t)p[e + 1] * c->da) * w;
        for (int64_t b = 0; b < ub; b += 64) __builtin_prefetch(nu + b);
        for (int64_t b = 0; b < db && p[e + 1] != p[j]; b += 64)
          __builtin_prefetch(nd + b);
      }
      load_row(c->dt, u, pu + r[e] * c->ua, c->su, n, row);
      add_sums(sum, row, next, n);
      float *t = sum;
      sum = next, next = t;
    }
    store_row(c->dt, d, at, c->sd, n, sum);
  }
}

/* Sums a row's targets' runs, a piece of RUN_PIECE slices of one run at a
   time. */
static void run_pieces(int64_t lo, int64_t hi, int worker, void *ctx) {
  const scatter *c = ctx;
  float *sum = (float *)(c->sums + worker * c->each), *row = sum + UNIT_ROW;
  float *next = row + UNIT_ROW;
  int64_t pieces = (c->len + RUN_PIECE - 1) / RUN_PIECE;
  for (int64_t v = lo; v < hi; v++) {
    int64_t q = v / pieces, i0 = v % pieces * RUN_PIECE;
    int64_t n = c->len - i0 < RUN_PIECE ? c->len - i0 : RUN_PIECE;
    int64_t j = c->runs[q], e = c->runs[q + 1];
    sum_sorted(c, c->pd + i0 * c->sd, c->pu + i0 * c->su, n, c->rp + j,
               c->rr + j, e - j, sum, row, next);
  }
}

/* Sums the row of slices from unit [v], whose positions step 0 along it:
   its positions sorted once, then a job over its targets' runs, each cut
   into pieces. */
static void narrow_rows(scatter *c, int64_t v, int64_t bytes) {
  int64_t p[4], m = c->m;
  run_at(c, v, p);
  c->pd = p[0];
  c->pu = p[2];
  const int64_t *x = (const int64_t *)c->a[2].base;
  uint64_t *rp = (uint64_t *)c->order, *rp2 = rp + m;
  int64_t *rr = c->order + 2 * m, *rr2 = rr + m, *runs = rr2 + m, k = 0;
  for (int64_t t = 0; t < m; t++) {
    uint64_t q = (uint64_t)x[p[1] + t * c->ia];
    if (q >= c->dim) continue;
    rp[k] = q;
    rr[k++] = t;
  }
  int moved = sort_steps(rp, rr, rp2, rr2, k);
  c->rp = moved ? rp2 : rp;
  c->rr = moved ? rr2 : rr;
  int64_t n = 0;
  for (int64_t j = 0; j < k; j++)
    if (j == 0 || c->rp[j] != c->rp[j - 1]) runs[n++] = j;
  runs[n] = k;
  c->runs = runs;
  nx_cpu_job(n * ((c->len + RUN_PIECE - 1) / RUN_PIECE), bytes,
             updates_cost(c->m * c->len), run_pieces, c);
}

/* Sums unit [u]'s run of [n] slices, its positions in [lo, lo + span):
   its worker's rows of floats, then its sort's words, the unit's own where
   the targets are parted, else after the rows. */
static void narrow_run(const scatter *c, int worker, int64_t u, int64_t pd,
                       int64_t pi, int64_t pu, int64_t n, uint64_t lo,
                       uint64_t span) {
  const int64_t *x = (const int64_t *)c->a[2].base;
  int64_t m = c->m, ia = c->ia;
  float *sum = (float *)(c->sums + worker * c->each), *row = sum + UNIT_ROW;
  float *next = row + UNIT_ROW;
  int64_t most = c->at ? c->at[u + 1] - c->at[u] : m;
  uint64_t *p = c->at ? (uint64_t *)(c->order + 4 * c->at[u])
                      : (uint64_t *)(next + UNIT_ROW);
  uint64_t *p2 = p + most;
  int64_t *r = (int64_t *)(p2 + most), *r2 = r + most;
  /* The count gave the unit [most] positions; one another thread changed
     since stops at the region's end rather than past it. */
  int64_t k = 0;
  for (int64_t t = 0; t < m; t++) {
    uint64_t q = (uint64_t)x[pi + t * ia];
    if (q - lo >= span || k == most) continue;
    p[k] = q;
    r[k++] = t;
  }
  int moved = sort_steps(p, r, p2, r2, k);
  sum_sorted(c, pd, pu, n, moved ? p2 : p, moved ? r2 : r, k, sum, row, next);
}

/* Sums in row order

   A float's, a narrow float's or a complex number's Add whose units are
   single slices, fewer than the threads, as a 1-d scatter's or a group-by's
   sum, sums each unit's updates on one thread in row order, into the
   destination itself, where the targets the units reach at once fit the
   host's largest cache: [t] elements of the dtype a unit, or a narrow
   float's [t] float32 sums and a flag each that an update reached it,
   whose reached targets it encodes once at the end. The bits are the
   sorted scatter's, whose bins apply each target's updates in the same
   order. Sorting the updates into bins on every thread costs more than it
   saves where the targets stay in the cache. Per update, by callgrind on
   fudge (its LL the 24 MiB L3, 2026-11-04), sorted and in row order:

     rows into groups              instructions   last-level misses
     float64, 1M into 100 to 100K  49-50 / 13-14  0.22-0.29 / 0-0.03
     float64, 10M into 100 to 100K    50 / 13        0.26 / 0.25
     float64, 10M into 1M (8 MB)      49 / 14        0.67 / 0.28
     float64, 10M into 3M (24 MB)     50 / 16        0.95 / 0.64
     float64, 10M into 10M (80 MB)    51 / 22        1.34 / 1.39
     bfloat16, 1M into 100            80 / 15        0.10 / 0.005
     complex128, 1M into 100          57 / 18        0.50 / 0.08

   The sort's own count, copy and second read of each update cost the
   instructions more. Past the cache the misses tie, but in row order they
   are reads at random on one thread, where the sort's stream on every
   thread, so the cache bounds row order. Wall times, which a loaded
   machine can't judge, are re-checked at the quiet re-record. Rows of
   several slices, an embedding gradient's, still sort: their entries are
   rows' steps, a sort's cost shared by a row's slices. */

/* Whether each unit is one slice: a row of one slice, or a narrow
   float's slice whose positions differ along its row (slices). */
static int one_slice(const scatter *c) {
  return c->len == 1 || c->run_len == 1;
}

/* The bytes a narrow float's flags of [t] targets take, padded to a
   line. */
static int64_t flags_of(int64_t t) { return (t + 63) & -64; }

/* Whether the scatter [c], of [units] units, sums each in row order on
   its thread. */
static int in_order(const scatter *c, int64_t units) {
  if (c->combine != NX_COMBINE_ADD || !one_slice(c)) return 0;
  int64_t w = 4;
  switch (c->dt) {
    case NX_FLOAT16: case NX_BFLOAT16: case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: case NX_FLOAT32: break;
    case NX_FLOAT64: case NX_COMPLEX64: w = 8; break;
    case NX_COMPLEX128: w = 16; break;
    default: return 0;
  }
  return units * (int64_t)c->dim * w <= nx_cpu_llc;
}

/* Updates a narrow float's unit in row order decodes at once. */
#define DECODED 256

/* Sums a narrow float's unit, the slice at [pd], [pi] and [pu], in row
   order in its worker's sums, whose flags start and end clear. The updates
   decode DECODED at a time, as a row converts. */
static void narrow_in_order(const scatter *c, int worker, int64_t pd,
                            int64_t pi, int64_t pu) {
  const nx_array *d = &c->a[0], *u = &c->a[3];
  const int64_t *x = (const int64_t *)c->a[2].base;
  int64_t t = (int64_t)c->dim, da = c->da, ia = c->ia;
  uint8_t *reached = c->sums + worker * c->each;
  float *sums = (float *)(reached + flags_of(t));
  float row[DECODED];
  for (int64_t k = 0; k < c->m; k += DECODED) {
    int64_t n = c->m - k < DECODED ? c->m - k : DECODED;
    load_row(c->dt, u, pu + k * c->ua, c->ua, n, row);
    const int64_t *at = x + pi + k * ia;
    for (int64_t j = 0; j < n; j++) {
      uint64_t p = (uint64_t)at[j * ia];
      if (p >= c->dim) continue;
      if (!reached[p]) {
        sums[p] = nx_bits_to_float(c->dt, code_at(d, pd + (int64_t)p * da));
        reached[p] = 1;
      }
      sums[p] = nx_add_f32(sums[p], row[j]);
    }
  }
  for (int64_t p = 0; p < t; p++) {
    if (!reached[p]) continue;
    code_put(d, pd + p * da, narrow_of_float(c->dt, sums[p]));
    reached[p] = 0;
  }
}

/* The first target of part [k] of the axis's targets. */
static uint64_t part_first(const scatter *c, int64_t k) {
  return c->dim * (uint64_t)k / (uint64_t)c->parts;
}

static void scatter_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  const scatter *c = ctx;
  for (int64_t u = lo; u < hi; u++) {
    /* The unit's part of the axis's targets, then its run of slices. */
    int64_t part = u % c->parts, p[4];
    uint64_t first = part_first(c, part), next = part_first(c, part + 1);
    int64_t n = run_at(c, u / c->parts, p);
    if (c->narrow && c->in_order)
      narrow_in_order(c, worker, p[0], p[1], p[2]);
    else if (c->narrow)
      narrow_run(c, worker, u, p[0], p[1], p[2], n, first, next - first);
    else c->run(c, p[0], p[1], p[2], n, first, next - first);
  }
}

/* Positions of a run that a job's piece counts by part (part_order). */
#define COUNT_PIECE (256 * 1024)

typedef struct {
  const scatter *c;
  int64_t pieces;  /* pieces of each run's positions */
  uint64_t *first; /* each part's first target, then the axis's extent */
  int64_t *counts; /* each piece's positions in each part, a row a piece */
} counting;

/* Counts each piece's positions by part: a position's part is first
   guessed from its share of the axis, then moved to the part whose targets
   hold it. */
static void count_parts(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const counting *k = ctx;
  const scatter *c = k->c;
  const int64_t *x = (const int64_t *)c->a[2].base;
  double share = (double)c->parts / (double)c->dim;
  for (int64_t v = lo; v < hi; v++) {
    int64_t p[4], j = v % k->pieces, *n = k->counts + v * c->parts;
    run_at(c, v / k->pieces, p);
    int64_t t1 = c->m * (j + 1) / k->pieces;
    for (int64_t t = c->m * j / k->pieces; t < t1; t++) {
      uint64_t q = (uint64_t)x[p[1] + t * c->ia];
      if (q >= c->dim) continue;
      int64_t g = (int64_t)((double)q * share);
      if (g >= c->parts) g = c->parts - 1;
      while (q < k->first[g]) g--;
      while (q >= k->first[g + 1]) g++;
      n[g]++;
    }
  }
}

/* Gives each unit of the [units] runs' parts its sort's words (narrow_run),
   four for each of its run's positions in its part: answers 0 where memory
   runs out. A run is cut into pieces of at most COUNT_PIECE positions, and
   into enough for the [threads] where the runs are few: a 1-d Add of 200K
   updates would count on one thread. The count reads [bytes] and costs
   [cost]. */
static int part_order(scatter *c, int64_t units, int64_t bytes, double cost,
                      int threads) {
  int64_t jobs = units * c->parts;
  int64_t pieces = (c->m + COUNT_PIECE - 1) / COUNT_PIECE;
  int64_t least = (threads + units - 1) / units;
  if (pieces < least) pieces = least;
  counting k = {.c = c, .pieces = pieces};
  k.first = malloc((size_t)(c->parts + 1) * sizeof(uint64_t));
  k.counts = calloc((size_t)(units * pieces * c->parts), sizeof(int64_t));
  c->at = malloc((size_t)(jobs + 1) * sizeof(int64_t));
  int ok = k.first != NULL && k.counts != NULL && c->at != NULL;
  if (ok) {
    for (int64_t g = 0; g <= c->parts; g++) k.first[g] = part_first(c, g);
    nx_cpu_job(units * pieces, bytes, cost, count_parts, &k);
    c->at[0] = 0;
    for (int64_t u = 0; u < jobs; u++) {
      int64_t v = u / c->parts * pieces, g = u % c->parts, n = 0;
      for (int64_t j = 0; j < pieces; j++) n += k.counts[(v + j) * c->parts + g];
      c->at[u + 1] = c->at[u] + n;
    }
    c->order = malloc((size_t)(4 * c->at[jobs] + 1) * sizeof(int64_t));
    ok = c->order != NULL;
  }
  free(k.first);
  free(k.counts);
  return ok;
}

/* Sorted scatters

   Where the runs are fewer than the threads and the axis long, as a 1-d
   scatter's is or an embedding gradient's, whose positions step 0 along
   its run, a run's updates are sorted by their targets' bins of 2^shift
   positions along the axis, a block at a time. A job cuts the block into
   chunks; each chunk's thread counts its updates by bin, then copies
   their positions into its bins' runs in their order, with the updates
   where a row of slices is one slice and their rows' steps where it has
   several, its last run perhaps of one slice, dropping positions outside
   the axis. A second job takes the bins: each
   copies its targets from into, in the first block, then applies its
   chunks' runs in chunk order with the scatter's own run, so that a
   target's updates apply in C order, into targets that fit half a core's
   L2. Threads that split the targets instead each read every update: 1-d
   scatters of 16 Mi updates ran 1.2 to 1.7 times slower so on an x86-64
   of six cores, and the embedding gradient's row 1.3 times. */

/* The bytes of a sorted entry: the update where a unit is one slice, else
   its row's step. */
static int64_t entry_bytes(const scatter *c) {
  return one_slice(c) ? c->w : 8;
}

/* The first step of chunk [k] of the block. */
static int64_t chunk_first(const scatter *c, int64_t k) {
  return c->steps * k / c->chunks;
}

/* Lays out a chunk's updates from [t0] to [t1] by bin, from the bins'
   firsts at [seg], with each a value [V] of the type [T], the update or
   its row's step: each bin's gather in [gp] and [gv], GATHER at a time. */
#define PLACE(T, V)                                                          \
  {                                                                          \
    T *gv = (T *)(gp + bins * GATHER);                                       \
    for (int64_t t = t0; t < t1; t++) {                                      \
      uint64_t p = (uint64_t)x[t * ia];                                      \
      if (p >= dim) continue;                                                \
      int64_t q = (int64_t)(p >> shift), g = fill[q];                        \
      if (seg[q] + g == end[q]) continue;                                    \
      fill[q] = g + 1;                                                       \
      gp[q * GATHER + g] = (int64_t)p;                                       \
      gv[q * GATHER + g] = V;                                                \
      if (g + 1 < GATHER) continue;                                          \
      memcpy(pos + seg[q], gp + q * GATHER, sizeof(int64_t) * GATHER);       \
      memcpy((T *)vals + seg[q], gv + q * GATHER, sizeof(T) * GATHER);       \
      seg[q] += GATHER;                                                      \
      fill[q] = 0;                                                           \
    }                                                                        \
    for (int64_t q = 0; q < bins; q++) {                                     \
      memcpy(pos + seg[q], gp + q * GATHER, sizeof(int64_t) * fill[q]);      \
      memcpy((T *)vals + seg[q], gv + q * GATHER, sizeof(T) * fill[q]);      \
      seg[q] += fill[q];                                                     \
    }                                                                        \
  }

static void sort_chunks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const scatter *c = ctx;
  int64_t bins = c->bins, ia = c->ia, ua = c->ua, b0 = c->b0;
  uint64_t dim = c->dim;
  int shift = c->shift;
  int64_t *pos = c->pos;
  uint8_t *vals = c->vals;
  const int64_t *x = (const int64_t *)c->a[2].base + c->pi + c->b0 * ia;
  const uint8_t *u = c->a[3].base + (c->pu + c->b0 * ua) * c->w;
  /* Counts of bin q at seg[q + 1], made the bins' firsts, then their
     cursors, and end[q] its counted end. They stay on this thread's
     stack: chunks' counters sharing a cache line made the sort several
     times slower. */
  int64_t seg[MOST_BINS + 1], fill[MOST_BINS], end[MOST_BINS];
  int64_t e = entry_bytes(c);
  for (int64_t k = lo; k < hi; k++) {
    int64_t t0 = chunk_first(c, k), t1 = chunk_first(c, k + 1);
    int64_t *gp = (int64_t *)(c->gather + k * c->gather_bytes);
    memset(seg, 0, (size_t)(bins + 1) * sizeof *seg);
    memset(fill, 0, (size_t)bins * sizeof *fill);
    for (int64_t t = t0; t < t1; t++) {
      uint64_t p = (uint64_t)x[t * ia];
      if (p < dim) seg[(p >> shift) + 1]++;
    }
    seg[0] = t0;
    for (int64_t q = 0; q < bins; q++) {
      seg[q + 1] += seg[q];
      end[q] = seg[q + 1];
    }
#define AT(T) ((const T *)u)[t * ua]
    if (!one_slice(c)) PLACE(int64_t, b0 + t)
    else switch (c->w) {
        case 1: PLACE(uint8_t, AT(uint8_t)) break;
        case 2: PLACE(uint16_t, AT(uint16_t)) break;
        case 4: PLACE(uint32_t, AT(uint32_t)) break;
        case 8: PLACE(uint64_t, AT(uint64_t)) break;
        default: PLACE(u128, AT(u128))
      }
#undef AT
    /* The positions are read twice; another thread writing them between
       the reads, which the caller must not do, could give a bin more
       positions than counted or fewer. A full bin drops the rest and a
       short one ends in entries no run applies, so such a call's result
       is wrong rather than memory written outside the bins. */
    for (int64_t q = 0; q < bins; q++) {
      int64_t gap = end[q] - seg[q];
      if (gap == 0) continue;
      for (int64_t j = seg[q]; j < end[q]; j++) pos[j] = -1;
      memset(vals + seg[q] * e, 0, (size_t)(gap * e));
    }
    memcpy(c->seg + k * bins, end, (size_t)bins * sizeof *end);
  }
}

#undef PLACE

/* Copies the run's targets of the bin from [first] from into, unless
   into is the destination. */
static void copy_bin(const scatter *c, uint64_t first) {
  if (c->a[1].alias) return;
  uint64_t end = first + ((uint64_t)1 << c->shift);
  if (end > c->dim) end = c->dim;
  nx_loop l = {.rank = 2,
               .extent = {(int64_t)(end - first), c->n},
               .first = {c->pd + (int64_t)first * c->da,
                         c->pt + (int64_t)first * c->ta}};
  l.step[0][0] = c->da;
  l.step[0][1] = c->sd;
  l.step[1][0] = c->ta;
  l.step[1][1] = c->st;
  l.rank = nx_coalesce_dims(2, 2, l.extent, l.step);
  nx_cpu_copy_loop(c->a[0].base, c->a[1].base, c->a[0].bits, &l,
                   target_cost(c->dt));
}

/* Applies each bin's runs with the scatter's run, over the sorted
   positions and updates. */
static void apply_bins(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const scatter *c = ctx;
  /* The run reads the destination's base, width and offset, and the
     positions' and updates' bases and widths. */
  const nx_array *d = &c->a[0];
  nx_array a[4] = {{.base = d->base, .bits = d->bits, .offset = d->offset},
                   {.bits = d->bits},
                   {.bits = 64},
                   {.base = c->a[3].base, .bits = c->a[3].bits}};
  scatter s = *c;
  s.a = a;
  s.ia = s.ua = 1;
  uint64_t span = (uint64_t)1 << c->shift;
  for (int64_t q = lo; q < hi; q++) {
    uint64_t first = (uint64_t)q << c->shift;
    if (c->b0 == 0) copy_bin(c, first);
    for (int64_t k = 0; k < c->chunks; k++) {
      const int64_t *seg = c->seg + k * c->bins;
      int64_t from = q == 0 ? chunk_first(c, k) : seg[q - 1];
      s.m = seg[q] - from;
      if (s.m == 0) continue;
      a[2].base = (uint8_t *)(c->pos + from);
      if (one_slice(c)) {
        a[3].base = c->vals + from * c->w;
        s.run(&s, c->pd, 0, 0, 1, first, span);
        continue;
      }
      /* A row of updates at a time, from its step, the next row's bytes
         fetched ahead where they are contiguous: rows read out of order
         stall on memory otherwise. */
      const int64_t *t = (const int64_t *)c->vals + from;
      int64_t m = s.m, bytes = c->su == 1 ? c->n * c->w : 0;
      s.m = 1;
      for (int64_t j = 0; j < m; j++) {
        if (j + 1 < m)
          for (int64_t b = 0; b < bytes; b += 64)
            __builtin_prefetch(c->a[3].base +
                               (c->pu + t[j + 1] * c->ua) * c->w + b);
        s.run(&s, c->pd, j, c->pu + t[j] * c->ua, c->n, first, span);
      }
    }
  }
}

/* Sums a narrow float's bins (Narrow sums), whose rows are of one slice:
   a bin's targets' float32 sums after a flag for each target, padded to a
   line, in its worker's scratch, or in the bin's own where the sums carry
   from block to block. The last block encodes each reached target once,
   walking the bin's targets where they are fewer than its entries or
   carried, else its entries. */
static void narrow_bins(int64_t lo, int64_t hi, int worker, void *ctx) {
  const scatter *c = ctx;
  const nx_array *d = &c->a[0];
  int carry = c->block < c->m, last = c->b0 + c->steps == c->m;
  int64_t span = (int64_t)1 << c->shift;
  for (int64_t q = lo; q < hi; q++) {
    uint8_t *reached = c->sums + (carry ? q : worker) * c->each;
    float *sums = (float *)(reached + flag_bytes(c->shift));
    uint64_t first = (uint64_t)q << c->shift;
    int64_t targets = (int64_t)(c->dim - first) < span
                          ? (int64_t)(c->dim - first) : span;
    int64_t entries = 0;
    if (c->b0 == 0) copy_bin(c, first);
    for (int pass = 0; pass < 1 + last; pass++) {
      if (pass == 1 && (carry || entries > targets)) {
        for (int64_t t = 0; t < targets; t++) {
          if (!reached[t]) continue;
          int64_t at = c->pd + ((int64_t)first + t) * c->da;
          code_put(d, at, narrow_of_float(c->dt, sums[t]));
          reached[t] = 0;
        }
        break;
      }
      for (int64_t k = 0; k < c->chunks; k++) {
        const int64_t *seg = c->seg + k * c->bins;
        int64_t from = q == 0 ? chunk_first(c, k) : seg[q - 1];
        entries += seg[q] - from;
        for (int64_t j = from; j < seg[q]; j++) {
          int64_t t = c->pos[j] - (int64_t)first;
          /* An entry outside the bin, left by a racing write of the
             positions (sort_chunks). */
          if ((uint64_t)t >= (uint64_t)span) continue;
          int64_t at = c->pd + c->pos[j] * c->da;
          if (pass == 1) {
            if (reached[t]) code_put(d, at, narrow_of_float(c->dt, sums[t]));
            reached[t] = 0;
            continue;
          }
          uint32_t u = c->w == 1 ? c->vals[j] : ((const uint16_t *)c->vals)[j];
          if (!reached[t]) {
            sums[t] = nx_bits_to_float(c->dt, code_at(d, at));
            reached[t] = 1;
          }
          sums[t] = nx_add_f32(sums[t], nx_bits_to_float(c->dt, u));
        }
      }
    }
  }
}

/* The bytes a sorted block of [steps] steps of rows of [n] slices takes to
   apply. */
static int64_t apply_bytes(const scatter *c, int64_t steps, int64_t n) {
  return steps * (8 + entry_bytes(c) + 3 * n * c->w);
}

/* The shift of the bins: as many targets as fit half a core's L2, the
   share the contraction's blocks take, fewer while the threads would have
   under BINS_PER_THREAD bins each, more past MOST_BINS bins. */
static int bin_shift(uint64_t dim, int64_t w, int threads) {
  int64_t most = nx_cpu_l2 / 2;
  int s = 0;
  while (s < 62 && ((int64_t)2 << s) * w <= most) s++;
  while (s > 0 && ((dim - 1) >> s) + 1 < (uint64_t)threads * BINS_PER_THREAD)
    s--;
  while (s < 62 && ((dim - 1) >> s) + 1 > MOST_BINS) s++;
  return s;
}

/* Scatters run [v] by sorted blocks. A block's updates' cost is shared
   between its sort and its application as their bytes are. */
static void sorted(scatter *c, int64_t v, int threads) {
  int64_t p[4];
  c->n = run_at(c, v, p);
  c->pd = p[0];
  c->pi = p[1];
  c->pu = p[2];
  c->pt = p[3];
  for (c->b0 = 0; c->b0 < c->m; c->b0 += c->block) {
    c->steps = c->m - c->b0 < c->block ? c->m - c->b0 : c->block;
    c->chunks = c->steps < threads ? c->steps : threads;
    int64_t e = entry_bytes(c);
    int64_t sorting = c->steps * 2 * (8 + e);
    int64_t applying = apply_bytes(c, c->steps, c->n);
    double per = updates_cost(c->steps * c->n) / (double)(sorting + applying);
    nx_cpu_job(c->chunks, sorting, per * (double)sorting, sort_chunks, c);
    nx_cpu_job(c->bins, applying, per * (double)applying,
               c->narrow ? narrow_bins : apply_bins, c);
  }
}

/* Owned targets

   An integer's Add wraps, and its Max and Min pick one of their operands,
   so the result is the same in any order of the updates. So do a float's
   and a narrow float's Max and Min, under the contract: on operands that
   aren't NaN they pick one by prog.mli's order, -0 below +0, which is
   total, so the extreme of a target's element and its updates is one
   operand's bits in any order (nx_maximum of +0 and -0 is +0 either way
   round, the AND of their bits). A NaN operand makes the result a NaN in
   any order, whose sign and payload nx_cpu.mli leaves free to change with
   the threads. A complex number's extremes don't fold: a NaN result is the
   first NaN operand in C order, whose other part prog.mli keeps, and the
   copies would give another's (1M complex128 Max into 100 targets, with
   NaNs of distinct imaginary parts, gave other bits than one thread in 38
   runs of 50).

   Where few runs of many updates reach few targets, each thread then
   folds its shares of the updates into a copy of the targets of its own,
   and the copies fold into the destination after: each update is read
   once, with no sort and no part of the targets. A target no update
   reaches keeps its bits through the fold, x + 0 or the extreme of x and
   x, a signalling NaN's included: nx_maximum and nx_minimum answer a NaN
   first operand with its bits, by selection and masks rather than
   arithmetic, and a narrow float's combine_one stores either operand's
   code, both x's. A boolean's Max and Min store 0 or 1, and would rewrite
   an unreached true byte of 2, so booleans sort. float4_e2m1fn's elements
   share bytes, and the copies take whole bytes per element, so it doesn't
   fold, as int4 and uint4 don't. A copy takes at most half a core's L2, as
   a sorted scatter's bin does, and the copies together no more elements
   than the updates, so their fold costs no more than the updates do. 1M
   int64 Adds into 100 targets on four threads took 47 instructions an
   update sorted and 11 so, as many as on one thread; 10M took 12.9 ms
   sorted on fudge's 14 threads under load 8 to 25, 6.6 on one thread, and
   2.4 so. 1M float64 Max into 100 targets took 62 sorted and 25 so, as
   many as on one thread, and bfloat16's 137 and 100. */

/* Updates of a run each share of it takes. */
#define OWN_SHARE (64 * 1024)

typedef struct {
  const scatter *c;
  int64_t shares; /* of each run */
  int64_t total;  /* the destination's elements, each copy's */
  uint8_t *own;   /* each worker's copy, total elements apart */
  uint8_t *used;  /* whether each worker's copy holds its targets */
} owning;

/* Whether [combine]'s result at [dt] is the same in any order of the
   updates, and its fold keeps an unreached target's bits. */
static int order_free(int combine, int dt) {
  if (combine == NX_COMBINE_SET) return 0;
  switch (dt) {
    case NX_INT8: case NX_UINT8: case NX_INT16: case NX_UINT16:
    case NX_INT32: case NX_UINT32: case NX_INT64: case NX_UINT64: return 1;
    case NX_FLOAT16: case NX_BFLOAT16: case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: case NX_FLOAT32: case NX_FLOAT64:
      return combine != NX_COMBINE_ADD;
    default: return 0;
  }
}

/* Folds shares into the worker's copy, which starts as Add's identity,
   zero, or as the destination, which Max and Min may meet again. */
static void own_shares(int64_t lo, int64_t hi, int worker, void *ctx) {
  const owning *o = ctx;
  const scatter *c = o->c;
  int64_t w = c->a[0].bits / 8, off = c->a[0].offset;
  uint8_t *own = o->own + worker * o->total * w;
  if (!o->used[worker]) {
    if (c->combine == NX_COMBINE_ADD) memset(own, 0, (size_t)(o->total * w));
    else memcpy(own, c->a[0].base + off * w, (size_t)(o->total * w));
    o->used[worker] = 1;
  }
  nx_array a[4] = {c->a[0], c->a[1], c->a[2], c->a[3]};
  a[0].base = own;
  scatter s = *c;
  s.a = a;
  for (int64_t u = lo; u < hi; u++) {
    int64_t p[4], j = u % o->shares;
    int64_t n = run_at(c, u / o->shares, p);
    int64_t t0 = c->m * j / o->shares, t1 = c->m * (j + 1) / o->shares;
    s.m = t1 - t0;
    c->run(&s, p[0] - off, p[1] + t0 * c->ia, p[2] + t0 * c->ua, n, 0,
           c->dim);
  }
}

/* Folds the copies' elements [lo, hi) into the destination's, each copy
   as one row of updates at position 0. */
static void fold_owned(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const owning *o = ctx;
  const scatter *c = o->c;
  int64_t w = c->a[0].bits / 8, zero = 0;
  nx_array a[4] = {c->a[0], c->a[1], c->a[2], c->a[3]};
  a[2].base = (uint8_t *)&zero;
  scatter s = {.a = a, .dt = c->dt, .combine = c->combine, .m = 1,
               .sd = 1, .su = 1};
  for (int64_t k = 0; o->used[k] != 2; k++) {
    if (!o->used[k]) continue;
    a[3].base = o->own + k * o->total * w;
    c->run(&s, c->a[0].offset + lo, 0, lo, hi - lo, 0, 1);
  }
}

/* Whether the scatter [c] of [updates] into [total] contiguous targets,
   of fewer runs than [threads], folds into owned copies. */
static int owns(const scatter *c, int64_t total, int64_t updates,
                int threads) {
  const nx_array *d = &c->a[0];
  return threads > 1 && order_free(c->combine, c->dt) &&
         (d->flags & NX_CONTIGUOUS) &&
         total * (d->bits / 8) <= nx_cpu_l2 / 2 && threads * total <= updates;
}

/* Scatters the [units] runs into owned copies on at most [threads]
   threads, the updates' [bytes] costing [cost]: answers 0 where memory
   runs out. */
static int owned(const scatter *c, int64_t units, int64_t total,
                 int threads, int64_t bytes, double cost) {
  int64_t w = c->a[0].bits / 8;
  owning o = {.c = c,
              .shares = (c->m + OWN_SHARE - 1) / OWN_SHARE,
              .total = total};
  o.own = malloc((size_t)(threads * total * w));
  /* Each worker's flag, then 2. */
  o.used = calloc((size_t)threads + 1, 1);
  if (o.own == NULL || o.used == NULL) {
    free(o.own);
    free(o.used);
    return 0;
  }
  o.used[threads] = 2;
  nx_cpu_job(units * o.shares, bytes, cost, own_shares, &o);
  int64_t folded = (threads + 2) * total * w;
  nx_cpu_job(total, folded, (double)folded, fold_owned, &o);
  free(o.own);
  free(o.used);
  return 1;
}

/* Fills the loop of slices from the shape [us] of rank [r], off [axis],
   with into's strides [ts], and answers its units. Only a sorted
   scatter's copy_bin reads into's steps; other scatters pass zeros, which
   never part an axis. */
static int64_t slices(scatter *c, const int64_t *us, int r, intnat axis,
                      const int64_t *ts) {
  const nx_array *a = c->a;
  nx_loop *l = &c->l;
  l->first[0] = a[0].offset;
  l->first[1] = a[2].offset;
  l->first[2] = a[3].offset;
  l->first[3] = a[1].offset;
  for (int i = 0; i < r; i++) {
    l->extent[i] = i == axis ? 1 : us[i];
    l->step[0][i] = a[0].dim[r + i];
    l->step[1][i] = a[2].dim[r + i];
    l->step[2][i] = a[3].dim[r + i];
    l->step[3][i] = ts[i];
  }
  l->rank = nx_coalesce_dims(4, r, l->extent, l->step);
  int lr = l->rank;
  c->sd = l->step[0][lr - 1];
  c->si = l->step[1][lr - 1];
  c->su = l->step[2][lr - 1];
  c->st = l->step[3][lr - 1];
  c->len = l->extent[lr - 1];
  /* A narrow float's sums sort a run's positions once (narrow_run): where
     they differ along the run, it is one slice. */
  c->run_len = c->narrow && c->si != 0 ? 1 : UNIT_ROW;
  c->pieces = (c->len + c->run_len - 1) / c->run_len;
  int64_t rows = 1;
  for (int i = 0; i < lr - 1; i++) rows *= l->extent[i];
  return rows * c->pieces;
}

/* Copies into to the destination, unless the two are one array. */
static void copy_into(const nx_array *a, int64_t total) {
  if (total == 0 || a[1].alias) return;
  nx_loop l;
  nx_coalesce(2, a, &l);
  nx_cpu_copy_loop(a[0].base, a[1].base, a[0].bits, &l,
                   target_cost(a[0].dtype));
}

value nx_cpu_scatter(value vcombine, value vaxis, value vd, value vinto,
                     value vi, value vu) {
  CAMLparam5(vcombine, vaxis, vd, vinto, vi);
  CAMLxparam1(vu);
  int combine = Int_val(vcombine);
  intnat axis = Long_val(vaxis);
  int dt = nx_array_dtype(vinto);
  int64_t ys[NX_MAX_RANK], ts[NX_MAX_RANK], is[NX_MAX_RANK], us[NX_MAX_RANK];
  int ry = shape_of(vd, ys), rt = shape_of(vinto, ts), ri = shape_of(vi, is),
      ru = shape_of(vu, us);
  if (!same_shape(ys, ry, ts, rt) || !same_shape(is, ri, us, ru) ||
      !fits_off_axis(us, ru, ts, rt, axis))
    CAMLreturn(Val_int(NX_SHAPE));
  nx_operand in[4] = {
      {vd, dt, 1}, {vinto, dt, 0}, {vi, NX_INT64, 0}, {vu, dt, 0}};
  nx_array a[4];
  int e = nx_read(4, in, a);
  if (e) CAMLreturn(Val_int(e));
  /* One rank: the destination's, into's, the positions' and the
     updates'. */
  int r = a[0].rank;
  int64_t total = 1, updates = 1;
  for (int i = 0; i < r; i++) {
    total *= ys[i];
    updates *= us[i];
  }
  double target = NX_CPU_STRIDED_COST * target_cost(dt);
  int released =
      nx_cpu_release((double)total * target + updates_cost(updates));
  if (total == 0 || updates == 0) {
    copy_into(a, total);
    nx_cpu_reacquire(released);
    nx_done(4, a);
    CAMLreturn(Val_int(NX_OK));
  }
  scatter c = {.a = a,
               .run = run_of(combine, dt),
               .dt = dt,
               .combine = combine,
               .m = us[axis],
               .ia = a[2].dim[r + axis],
               .ua = a[3].dim[r + axis],
               .da = a[0].dim[r + axis],
               .ta = a[1].dim[r + axis],
               .dim = (uint64_t)ys[axis],
               .narrow = combine == NX_COMBINE_ADD && is_narrow(dt)};
  static const int64_t unread[NX_MAX_RANK];
  int64_t units = slices(&c, us, r, axis, unread);
  int64_t bytes = updates * (8 + a[3].bits / 8 + a[0].bits / 8);
  double cost = updates_cost(updates);
  int threads = nx_cpu_threads(bytes, cost);
  /* Fewer units than threads, over many updates, or any for a narrow
     float's sums. Units of one slice, as a 1-d scatter's or a narrow
     float's whose positions differ along its rows, and rows of slices
     whose positions step 0 along them, as an embedding gradient's, sort
     their updates; a dtype of under a byte shares its bytes across bins,
     so doesn't; a narrow float's rows of several slices sort them by
     target once a row (narrow_rows). Where the result is the same in any
     order of the updates and the targets are few, each thread folds its
     shares into targets of its own (Owned targets). A float's Add of
     single slices whose targets fit the cache sums each on one thread
     instead of sorting (Sums in row order). Otherwise each thread takes a
     part of the targets and walks every update of its run. */
  c.parts = 1;
  int few = units < threads && (c.m >= PART_UPDATES || c.narrow);
  int rows = few && c.narrow && c.len > 1 && c.si == 0;
  int own = few && !rows && owns(&c, total, updates, threads);
  c.in_order = few && !rows && !own && in_order(&c, units);
  int sort = few && !rows && !own && !c.in_order &&
             (one_slice(&c) || c.si == 0) && a[0].bits >= 8;
  if (few && !sort && !rows && !own && !c.in_order)
    c.parts = (threads + units - 1) / units;
  /* Into's steps may part the loop's axes further; a row of one slice, or
     of positions stepping 0, stays one. */
  if (sort) units = slices(&c, us, r, axis, a[1].dim + r);
  if ((uint64_t)c.parts > c.dim) c.parts = (int64_t)c.dim;
  if (!sort) copy_into(a, total);
  /* A worker's scratch for a narrow float's sums: its targets' flags and
     float32 sums in row order; else three rows of floats, then two words
     for each of its unit's positions and two for their sort; a row's sort
     keeps those words once, and a word more for each run, and parted
     targets' units each for their part's positions. */
  int64_t jobs = units * c.parts;
  int workers = jobs < threads ? (int)jobs : threads;
  int parted = c.narrow && c.parts > 1;
  if (c.narrow && c.in_order)
    c.each = flags_of((int64_t)c.dim) + 4 * (int64_t)c.dim;
  else if (c.narrow && !sort)
    c.each = (rows || parted ? 0 : 32 * c.m) + 12 * UNIT_ROW;
  /* Parted targets' count of their positions takes a share of the
     updates' cost, as its bytes are of the positions' and the updates'. */
  int64_t counted = parted ? units * c.m * 8 : 0;
  double per = cost / (double)(bytes + counted);
  int laid = !parted ||
             part_order(&c, units, counted, per * (double)counted, threads);
  if (rows) {
    workers = threads;
    c.order = malloc((size_t)(5 * c.m + 1) * sizeof(int64_t));
  }
  if (sort) {
    c.w = a[0].bits / 8;
    int64_t width = c.len < UNIT_ROW ? c.len : UNIT_ROW;
    c.shift = bin_shift(c.dim, c.narrow ? 4 : width * c.w, threads);
    c.bins = (int64_t)((c.dim - 1) >> c.shift) + 1;
    /* A narrow float's sums carry from block to block in every bin's
       scratch (narrow_bins), of the order of the targets; over as many
       targets as updates or more, one block takes the axis. */
    c.block = c.narrow && (int64_t)c.dim >= c.m ? c.m
              : (int64_t)c.dim < FEWEST_BLOCK  ? FEWEST_BLOCK
              : (int64_t)c.dim > MOST_BLOCK    ? MOST_BLOCK
                                               : (int64_t)c.dim;
    /* A bin's scratch for a narrow float's sums, on the threads of its
       block's job, or each bin's own. */
    if (c.narrow) {
      int64_t apply = apply_bytes(&c, c.m, 1);
      c.each = flag_bytes(c.shift) + ((int64_t)4 << c.shift);
      workers = c.block < c.m ? (int)c.bins
                              : nx_cpu_threads(apply, updates_cost(c.m));
    }
    int64_t most = c.m < c.block ? c.m : c.block, e = entry_bytes(&c);
    c.gather_bytes = c.bins * GATHER * (8 + e);
    c.pos = malloc((size_t)most * sizeof(int64_t));
    c.vals = malloc((size_t)(most * e));
    c.seg = malloc((size_t)(threads * c.bins) * sizeof(int64_t));
    c.gather = malloc((size_t)(threads * c.gather_bytes));
  }
  if (c.narrow) c.sums = calloc((size_t)workers, (size_t)c.each);
  if ((c.narrow && c.sums == NULL) || (rows && c.order == NULL) || !laid ||
      (sort && (c.pos == NULL || c.vals == NULL || c.seg == NULL ||
                c.gather == NULL))) {
    free(c.sums);
    free(c.order);
    free(c.at);
    free(c.pos);
    free(c.vals);
    free(c.seg);
    free(c.gather);
    nx_cpu_reacquire(released);
    nx_done(4, a);
    caml_raise_out_of_memory();
  }
  int ok = 1;
  if (rows)
    for (int64_t v = 0; v < units; v += c.pieces) narrow_rows(&c, v, bytes);
  else if (sort)
    for (int64_t v = 0; v < units; v++) sorted(&c, v, threads);
  else if (own)
    ok = owned(&c, units, total, threads, bytes, cost);
  else
    nx_cpu_job(jobs, bytes, per * (double)bytes, scatter_units, &c);
  free(c.sums);
  free(c.order);
  free(c.at);
  free(c.pos);
  free(c.vals);
  free(c.seg);
  free(c.gather);
  nx_cpu_reacquire(released);
  nx_done(4, a);
  if (!ok) caml_raise_out_of_memory();
  CAMLreturn(Val_int(NX_OK));
}

value nx_cpu_scatter_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_scatter(argv[0], argv[1], argv[2], argv[3], argv[4],
                        argv[5]);
}
