/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Reductions and scans of one operand by Sum, Prod, Max or Min.

   Association. An output's terms are numbered t = 0, 1, … in C order of the
   reduced axes' indices. They fall into blocks of NX_CPU_FOLD_BLOCK
   consecutive terms and, within a block, into NX_CPU_LANES lanes from the
   identity, term t into lane t mod 16. Lane i takes lane i + 8 for i < 8,
   then lane i + 4, i + 2 and i + 1: lane 0 is the block's value. The blocks
   combine by the binary tree whose left part holds the largest power of two
   of blocks below their count. A binary counter computes it in one pass: it
   combines the two newest values once per trailing zero of the number of
   blocks done, then what remains, newest first.

   A scan cuts each slice along its axis into chunks of SCAN_CHUNK terms from
   its start. A chunk's total is its terms reduced in the order above. The
   carry into chunk 0 is the identity, into chunk c + 1 the carry into c
   combined with c's total, and result i of chunk c is the carry combined
   with the chunk's terms up to i, left to right.

   Only float sums and products depend on this order: integers wrap and
   extremes are exact. Neither the layout, the path nor the thread count
   changes it. A thread takes a group of blocks, a power of two of them from
   a multiple of that power, and such a group's value is a subtree of the
   blocks' tree: the groups' values combine by the same counter.

   NaN. A float result that is NaN is replaced by its first NaN term in
   index order, found by a second walk taken only then. A scan's results
   stay NaN once they are, since adding, multiplying and the extremes keep
   a NaN: where a slice's last result is NaN, every result from its first
   NaN term on is that term.

   Paths. The operand and the destination, read at the operand's shape with
   step 0 along the reduced axes, coalesce: reduced and kept axes never
   merge, and the axes keep their order. A unit of work folds W outputs:
   - one output, its runs of terms along the innermost reduced axis into
     lanes, where a reduced axis steps least through the operand;
   - W outputs along the kept axis that steps least, each term a row of W
     elements combined into one of 16 accumulator rows, elsewhere and where
     outputs have few terms. A 2x2 window's maximum takes this path. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/mlvalues.h>

#include "cpu.h"

#define BLOCK NX_CPU_FOLD_BLOCK
#define LANES NX_CPU_LANES
#define SCAN_CHUNK 4096

/* A streaming unit's row of outputs: 512 bytes, 128 float32. Byte buffers
   hold elements of the dtype, read and written as such, so they are aligned
   as the widest. */
#define ROW 512

/* The depth of the blocks' tree: 2^48 blocks exceed any array. */
#define LEVELS 48

/* Fewer terms per output take the streaming path. */
#define FEW 16

/* A scan's slices a unit runs side by side where its axis steps least. */
#define SLICES 4

/* Full blocks whose values the table computes in one call. */
#define RUN 64

/* Units the groups of blocks aim for where outputs are few. */
#define UNITS 64

typedef struct {
  const nx_cpu_fold *f;
  int dtype, w, nan;
  _Alignas(8) uint8_t id[8];
  const nx_array *a; /* a[0] the destination, a[1] the operand */
  /* The reduced axes in C order: extents and the operand's steps. */
  int nr;
  int64_t re[NX_MAX_RANK], rs[NX_MAX_RANK];
  /* The kept axes, the row axis last: extents, the operand's and the
     destination's steps. */
  int nk;
  int64_t ke[NX_MAX_RANK], ks[NX_MAX_RANK], kd[NX_MAX_RANK];
  int64_t x0, d0;
  int64_t rd; /* a scan's destination step along its axis */
  int slices; /* a unit's outputs are slices scanned side by side */
  int64_t terms, blocks;
  int64_t row;    /* outputs per unit along the row axis; 1 per output */
  int64_t chunks; /* units along the row axis */
  int64_t outer;  /* units along the other kept axes */
  int64_t group;  /* blocks per group, a power of two */
  int64_t groups;
  uint8_t *scratch; /* each unit's group's values, where groups > 1 */
} fold;

static void identity(int monoid, int dt, uint8_t *id) {
  memset(id, 0, 8);
  int w = nx_cpu_width(dt);
  switch (monoid) {
    case NX_SUM: return;
    case NX_PROD:
      if (dt == NX_FLOAT32) *(float *)id = 1.f;
      else if (dt == NX_FLOAT64) *(double *)id = 1.;
      else id[0] = 1; /* little-endian hosts: the low byte */
      return;
    case NX_MAX:
      if (dt == NX_FLOAT32) *(float *)id = -INFINITY;
      else if (dt == NX_FLOAT64) *(double *)id = -INFINITY;
      else if (nx_dtype_row_of(dt).kind == NX_KIND_SIGNED) id[w - 1] = 0x80;
      return;
    default: /* NX_MIN */
      if (dt == NX_FLOAT32) *(float *)id = INFINITY;
      else if (dt == NX_FLOAT64) *(double *)id = INFINITY;
      else if (dt == NX_BOOL) id[0] = 1;
      else {
        memset(id, 0xFF, w);
        if (nx_dtype_row_of(dt).kind == NX_KIND_SIGNED) id[w - 1] = 0x7F;
      }
  }
}

/* [n] identities at [d], copied in doubling runs. */
static void fill(const fold *q, uint8_t *d, int64_t n) {
  if (n == 0) return;
  memcpy(d, q->id, q->w);
  for (int64_t k = 1; k < n; k *= 2)
    memcpy(d + k * q->w, d, (k < n - k ? k : n - k) * q->w);
}

static int is_nan(const fold *q, const uint8_t *p) {
  if (q->dtype == NX_FLOAT32) {
    float v;
    memcpy(&v, p, 4);
    return v != v;
  }
  double v;
  memcpy(&v, p, 8);
  return v != v;
}

/* Whether one of the [n] values at [v] is NaN, in one pass the compiler
   vectorises. */
static int any_nan(const fold *q, const uint8_t *v, int64_t n) {
  int bad = 0;
  if (q->dtype == NX_FLOAT32) {
    const float *f = (const float *)v;
    for (int64_t j = 0; j < n; j++) bad |= f[j] != f[j];
  } else {
    const double *f = (const double *)v;
    for (int64_t j = 0; j < n; j++) bad |= f[j] != f[j];
  }
  return bad;
}

static const uint8_t *at(const fold *q, int64_t p) {
  return q->a[1].base + p * q->w;
}

/* The operand's position of term [t] of the output at [p]. */
static int64_t term(const fold *q, int64_t p, int64_t t) {
  for (int i = q->nr - 1; i >= 0; i--) {
    p += t % q->re[i] * q->rs[i];
    t /= q->re[i];
  }
  return p;
}

/* Blocks */

/* Folds terms [t0, t0 + n) of the output at [p] into the lanes [l], run by
   run along the innermost reduced axis. */
static void runs(const fold *q, int64_t p, int64_t t0, int64_t n, uint8_t *l) {
  int r = q->nr - 1;
  int64_t inner = q->re[r], s = q->rs[r];
  for (int64_t t = t0; t < t0 + n;) {
    int64_t i = t % inner, len = inner - i;
    if (len > t0 + n - t) len = t0 + n - t;
    q->f->lanes(at(q, term(q, p, t)), s, len, l, (int)(t & (LANES - 1)));
    t += len;
  }
}

/* The lanes' tree over [W] outputs: 16 rows of [W] values, [used] of them
   holding terms; the others are the identity, which changes no value. */
static void lane_tree(const fold *q, uint8_t *l, int64_t W, int used) {
  int64_t row = W * q->w;
  for (int h = LANES / 2; h >= 1; h /= 2)
    for (int i = 0; i < h && i + h < used; i++)
      q->f->combine(l + i * row, l + (i + h) * row, 1, W);
}

/* Block [b]'s values of the unit's [W] outputs, the first at [p] and each
   next [s] further along the operand, into [v]. */
static void block(const fold *q, int64_t p, int64_t s, int64_t W, int64_t b,
                  uint8_t *v) {
  int64_t t0 = b * BLOCK, n = q->terms - t0 < BLOCK ? q->terms - t0 : BLOCK;
  int used = n < LANES ? (int)n : LANES;
  if (W == 1) {
    _Alignas(16) uint8_t l[LANES * 8];
    fill(q, l, LANES);
    runs(q, p, t0, n, l);
    lane_tree(q, l, 1, used);
    memcpy(v, l, q->w);
    return;
  }
  _Alignas(16) uint8_t l[LANES * ROW];
  int64_t row = W * q->w;
  for (int i = 0; i < used; i++) fill(q, l + i * row, W);
  /* Along one reduced axis, lane i's terms lie LANES apart: the table adds
     each lane's in order, its accumulators in registers. */
  if (q->nr == 1) {
    int64_t st = q->rs[0];
    for (int i = 0; i < used; i++)
      q->f->column(l + i * row, at(q, p + (t0 + i) * st), LANES * st,
                   (n - i + LANES - 1) / LANES, s, W);
    lane_tree(q, l, W, used);
    memcpy(v, l, row);
    return;
  }
  /* The terms' positions step through the reduced axes as an odometer. */
  int64_t idx[NX_MAX_RANK], pos = term(q, p, t0), r = q->nr - 1;
  for (int64_t i = r, t = t0; i >= 0; i--) {
    idx[i] = t % q->re[i];
    t /= q->re[i];
  }
  for (int64_t t = t0; t < t0 + n; t++) {
    q->f->combine(l + (t & (LANES - 1)) * row, at(q, pos), s, W);
    int64_t i = r;
    pos += q->rs[i];
    while (++idx[i] == q->re[i] && i > 0) {
      pos -= q->re[i] * q->rs[i];
      idx[i--] = 0;
      pos += q->rs[i];
    }
  }
  lane_tree(q, l, W, used);
  memcpy(v, l, row);
}

/* The value of blocks [b0, b1) of the unit's outputs, by the counter, into
   [v]: the identity for no block. */
static void blocks(const fold *q, int64_t p, int64_t s, int64_t W, int64_t b0,
                   int64_t b1, uint8_t *v) {
  _Alignas(16) uint8_t stack[LEVELS][ROW];
  _Alignas(16) uint8_t run[RUN * 8];
  int top = 0;
  int64_t row = W * q->w;
  /* One output's full blocks of contiguous terms take the table's blocks,
     RUN at a time. */
  int64_t full = W == 1 && q->nr == 1 && q->rs[0] == 1 ? q->terms / BLOCK : 0;
  for (int64_t b = b0; b < b1;) {
    int64_t n = (b1 < full ? b1 : full) - b;
    if (n > RUN) n = RUN;
    if (n > 0)
      q->f->blocks(at(q, p + b * BLOCK), n, q->id, run);
    else
      n = 1;
    for (int64_t i = 0; i < n; i++, b++) {
      if (b < full) memcpy(stack[top++], run + i * q->w, q->w);
      else block(q, p, s, W, b, stack[top++]);
      for (int64_t k = b - b0 + 1; (k & 1) == 0; k >>= 1, top--)
        q->f->combine(stack[top - 2], stack[top - 1], 1, W);
    }
  }
  for (; top > 1; top--) q->f->combine(stack[top - 2], stack[top - 1], 1, W);
  if (top == 0) fill(q, v, W);
  else memcpy(v, stack[0], row);
}

/* Units */

/* Unit [u]'s outputs: the first one's positions in the operand and the
   destination, how many, and the operand's and destination's steps from
   one to the next. Units run outer index, then chunk, then group. */
typedef struct {
  int64_t p, d, W, s, sd, g;
} unit;

static unit unit_of(const fold *q, int64_t u) {
  unit x = {q->x0, q->d0, 1, 0, 0, u % q->groups};
  u /= q->groups;
  int last = q->nk - 1;
  if (q->row > 1) {
    int64_t c = u % q->chunks;
    u /= q->chunks;
    x.s = q->ks[last];
    x.sd = q->kd[last];
    x.W = q->ke[last] - c * q->row < q->row ? q->ke[last] - c * q->row : q->row;
    x.p += c * q->row * x.s;
    x.d += c * q->row * x.sd;
    last--;
  }
  for (int i = last; i >= 0; i--) {
    x.p += u % q->ke[i] * q->ks[i];
    x.d += u % q->ke[i] * q->kd[i];
    u /= q->ke[i];
  }
  return x;
}

/* Stores the unit's values [v], each NaN replaced by its first NaN term. */
static void store(const fold *q, const unit *x, const uint8_t *v) {
  uint8_t *d = q->a[0].base + x->d * q->w;
  if (x->sd == 1) memcpy(d, v, x->W * q->w);
  else
    for (int64_t j = 0; j < x->W; j++)
      memcpy(d + j * x->sd * q->w, v + j * q->w, q->w);
  if (!q->nan || !any_nan(q, v, x->W)) return;
  for (int64_t j = 0; j < x->W; j++) {
    if (!is_nan(q, v + j * q->w)) continue;
    int64_t p = x->p + j * x->s, t = 0;
    while (t < q->terms && !is_nan(q, at(q, term(q, p, t)))) t++;
    if (t < q->terms) memcpy(d + j * x->sd * q->w, at(q, term(q, p, t)), q->w);
  }
}

/* Outputs of a lane or fewer each, one block: their terms' positions from
   an output's first are the same for every output, and the table folds
   them output by output, its lanes in registers. */
static void few(const fold *q, const unit *x) {
  int64_t off[LANES];
  for (int64_t t = 0; t < q->terms; t++) off[t] = term(q, 0, t);
  uint8_t *d = q->a[0].base + x->d * q->w;
  q->f->few(at(q, x->p), off, (int)q->terms, x->s, x->W, q->id, d, x->sd);
  if (!q->nan || (x->sd == 1 && !any_nan(q, d, x->W))) return;
  for (int64_t j = 0; j < x->W; j++) {
    uint8_t *e = d + j * x->sd * q->w;
    if (!is_nan(q, e)) continue;
    int64_t t = 0;
    while (t < q->terms && !is_nan(q, at(q, x->p + j * x->s + off[t]))) t++;
    if (t < q->terms) memcpy(e, at(q, x->p + j * x->s + off[t]), q->w);
  }
}

static void reduce_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const fold *q = ctx;
  _Alignas(16) uint8_t v[ROW];
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    if (q->terms <= LANES) {
      few(q, &x);
      continue;
    }
    int64_t b0 = x.g * q->group, b1 = b0 + q->group;
    if (b1 > q->blocks) b1 = q->blocks;
    if (q->groups == 1) {
      blocks(q, x.p, x.s, x.W, b0, b1, v);
      store(q, &x, v);
    } else
      blocks(q, x.p, x.s, x.W, b0, b1, q->scratch + u * q->row * q->w);
  }
}

/* Combines each unit's groups by the counter and stores them. */
static void finish_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const fold *q = ctx;
  int64_t row = q->row * q->w;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u * q->groups);
    _Alignas(16) uint8_t stack[LEVELS][ROW];
    int top = 0;
    for (int64_t g = 0; g < q->groups; g++) {
      memcpy(stack[top++], q->scratch + (u * q->groups + g) * row, row);
      for (int64_t k = g + 1; (k & 1) == 0; k >>= 1, top--)
        q->f->combine(stack[top - 2], stack[top - 1], 1, x.W);
    }
    for (; top > 1; top--) q->f->combine(stack[top - 2], stack[top - 1], 1, x.W);
    store(q, &x, stack[0]);
  }
}

static int64_t magnitude(int64_t x) { return x < 0 ? -x : x; }

/* Runs [body] over [total] units on at most [threads] threads, moving
   [bytes]. */
static void job(int64_t total, int64_t bytes, int threads, rig_pool_body body,
                void *ctx) {
  if (threads == 1) body(0, total, 0, ctx);
  else nx_cpu_job(total, bytes, bytes, body, ctx);
}

/* Adds the axis of extent [e] and steps [s] (operand) and [d] (destination)
   to the kept axes, if it has more than one element. */
static void keep(fold *q, int64_t e, int64_t s, int64_t d) {
  if (e <= 1) return;
  q->ke[q->nk] = e;
  q->ks[q->nk] = s;
  q->kd[q->nk++] = d;
}

/* Picks the path for [q]'s axes and lays out its units. The kept axis that
   steps least through the operand becomes the row axis, last, where it
   steps less than every reduced axis or the outputs have few terms. A
   scan's slices otherwise run SLICES side by side along that axis, so that
   as many running sums are in flight. */
static void choose(fold *q, int scan) {
  int64_t outputs = 1;
  for (int i = 0; i < q->nk; i++) outputs *= q->ke[i];
  int least = -1;
  int64_t kept = INT64_MAX, red = INT64_MAX;
  for (int i = 0; i < q->nk; i++)
    if (magnitude(q->ks[i]) < kept) {
      least = i;
      kept = magnitude(q->ks[i]);
    }
  for (int i = 0; i < q->nr; i++)
    if (q->re[i] > 1 && magnitude(q->rs[i]) < red) red = magnitude(q->rs[i]);
  q->row = 1;
  q->chunks = 1;
  q->outer = outputs;
  q->blocks = (q->terms + BLOCK - 1) / BLOCK;
  int rows = least >= 0 && (kept < red || q->terms < FEW);
  if (least < 0 || (!rows && !scan)) return;
  int64_t e = q->ke[least], s = q->ks[least], d = q->kd[least];
  for (int i = least; i < q->nk - 1; i++) {
    q->ke[i] = q->ke[i + 1];
    q->ks[i] = q->ks[i + 1];
    q->kd[i] = q->kd[i + 1];
  }
  q->ke[q->nk - 1] = e;
  q->ks[q->nk - 1] = s;
  q->kd[q->nk - 1] = d;
  q->slices = !rows;
  int64_t most = q->slices ? SLICES : ROW / q->w;
  q->row = most < e ? most : e;
  q->chunks = (e + q->row - 1) / q->row;
  q->outer = outputs / e;
}

/* Lays out a reduction from the coalesced loop [l] over the destination and
   the operand, the destination stepping 0 along the reduced axes. Where
   units are few and blocks many, each unit takes a group of blocks, its
   size from the shape alone. */
static void plan_reduce(fold *q, const nx_loop *l) {
  q->nr = q->nk = 0;
  for (int i = 0; i < l->rank; i++)
    if (l->step[0][i] != 0) keep(q, l->extent[i], l->step[1][i], l->step[0][i]);
    else if (l->extent[i] > 1) {
      q->re[q->nr] = l->extent[i];
      q->rs[q->nr++] = l->step[1][i];
    }
  if (q->nr == 0) {
    q->re[0] = 1;
    q->rs[0] = 0;
    q->nr = 1;
  }
  q->x0 = l->first[1];
  q->d0 = l->first[0];
  choose(q, 0);
  int64_t units = q->outer * q->chunks;
  q->group = q->blocks > 0 ? q->blocks : 1;
  if (units < UNITS && q->blocks > 1) {
    int64_t want = (UNITS + units - 1) / units;
    int64_t per = (q->blocks + want - 1) / want;
    for (q->group = 1; q->group < per;) q->group *= 2;
  }
  q->groups = q->blocks > 0 ? (q->blocks + q->group - 1) / q->group : 1;
}

/* The destination's elements, all set to the identity: a reduction of no
   term. */
static void fill_dst(fold *q) {
  const nx_array *d = &q->a[0];
  int64_t extent[NX_MAX_RANK], step[1][NX_MAX_RANK];
  for (int i = 0; i < d->rank; i++) {
    extent[i] = d->dim[i];
    step[0][i] = d->dim[d->rank + i];
  }
  int r = nx_coalesce_dims(1, d->rank, extent, step);
  int64_t n = 1;
  for (int i = 0; i < r; i++) n *= extent[i];
  for (int64_t k = 0; k < n; k++) {
    int64_t p = d->offset, rest = k;
    for (int i = r - 1; i >= 0; i--) {
      p += rest % extent[i] * step[0][i];
      rest /= extent[i];
    }
    memcpy(d->base + p * q->w, q->id, q->w);
  }
}

/* Scans */

/* Chunk [c]'s total of the unit's slices, into [v]: slice by slice where
   they run side by side. */
static void total(const fold *q, const unit *x, int64_t c, uint8_t *v) {
  int64_t b0 = c * (SCAN_CHUNK / BLOCK), b1 = b0 + SCAN_CHUNK / BLOCK;
  if (b1 > q->blocks) b1 = q->blocks;
  if (!q->slices) {
    blocks(q, x->p, x->s, x->W, b0, b1, v);
    return;
  }
  for (int64_t j = 0; j < x->W; j++)
    blocks(q, x->p + j * x->s, 0, 1, b0, b1, v + j * q->w);
}

/* Chunk [c]'s results of the unit's slices, from the carry [a], which it
   leaves holding the chunk's last results: side by side, or a row of
   outputs at a time. */
static void rescan(const fold *q, const unit *x, int64_t c, uint8_t *a) {
  int64_t t0 = c * SCAN_CHUNK;
  int64_t n = q->terms - t0 < SCAN_CHUNK ? q->terms - t0 : SCAN_CHUNK;
  uint8_t *y = q->a[0].base + x->d * q->w;
  if (q->slices || x->W == 1) {
    q->f->scan(a, at(q, x->p + t0 * q->rs[0]), q->rs[0], x->s,
               y + t0 * q->rd * q->w, q->rd, x->sd, n, (int)x->W);
    return;
  }
  for (int64_t t = t0; t < t0 + n; t++) {
    uint8_t *row = y + t * q->rd * q->w;
    q->f->combine(a, at(q, x->p + t * q->rs[0]), x->s, x->W);
    if (x->sd == 1) memcpy(row, a, x->W * q->w);
    else
      for (int64_t j = 0; j < x->W; j++)
        memcpy(row + j * x->sd * q->w, a + j * q->w, q->w);
  }
}

/* Each of the unit's slices whose last result is NaN takes its first NaN
   term from that term on. */
static void settle(const fold *q, const unit *x) {
  if (!q->nan || q->terms == 0) return;
  uint8_t *y = q->a[0].base;
  for (int64_t j = 0; j < x->W; j++) {
    int64_t p = x->p + j * x->s, d = x->d + j * x->sd;
    if (!is_nan(q, y + (d + (q->terms - 1) * q->rd) * q->w)) continue;
    int64_t k = 0;
    while (k < q->terms && !is_nan(q, at(q, p + k * q->rs[0]))) k++;
    for (int64_t t = k; t < q->terms; t++)
      memcpy(y + (d + t * q->rd) * q->w, at(q, p + k * q->rs[0]), q->w);
  }
}

/* One pass per unit: each chunk's results from the carry, which then takes
   the chunk's total. The last chunk's total carries nowhere. */
static void scan_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const fold *q = ctx;
  _Alignas(16) uint8_t carry[ROW], a[ROW], v[ROW];
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    fill(q, carry, x.W);
    for (int64_t c = 0; c * SCAN_CHUNK < q->terms; c++) {
      memcpy(a, carry, x.W * q->w);
      rescan(q, &x, c, a);
      if ((c + 1) * SCAN_CHUNK >= q->terms) break;
      total(q, &x, c, v);
      q->f->combine(carry, v, 1, x.W);
    }
    settle(q, &x);
  }
}

/* Few slices, long axes: every chunk's total in one job, each unit's
   carries in order, every chunk's results in a second job. A unit's group
   is its chunk. */
static void total_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const fold *q = ctx;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    if (x.g < q->groups - 1) total(q, &x, x.g, q->scratch + u * q->row * q->w);
  }
}

static void rescan_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const fold *q = ctx;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    rescan(q, &x, x.g, q->scratch + u * q->row * q->w);
  }
}

/* One slice's chunks four at a time, side by side, where a unit is one
   slice: their carries lie side by side in the scratch. A short last chunk
   runs alone. */
static void rescan_quads(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const fold *q = ctx;
  int64_t quads = (q->groups + 3) / 4;
  uint8_t *y = q->a[0].base;
  for (int64_t v = lo; v < hi; v++) {
    int64_t u = v / quads, c = v % quads * 4;
    unit x = unit_of(q, u * q->groups);
    uint8_t *a = q->scratch + (u * q->groups + c) * q->w;
    int k = q->groups - c < 4 ? (int)(q->groups - c) : 4;
    int full = (c + k) * SCAN_CHUNK <= q->terms ? k : k - 1;
    int64_t t0 = c * SCAN_CHUNK;
    if (full > 0)
      q->f->scan(a, at(q, x.p + t0 * q->rs[0]), q->rs[0], SCAN_CHUNK * q->rs[0],
                 y + (x.d + t0 * q->rd) * q->w, q->rd, SCAN_CHUNK * q->rd,
                 SCAN_CHUNK, full);
    if (full < k) rescan(q, &x, c + full, a + full * q->w);
  }
}

static void carries(const fold *q, int64_t units) {
  int64_t row = q->row * q->w;
  _Alignas(16) uint8_t carry[ROW], v[ROW];
  for (int64_t u = 0; u < units; u++) {
    unit x = unit_of(q, u * q->groups);
    fill(q, carry, x.W);
    for (int64_t c = 0; c < q->groups; c++) {
      uint8_t *s = q->scratch + (u * q->groups + c) * row;
      memcpy(v, s, row);
      memcpy(s, carry, row);
      if (c < q->groups - 1) q->f->combine(carry, v, 1, x.W);
    }
  }
}

/* Lays out a scan along the axis [axis] of the destination [d] and the
   operand [x]: the axes before it and those after it coalesce apart. */
static void plan_scan(fold *q, const nx_array *d, const nx_array *x, int axis) {
  int r = x->rank;
  q->nr = 1;
  q->re[0] = x->dim[axis];
  q->rs[0] = x->dim[r + axis];
  q->rd = d->dim[r + axis];
  q->x0 = x->offset;
  q->d0 = d->offset;
  q->nk = 0;
  int parts[2][2] = {{0, axis}, {axis + 1, r}};
  for (int k = 0; k < 2; k++) {
    int64_t extent[NX_MAX_RANK], step[2][NX_MAX_RANK];
    int n = parts[k][1] - parts[k][0];
    for (int i = 0; i < n; i++) {
      int a = parts[k][0] + i;
      extent[i] = x->dim[a];
      step[0][i] = d->dim[r + a];
      step[1][i] = x->dim[r + a];
    }
    n = nx_coalesce_dims(2, n, extent, step);
    for (int i = 0; i < n; i++) keep(q, extent[i], step[1][i], step[0][i]);
  }
  choose(q, 1);
  int64_t chunks = (q->terms + SCAN_CHUNK - 1) / SCAN_CHUNK;
  q->groups = q->outer * q->chunks < UNITS && chunks > 1 ? chunks : 1;
}

/* Entries */

/* The core case of the descriptor [s]: its monoid, or -1 for a case nx.cpu
   declines. It reads [s], its axes into [axes], before anything that can
   move it. */
static int core(value s, int family, int dt, int *axes, int *naxes) {
  const nx_spec_loop *m = (const nx_spec_loop *)String_val(s);
  if (m->family != family || m->nloads != 1 || m->nreductions != 1) return -1;
  if (m->loads[0] != 0) return -1;
  const nx_spec_reduction *r = nx_spec_loop_reductions(m);
  const nx_prog *p = nx_spec_loop_prog(m);
  if (r->kind > NX_MIN || r->output != 0 || r->dtype != dt) return -1;
  if (!nx_prog_is_operand(p) || nx_prog_ins(p)[0] != dt) return -1;
  if (nx_cpu_table->fold[r->kind][dt].lanes == NULL) return -1;
  *naxes = m->naxes;
  for (int i = 0; i < m->naxes; i++) axes[i] = nx_spec_loop_axes(m)[i];
  return r->kind;
}

/* Whether the destination's shape [y] of rank [yr] is the result's for an
   operand of shape [x] of rank [xr]: [x] without [axes] for a reduction,
   [x] for a scan along its axis. Sets the terms per output and the
   outputs. */
static int fits(int scan, const int64_t *x, int xr, const int64_t *y, int yr,
                const int *axes, int naxes, int64_t *terms, int64_t *outputs) {
  int j = 0, k = 0;
  *terms = *outputs = 1;
  for (int i = 0; i < xr; i++) {
    if (j < naxes && axes[j] == i) {
      *terms *= x[i];
      j++;
      if (!scan) continue;
    } else
      *outputs *= x[i];
    if (k >= yr || y[k++] != x[i]) return 0;
  }
  return j == naxes && k == yr;
}

/* Reduces or scans, as [family] says, the operand in [ops] into the
   destination in [dsts]: on one thread where [threads] is 1, on as many as
   the job gives where it is 0. */
static value run(value s, value dsts, value ops, int family, int threads) {
  CAMLparam3(s, dsts, ops);
  int scan = family == NX_SPEC_SCAN;
  if (Wosize_val(dsts) != 1 || Wosize_val(ops) != 1)
    CAMLreturn(Val_int(NX_DECLINED));
  value vd = Field(Field(dsts, 0), 0), vx = Field(Field(ops, 0), 0);
  int dt = nx_array_dtype(vx), axes[NX_MAX_RANK], naxes;
  int monoid = core(s, family, dt, axes, &naxes);
  if (monoid < 0) CAMLreturn(Val_int(NX_DECLINED));
  /* The shapes, before the door; an extreme has a term for each output. */
  int64_t x[2 * NX_MAX_RANK], y[2 * NX_MAX_RANK], off, terms, outputs;
  int xr = nx_array_layout(vx, x, &off), yr = nx_array_layout(vd, y, &off);
  if (!fits(scan, x, xr, y, yr, axes, naxes, &terms, &outputs))
    CAMLreturn(Val_int(NX_SHAPE));
  if (!scan && terms == 0 && outputs > 0 && monoid >= NX_MAX)
    CAMLreturn(Val_int(NX_SHAPE));
  nx_operand in[2] = {{vd, dt, 1}, {vx, dt, 0}};
  nx_array a[2];
  int e = nx_read(2, in, a);
  if (e) CAMLreturn(Val_int(e));
  fold q = {.f = &nx_cpu_table->fold[monoid][dt],
            .dtype = dt,
            .w = nx_cpu_width(dt),
            .nan = dt == NX_FLOAT32 || dt == NX_FLOAT64,
            .a = a,
            .terms = terms};
  identity(monoid, dt, q.id);
  int64_t bytes = terms * outputs * q.w;
  if (outputs > 0 && terms == 0 && !scan) fill_dst(&q);
  if (outputs > 0 && terms > 0) {
    if (scan) plan_scan(&q, &a[0], &a[1], axes[0]);
    else {
      nx_loop l = {.rank = xr, .first = {a[0].offset, a[1].offset}};
      for (int i = 0, k = 0, j = 0; i < xr; i++) {
        l.extent[i] = a[1].dim[i];
        l.step[1][i] = a[1].dim[xr + i];
        int reduced = j < naxes && axes[j] == i;
        j += reduced;
        l.step[0][i] = reduced ? 0 : a[0].dim[yr + k++];
      }
      l.rank = nx_coalesce_dims(2, xr, l.extent, l.step);
      plan_reduce(&q, &l);
    }
    int64_t units = q.outer * q.chunks;
    if (q.groups > 1) {
      q.scratch = malloc((size_t)(units * q.groups * q.row * q.w));
      if (q.scratch == NULL) {
        nx_done(2, a);
        caml_raise_out_of_memory();
      }
    }
    if (q.groups == 1)
      job(units, bytes, threads, scan ? scan_units : reduce_units, &q);
    else if (!scan) {
      job(units * q.groups, bytes, threads, reduce_units, &q);
      job(units, units * q.groups * q.row * q.w, threads, finish_units, &q);
    } else {
      job(units * q.groups, bytes, threads, total_units, &q);
      carries(&q, units);
      if (q.row == 1)
        job(units * ((q.groups + 3) / 4), bytes, threads, rescan_quads, &q);
      else
        job(units * q.groups, bytes, threads, rescan_units, &q);
      for (int64_t u = 0; u < units; u++) {
        unit x = unit_of(&q, u * q.groups);
        settle(&q, &x);
      }
    }
    free(q.scratch);
  }
  nx_done(2, a);
  CAMLreturn(Val_int(NX_OK));
}

value nx_cpu_reduce_on(value s, value dsts, value ops, int threads) {
  return run(s, dsts, ops, NX_SPEC_REDUCE, threads);
}

value nx_cpu_scan_on(value s, value dsts, value ops, int threads) {
  return run(s, dsts, ops, NX_SPEC_SCAN, threads);
}

value nx_cpu_reduce(value s, value dsts, value ops) {
  return nx_cpu_reduce_on(s, dsts, ops, 0);
}

value nx_cpu_scan(value s, value dsts, value ops) {
  return nx_cpu_scan_on(s, dsts, ops, 0);
}
