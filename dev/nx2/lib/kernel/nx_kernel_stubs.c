/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <caml/mlvalues.h>

#include "nx_array.h"
#include "nx_contract_view_fill.h"
#include "nx_prog.h"

_Static_assert(NX_OP1_COUNT == 31 && NX_OP2_COUNT == 18 && NX_OP3_COUNT == 2,
               "one code per kind");

/* The bits nx_dtype.h's store of [x] writes into an element of the float
   dtype [dt] of at most 16 bits. */
intnat nx_kernel_narrow_bits(intnat dt, double x) {
  return (intnat)nx_double_to_bits((int)dt, x);
}

value nx_kernel_narrow_bits_byte(value dt, value x) {
  return Val_long(nx_kernel_narrow_bits(Long_val(dt), Double_val(x)));
}

_Static_assert(NX_VIEW_DST == 3 && NX_VIEW_CONTRACTED == 3,
               "operand and axis indices");

/* nx_contract_view_fill, in one pass over the pairs [vbatch] and
   [vcontracting] and the layouts of [y], [a], [b] and the init [i], an
   option, into [w]: the paired axes marked in a mask per side, each
   operand's axes of each group gathered, every group's extents checked
   before any merges, then each group coalesced (nx_coalesce_dims) into
   its runs and written. It reads each pair once, so another domain's write
   to the arrays cannot change what it checked. */
enum { A = NX_VIEW_A, B = NX_VIEW_B, I = NX_VIEW_INIT, Y = NX_VIEW_DST };
enum {
  BATCH = NX_VIEW_BATCH,
  ROW = NX_VIEW_ROW,
  COLUMN = NX_VIEW_COLUMN,
  CONTRACTED = NX_VIEW_CONTRACTED
};

typedef nx_layout layout;

/* Groups the [count] axes of group [g] over the [n] operands [ops], whose
   axes are [axes][op][g], into its runs in [r] (nx_contract_view_fill.h):
   one run where there are at most one axis, as nx_coalesce_dims writes it,
   an absent or single axis of extent 1 stepping 0, as does an empty one.
   The operands not in [ops] step 0. */
static void group(nx_contract_runs *r, int g, int count, int n,
                  const int *ops, const layout *l,
                  int8_t axes[4][4][NX_MAX_RANK]) {
  int64_t ext[NX_MAX_RANK], st[4][NX_MAX_RANK];
  int runs = 1;
  if (count <= 1) {
    ext[0] = count ? l[ops[0]].dim[axes[ops[0]][g][0]] : 1;
    for (int p = 0; p < n; p++) {
      const layout *lp = &l[ops[p]];
      st[p][0] = ext[0] > 1 ? lp->dim[lp->rank + axes[ops[p]][g][0]] : 0;
    }
  } else {
    for (int k = 0; k < count; k++) {
      ext[k] = l[ops[0]].dim[axes[ops[0]][g][k]];
      for (int p = 0; p < n; p++) {
        const layout *lp = &l[ops[p]];
        st[p][k] = lp->dim[lp->rank + axes[ops[p]][g][k]];
      }
    }
    runs = nx_coalesce_dims(n, count, ext, st);
  }
  r->runs[g] = runs;
  for (int o = A; o <= Y; o++)
    for (int k = 0; k < runs; k++) r->stride[o][g][k] = 0;
  for (int k = 0; k < runs; k++) {
    r->extent[g][k] = ext[k];
    for (int p = 0; p < n; p++) r->stride[ops[p]][g][k] = st[p][k];
  }
}

/* Whether the operands [ops] agree on group [g]'s [count] extents. */
static int fits(int g, int count, int n, const int *ops, const layout *l,
                int8_t axes[4][4][NX_MAX_RANK]) {
  for (int p = 1; p < n; p++)
    for (int k = 0; k < count; k++)
      if (l[ops[p]].dim[axes[ops[p]][g][k]] !=
          l[ops[0]].dim[axes[ops[0]][g][k]])
        return 0;
  return 1;
}

/* Whether operand [o] spans group [g]: batch all, rows a, init and dst,
   columns b, init and dst, contracted a and b. */
static const int spans[4][4] = {
    [A] = {[BATCH] = 1, [ROW] = 1, [CONTRACTED] = 1},
    [B] = {[BATCH] = 1, [COLUMN] = 1, [CONTRACTED] = 1},
    [I] = {[BATCH] = 1, [ROW] = 1, [COLUMN] = 1},
    [Y] = {[BATCH] = 1, [ROW] = 1, [COLUMN] = 1}};

/* The view where each group holds [count[g]] of at most one axis, as a
   product of matrices' groups do: each group's extent is its first
   operand's (b's for the columns, a's for the others), which every other
   operand's must equal, and each operand steps along its axis, 0 over at
   most one element, as fits and group check and write them, without
   their lists. On the M1 Max it took an 8³ product's fill from 38 to 18
   ns. */
static int single(nx_contract_view *w, const layout *l, int init,
                  const int *count, int8_t axes[4][4][NX_MAX_RANK]) {
  int64_t ext[4];
  for (int g = BATCH; g <= CONTRACTED; g++) {
    int o = g == COLUMN ? B : A;
    ext[g] = count[g] ? l[o].dim[axes[o][g][0]] : 1;
  }
  for (int o = A; o <= Y; o++)
    for (int g = BATCH; g <= CONTRACTED; g++)
      if ((o != I || init) && count[g] && spans[o][g] &&
          l[o].dim[axes[o][g][0]] != ext[g])
        return NX_SHAPE;
  memset(w, 0, sizeof *w);
  for (int o = A; o <= Y; o++) {
    if (o == I && !init) continue;
    w->offset[o] = l[o].offset;
    for (int g = BATCH; g <= CONTRACTED; g++)
      if (count[g] && spans[o][g] && ext[g] > 1)
        w->stride[o][g] = l[o].dim[l[o].rank + axes[o][g][0]];
  }
  for (int g = BATCH; g <= CONTRACTED; g++) w->extent[g] = ext[g];
  return NX_OK;
}

/* A contraction's pairs read and checked: the layouts, each operand's
   axes by group, the axes in each group and whether it has an init. */
typedef struct {
  layout l[4];
  int8_t axes[4][4][NX_MAX_RANK];
  int count[4];
  int init;
} pairs;

/* Checks the [nb] batch pairs then the [nc] contracting pairs [ps], a's
   axis then b's, against the layouts in [q], and gathers each operand's
   axes by group: NX_OK, or NX_SHAPE where they do not fit the layouts'
   ranks or an axis is in two pairs. */
static int pairs_of(pairs *q, int nb, int nc, const int (*ps)[2]) {
  layout *l = q->l;
  if (nb + nc > l[A].rank || nb + nc > l[B].rank) return NX_SHAPE;
  int fa = l[A].rank - nb - nc, fb = l[B].rank - nb - nc;
  int ry = nb + fa + fb;
  uint64_t na = 0, nbm = 0;
  for (int k = 0; k < nb + nc; k++) {
    int ia = ps[k][0], ib = ps[k][1];
    if (ia < 0 || ia >= l[A].rank || ib < 0 || ib >= l[B].rank ||
        na >> ia & 1 || nbm >> ib & 1)
      return NX_SHAPE;
    na |= (uint64_t)1 << ia;
    nbm |= (uint64_t)1 << ib;
    int g = k < nb ? BATCH : CONTRACTED, at = k < nb ? k : k - nb;
    q->axes[A][g][at] = (int8_t)ia;
    q->axes[B][g][at] = (int8_t)ib;
  }
  if (l[Y].rank != ry || (q->init && l[I].rank != ry)) return NX_SHAPE;
  for (int ax = 0, n = 0; ax < l[A].rank; ax++)
    if (!(na >> ax & 1)) q->axes[A][ROW][n++] = ax;
  for (int ax = 0, n = 0; ax < l[B].rank; ax++)
    if (!(nbm >> ax & 1)) q->axes[B][COLUMN][n++] = ax;
  for (int o = I; o <= Y; o++) {
    for (int k = 0; k < nb; k++) q->axes[o][BATCH][k] = k;
    for (int k = 0; k < fa; k++) q->axes[o][ROW][k] = nb + k;
    for (int k = 0; k < fb; k++) q->axes[o][COLUMN][k] = nb + fa + k;
  }
  q->count[BATCH] = nb;
  q->count[ROW] = fa;
  q->count[COLUMN] = fb;
  q->count[CONTRACTED] = nc;
  return NX_OK;
}

/* Reads the pairs [vbatch] and [vcontracting] and the layouts of [y], [a],
   [b] and the init [i] into [q] (pairs_of). */
static int read_pairs(pairs *q, value vbatch, value vcontracting, value y,
                      value a, value b, value i) {
  int nb = (int)Wosize_val(vbatch), nc = (int)Wosize_val(vcontracting);
  layout *l = q->l;
  q->init = Is_block(i);
  l[A].rank = nx_array_layout(a, l[A].dim, &l[A].offset);
  l[B].rank = nx_array_layout(b, l[B].dim, &l[B].offset);
  l[Y].rank = nx_array_layout(y, l[Y].dim, &l[Y].offset);
  l[I].rank =
      q->init ? nx_array_layout(Field(i, 0), l[I].dim, &l[I].offset) : 0;
  if (nb + nc > l[A].rank || nb + nc > l[B].rank) return NX_SHAPE;
  int ps[NX_MAX_RANK][2];
  for (int k = 0; k < nb + nc; k++) {
    value p = Field(k < nb ? vbatch : vcontracting, k < nb ? k : k - nb);
    intnat ia = Long_val(Field(p, 0)), ib = Long_val(Field(p, 1));
    /* Out of an int's range is out of the ranks'. */
    ps[k][0] = ia < 0 || ia > NX_MAX_RANK ? -1 : (int)ia;
    ps[k][1] = ib < 0 || ib > NX_MAX_RANK ? -1 : (int)ib;
  }
  return pairs_of(q, nb, nc, ps);
}

/* Groups [q] into [r]: NX_OK, or NX_SHAPE where the operands' extents
   differ within a group. Each group's operands: batch over all, rows over
   a, init and dst, columns over b, init and dst, contracted over a and
   b. */
static int runs_of(nx_contract_runs *r, pairs *q) {
  int batch[4] = {A, B, Y, I}, row[3] = {A, Y, I}, column[3] = {B, Y, I};
  int contracted[2] = {A, B};
  int nbatch = 3 + q->init, nrow = 2 + q->init;
  const int *c = q->count;
  /* Every group fits before any merges: a misfit is refused even behind a
     group that does not merge. */
  if (!fits(BATCH, c[BATCH], nbatch, batch, q->l, q->axes) ||
      !fits(ROW, c[ROW], nrow, row, q->l, q->axes) ||
      !fits(COLUMN, c[COLUMN], nrow, column, q->l, q->axes) ||
      !fits(CONTRACTED, c[CONTRACTED], 2, contracted, q->l, q->axes))
    return NX_SHAPE;
  group(r, BATCH, c[BATCH], nbatch, batch, q->l, q->axes);
  group(r, ROW, c[ROW], nrow, row, q->l, q->axes);
  group(r, COLUMN, c[COLUMN], nrow, column, q->l, q->axes);
  group(r, CONTRACTED, c[CONTRACTED], 2, contracted, q->l, q->axes);
  for (int o = A; o <= Y; o++)
    r->offset[o] = o == I && !q->init ? 0 : q->l[o].offset;
  return NX_OK;
}

/* The view or the runs of the pairs read into [q] (nx_contract_fill). */
static int fill_of(nx_contract_view *w, nx_contract_runs *r, pairs *q) {
  const int *c = q->count;
  if (c[BATCH] <= 1 && c[ROW] <= 1 && c[COLUMN] <= 1 && c[CONTRACTED] <= 1)
    return single(w, q->l, q->init, c, q->axes);
  int e;
  if ((e = runs_of(r, q))) return e;
  for (int g = BATCH; g <= CONTRACTED; g++)
    if (r->runs[g] != 1) return NX_DECLINED;
  memset(w, 0, sizeof *w);
  for (int o = A; o <= Y; o++) {
    w->offset[o] = r->offset[o];
    for (int g = BATCH; g <= CONTRACTED; g++)
      w->stride[o][g] = r->stride[o][g][0];
  }
  for (int g = BATCH; g <= CONTRACTED; g++) w->extent[g] = r->extent[g][0];
  return NX_OK;
}

int nx_contract_fill(nx_contract_view *w, nx_contract_runs *r, value vbatch,
                     value vcontracting, value y, value a, value b,
                     value i) {
  pairs q;
  int e = read_pairs(&q, vbatch, vcontracting, y, a, b, i);
  if (e) return e;
  return fill_of(w, r, &q);
}

int nx_contract_fill_layouts(nx_contract_view *w, nx_contract_runs *r,
                             int nb, int nc, const int (*ps)[2],
                             const nx_layout *y, const nx_layout *a,
                             const nx_layout *b, const nx_layout *i) {
  pairs q;
  q.l[A] = *a;
  q.l[B] = *b;
  q.l[Y] = *y;
  q.init = i != NULL;
  q.l[I].rank = 0;
  if (i) q.l[I] = *i;
  int e = pairs_of(&q, nb, nc, ps);
  if (e) return e;
  return fill_of(w, r, &q);
}

int nx_contract_view_fill(nx_contract_view *w, value vbatch,
                          value vcontracting, value y, value a, value b,
                          value i) {
  nx_contract_runs r;
  return nx_contract_fill(w, &r, vbatch, vcontracting, y, a, b, i);
}
