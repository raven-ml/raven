/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stddef.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_array.h"
#include "nx_contract_view_fill.h"
#include "nx_prog.h"

_Static_assert(NX_OP1_COUNT == 28 && NX_OP2_COUNT == 18 && NX_OP3_COUNT == 2,
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
   before any merges, then each group coalesced (nx_coalesce_dims) and
   written. It reads each pair once, so another domain's write to the
   arrays cannot change what it checked. */
enum { A = NX_VIEW_A, B = NX_VIEW_B, I = NX_VIEW_INIT, Y = NX_VIEW_DST };
enum {
  BATCH = NX_VIEW_BATCH,
  ROW = NX_VIEW_ROW,
  COLUMN = NX_VIEW_COLUMN,
  CONTRACTED = NX_VIEW_CONTRACTED
};

typedef struct {
  int rank;
  int64_t dim[2 * NX_MAX_RANK];
  int64_t offset;
} layout;

/* Groups the [count] axes of group [g] over the [n] operands [ops], whose
   axes are [axes][op][g], into [w]; answers whether they make one run. A
   group of at most one axis is one as nx_coalesce_dims writes it: an
   absent or single axis of extent 1 steps 0, as does an empty one. */
static int group(nx_contract_view *w, int g, int count, int n, const int *ops,
                 const layout *l, int8_t axes[4][4][NX_MAX_RANK]) {
  if (count <= 1) {
    int64_t d = count ? l[ops[0]].dim[axes[ops[0]][g][0]] : 1;
    w->extent[g] = d;
    for (int p = 0; p < n; p++) {
      const layout *lp = &l[ops[p]];
      int64_t s = d > 1 ? lp->dim[lp->rank + axes[ops[p]][g][0]] : 0;
      w->stride[ops[p]][g] = s;
    }
    return 1;
  }
  int64_t ext[NX_MAX_RANK], st[4][NX_MAX_RANK];
  for (int k = 0; k < count; k++) {
    const layout *l0 = &l[ops[0]];
    ext[k] = l0->dim[axes[ops[0]][g][k]];
    for (int p = 0; p < n; p++) {
      const layout *lp = &l[ops[p]];
      st[p][k] = lp->dim[lp->rank + axes[ops[p]][g][k]];
    }
  }
  if (nx_coalesce_dims(n, count, ext, st) != 1) return 0;
  w->extent[g] = ext[0];
  for (int p = 0; p < n; p++) w->stride[ops[p]][g] = st[p][0];
  return 1;
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

int nx_contract_view_fill(nx_contract_view *w, value vbatch,
                          value vcontracting, value y, value a, value b,
                          value i) {
  int nb = (int)Wosize_val(vbatch), nc = (int)Wosize_val(vcontracting);
  int init = Is_block(i);
  layout l[4];
  l[A].rank = nx_array_layout(a, l[A].dim, &l[A].offset);
  l[B].rank = nx_array_layout(b, l[B].dim, &l[B].offset);
  l[Y].rank = nx_array_layout(y, l[Y].dim, &l[Y].offset);
  l[I].rank = init ? nx_array_layout(Field(i, 0), l[I].dim, &l[I].offset) : 0;
  if (nb + nc > l[A].rank || nb + nc > l[B].rank) return NX_SHAPE;
  int fa = l[A].rank - nb - nc, fb = l[B].rank - nb - nc;
  int ry = nb + fa + fb;
  uint64_t na = 0, nbm = 0;
  int8_t axes[4][4][NX_MAX_RANK];
  for (int k = 0; k < nb + nc; k++) {
    value p = Field(k < nb ? vbatch : vcontracting, k < nb ? k : k - nb);
    intnat ia = Long_val(Field(p, 0)), ib = Long_val(Field(p, 1));
    if (ia < 0 || ia >= l[A].rank || ib < 0 || ib >= l[B].rank ||
        na >> ia & 1 || nbm >> ib & 1)
      return NX_SHAPE;
    na |= (uint64_t)1 << ia;
    nbm |= (uint64_t)1 << ib;
    int g = k < nb ? BATCH : CONTRACTED, at = k < nb ? k : k - nb;
    axes[A][g][at] = (int)ia;
    axes[B][g][at] = (int)ib;
  }
  if (l[Y].rank != ry || (init && l[I].rank != ry)) return NX_SHAPE;
  for (int ax = 0, n = 0; ax < l[A].rank; ax++)
    if (!(na >> ax & 1)) axes[A][ROW][n++] = ax;
  for (int ax = 0, n = 0; ax < l[B].rank; ax++)
    if (!(nbm >> ax & 1)) axes[B][COLUMN][n++] = ax;
  for (int o = I; o <= Y; o++) {
    for (int k = 0; k < nb; k++) axes[o][BATCH][k] = k;
    for (int k = 0; k < fa; k++) axes[o][ROW][k] = nb + k;
    for (int k = 0; k < fb; k++) axes[o][COLUMN][k] = nb + fa + k;
  }
  /* Each group's operands: batch over all, rows over a, init and dst,
     columns over b, init and dst, contracted over a and b. */
  int batch[4] = {A, B, Y, I}, row[3] = {A, Y, I}, column[3] = {B, Y, I};
  int contracted[2] = {A, B};
  int nbatch = 3 + init, nrow = 2 + init;
  /* Every group fits before any merges: a misfit is refused even behind a
     group that does not merge. */
  if (!fits(BATCH, nb, nbatch, batch, l, axes) ||
      !fits(ROW, fa, nrow, row, l, axes) ||
      !fits(COLUMN, fb, nrow, column, l, axes) ||
      !fits(CONTRACTED, nc, 2, contracted, l, axes))
    return NX_SHAPE;
  /* What an operand lacks reads 0: its axes, and an absent init. */
  memset(w, 0, sizeof *w);
  w->offset[A] = l[A].offset;
  w->offset[B] = l[B].offset;
  w->offset[Y] = l[Y].offset;
  if (init) w->offset[I] = l[I].offset;
  int grouped = group(w, BATCH, nb, nbatch, batch, l, axes) &&
                group(w, ROW, fa, nrow, row, l, axes) &&
                group(w, COLUMN, fb, nrow, column, l, axes) &&
                group(w, CONTRACTED, nc, 2, contracted, l, axes);
  return grouped ? NX_OK : NX_DECLINED;
}

/* Contract_view.v: Ok of the view of the pairs [vbatch] and
   [vcontracting], the result [y], the init [i], an option, and [a] and [b],
   as an int array in nx_kernel.ml's order (the extents, the offsets, the
   strides by operand then axis, then whether it has an init), or Error of
   the answer nx_contract_view_fill gives. */
enum { VIEW_FIELDS = 25, AT_OFFSET = 4, AT_STRIDE = 8, AT_HAS_INIT = 24 };
enum { OK_TAG, ERROR_TAG };

value nx_kernel_view(value vbatch, value vcontracting, value y, value i,
                     value a, value b) {
  CAMLparam5(vbatch, vcontracting, y, i, a);
  CAMLxparam1(b);
  CAMLlocal2(v, r);
  nx_contract_view w;
  int answer = nx_contract_view_fill(&w, vbatch, vcontracting, y, a, b, i);
  if (answer != NX_OK) {
    r = caml_alloc_small(1, ERROR_TAG);
    Field(r, 0) = Val_int(answer);
    CAMLreturn(r);
  }
  v = caml_alloc_small(VIEW_FIELDS, 0);
  for (int k = 0; k < 4; k++) {
    Field(v, k) = Val_long(w.extent[k]);
    Field(v, AT_OFFSET + k) = Val_long(w.offset[k]);
    for (int x = 0; x < 4; x++)
      Field(v, AT_STRIDE + 4 * k + x) = Val_long(w.stride[k][x]);
  }
  Field(v, AT_HAS_INIT) = Val_bool(Is_block(i));
  r = caml_alloc_small(1, OK_TAG);
  Field(r, 0) = v;
  CAMLreturn(r);
}

value nx_kernel_view_byte(value *argv, int argn) {
  (void)argn;
  return nx_kernel_view(argv[0], argv[1], argv[2], argv[3], argv[4],
                        argv[5]);
}
