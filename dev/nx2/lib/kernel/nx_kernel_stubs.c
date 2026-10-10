/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stddef.h>
#include <string.h>

#include <caml/mlvalues.h>

#include "nx_array.h"
#include "nx_spec.h"

/* spec.ml writes nx_spec_contract at these byte offsets. */
_Static_assert(offsetof(nx_spec_contract, family) == 0, "at_family");
_Static_assert(offsetof(nx_spec_contract, acc) == 4, "at_acc");
_Static_assert(offsetof(nx_spec_contract, out) == 8, "at_out");
_Static_assert(offsetof(nx_spec_contract, init) == 12, "at_init");
_Static_assert(offsetof(nx_spec_contract, nbatch) == 16, "at_nbatch");
_Static_assert(offsetof(nx_spec_contract, ncontracting) == 20,
               "at_ncontracting");
_Static_assert(offsetof(nx_spec_contract, pairs) == 24, "at_pairs");
_Static_assert(sizeof(((nx_spec_contract *)0)->pairs[0]) == 8,
               "two int32 per pair");

/* prog.ml writes nx_prog at these byte offsets. */
_Static_assert(offsetof(nx_prog, nodes) == 16, "header");
_Static_assert(sizeof(nx_prog_node) == 40, "record");
_Static_assert(offsetof(nx_prog_node, bits) == 24, "at_bits");
_Static_assert(NX_OP1_COUNT == 28 && NX_OP2_COUNT == 18 && NX_OP3_COUNT == 2,
               "one code per kind");

/* spec.ml writes nx_spec_loop and nx_spec_pad at these byte offsets. */
_Static_assert(offsetof(nx_spec_loop, nloads) == 4, "at_nloads");
_Static_assert(offsetof(nx_spec_loop, naxes) == 8, "at_naxes");
_Static_assert(offsetof(nx_spec_loop, nreductions) == 12, "at_nreductions");
_Static_assert(offsetof(nx_spec_loop, at_prog) == 16, "at_prog");
_Static_assert(offsetof(nx_spec_loop, prog_len) == 20, "at_prog_len");
_Static_assert(offsetof(nx_spec_loop, loads) == 24, "at_loads");
_Static_assert(sizeof(nx_spec_reduction) == 12, "three int32 per reduction");
_Static_assert(NX_ARGMIN == 7, "one code per reduction");
_Static_assert(offsetof(nx_spec_pad, fill) == 8, "at_fill");
_Static_assert(offsetof(nx_spec_pad, geometry) == 24, "at_geometry");

/* The bits nx_dtype.h's store of [x] writes into an element of the float
   dtype [dt] of at most 16 bits. */
intnat nx_kernel_narrow_bits(intnat dt, double x) {
  return (intnat)nx_double_to_bits((int)dt, x);
}

value nx_kernel_narrow_bits_byte(value dt, value x) {
  return Val_long(nx_kernel_narrow_bits(Long_val(dt), Double_val(x)));
}

/* spec.ml writes nx_contract_view at these byte offsets. */
_Static_assert(offsetof(nx_contract_view, extent) == 0, "at_extent");
_Static_assert(offsetof(nx_contract_view, offset) == 32, "at_offset");
_Static_assert(offsetof(nx_contract_view, stride) == 64, "at_stride");
_Static_assert(NX_VIEW_DST == 3 && NX_VIEW_CONTRACTED == 3,
               "operand and axis indices");
_Static_assert(sizeof(nx_contract_view) == 192, "view_bytes");

/* Contract_view.fill, in one pass over the descriptor [s] and the layouts
   of [y], [a], [b] and [i] (dst again without an init), into the view [v]:
   the paired axes marked in a mask per side, each operand's axes of each
   group gathered, every group's extents checked before any merges, then
   each group coalesced (nx_coalesce_dims) and written. Answers FILLED,
   UNMERGED for a group that is no single run, or why the call does not fit
   the descriptor. It allocates nothing. */
enum { RANKS = -2, EXTENTS = -1, UNMERGED = 0, FILLED = 1 };
enum { A = NX_VIEW_A, B = NX_VIEW_B, I = NX_VIEW_INIT, Y = NX_VIEW_DST };
enum {
  BATCH = NX_VIEW_BATCH,
  ROW = NX_VIEW_ROW,
  COLUMN = NX_VIEW_COLUMN,
  CONTRACTED = NX_VIEW_CONTRACTED
};

/* The byte after the view in Contract_view.t: whether it has an init. */
#define AT_HAS_INIT 192
_Static_assert(sizeof(nx_contract_view) == AT_HAS_INIT, "has_init's place");

typedef struct {
  int rank;
  int64_t dim[2 * NX_MAX_RANK];
  int64_t offset;
} layout;

/* Groups the [count] axes of group [g] over the [n] operands [ops], whose
   axes are [axes][op][g], into [w]; answers whether they make one run. */
static int group(nx_contract_view *w, int g, int count, int n, const int *ops,
                 const layout *l, int axes[4][4][NX_MAX_RANK]) {
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
                int axes[4][4][NX_MAX_RANK]) {
  for (int p = 1; p < n; p++)
    for (int k = 0; k < count; k++)
      if (l[ops[p]].dim[axes[ops[p]][g][k]] !=
          l[ops[0]].dim[axes[ops[0]][g][k]])
        return 0;
  return 1;
}

intnat nx_kernel_view_fill(value s, value v, value y, value a, value b,
                           value i) {
  const nx_spec_contract *c = (const nx_spec_contract *)String_val(s);
  int nb = c->nbatch, nc = c->ncontracting, init = c->init;
  layout l[4];
  l[A].rank = nx_array_layout(a, l[A].dim, &l[A].offset);
  l[B].rank = nx_array_layout(b, l[B].dim, &l[B].offset);
  l[Y].rank = nx_array_layout(y, l[Y].dim, &l[Y].offset);
  l[I].rank = nx_array_layout(i, l[I].dim, &l[I].offset);
  int fa = l[A].rank - nb - nc, fb = l[B].rank - nb - nc;
  int ry = nb + fa + fb;
  /* A pair past an operand's rank sets a bit at or past it. */
  uint64_t na = 0, nbm = 0;
  int axes[4][4][NX_MAX_RANK];
  for (int k = 0; k < nb + nc; k++) {
    int ia = c->pairs[k][0], ib = c->pairs[k][1];
    na |= (uint64_t)1 << ia;
    nbm |= (uint64_t)1 << ib;
    int g = k < nb ? BATCH : CONTRACTED, at = k < nb ? k : k - nb;
    axes[A][g][at] = ia;
    axes[B][g][at] = ib;
  }
  if (fa < 0 || fb < 0 || na >> l[A].rank || nbm >> l[B].rank ||
      l[Y].rank != ry || (init && l[I].rank != ry))
    return RANKS;
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
    return EXTENTS;
  uint8_t *bytes = Bytes_val(v);
  nx_contract_view *w = (nx_contract_view *)bytes;
  /* What an operand lacks reads 0: its axes, and an absent init. */
  memset(w, 0, sizeof *w);
  *(int64_t *)(bytes + AT_HAS_INIT) = init;
  w->offset[A] = l[A].offset;
  w->offset[B] = l[B].offset;
  w->offset[Y] = l[Y].offset;
  if (init) w->offset[I] = l[I].offset;
  return group(w, BATCH, nb, nbatch, batch, l, axes) &&
         group(w, ROW, fa, nrow, row, l, axes) &&
         group(w, COLUMN, fb, nrow, column, l, axes) &&
         group(w, CONTRACTED, nc, 2, contracted, l, axes);
}

value nx_kernel_view_fill_byte(value *argv, int argn) {
  (void)argn;
  return Val_long(
      nx_kernel_view_fill(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]));
}
