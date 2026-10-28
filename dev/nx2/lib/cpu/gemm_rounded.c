/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Results of another dtype

   A dst of another dtype than acc takes the sum in acc rounded once, as a
   cast of it would round it: R is scratch in acc, laid out C-contiguous
   over the batch's, rows' and columns' runs, and a cast walk takes it
   into dst once every product is in. A 4-bit acc sums in 8 bits
   (summed_in): R is then narrowed to acc, by a cast into dst where dst
   is of acc, else into scratch of acc cast into dst in turn. */

#include "gemm.h"

/* [r]'s view as runs: one a group. */
void nx_gemm_runs_of_view(nx_contract_runs *r, const nx_contract_view *v) {
  for (int g = BATCH; g <= CON; g++) {
    r->runs[g] = 1;
    r->extent[g][0] = v->extent[g];
    for (int o = A; o <= DST; o++) r->stride[o][g][0] = v->stride[o][g];
  }
  for (int o = A; o <= DST; o++) r->offset[o] = v->offset[o];
}

/* What a cast of [x] into [y] costs an element (nx_cpu_cast_cost). */
static double cast_cost(const nx_array *y, const nx_array *x) {
  return nx_cpu_cast_cost(x->dtype, y->dtype);
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
      nx_cpu_cast_loop(&t, y, &lt, cast_cost(&t, y));
    }
  }
  if (done) {
    const nx_array *x = narrows ? &t : y;
    l.rank = nx_coalesce_dims(2, rank, l.extent, l.step);
    nx_cpu_cast_loop(dst, x, &l, cast_cost(dst, x));
  }
  if (t.base) nx_cpu_give(t.base);
  return done;
}

/* The contraction of [r]'s groups over [op] in [acc], summed in [sum],
   into op[DST] of another dtype than [sum], through scratch R: 0 if an
   allocation failed. */
int nx_gemm_rounded_on(int acc, int sum, int order, nx_contract_runs *r,
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
     cast so, contracted last, so that the runs merge and nx_gemm_runs_on copies
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
    done = nx_gemm_stage_runs(r, o, gs[o], acc, op[o], NULL, &z[o]);
    if (done) ops[o] = &z[o];
  }
  if (done)
    done = nx_gemm_runs_on(sum, order, ops[A]->dtype, ops[B]->dtype, r, ops, s);
  for (int o = A; o <= INIT; o++)
    if (ops[o] == &z[o]) nx_cpu_give(z[o].base);
  int narrows = acc != sum && op[DST]->dtype != acc;
  if (done)
    done = store_rounded(op[DST], &y, acc, narrows, first, rank, extent, step);
  nx_cpu_give(y.base);
  return done;
}
