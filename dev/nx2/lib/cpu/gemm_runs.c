/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Groups of several runs

   A group whose axes do not coalesce into one, over every operand that
   has it, is several runs (nx_contract_runs). The last run of each group
   is the view's axis and the batch's, rows' and columns' other runs an
   outer loop, each of whose positions is a contraction of the view: an
   output's sum is the same whichever contraction computes it. Attention's
   heads, a [b; s; h; d] array read as [b; h; s; d], are so a batch of h
   with b outside, read in place.

   The contracted runs instead make one. Where the kernels read a and b
   along spans (nx_gemm_spanned), each reads its blocks run by run
   (nx_gemm_pack_at), and the runs of the rows that a alone makes, or of
   the columns b alone does, are read so too: a convolution over windows
   of its padded input is one product of all its windows, as an explicit
   copy of them would be, and reads the input where it lies. Elsewhere a
   and b are copied, every
   element of them, into buffers where their runs lie in C order, the
   contracted ones last, and those then coalesce. Either keeps the
   products' numbering, so every order gives its bits; a loop over the
   outer contracted runs, each from the last's sums, would keep chain
   order's bits alone. */

#include <string.h>

#include "gemm.h"

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
  const reads *sp;         /* a's and b's spans, or NULL */
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
    if (!nx_gemm_contract_view(q->acc, q->order, q->da, q->db, &v, q->op,
                               q->s ? s : NULL, q->sp))
      atomic_store(&q->failed, 1);
  }
}

/* Multiplies the copy at [y] of a scaled operand, of acc [dt], laid out as
   nx_gemm_stage_runs lays it, its groups [gs] whole, by its scales [s]. */
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
int nx_gemm_stage_runs(nx_contract_runs *r, int o, const int *gs, int dt,
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
  /* A staged element costs a copy's or a cast's kind, in the wider
     dtype. */
  int wide = nx_cpu_width(dt) > nx_cpu_width(src->dtype) ? dt : src->dtype;
  if (dt == src->dtype)
    nx_cpu_copy_loop(base, src->base, bits, &l,
                     nx_cpu_kind_cost(1, NX_OP1_COPY, dt));
  else nx_cpu_cast_loop(x, src, &l, nx_cpu_kind_cost(1, NX_OP1_CAST, wide));
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

/* Makes [r]'s group [x] one run, which a, where bit A of [read] is set,
   and b, where bit B is, read along their own runs coalesced into [sp],
   each other operand along its one; leaves [r] as it was where another
   makes several. */
static void spread(nx_contract_runs *r, int x, int read, reads *sp) {
  int n = r->runs[x];
  int64_t one[4], whole = 1;
  spans own[2];
  for (int o = A; o <= DST; o++) {
    int64_t ext[NX_MAX_RANK], st[1][NX_MAX_RANK];
    memcpy(ext, r->extent[x], (size_t)n * sizeof *ext);
    memcpy(st[0], r->stride[o][x], (size_t)n * sizeof *ext);
    int k = nx_coalesce_dims(1, n, ext, st);
    one[o] = k == 1 ? st[0][0] : 0;
    if (o <= B && (read >> o & 1)) {
      own[o].n = k;
      memcpy(own[o].extent, ext, (size_t)k * sizeof *ext);
      memcpy(own[o].stride, st[0], (size_t)k * sizeof *ext);
    } else if (k > 1) return;
  }
  for (int o = A; o <= B; o++)
    if (read >> o & 1) sp->of[o][x] = own[o];
  for (int k = 0; k < n; k++) whole *= r->extent[x][k];
  r->runs[x] = 1;
  r->extent[x][0] = whole;
  for (int o = A; o <= DST; o++) r->stride[o][x][0] = one[o];
}

/* The contraction of [r]'s groups over [op], as nx_gemm_contract_view's,
   a and b scaled by [s] where it is not NULL: 0 if an allocation failed.
   A scaled operand copied (nx_gemm_stage_runs) is copied scaled, and read
   unscaled. */
int nx_gemm_runs_on(int acc, int order, int da, int db, nx_contract_runs *r,
                    const nx_array *const *op, const scale *s) {
  for (int g = BATCH; g <= COL; g++)
    for (int k = 0; k < r->runs[g]; k++)
      if (r->extent[g][k] == 0) return 1;
  nx_array x[2];
  const nx_array *ops[4] = {op[A], op[B], op[INIT], op[DST]};
  scale kept[2] = {{0}};
  if (s) memcpy(kept, s, sizeof kept);
  reads sp = {0};
  int packs = nx_gemm_spanned(acc, da, db, s);
  int staged = r->runs[CON] > 1 && !merges(r), done = 0;
  if (packs) {
    if (staged) spread(r, CON, 1 << A | 1 << B, &sp);
    if (r->runs[ROW] > 1) spread(r, ROW, 1 << A, &sp);
    if (r->runs[COL] > 1) spread(r, COL, 1 << B, &sp);
    staged = 0;
  }
  if (staged) {
    static const int gs[2][3] = {{BATCH, ROW, CON}, {BATCH, COL, CON}};
    for (int o = A; o <= B; o++) {
      const scale *so = kept[o].v ? &kept[o] : NULL;
      int dt = so ? acc : op[o]->dtype;
      if (!nx_gemm_stage_runs(r, o, gs[o], dt, op[o], so, &x[o])) {
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
             .s = scaled ? kept : NULL, .sp = packs ? &sp : NULL};
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
  double cost =
      nx_gemm_cost(g->kernel.flops_per_byte, e, m, n, k, nx_cpu_width(acc));
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
