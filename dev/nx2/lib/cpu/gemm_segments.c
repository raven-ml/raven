/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Segments

   A segmented contraction is, for each batch element and group, the
   contraction of views of its operands: Rows' group e is a's rows
   [lo, hi) along a_axis against b's index e along b_axis, Contracting's
   the pair's window [lo, hi) of a and b into dst's slice e. Each is
   computed as an unsegmented call of those views is, so it sums as that
   call does, and the groups write apart. Rows' rows in no group are a
   view of no products: init as acc holds it, or +0.

   The views are cut from the call's frame: its operands' layouts, Rows'
   dst and init given an axis of b's g groups, stepping 0, where b_axis
   lies among the result's columns, and Contracting's without their
   groups' axis, which each slice steps along. */

#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>

#include "gemm.h"

typedef struct {
  int nb, nc, rows, init;     /* rows: Rows, else Contracting */
  int ps[NX_MAX_RANK][2];     /* the batch pairs, then the contracting */
  int at[2];                  /* the cut axis of a and of b */
  int ya, yb;                 /* Rows: dst's axes of a_axis and b_axis */
  int64_t g, lead[4];         /* Contracting: dst's and init's steps
                                 along the groups' axis */
  nx_layout l[4];             /* by view operand */
  double x[4];                /* the extents' products, by group */
} frame;

/* [y] with an axis of [n] stepping 0 inserted at [at], into [l]. */
static void inserted(nx_layout *l, const nx_layout *y, int at, int64_t n) {
  int r = y->rank;
  l->rank = r + 1;
  l->offset = y->offset;
  for (int d = 0, e = 0; d <= r; d++) {
    int here = d == at;
    l->dim[d] = here ? n : y->dim[e];
    l->dim[r + 1 + d] = here ? 0 : y->dim[r + e];
    e += !here;
  }
}

/* [y] without its first axis, into [l]; its step along it at [lead]. */
static void dropped(nx_layout *l, const nx_layout *y, int64_t *lead) {
  int r = y->rank - 1;
  l->rank = r;
  l->offset = y->offset;
  *lead = y->dim[r + 1];
  for (int d = 0; d < r; d++) {
    l->dim[d] = y->dim[d + 1];
    l->dim[r + d] = y->dim[r + 2 + d];
  }
}

/* [l] cut along [axis] to [n] indices from [start]. */
static void cut(nx_layout *l, int axis, int64_t start, int64_t n) {
  l->offset += start * l->dim[l->rank + axis];
  l->dim[axis] = n;
}

/* The count of side [o]'s axes below [x] no pair of [f] names, where [x]
   is one, 0 <= x < [rank]; else -1. */
static int free_below(const frame *f, int o, int rank, intnat x) {
  if (x < 0 || x >= rank) return -1;
  int n = 0;
  for (int d = 0; d <= x; d++) {
    int paired = 0;
    for (int k = 0; k < f->nb + f->nc; k++) paired |= f->ps[k][o] == d;
    if (d == x) return paired ? -1 : n;
    n += !paired;
  }
  return -1;
}

/* Reads the frame [f] of the segments [vs], an Nx_kernel.segments, and
   the layout of its offsets into [off], from the pairs, [vy], [va], [vb]
   and the init [vi]: NX_OK, or NX_SHAPE where they do not fit as
   Shape.contract states. Out of line: its layouts stay off the
   external's frame. */
static __attribute__((noinline)) int frame_of(frame *f, value vbatch,
                                              value vcon, value vs, value vy,
                                              value va, value vb, value vi,
                                              nx_layout *off) {
  nx_layout *l = f->l, y, i;
  f->nb = (int)Wosize_val(vbatch);
  f->nc = (int)Wosize_val(vcon);
  f->init = Is_block(vi);
  f->rows = Tag_val(vs) == 0;
  int nb = f->nb, nc = f->nc;
  l[A].rank = nx_array_layout(va, l[A].dim, &l[A].offset);
  l[B].rank = nx_array_layout(vb, l[B].dim, &l[B].offset);
  y.rank = nx_array_layout(vy, y.dim, &y.offset);
  i.rank = f->init ? nx_array_layout(Field(vi, 0), i.dim, &i.offset) : 0;
  off->rank = nx_array_layout(Field(vs, 0), off->dim, &off->offset);
  if (nb + nc > l[A].rank || nb + nc > l[B].rank) return NX_SHAPE;
  for (int k = 0; k < nb + nc; k++) {
    value p = Field(k < nb ? vbatch : vcon, k < nb ? k : k - nb);
    for (int o = A; o <= B; o++) {
      intnat x = Long_val(Field(p, o));
      f->ps[k][o] = x < 0 || x >= l[o].rank ? -1 : (int)x;
    }
  }
  /* The offsets: the batch extents, then g + 1. */
  if (off->rank != nb + 1 || off->dim[nb] < 1) return NX_SHAPE;
  for (int k = 0; k < nb; k++)
    if (f->ps[k][A] < 0 || off->dim[k] != l[A].dim[f->ps[k][A]])
      return NX_SHAPE;
  f->g = off->dim[nb] - 1;
  if (f->rows) {
    intnat xa = Long_val(Field(vs, 1)), xb = Long_val(Field(vs, 2));
    int fa = l[A].rank - nb - nc;
    int ba = free_below(f, A, l[A].rank, xa);
    int bb = free_below(f, B, l[B].rank, xb);
    if (ba < 0 || bb < 0 || l[B].dim[xb] != f->g) return NX_SHAPE;
    f->at[A] = (int)xa;
    f->at[B] = (int)xb;
    f->ya = nb + ba;
    f->yb = nb + fa + bb;
    if (f->yb > y.rank || y.rank >= NX_MAX_RANK ||
        (f->init && (f->yb > i.rank || i.rank >= NX_MAX_RANK)))
      return NX_SHAPE;
    inserted(&l[DST], &y, f->yb, f->g);
    if (f->init) inserted(&l[INIT], &i, f->yb, f->g);
  } else {
    intnat p = Long_val(Field(vs, 1));
    if (p < 0 || p >= nc || y.rank < 1 || y.dim[0] != f->g ||
        (f->init && (i.rank < 1 || i.dim[0] != f->g)))
      return NX_SHAPE;
    f->at[A] = f->ps[nb + p][A];
    f->at[B] = f->ps[nb + p][B];
    dropped(&l[DST], &y, &f->lead[DST]);
    if (f->init) dropped(&l[INIT], &i, &f->lead[INIT]);
  }
  nx_contract_view v;
  nx_contract_runs r;
  int fit = nx_contract_fill_layouts(&v, &r, nb, nc,
                                     (const int (*)[2])f->ps, &l[DST], &l[A],
                                     &l[B], f->init ? &l[INIT] : NULL);
  if (fit == NX_SHAPE) return NX_SHAPE;
  for (int g = BATCH; g <= CON; g++) {
    f->x[g] = fit == NX_OK ? (double)v.extent[g] : 1;
    for (int k = 0; fit != NX_OK && k < r.runs[g]; k++)
      f->x[g] *= (double)r.extent[g][k];
  }
  /* Rows' columns hold b's g groups, of which a row takes one. */
  if (f->rows && f->g > 0) f->x[COL] /= (double)f->g;
  return NX_OK;
}

/* A segmented call's work: its frame, dtypes and operands, and its scaled
   sides' axes and scales, sc[o] NULL for an unscaled side. */
typedef struct {
  const frame *f;
  int acc, order, out, dts[3];
  const nx_array *const *op;
  const scale_axes *sx;
  decoded d[2];
  const nx_array *sc[2];
} segmented;

/* Batch element [beta]'s view of group [e] over [n] indices of the cut
   axis from [lo], of no products where [none]: 0 if an allocation
   failed. */
static int part(const segmented *j, const int64_t *beta, int64_t e,
                int64_t lo, int64_t n, int none) {
  const frame *f = j->f;
  nx_layout l[4];
  memcpy(l, f->l, sizeof l);
  int64_t start[2][NX_MAX_RANK] = {{0}};
  for (int k = 0; k < f->nb; k++) {
    for (int o = A; o <= B; o++) {
      cut(&l[o], f->ps[k][o], beta[k], 1);
      start[o][f->ps[k][o]] = beta[k];
    }
    cut(&l[DST], k, beta[k], 1);
    if (f->init) cut(&l[INIT], k, beta[k], 1);
  }
  cut(&l[A], f->at[A], lo, n);
  start[A][f->at[A]] = lo;
  if (f->rows) {
    cut(&l[B], f->at[B], e, 1);
    start[B][f->at[B]] = e;
    for (int o = INIT; o <= DST; o++) {
      if (o == INIT && !f->init) continue;
      cut(&l[o], f->ya, lo, n);
      cut(&l[o], f->yb, e, 1);
    }
  } else {
    cut(&l[B], f->at[B], lo, n);
    start[B][f->at[B]] = lo;
    l[DST].offset += e * f->lead[DST];
    if (f->init) l[INIT].offset += e * f->lead[INIT];
  }
  nx_contract_view v;
  nx_contract_runs r;
  int fit = nx_contract_fill_layouts(&v, &r, f->nb, f->nc,
                                     (const int (*)[2])f->ps, &l[DST], &l[A],
                                     &l[B], f->init ? &l[INIT] : NULL);
  int runs = fit == NX_DECLINED;
  if (none && runs) {
    r.runs[CON] = 1;
    r.extent[CON][0] = 0;
  } else if (none) v.extent[CON] = 0;
  scale s[2] = {{0}};
  int scaled = 0;
  for (int o = A; o <= B && !none; o++)
    if (j->sc[o]) {
      nx_gemm_scale_at(&j->d[o], &j->sx[o], o, l[o].dim, start[o], &s[o]);
      scaled = 1;
    }
  return computed(j->acc, j->order, j->out, j->dts, runs, &v, &r, j->op,
                  scaled ? s : NULL);
}

/* The groups of each batch element under the offsets [off]: 0 if an
   allocation failed. The offsets are copied first, as dst may be them. */
static int groups_on(segmented *j, const nx_array *off) {
  const frame *f = j->f;
  int nb = f->nb;
  int64_t w = f->g + 1, count = 1, beta[NX_MAX_RANK] = {0};
  for (int k = 0; k < nb; k++) count *= off->dim[k];
  if (count == 0) return 1;
  int64_t *o = malloc((size_t)(count * w) * sizeof *o);
  if (o == NULL) return 0;
  const int64_t *src = (const int64_t *)off->base;
  for (int64_t q = 0; q < count * w; q++) {
    int64_t at = off->offset;
    for (int64_t d = nb, x = q; d >= 0; d--) {
      at += x % off->dim[d] * off->dim[nb + 1 + d];
      x /= off->dim[d];
    }
    o[q] = src[at];
  }
  int64_t n = f->l[A].dim[f->at[A]];
  int done = 1;
  for (int64_t q = 0; q < count && done; q++) {
    const int64_t *of = o + q * w;
    for (int64_t d = nb - 1, x = q; d >= 0; d--) {
      beta[d] = x % off->dim[d];
      x /= off->dim[d];
    }
    /* Group e holds [max(of[e], of[0]), min(of[e + 1..g])), within the
       axis: below, [hi] is the least offset above e, at most n. */
    int64_t hi = of[f->g] < n ? of[f->g] : n;
    for (int64_t e = f->g - 1; e >= 0 && done; e--) {
      int64_t lo = of[e] > of[0] ? of[e] : of[0];
      lo = lo < 0 ? 0 : lo > n ? n : lo;
      int64_t len = hi > lo ? hi - lo : 0;
      if (len > 0 || !f->rows) done = part(j, beta, e, len ? lo : 0, len, 0);
      if (of[e] < hi) hi = of[e];
    }
    if (!f->rows || !done) continue;
    /* Rows' rows in no group: below of[0] and from of[g]. */
    int64_t l0 = of[0] < 0 ? 0 : of[0] > n ? n : of[0];
    int64_t h0 = of[f->g] < l0 ? l0 : of[f->g] > n ? n : of[f->g];
    if (f->g == 0) h0 = l0;
    if (l0 > 0) done = part(j, beta, 0, 0, l0, 1);
    if (done && h0 < n) done = part(j, beta, 0, h0, n - h0, 1);
  }
  free(o);
  return done;
}

/* The external for a call with segments [vseg], an option: frames the
   call (frame_of), answering what it answers, refuses as contract_body
   does, then reads the operands and the offsets through
   the door and computes each group (groups_on). Out of line, as
   contract_scaled. */
__attribute__((noinline)) value nx_gemm_contract_segmented(
    value vbatch, value vcontracting, value vacc, value vdst, value vi,
    value va, value vb, value vsa, value vsb, value vseg) {
  int acc = Int_val(vacc), init = Is_block(vi);
  value vinit = init ? Field(vi, 0) : va;
  value vs = Field(vseg, 0);
  frame f;
  nx_layout offl;
  int fit = frame_of(&f, vbatch, vcontracting, vs, vdst, va, vb, vi, &offl);
  if (fit != NX_OK) return Val_int(fit);
  value vsc[2] = {vsa, vsb};
  int scaled = Is_block(vsa) || Is_block(vsb);
  if (scaled && !nx_gemm_scales_fit(va, vb, vsc)) return Val_int(NX_SHAPE);
  scale_axes sx[2];
  for (int o = A; o <= B; o++)
    if (Is_block(vsc[o]))
      nx_gemm_axes_of(Field(vsc[o], 0), o == A ? va : vb, vbatch, vcontracting,
                      o, &sx[o]);
  segmented j = {.f = &f, .acc = acc, .out = nx_array_dtype(vdst), .sx = sx,
                 .dts = {nx_array_dtype(va), nx_array_dtype(vb),
                         nx_array_dtype(vinit)}};
  nx_operand in[7];
  in[0] = (nx_operand){va, j.dts[0], 0};
  in[1] = (nx_operand){vb, j.dts[1], 0};
  in[2] = (nx_operand){vdst, j.out, 1};
  in[3] = (nx_operand){vinit, j.dts[2], 0};
  int n = 3 + init, at[2] = {-1, -1};
  for (int o = A; o <= B; o++)
    if (Is_block(vsc[o])) {
      value sv = Field(Field(vsc[o], 0), 0);
      at[o] = n;
      in[n++] = (nx_operand){sv, nx_array_dtype(sv), 0};
    }
  int ao = n;
  in[n++] = (nx_operand){Field(vs, 0), NX_INT64, 0};
  nx_array a[7];
  int e = nx_read(n, in, a);
  if (e) return Val_int(e);
  const nx_array *op[4] = {&a[0], &a[1], init ? &a[3] : NULL, &a[2]};
  j.op = op;
  j.order = nx_gemm_order_of(acc, f.x);
  /* Each part's set-up, a product's, for each batch element's groups and
     Rows' two runs of rows in none. */
  double parts = (double)nx_cpu_elements(&a[ao]) + 2;
  double cost = nx_gemm_call_cost(acc, j.order, f.x) + parts * PRODUCT_COST;
  for (int o = A; o <= B; o++)
    if (at[o] >= 0) cost += nx_gemm_decode_cost(acc, &a[at[o]]);
  int released = nx_cpu_release(cost);
  int done = 1;
  for (int o = A; o <= B && done; o++)
    if (at[o] >= 0) {
      j.sc[o] = &a[at[o]];
      done = nx_gemm_decoded_of(acc, f.l[o].rank, f.l[o].dim, j.sc[o], &sx[o],
                                &j.d[o]);
      if (!done) j.sc[o] = NULL;
    }
  if (done) done = groups_on(&j, &a[ao]);
  for (int o = A; o <= B; o++)
    if (j.sc[o]) nx_cpu_aligned_free(j.d[o].mem);
  nx_cpu_reacquire(released);
  nx_done(n, a);
  if (!done) caml_raise_out_of_memory();
  return Val_int(NX_OK);
}
