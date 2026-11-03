/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Segments

   A segmented contraction is, for each batch element and group, the
   contraction of views of its operands: Rows' group e is a's rows with id
   e along a_axis against b's index e along b_axis, Contracting's the
   pair's indices with id e of a and b into dst's slice e. Each is
   computed as an unsegmented call of those views is, so it sums as that
   call does, and the groups write apart. Rows' rows in no group are a
   view of no products: init as acc holds it, or +0.

   A group's indices lie anywhere along the axis, so the call first groups
   them by counting, as a stable permutation: each group's indices in
   index order, the groups in order, then the indices in no group. It
   copies the operands the ids cut into C-contiguous scratch in that
   order, Rows' a and init along their rows and Contracting's a and b
   along the pair, so that each group is a window of the scratch, and
   Rows' result rows go back to dst at their indices. A gathered operand's
   scales are gathered with it.

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
   the layout of its ids into [off], from the pairs, [vy], [va], [vb]
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
  /* The ids: the batch extents, then the cut axis's. */
  if (off->rank != nb + 1) return NX_SHAPE;
  for (int k = 0; k < nb; k++)
    if (f->ps[k][A] < 0 || off->dim[k] != l[A].dim[f->ps[k][A]])
      return NX_SHAPE;
  if (f->rows) {
    intnat xa = Long_val(Field(vs, 1)), xb = Long_val(Field(vs, 2));
    int fa = l[A].rank - nb - nc;
    int ba = free_below(f, A, l[A].rank, xa);
    int bb = free_below(f, B, l[B].rank, xb);
    if (ba < 0 || bb < 0 || off->dim[nb] != l[A].dim[xa]) return NX_SHAPE;
    f->g = l[B].dim[xb];
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
    intnat g = Long_val(Field(vs, 1)), p = Long_val(Field(vs, 2));
    if (p < 0 || p >= nc || g < 0 || y.rank < 1 || y.dim[0] != g ||
        (f->init && (i.rank < 1 || i.dim[0] != g)))
      return NX_SHAPE;
    f->g = g;
    f->at[A] = f->ps[nb + p][A];
    f->at[B] = f->ps[nb + p][B];
    if (f->at[A] < 0 || off->dim[nb] != l[A].dim[f->at[A]]) return NX_SHAPE;
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
  int lazy; /* Rows' b decodes each group's scales as its part runs */
} segmented;

/* [sc], the scales of an operand of extents [x], cut to the block holding
   index [i] of axis [ax], as of an operand of one index there. */
static void cut_scale(nx_array *sc, int64_t *x, int ax, int64_t i) {
  int64_t ratio = sc->dim[ax] ? x[ax] / sc->dim[ax] : 1;
  sc->offset += i / ratio * sc->dim[sc->rank + ax];
  sc->dim[ax] = 1;
  x[ax] = 1;
}

/* Rows' b's scales of batch element [beta]'s group [e] alone, decoded into
   [d]: the scales' block that holds index [e] along b_axis and [beta]'s
   along each batch axis, as of an operand of one index there, so that a
   call decodes the scales of the groups it multiplies, each once, not of
   every group. 0 if the memory could not be allocated. */
static int group_scales(decoded *d, const segmented *j, const int64_t *beta,
                        int64_t e) {
  const frame *f = j->f;
  int r = f->l[B].rank;
  nx_array sc = *j->sc[B];
  int64_t x[NX_MAX_RANK];
  for (int k = 0; k < r; k++) x[k] = f->l[B].dim[k];
  for (int k = 0; k < f->nb; k++) cut_scale(&sc, x, f->ps[k][B], beta[k]);
  cut_scale(&sc, x, f->at[B], e);
  return nx_gemm_decoded_of(j->acc, r, x, &sc, &j->sx[B], d);
}

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
  decoded own = {0};
  for (int o = A; o <= B && !none; o++)
    if (j->sc[o]) {
      const decoded *d = &j->d[o];
      if (o == B && j->lazy) {
        if (!group_scales(&own, j, beta, e)) return 0;
        start[B][f->at[B]] = 0;
        for (int k = 0; k < f->nb; k++) start[B][f->ps[k][B]] = 0;
        d = &own;
      }
      nx_gemm_scale_at(d, &j->sx[o], o, l[o].dim, start[o], &s[o]);
      scaled = 1;
    }
  int done = computed(j->acc, j->order, j->out, j->dts, runs, &v, &r, j->op,
                      scaled ? s : NULL);
  if (own.mem) nx_cpu_aligned_free(own.mem);
  return done;
}

/* Grouping */

static int64_t numel(const nx_layout *l) {
  int64_t n = 1;
  for (int d = 0; d < l->rank; d++) n *= l->dim[d];
  return n;
}

/* [l] laid out C-contiguous from 0 over its own extents, into [c]. */
static void packed(nx_layout *c, const nx_layout *l) {
  int r = l->rank;
  int64_t n = 1;
  *c = *l;
  c->offset = 0;
  for (int d = r - 1; d >= 0; d--) {
    c->dim[r + d] = n;
    n *= l->dim[d];
  }
}

/* [y] without its axis [at], into [l]. */
static void removed(nx_layout *l, const nx_layout *y, int at) {
  int r = y->rank - 1;
  l->rank = r;
  l->offset = y->offset;
  for (int d = 0, e = 0; d <= r; d++) {
    if (d == at) continue;
    l->dim[e] = y->dim[d];
    l->dim[r + e] = y->dim[r + 1 + d];
    e++;
  }
}

/* Batch element [q]'s index, in C order of the [nb] extents [x], into
   [beta]. */
static void batch_of(int64_t *beta, const int64_t *x, int nb, int64_t q) {
  for (int k = nb - 1; k >= 0; k--) {
    beta[k] = q % x[k];
    q /= x[k];
  }
}

/* The cut axis's indices grouped, for each of [count] batch elements of
   [n] indices into [g] groups: element q's from q n in [perm], each
   group's in index order, the groups in order, then those in no group;
   group e's first position in it at q (g + 2) + e in [start], and at
   q (g + 2) + g the count of indices in a group. */
typedef struct {
  int64_t count, n, g;
  int64_t *perm, *start;
} grouping;

/* Groups the ids [id], [count] rows of [n], by counting: O(n + g) a
   row. Group e's count lands at e + 2, its start, by the scan, at e + 1,
   where it moves to its end as its indices land, which is group e + 1's
   start: so each start ends at its own place. */
static void group(grouping *gr, const int64_t *id) {
  int64_t n = gr->n, g = gr->g;
  for (int64_t q = 0; q < gr->count; q++) {
    const int64_t *x = id + q * n;
    int64_t *st = gr->start + q * (g + 2), *pm = gr->perm + q * n;
    memset(st, 0, (size_t)(g + 2) * sizeof *st);
    for (int64_t r = 0; r < n; r++)
      if (x[r] >= 0 && x[r] < g) st[x[r] + 2]++;
    for (int64_t e = 2; e < g + 2; e++) st[e] += st[e - 1];
    int64_t none = st[g + 1];
    for (int64_t r = 0; r < n; r++) {
      if (x[r] >= 0 && x[r] < g) pm[st[x[r] + 1]++] = r;
      else pm[none++] = r;
    }
  }
}

/* Copies, for each batch element q and position p along the cut axis
   [ax], the slice of [x] at index perm(p) into [s]'s at p, where
   [gather], else the slice of [s] at p into [x]'s at perm(p): [xl] and
   [sl] lay out the same extents, their batch axes [bx] in pair order. */
static void moved(const nx_array *s, const nx_layout *sl, const nx_array *x,
                  const nx_layout *xl, int ax, const int *bx, int nb,
                  const grouping *gr, int gather) {
  int r = sl->rank;
  const int64_t *ss = sl->dim + r, *xs = xl->dim + r;
  nx_loop l;
  int64_t x0[NX_MAX_RANK];
  for (int d = 0; d < r; d++) {
    l.extent[d] = sl->dim[d];
    l.step[0][d] = gather ? ss[d] : xs[d];
    l.step[1][d] = gather ? xs[d] : ss[d];
  }
  l.extent[ax] = 1;
  for (int k = 0; k < nb; k++) {
    x0[k] = sl->dim[bx[k]];
    l.extent[bx[k]] = 1;
  }
  int64_t slice = 1;
  for (int d = 0; d < r; d++) slice *= l.extent[d];
  if (slice == 0 || gr->n == 0) return;
  l.rank = nx_coalesce_dims(2, r, l.extent, l.step);
  uint8_t *to = gather ? s->base : x->base;
  const uint8_t *from = gather ? x->base : s->base;
  double per = nx_cpu_kind_cost(1, NX_OP1_COPY, x->dtype, x->dtype);
  int64_t beta[NX_MAX_RANK];
  for (int64_t q = 0; q < gr->count; q++) {
    batch_of(beta, x0, nb, q);
    int64_t s0 = sl->offset, z0 = xl->offset;
    for (int k = 0; k < nb; k++) {
      s0 += beta[k] * ss[bx[k]];
      z0 += beta[k] * xs[bx[k]];
    }
    const int64_t *pm = gr->perm + q * gr->n;
    for (int64_t p = 0; p < gr->n; p++) {
      int64_t at = s0 + p * ss[ax], src = z0 + pm[p] * xs[ax];
      l.first[0] = gather ? at : src;
      l.first[1] = gather ? src : at;
      nx_cpu_copy_loop(to, from, x->bits, &l, per);
    }
  }
}

/* [x], laid out as [xl], into scratch [s] laid out C-contiguous as [sl],
   in the order of [gr] along [ax] where [gather], else uninitialised: 0
   if it could not be allocated. */
static int staged(nx_array *s, nx_layout *sl, const nx_array *x,
                  const nx_layout *xl, int ax, const int *bx, int nb,
                  const grouping *gr, int gather) {
  packed(sl, xl);
  *s = *x;
  s->base = alloc((numel(sl) * x->bits + 7) / 8);
  if (s->base == NULL) return 0;
  s->offset = 0;
  s->alias = 0;
  if (gather) moved(s, sl, x, xl, ax, bx, nb, gr, 1);
  return 1;
}

/* The scales [d] of an operand of rank [r] and extents [x], its axes
   [sx], gathered as its axis [ax] is by [gr], into [o], whose o->mem
   nx_cpu_aligned_free gives back: a scale an index along [ax] and the
   batch axes [bx], its source index's, as nx_gemm_decoded_of lays them
   out. 0 if its memory could not be allocated. */
static int scales_moved(decoded *o, const decoded *d, int acc, int r,
                        const int64_t *x, const scale_axes *sx, int ax,
                        const int *bx, int nb, const grouping *gr) {
  int w = nx_cpu_width(acc);
  int64_t dim[NX_MAX_RANK], xb[NX_MAX_RANK], count = 1, positions = 0;
  for (int k = 0; k < r; k++) dim[k] = x[k] ? x[k] / d->ratio[k] : 0;
  dim[ax] = x[ax];
  for (int k = 0; k < nb; k++) {
    dim[bx[k]] = x[bx[k]];
    xb[k] = x[bx[k]];
  }
  for (int k = r - 1; k >= 0; k--) {
    o->step[k] = count;
    o->ratio[k] = dim[k] ? x[k] / dim[k] : 1;
    count *= dim[k];
  }
  for (int t = 0; t < 3; t++) {
    int64_t len = 1;
    for (int k = 0; k < sx->n[t]; k++) len *= x[sx->ax[t][k]];
    positions += len > 0 ? len : 1;
  }
  int64_t values = ceil_div(count * w, 8);
  o->mem = alloc((values + positions) * 8);
  if (o->mem == NULL) return 0;
  o->t = o->mem + values;
  const uint8_t *from = (const uint8_t *)d->mem;
  uint8_t *to = (uint8_t *)o->mem;
  int64_t idx[NX_MAX_RANK];
  for (int64_t u = 0; u < count; u++) {
    batch_of(idx, dim, r, u);
    int64_t q = 0, at = 0;
    for (int k = 0; k < nb; k++) q = q * xb[k] + idx[bx[k]];
    /* The source's index along each axis: a block's first, but along
       [ax], where it is the permutation's. */
    for (int k = 0; k < r; k++) {
      int64_t i = k == ax ? gr->perm[q * gr->n + idx[k]] : idx[k] * o->ratio[k];
      at += i / d->ratio[k] * d->step[k];
    }
    memcpy(to + u * w, from + at * w, (size_t)w);
  }
  return 1;
}

/* Groups of an empty b: each row of each batch element has no
   products. */
static int none_on(const segmented *j, const int64_t *x, int64_t count,
                   int64_t n) {
  int64_t beta[NX_MAX_RANK];
  int done = 1;
  for (int64_t q = 0; q < count && done; q++) {
    batch_of(beta, x, j->f->nb, q);
    done = part(j, beta, 0, 0, n, 1);
  }
  return done;
}

/* The groups of [gr] computed over the grouped scratch of [j]'s operands
   (groups_on): each batch element's groups, Rows' only where it has
   indices and then its indices in no group. */
static int parts_on(const segmented *j, const int64_t *x,
                    const grouping *gr) {
  const frame *f = j->f;
  int64_t beta[NX_MAX_RANK], g = gr->g, n = gr->n;
  int done = 1;
  for (int64_t q = 0; q < gr->count && done; q++) {
    batch_of(beta, x, f->nb, q);
    const int64_t *st = gr->start + q * (g + 2);
    for (int64_t e = 0; e < g && done; e++) {
      int64_t len = st[e + 1] - st[e];
      if (len > 0 || !f->rows) done = part(j, beta, e, len ? st[e] : 0, len, 0);
    }
    if (f->rows && done && st[g] < n)
      done = part(j, beta, 0, st[g], n - st[g], 1);
  }
  return done;
}

/* The groups of each batch element under the ids [ids]: 0 if an
   allocation failed. The ids are copied first, as dst may be them. */
static int groups_on(const segmented *j, const nx_array *ids) {
  const frame *f = j->f;
  int nb = f->nb, rows = f->rows;
  int64_t count = 1, n = ids->dim[nb], g = f->g;
  for (int k = 0; k < nb; k++) count *= ids->dim[k];
  if (count == 0 || nx_cpu_elements(j->op[DST]) == 0) return 1;
  if (rows && numel(&f->l[B]) == 0) return none_on(j, ids->dim, count, n);
  int64_t *id = malloc((size_t)(count * n + 1) * sizeof *id);
  int64_t *perm = malloc((size_t)(count * n + 1) * sizeof *perm);
  int64_t *start = malloc((size_t)(count * (g + 2)) * sizeof *start);
  int done = id != NULL && perm != NULL && start != NULL;
  if (done) {
    const int64_t *src = (const int64_t *)ids->base;
    for (int64_t u = 0; u < count * n; u++) {
      int64_t at = ids->offset;
      for (int64_t d = nb, x = u; d >= 0; d--) {
        at += x % ids->dim[d] * ids->dim[nb + 1 + d];
        x /= ids->dim[d];
      }
      id[u] = src[at];
    }
  }
  grouping gr = {count, n, g, perm, start};
  if (done) group(&gr, id);
  free(id);
  /* The scratch operands, frame and scales the groups are computed
     over. */
  frame fs = *f;
  segmented js = *j;
  js.f = &fs;
  nx_array s[4] = {{0}};
  const nx_array *op[4] = {j->op[0], j->op[1], j->op[2], j->op[3]};
  js.op = op;
  decoded moved_d[2] = {{0}};
  int bx[3][NX_MAX_RANK];
  for (int k = 0; k < nb; k++) {
    bx[A][k] = f->ps[k][A];
    bx[B][k] = f->ps[k][B];
    bx[2][k] = k;
  }
  nx_layout yl = {0}, yc = {0}, il, ic;
  for (int o = A; o <= B && done; o++) {
    if (o == B && rows) break;
    done = staged(&s[o], &fs.l[o], j->op[o], &f->l[o], f->at[o], bx[o], nb,
                  &gr, 1);
    op[o] = &s[o];
    if (done && j->sc[o]) {
      done = scales_moved(&moved_d[o], &j->d[o], j->acc, f->l[o].rank,
                          f->l[o].dim, &j->sx[o], f->at[o], bx[o], nb, &gr);
      js.d[o] = moved_d[o];
    }
  }
  if (rows && done && f->init) {
    removed(&il, &f->l[INIT], f->yb);
    done = staged(&s[INIT], &ic, j->op[INIT], &il, f->ya, bx[2], nb, &gr, 1);
    inserted(&fs.l[INIT], &ic, f->yb, g);
    op[INIT] = &s[INIT];
  }
  if (rows && done) {
    removed(&yl, &f->l[DST], f->yb);
    done = staged(&s[DST], &yc, j->op[DST], &yl, f->ya, bx[2], nb, &gr, 0);
    inserted(&fs.l[DST], &yc, f->yb, g);
    op[DST] = &s[DST];
  }
  if (done) done = parts_on(&js, ids->dim, &gr);
  if (rows && done) moved(&s[DST], &yc, j->op[DST], &yl, f->ya, bx[2], nb, &gr, 0);
  for (int o = 0; o < 4; o++)
    if (s[o].base) nx_cpu_aligned_free(s[o].base);
  for (int o = A; o <= B; o++)
    if (moved_d[o].mem) nx_cpu_aligned_free(moved_d[o].mem);
  free(perm);
  free(start);
  return done;
}

/* The external for a call with segments [vseg], an option: frames the
   call (frame_of), answering what it answers, refuses as contract_body
   does, then reads the operands and the ids through the door and
   computes each group (groups_on). Out of line, as
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
     Rows' run of rows in none; and the grouped operands' copies, Rows'
     result twice. */
  const nx_array *ids = &a[ao];
  double batches = 1, rows = (double)ids->dim[f.nb], groups = (double)f.g;
  for (int k = 0; k < f.nb; k++) batches *= (double)ids->dim[k];
  if (f.rows && rows < groups) groups = rows;
  double parts = batches * (groups + 1);
  double copied = f.rows ? nx_cpu_bytes(&a[0]) + 2 * nx_cpu_bytes(&a[2]) +
                               (init ? nx_cpu_bytes(&a[3]) : 0)
                         : nx_cpu_bytes(&a[0]) + nx_cpu_bytes(&a[1]);
  double cost = nx_gemm_call_cost(acc, j.order, f.x) + parts * PRODUCT_COST +
                copied * NX_CPU_STRIDED_COST;
  for (int o = A; o <= B; o++)
    if (at[o] >= 0) cost += nx_gemm_decode_cost(acc, &a[at[o]]);
  int released = nx_cpu_release(cost);
  int done = 1;
  for (int o = A; o <= B && done; o++)
    if (at[o] >= 0) {
      j.sc[o] = &a[at[o]];
      if (o == B && f.rows) {
        j.lazy = 1;
        continue;
      }
      done = nx_gemm_decoded_of(acc, f.l[o].rank, f.l[o].dim, j.sc[o], &sx[o],
                                &j.d[o]);
      if (!done) j.sc[o] = NULL;
    }
  if (done) done = groups_on(&j, ids);
  for (int o = A; o <= B; o++)
    if (j.sc[o] && !(o == B && j.lazy)) nx_cpu_aligned_free(j.d[o].mem);
  nx_cpu_reacquire(released);
  nx_done(n, a);
  if (!done) caml_raise_out_of_memory();
  return Val_int(NX_OK);
}
