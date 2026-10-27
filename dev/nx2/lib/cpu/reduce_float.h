/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The kernels only the float dtypes have: sums of few products, and
   Logsumexp's and Moments' reductions and scans. reduce_body.h includes
   this file once per float type, after Arg's kernels, whose top_max_ is
   Logsumexp's first pass, with

   - T, the type, and S(name), which names name at the type: S(products)
     is products_f32;
   - MADE, the NaN nx_kinds.h makes from numbers, the quiet one of positive
     sign;
   - CUT, the least integer whose exp is a normal T.

   Each scan includes reduce_network.h for its statistic. */

/* Sums of few products, a vector of the target's width at a time. Lane t
   starts from +0 and takes product t, then the lanes' tree, as few gives
   them for the products as terms: +0 plus a product is the product but at
   -0, whose lane holds +0, and a sum is -0 only where both its operands
   are, so the tree over the products themselves, plus +0 once, has the
   same bits. Each count of products is compiled apart, its tree unrolled
   in registers. */
typedef T S(pv) __attribute__((vector_size(NX_VECTOR_BYTES)));

typedef T S(pm)
    __attribute__((vector_size(NX_VECTOR_BYTES), aligned(1), may_alias));

static inline __attribute__((always_inline)) void S(products_n)(
    int n, int64_t w, int64_t h, const T *a, const int64_t *oa,
    int64_t sa, const T *b, const int64_t *ob, int64_t sb, T *y,
    int64_t sy, int64_t sy2) {
  enum { V = NX_VECTOR_BYTES / sizeof(T) };
  /* The rows apart from the offsets, which a store could alias. */
  const T *pa[NX_CPU_LANES], *pb[NX_CPU_LANES];
  for (int t = 0; t < n; t++) {
    pa[t] = a + oa[t];
    pb[t] = b + ob[t];
  }
  for (int64_t i = 0; i < h; i++, y += sy2) {
    int64_t j = 0;
    for (; j + V <= w; j += V) {
      S(pv) l[NX_CPU_LANES];
      UNROLL(16)
      for (int t = 0; t < n; t++)
        l[t] = *(const S(pm) *)(pa[t] + j) * *(const S(pm) *)(pb[t] + j);
      UNROLL(4)
      for (int g = 8; g >= 1; g /= 2) {
        UNROLL(8)
        for (int t = 0; t < g && t + g < n; t++) l[t] += l[t + g];
      }
      S(pv) r = l[0] + (S(pv)){0};
      if (sy == 1) *(S(pm) *)(y + j) = r;
      else
        for (int k = 0; k < V; k++) y[(j + k) * sy] = r[k];
    }
    for (; j < w; j++) {
      T l[NX_CPU_LANES];
      UNROLL(16)
      for (int t = 0; t < n; t++) l[t] = pa[t][j] * pb[t][j];
      UNROLL(4)
      for (int g = 8; g >= 1; g /= 2) {
        UNROLL(8)
        for (int t = 0; t < g && t + g < n; t++) l[t] += l[t + g];
      }
      T r = l[0] + 0;
      y[j * sy] = r;
    }
    for (int t = 0; t < n; t++) {
      pa[t] += sa;
      pb[t] += sb;
    }
  }
}

#define PRODUCTS_CASE(n)                                                     \
  case n:                                                                    \
    S(products_n)(n, w, h, a, oa, sa, b, ob, sb, y, sy, sy2);                \
    return;

static void S(products)(int n, int64_t w, int64_t h, const uint8_t *a_,
                        const int64_t *oa, int64_t sa, const uint8_t *b_,
                        const int64_t *ob, int64_t sb, uint8_t *y_,
                        int64_t sy, int64_t sy2) {
  const T *a = (const T *)a_, *b = (const T *)b_;
  T *y = (T *)y_;
  switch (n) {
    PRODUCTS_CASE(1) PRODUCTS_CASE(2) PRODUCTS_CASE(3) PRODUCTS_CASE(4)
    PRODUCTS_CASE(5) PRODUCTS_CASE(6) PRODUCTS_CASE(7) PRODUCTS_CASE(8)
    PRODUCTS_CASE(9) PRODUCTS_CASE(10) PRODUCTS_CASE(11) PRODUCTS_CASE(12)
    PRODUCTS_CASE(13) PRODUCTS_CASE(14) PRODUCTS_CASE(15) PRODUCTS_CASE(16)
  }
}

#undef PRODUCTS_CASE

/* Moments scans. A unit scans [w] slices side by side by the network of
   nx_kernel.mli, a state rows of [w] lanes: its mu, S and n. Every lane's
   state at a place in the network counts the same terms, so n is held once,
   in its row's first element, and a join's weight nb / n is one division
   for all its lanes.

   The terms enter in order, term t at index t of level 0. A state at an
   odd index joins the one before it, and the pair enters the next level at
   half its index. Where its index is even, the level's result is the state
   itself at index 0, else the next level's last result joined with it; a
   pair's result is the next level's. So a level holds the even state that
   awaits its partner and its last result, and the scratch three states per
   level: the waiting one, the arriving one and the joined one.

     t        0    1    2        3     4
     level 0  s0   s1   s2       s3    s4
     level 1       p0            p1
     level 2                     q0
     y        s0   p0   p0.s2    q0    q0.s4

   where p0 is s0.s1, p1 s2.s3, q0 p0.p1 and a.b the join of a and b.

   The fast pass computes by plain arithmetic, which gives the kinds' bits
   wherever no value is a NaN or an infinity. Where one of its results is
   not finite, the exact pass computes the unit again with the kinds' ops,
   one kind per loop. */

typedef void (*S(join))(int64_t, const T *, const T *, T *);

/* State [b] after state [a], into [r], each [w] elements a row. */
static void S(moments_fast)(int64_t w, const T *restrict a,
                            const T *restrict b, T *restrict r) {
  T na = a[2 * w], nb = b[2 * w], k = na + nb, f = nb / k, g = na * f;
  for (int64_t l = 0; l < w; l++) {
    T ma = a[l], d = b[l] - ma;
    r[l] = ma + d * f;
    r[w + l] = (a[w + l] + b[w + l]) + (d * d) * g;
  }
  r[2 * w] = k;
}

static void S(moments_exact)(int64_t w, const T *restrict a,
                             const T *restrict b, T *restrict r) {
  const T *ma = a, *va = a + w, *mb = b, *vb = b + w;
  T na = a[2 * w], nb = b[2 * w], k = S(nx_add)(na, nb);
  T f = S(nx_fdiv)(nb, k), g = S(nx_mul)(na, f);
  T d[NX_CPU_NETWORK_LANES], t[NX_CPU_NETWORK_LANES],
      sum[NX_CPU_NETWORK_LANES];
  LANES_OF(d, S(nx_sub)(mb[l], ma[l]));
  LANES_OF(t, S(nx_mul)(d[l], f));
  LANES_OF(t, S(nx_add)(ma[l], t[l]));
  LANES_OF(sum, S(nx_add)(ma[l], mb[l]));
  LANES_OF(r, (FINITE(ma[l]) & FINITE(mb[l])) ? t[l] : sum[l]);
  LANES_OF(d, S(nx_mul)(d[l], d[l]));
  LANES_OF(d, S(nx_mul)(d[l], g));
  LANES_OF(t, S(nx_add)(va[l], vb[l]));
  LANES_OF(r + w, S(nx_add)(t[l], d[l]));
  r[2 * w] = k;
}

static inline S(join) S(moments_join)(arith a) {
  return a == EXACT ? S(moments_exact) : S(moments_fast);
}

/* The states of the terms [x] of each lane, shifted by [c]. */
static void S(moments_leaf)(arith a, int64_t w, const T *restrict x,
                            const T *restrict c, T *restrict r) {
  if (a == EXACT) LANES_OF(r, S(nx_sub)(x[l], c[l]));
  else LANES_OF(r, x[l] - c[l]);
  LANES_OF(r + w, 0);
  r[2 * w] = 1;
}

/* [tn] rows of [w] elements from [y] into [r], row i lane l from
   y[i ys + l yl]. Element by element: nx_copy_box's transposed tiles
   took scan-moments-f32-4096x1024-axis1 1.04 times as long on fudge. */
static void S(rows)(int64_t tn, int64_t w, const T *restrict y,
                    int64_t ys, int64_t yl, T *restrict r) {
  for (int64_t l = 0; l < w; l++)
    for (int64_t i = 0; i < tn; i++) r[i * w + l] = y[i * ys + l * yl];
}

/* [tn] rows of [w] elements from [r] into [y], as rows_ reads them. */
static void S(unrows)(int64_t tn, int64_t w, const T *restrict r, T *y,
                      int64_t ys, int64_t yl) {
  for (int64_t l = 0; l < w; l++)
    for (int64_t i = 0; i < tn; i++) y[i * ys + l * yl] = r[i * w + l];
}

/* The results at a level's [n] odd states, from the next level's:
   o[2 i] = u[i]. */
static void S(odd)(int64_t n, const T *restrict u, T *restrict o) {
  for (int64_t i = 0; i < n; i++) o[2 * i] = u[i];
}

/* Takes the [t]th state of a streaming network, at [in[0]], on through
   its levels, each level's [w] lanes' states 3 w elements: joins it with
   the states [held] there, and leaves the result at [t] at [got[0]]. A
   network over a tile past a slice's first, from its first term, takes
   [carry], the network's result before the tile, as the result before
   each level's first state; NULL elsewhere. */
static void S(stream)(S(join) join, int64_t w, int64_t t, T **held,
                      T **in, T **joined, const T **got,
                      const T *carry) {
  int k = 0;
  int64_t j = t;
  for (; j & 1; k++, j >>= 1) join(w, held[k], in[k], in[k + 1]);
  const T *before = j == 0 ? carry : got[k + 1];
  if (before == NULL) got[k] = in[k];
  else {
    join(w, before, in[k], joined[k]);
    got[k] = joined[k];
  }
  T *e = held[k];
  held[k] = in[k];
  in[k] = e;
  for (int h = 0; h < k; h++) got[h] = got[k];
}

/* The centers [c] of [w] slices, each its first term where finite, else
   0: from [x], lanes [xl] apart, or [src], through [tx]. */
static void S(moments_centers)(const T *x, int64_t xl, int64_t w,
                               const nx_cpu_terms *src, T *tx, T *c) {
  if (src != NULL) {
    src->fill(src->ctx, 0, 1, (uint8_t *)tx);
    x = tx;
    xl = 1;
  }
  LANES_OF(c, FINITE(x[l * xl]) ? x[l * xl] : 0);
}

/* Term m - 1's means c + mu and variances S over the count, of state
   [r], into [dm] and [dv]; in the exact pass mu less itself where mu is
   not finite. */
static inline int S(moments_put)(arith a, int64_t w, const T *c,
                                 const T *r, T m, T *dm, T *dv) {
  if (a == EXACT) {
    T nan[NX_CPU_NETWORK_LANES];
    LANES_OF(dm, S(nx_add)(c[l], r[l]));
    LANES_OF(dv, S(nx_fdiv)(r[w + l], m));
    LANES_OF(nan, S(nx_sub)(r[l], r[l]));
    LANES_OF(dv, FINITE(r[l]) ? dv[l] : nan[l]);
    return 0;
  }
  int bad = 0;
  for (int64_t l = 0; l < w; l++) {
    T mean = c[l] + r[l], var = r[w + l] / m;
    bad |= !(FINITE(mean) & FINITE(var));
    dm[l] = mean;
    dv[l] = var;
  }
  return bad;
}

/* The state at level NETWORK_P of the PT terms of [w] slices from term
   [t0], their states [leaf] makes, centred on [c] where the statistic
   is, joined as the network joins them, each state [size] elements: into
   [total]. Terms go through rows of the scratch as pass_'s do. */
static void S(total)(S(join) join,
                     void (*leaf)(int64_t, const T *, const T *, T *),
                     int64_t size, const T *x, int64_t xs, int64_t xl,
                     const T *c, int64_t t0, int64_t w, T *scratch,
                     const nx_cpu_terms *src, T *total) {
  int64_t tile = NX_CPU_NETWORK_TILE;
  T *held[NETWORK_P + 1], *in[NETWORK_P + 1];
  for (int k = 0; k <= NETWORK_P; k++) {
    held[k] = scratch + 2 * k * size;
    in[k] = held[k] + size;
  }
  T *tx = scratch + 2 * (NETWORK_P + 1) * size;
  int rows = src != NULL || xl != 1;
  for (int64_t i0 = 0; i0 < PT; i0 += tile) {
    if (src != NULL) src->fill(src->ctx, t0 + i0, tile, (uint8_t *)tx);
    else if (xl != 1) S(rows)(tile, w, x + (t0 + i0) * xs, xs, xl, tx);
    for (int64_t i = 0; i < tile; i++) {
      leaf(w, rows ? tx + i * w : x + (t0 + i0 + i) * xs, c, in[0]);
      int k = 0;
      for (int64_t j = i0 + i; j & 1; k++, j >>= 1)
        join(w, held[k], in[k], in[k + 1]);
      T *e = held[k];
      held[k] = in[k];
      in[k] = e;
    }
  }
  memcpy(total, held[NETWORK_P], size * sizeof(T));
}

/* The network's results over the [n] states at [states], each [size]
   elements, in their place. */
static void S(carries)(S(join) join, int64_t size, int64_t w,
                       int64_t n, T *states, T *scratch) {
  int levels = nx_cpu_network_levels(n);
  T *held[64], *in[64], *joined[64];
  const T *got[64];
  for (int k = 0; k < levels; k++) {
    held[k] = scratch + 3 * k * size;
    in[k] = held[k] + size;
    joined[k] = in[k] + size;
  }
  for (int64_t q = 0; q < n; q++) {
    memcpy(in[0], states + q * size, size * sizeof(T));
    S(stream)(join, w, q, held, in, joined, got, NULL);
    memcpy(states + q * size, got[0], size * sizeof(T));
  }
}

/* A tile's [L] terms' states: the terms less the center [c], their sums
   0. */
static inline void S(moments_leaves)(int64_t L, T c, T *restrict sm,
                                     T *restrict ss) {
  for (int64_t i = 0; i < L; i++) {
    sm[i] -= c;
    ss[i] = 0;
  }
}

/* Level k's [m] states from level k - 1's pairs, each of 2^(k - 1)
   terms: arguments, whose restrict lets the compiler vectorise. */
static inline void S(moments_up)(int k, int64_t m, const T *restrict am,
                                 const T *restrict as, T *restrict um,
                                 T *restrict us) {
  T half = (T)(1 << (k - 1)), w = half + half, f = half / w;
  T g = half * f;
  for (int64_t i = 0; i < m; i++) {
    T u = am[2 * i], d = am[2 * i + 1] - u;
    um[i] = u + d * f;
    us[i] = (as[2 * i] + as[2 * i + 1]) + (d * d) * g;
  }
}

/* A full tile's state, its mean [m], its sum [s] and its count. */
static inline void S(moments_top)(T m, T s, T *total) {
  total[0] = m;
  total[1] = s;
  total[2] = (T)PT;
}

static inline void S(moments_seed)(int k, const T *bm, const T *bs,
                                   T *const *o) {
  o[0][1] = bm[0];
  o[1][1] = bs[0];
  o[2][1] = (T)(1 << k);
}

/* Level k's results at the even positions from [i0] on, of [e] of
   them: the next level's results before them [um], [us], [un], joined
   with their states [bm], [bs] of [nb] terms. */
static void S(moments_evens)(int64_t i0, int64_t e, T nb,
                             const T *restrict um, const T *restrict us,
                             const T *restrict un, const T *restrict bm,
                             const T *restrict bs, T *restrict om,
                             T *restrict os, T *restrict on) {
  for (int64_t i = i0; i < e; i++) {
    T na = un[i], w = na + nb, f = nb / w, g = na * f;
    T u = um[i], d = bm[2 * i] - u;
    om[2 * i + 1] = u + d * f;
    os[2 * i + 1] = (us[i] + bs[2 * i]) + (d * d) * g;
    on[2 * i + 1] = w;
  }
}

static inline void S(moments_down)(int k, int64_t i0, int64_t e,
                                   T *const *u, const T *bm,
                                   const T *bs, T *const *o) {
  S(moments_evens)(i0, e, (T)(1 << k), u[0], u[1], u[2], bm, bs, o[0],
                   o[1], o[2]);
}

/* The means and variances of [n] terms from term [t0], their results
   [rm], [rs] centred on [c], into [ym], [yv], stepping [ms], [vs]: 1
   where one is not finite. Apart, whose restrict lets the compiler
   vectorise the divisions. */
static int S(moments_outs)(int64_t n, T c, int64_t t0,
                           const T *restrict rm, const T *restrict rs,
                           T *restrict ym, int64_t ms, T *restrict yv,
                           int64_t vs) {
  int bad = 0;
  for (int64_t i = 0; i < n; i++) {
    T mean = c + rm[i], var = rs[i] / ((T)(t0 + i) + 1);
    bad |= !(FINITE(mean) & FINITE(var));
    ym[i * ms] = mean;
    yv[i * vs] = var;
  }
  return bad;
}

static inline int S(moments_results)(int64_t L, T c, int64_t t0,
                                     T *const *r, T *ym, int64_t ms,
                                     T *yv, int64_t vs) {
  return S(moments_outs)(L, c, t0, r[0], r[1], ym + t0 * ms, ms,
                         yv + t0 * vs, vs);
}

#define N(name) S(moments_##name)
#define N_SIZE 3
#define N_FIELDS 3
#define N_OUTS 2
#include "reduce_network.h"
#undef N
#undef N_SIZE
#undef N_FIELDS
#undef N_OUTS

/* Logsumexp and Moments */

/* exp [y], or 0 below CUT, computed at CUT: no lane makes a subnormal,
   which costs x86 a hundred cycles. */
static inline T S(ex)(T y) {
  T e = S(nx_exp)(y < CUT ? CUT : y);
  return y < CUT ? 0 : e;
}

KERNEL S(top)(const uint8_t *x, int64_t n, int64_t t0, double c,
              nx_cpu_state *st) {
  (void)t0;
  (void)c;
  *st = (nx_cpu_state){.s = S(top_max)((const T *)x, n)};
}

static void S(top_merge)(nx_cpu_state *a, const nx_cpu_state *b) {
  T ma = (T)a->s, mb = (T)b->s;
  a->s = ma != ma ? ma : mb != mb ? mb : ma < mb ? mb : ma;
}

static void S(top_finish)(const nx_cpu_state *st, double c, uint8_t *d0,
                          uint8_t *d1) {
  (void)c;
  (void)d1;
  double m = FINITE(st->s) ? st->s : 0;
  memcpy(d0, &m, sizeof m);
}

KERNEL S(lse)(const uint8_t *x_, int64_t n, int64_t t0, double c_,
              nx_cpu_state *st) {
  (void)t0;
  const T *x = (const T *)x_;
  T c = (T)c_, l[16] = {0};
  int64_t i = 0;
  for (; i + 16 <= n; i += 16)
    UNROLL(16) for (int k = 0; k < 16; k++)
      l[k] += S(ex)(x[i + k] - c);
  for (; i < n; i++) l[i & 15] += S(ex)(x[i] - c);
  NX_CPU_LANE_TREE(i, j, l[i] += l[j]);
  *st = (nx_cpu_state){.s = l[0]};
}

static void S(lse_merge)(nx_cpu_state *a, const nx_cpu_state *b) {
  a->s = (T)a->s + (T)b->s;
}

static void S(lse_finish)(const nx_cpu_state *st, double c, uint8_t *d0,
                          uint8_t *d1) {
  (void)d1;
  T r = (T)c + S(nx_log)((T)st->s);
  memcpy(d0, &r, sizeof r);
}

/* The lanes' tree over lanes 0 to 7 [a] and 8 to 15 [b]. */
static inline T S(halves)(const T *a, const T *b) {
  T l[16];
  memcpy(l, a, 8 * sizeof(T));
  memcpy(l + 8, b, 8 * sizeof(T));
  NX_CPU_LANE_TREE(i, j, l[i] += l[j]);
  return l[0];
}

/* Moments' loops over the lanes unroll by 8, under a bound, i <= n - 16,
   that cannot overflow. Without the pragma gcc keeps the lanes in
   memory at -O2. At -O3 it unrolls Var's loop over four arrays of 8
   lanes whole and vectorises the loop over i instead, with permutes;
   over two arrays of 16 lanes it does not. Under OCaml's -fwrapv it
   cannot count the steps of i + 16 <= n, which may wrap, and at -O3 on
   arm64 leaves the lanes in memory. */
KERNEL S(mean)(const uint8_t *x_, int64_t n, int64_t t0, double c,
               nx_cpu_state *st) {
  (void)t0;
  (void)c;
  const T *x = (const T *)x_;
  T a[8] = {0}, b[8] = {0};
  int64_t i = 0;
  for (; i <= n - 16; i += 16)
    UNROLL(8) for (int k = 0; k < 8; k++) {
      a[k] += x[i + k];
      b[k] += x[i + 8 + k];
    }
  for (; i < n; i++) (i & 8 ? b : a)[i & 7] += x[i];
  *st = (nx_cpu_state){.n = n, .s = S(halves)(a, b)};
}

static void S(sum_merge)(nx_cpu_state *a, const nx_cpu_state *b) {
  a->n += b->n;
  a->s = (T)a->s + (T)b->s;
}

/* The count as the dtype rounds it: Nx_kernel.reduction's Moments
   divides the sum by the count, in the dtype. */
static void S(mean_finish)(const nx_cpu_state *st, double c,
                           uint8_t *d0, uint8_t *d1) {
  (void)c;
  (void)d1;
  double m = (T)st->s / (T)st->n;
  memcpy(d0, &m, sizeof m);
}

KERNEL S(var)(const uint8_t *x_, int64_t n, int64_t t0, double c_,
              nx_cpu_state *st) {
  (void)t0;
  const T *x = (const T *)x_;
  T c = (T)c_, s[16] = {0}, p[16] = {0};
  int64_t i = 0;
  for (; i <= n - 16; i += 16)
    UNROLL(8) for (int k = 0; k < 16; k++) {
      T d = x[i + k] - c;
      s[k] += d * d;
      p[k] += d;
    }
  for (; i < n; i++) {
    T d = x[i] - c;
    s[i & 15] += d * d;
    p[i & 15] += d;
  }
  *st = (nx_cpu_state){.n = n, .s = S(halves)(s, s + 8),
                       .d = S(halves)(p, p + 8)};
}

static void S(var_merge)(nx_cpu_state *a, const nx_cpu_state *b) {
  S(sum_merge)(a, b);
  a->d = (T)a->d + (T)b->d;
}

/* A mean that is a NaN is the variance too. Squares summed to +inf are
   the variance, where d * (d / n) may overflow too and v be a NaN. */
static void S(var_finish)(const nx_cpu_state *st, double c, uint8_t *d0,
                          uint8_t *d1) {
  T mean = MADE, var = MADE;
  if (st->n > 0) {
    T n = (T)st->n, d = (T)st->d, s = (T)st->s;
    T v = (s - d * (d / n)) / n;
    mean = (T)c;
    var = c != c ? mean : s == (T)INFINITY ? s : v < 0 ? 0 : v;
  }
  memcpy(d0, &mean, sizeof mean);
  memcpy(d1, &var, sizeof var);
}

/* Logsumexp's network (nx_cpu.mli). The sum of state [a], of
   maximum [ma] and sum [sa], then state [b], [m] the two's maximum:
   where m is finite, the sum of the side whose maximum is m as it is,
   its exp 0 being 1, plus the other's rescaled; 0 where m is -∞; m
   itself elsewhere, +∞ as the network makes it, or a NaN no result
   reads. Branch free, so that its loops vectorise. */
static inline T S(lse_sum)(T ma, T sa, T mb, T sb, T m) {
  int hi = ma == m;
  T lo = hi ? mb : ma, sl = hi ? sb : sa, sh = hi ? sa : sb;
  T s = sh + sl * S(ex)(lo - m);
  return FINITE(m) ? s : m == -(T)INFINITY ? 0 : m;
}

/* A term's sum: exp 0 where finite, 0 for -∞, else the term. */
static inline T S(lse_term)(T x) {
  return FINITE(x) ? 1 : x == -(T)INFINITY ? 0 : x;
}

static inline T S(lse_result)(T m, T s) {
  T r = m + S(nx_log)(s);
  return FINITE(m) ? r : m;
}

/* The network's hooks (reduce_network.h): Logsumexp has one arithmetic
   and no center. A state is a row of [w] maxima then [w] sums. */
/* State [b] after state [a], into [r]. */
static void S(lse_pair)(int64_t w, const T *restrict a,
                        const T *restrict b, T *restrict r) {
  for (int64_t l = 0; l < w; l++) {
    T m = S(nx_maximum)(a[l], b[l]);
    r[w + l] = S(lse_sum)(a[l], a[w + l], b[l], b[w + l], m);
    r[l] = m;
  }
}

static inline S(join) S(lse_join)(arith a) {
  (void)a;
  return S(lse_pair);
}

static inline void S(lse_leaf)(arith a, int64_t w, const T *x,
                               const T *c, T *r) {
  (void)a;
  (void)c;
  for (int64_t l = 0; l < w; l++) {
    r[l] = x[l];
    r[w + l] = S(lse_term)(x[l]);
  }
}

static inline void S(lse_centers)(const T *x, int64_t xl, int64_t w,
                                  const nx_cpu_terms *src, T *tx, T *c) {
  (void)x;
  (void)xl;
  (void)src;
  (void)tx;
  LANES_OF(c, 0);
}

static inline int S(lse_put)(arith a, int64_t w, const T *c, const T *r,
                             T m, T *dm, T *dv) {
  (void)a;
  (void)c;
  (void)m;
  (void)dv;
  for (int64_t l = 0; l < w; l++) dm[l] = S(lse_result)(r[l], r[w + l]);
  return 0;
}

static inline void S(lse_leaves)(int64_t L, T c, T *restrict sm,
                                 T *restrict ss) {
  (void)c;
  for (int64_t i = 0; i < L; i++) ss[i] = S(lse_term)(sm[i]);
}

/* Pairs of level k - 1's [2m] states into level k's [m]. */
static inline void S(lse_up)(int k, int64_t m, const T *restrict am,
                             const T *restrict as, T *restrict um,
                             T *restrict us) {
  (void)k;
  for (int64_t i = 0; i < m; i++) {
    T ma = am[2 * i], mb = am[2 * i + 1], g = S(nx_maximum)(ma, mb);
    us[i] = S(lse_sum)(ma, as[2 * i], mb, as[2 * i + 1], g);
    um[i] = g;
  }
}

static inline void S(lse_top)(T m, T s, T *total) {
  total[0] = m;
  total[1] = s;
}

static inline void S(lse_seed)(int k, const T *bm, const T *bs,
                               T *const *o) {
  (void)k;
  o[0][1] = bm[0];
  o[1][1] = bs[0];
}

/* Level k's results at the even positions from [i0] on, of [e] of
   them: the next level's results before them [um], [us], joined with
   their states [bm], [bs]. */
static void S(lse_evens)(int64_t i0, int64_t e, const T *restrict um,
                         const T *restrict us, const T *restrict bm,
                         const T *restrict bs, T *restrict om,
                         T *restrict os) {
  for (int64_t i = i0; i < e; i++) {
    T ma = um[i], mb = bm[2 * i], g = S(nx_maximum)(ma, mb);
    os[2 * i + 1] = S(lse_sum)(ma, us[i], mb, bs[2 * i], g);
    om[2 * i + 1] = g;
  }
}

static inline void S(lse_down)(int k, int64_t i0, int64_t e, T *const *u,
                               const T *bm, const T *bs, T *const *o) {
  (void)k;
  S(lse_evens)(i0, e, u[0], u[1], bm, bs, o[0], o[1]);
}

/* The results of [n] terms from term [t0], their maxima [rm] and sums
   [rs], into [y], stepping [ys]. */
static void S(lse_outs)(int64_t n, int64_t t0, const T *restrict rm,
                        const T *restrict rs, T *y, int64_t ys) {
  for (int64_t i = 0; i < n; i++)
    y[(t0 + i) * ys] = S(lse_result)(rm[i], rs[i]);
}

static inline int S(lse_results)(int64_t L, T c, int64_t t0, T *const *r,
                                 T *ym, int64_t ms, T *yv, int64_t vs) {
  (void)c;
  (void)yv;
  (void)vs;
  S(lse_outs)(L, t0, r[0], r[1], ym, ms);
  return 0;
}

#define N(name) S(lse_##name)
#define N_SIZE 2
#define N_FIELDS 2
#define N_OUTS 1
#include "reduce_network.h"
#undef N
#undef N_SIZE
#undef N_FIELDS
#undef N_OUTS
