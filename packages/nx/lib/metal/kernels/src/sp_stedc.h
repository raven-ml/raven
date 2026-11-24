/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The eigenvalues of real symmetric tridiagonal matrices T, ascending,
   and with them Q, T = Q diag λ Qᵀ, by divide and conquer as LAPACK's
   xSTEDC computes them (Cuppen 1981; Gu and Eisenstat 1995); or by
   bisection on Sturm counts where only the values are asked.

   Divide and conquer: T is torn at every multiple of L = 32 into leaves,
   T = diag(T1, …) + Σ ρ u uᵀ, each leaf solved by Jacobi (sp_jacobi.h),
   then neighbouring blocks merged level by level, all the merges of a
   level, of every matrix of the batch, in the launches of a phase:

     prepare   one thread a merge: z from the halves' vectors, the
               halves' merged order, the tolerance
     deflate   one thread a merge: the values whose part of z is
               negligible or that a rotation makes equal to a neighbour
               set aside (xLAED2), a value a step, in steps that resume
               across launches
     rotate    one thread a row: the rotations applied to Q's columns
     rank      one thread a set-aside value: its rank among them
     gather    one thread an element: the set-aside columns into Qt
     roots     one thread a root of 1 + ρ Σ z_j² / (d_j − λ) = 0
               (xLAED4), its d_j − λ into S's column, in steps that
               resume across launches
     weights   one thread a root: z recomputed from the roots (Gu and
               Eisenstat), so that the vectors are orthogonal
     vectors   one thread a root: S's column, normalised
     product   Q' = Q[:, kept] S over the merge's rows (the O(n³))
     sort      one thread a merge: roots and set-aside values merged
               ascending
     place     one thread an element: Q's columns in that order

   Every value is computed in a fixed order by one thread, so the bits do
   not depend on scheduling. Matrices are column-major here, element
   (i, j) of an n x n Q at q[i + j n]. A launch splits by its units, from
   [first] to [count] − 1, the bits not.

   Workspace a matrix (dc_ws), [per] bytes apart: Q, Qt (the product's
   output), S, n² each; for each index d, e, z, λ, w, set-aside values;
   cols, k and rotation lists; the failure flag and exponent; each root's
   secular state.

   This follows nx.amd's sp_stedc.h. */

#ifndef NX_METAL_SP_STEDC_H
#define NX_METAL_SP_STEDC_H

#include "sp_common.h"

namespace sp {

#define LEAF NX_SP_LEAF

struct dc_ws {
  device float *d, *e, *q, *qt, *s, *z, *dl, *w, *lam, *dd, *rc, *rs;
  device int *cols, *kq, *rp, *rn, *nr, *fail;
  device int *ex, *sec;
};

static inline dc_ws dc_of(device uchar *base, long b, int n, ulong per) {
  dc_ws ws;
  const ulong nn = ulong(n) * n;
  device uchar *p = base + ulong(b) * per;
  ws.q = (device float *)p;
  ws.qt = ws.q + nn;
  ws.s = ws.qt + nn;
  ws.d = ws.s + nn;
  ws.e = ws.d + n;
  ws.z = ws.e + n;
  ws.dl = ws.z + n;
  ws.w = ws.dl + n;
  ws.lam = ws.w + n;
  ws.dd = ws.lam + n;
  ws.rc = ws.dd + n;
  ws.rs = ws.rc + n;
  ws.cols = (device int *)(ws.rs + 2 * n);
  ws.kq = ws.cols + n;
  ws.rp = ws.kq + n;
  ws.rn = ws.rp + n;
  ws.nr = ws.rn + n;
  ws.fail = ws.nr + n;
  ws.ex = ws.fail + 1;
  ws.sec = ws.ex + 1;
  return ws;
}

#define Q(x, i, j) (x)[(ulong)(i) + (ulong)(j) * n]

/* The merge of level [lev] (1 = pairs of leaves) that index [x] of a
   matrix of order n falls in: its rows [a, b), split at [m]; false where
   that block has no right half at this level. */
static inline bool merge_of(int x, int lev, int n, thread int &a,
                            thread int &m, thread int &b) {
  const int span = LEAF << lev;
  a = x / span * span;
  m = a + (span >> 1);
  b = metal::min(a + span, n);
  return m < b;
}

/* Scale: T's largest magnitude brought into [1, 2) by a power of 2, the
   exponent kept; the tears at every leaf boundary; the leaves as dense
   L x L blocks (row-major) for Jacobi, the last one, of [last] values,
   apart. One thread a matrix for the scale, then a thread an element;
   [big] a float of threadgroup memory. */
static inline void dc_leaves(grp g, long gx, long ngx, threadgroup float *big,
                             device uchar *wsp, device const float *dsrc,
                             device const float *esrc, ulong stride,
                             device float *full, device float *tail, int n,
                             ulong per, long first, long count) {
  const int nl = (n + LEAF - 1) / LEAF, last = n - (nl - 1) * LEAF;
  for (long b = first + gx; b < count; b += ngx) {
    const dc_ws ws = dc_of(wsp, b, n, per);
    for (int i = g.tid; i < n; i += g.nthr) {
      ws.d[i] = dsrc[ulong(b) * stride + i];
      ws.e[i] = i < n - 1 ? esrc[ulong(b) * stride + i] : 0.0f;
    }
    barrier();
    if (g.tid == 0) {
      float x = 0;
      for (int i = 0; i < n; i++) {
        x = fmax_(x, fabs_(ws.d[i]));
        if (i < n - 1) x = fmax_(x, fabs_(ws.e[i]));
      }
      *big = x;
      ws.fail[0] = 0;
    }
    barrier();
    const int ex = *big > 0 ? -ilogb_(*big) : 0;
    if (g.tid == 0) ws.ex[0] = ex;
    for (int i = g.tid; i < n; i += g.nthr) {
      ws.d[i] = ldexp_(ws.d[i], ex);
      if (i < n - 1) ws.e[i] = ldexp_(ws.e[i], ex);
    }
    barrier();
    /* Each leaf's diagonal, less |ρ| at a torn end. */
    for (int i = g.tid; i < n; i += g.nthr) {
      const int leaf = i / LEAF, li = i % LEAF,
                size = leaf == nl - 1 ? last : LEAF;
      float di = ws.d[i];
      if (li == 0 && leaf > 0) di -= fabs_(ws.e[i - 1]);
      if (li == size - 1 && leaf < nl - 1) di -= fabs_(ws.e[i]);
      device float *blk =
          leaf == nl - 1 ? tail + ulong(b) * last * last
                         : full + (ulong(b) * (nl - 1) + leaf) * LEAF * LEAF;
      for (int j = 0; j < size; j++) {
        float x = 0;
        if (j == li) x = di;
        if (j == li + 1) x = ws.e[i];
        if (j == li - 1) x = ws.e[i - 1];
        blk[li * size + j] = x;
      }
    }
    barrier();
  }
}

/* The leaves' eigenvalues into d and vectors into Q's diagonal blocks,
   Q zero elsewhere; a leaf whose Jacobi failed (NaN, its input finite)
   fails the matrix. One thread an element of Q. */
static inline void dc_scatter(grp g, long gx, long ngx, device uchar *wsp,
                              device const float *wf, device const float *vf,
                              device const float *wt, device const float *vt,
                              int n, ulong per, long first, long count) {
  const int nl = (n + LEAF - 1) / LEAF, last = n - (nl - 1) * LEAF;
  const ulong nn = ulong(n) * n;
  const bool small = is_small(count);
  for (long x = first + gx * g.nthr + g.tid; x < count; x += ngx * g.nthr) {
    long b = x;
    const ulong k = ulong(split(b, long(nn), small));
    const dc_ws ws = dc_of(wsp, b, n, per);
    long col = long(k);
    const int i = int(split(col, n, small)), j = int(col);
    const int leaf = i / LEAF, size = leaf == nl - 1 ? last : LEAF;
    float v = 0;
    if (j / LEAF == leaf) {
      const int li = i % LEAF, lj = j % LEAF;
      v = leaf == nl - 1
              ? vt[ulong(b) * last * last + li * size + lj]
              : vf[(ulong(b) * (nl - 1) + leaf) * LEAF * LEAF + li * size + lj];
      if (i == j) {
        const float w = leaf == nl - 1
                            ? wt[ulong(b) * last + li]
                            : wf[(ulong(b) * (nl - 1) + leaf) * LEAF + li];
        ws.d[i] = w;
        if (w != w) ws.fail[0] = 1;
      }
    }
    ws.q[k] = v;
  }
}

/* Deflation's state of a merge between dc_prepare and the end of
   dc_deflate, in the secular state words of the merge's first index,
   which dc_secular's first launch writes before it reads them: the
   decisions' mode, the next value's place j in the merged order, the
   kept value pj that waits on the next, the counts k (kept), nd (set
   aside) and nrot (rotations), tol and ρ. */
enum { DF_MODE, DF_J, DF_PJ, DF_K, DF_ND, DF_NROT, DF_TOL, DF_RHO, DF_WORDS };
static_assert(DF_WORDS <= NX_SP_SECULAR, "deflation's state");

/* The modes: no value kept yet; pj kept, pending; every value set aside,
   as z is negligible; done. */
enum { DF_FIND, DF_PAIR, DF_ALL, DF_DONE };

/* Deflate (xLAED2), its start: z, ρ, the halves' merged order and tol,
   and the decisions' state. A thread a merge, the units (matrix, merge)
   from [first] to [count] − 1, [du] the divisor word of a matrix's
   merges. */
static inline void dc_prepare(grp g, long gx, long ngx, device uchar *wsp,
                              int n, int lev, ulong per, ulong du, long first,
                              long count) {
  const int span = LEAF << lev, merges = (n + span - 1) / span;
  const bool small = is_small(count);
  const float eps = lim::eps();
  for (long u = first + gx * g.nthr + g.tid; u < count; u += ngx * g.nthr) {
    long b = u;
    const int at = int(split(b, merges, du, small)) * span;
    int a, m, e;
    if (!merge_of(at, lev, n, a, m, e)) continue;
    const dc_ws ws = dc_of(wsp, b, n, per);
    const int N = e - a, n1 = m - a;
    device float *z = ws.z + a, *d = ws.d + a;
    device float *q = ws.q;
    float rho = ws.e[m - 1];
    for (int j = 0; j < N; j++) {
      float x = j < n1 ? Q(q, m - 1, a + j) : Q(q, m, a + j);
      if (rho < 0 && j >= n1) x = -x;
      z[j] = x * (1 / sqrt_(2.0f));
    }
    rho = fabs_(2 * rho);
    /* The halves merged ascending, the first of equals from the left,
       into kq[a…e) as scratch, whose first word dc_deflate's end takes
       for k. */
    device int *ix = ws.kq + a;
    for (int i = 0, l = 0, r = n1; i < N; i++)
      ix[i] = (r >= N || (l < n1 && d[l] <= d[r])) ? l++ : r++;
    float zmax = 0, dmax = 0;
    for (int i = 0; i < N; i++) {
      zmax = fmax_(zmax, fabs_(z[i]));
      dmax = fmax_(dmax, fabs_(d[i]));
    }
    const float tol = 8 * eps * fmax_(dmax, zmax);
    device int *st = ws.sec + ulong(a) * NX_SP_SECULAR;
    st[DF_MODE] = rho * zmax <= tol ? DF_ALL : DF_FIND;
    st[DF_J] = st[DF_PJ] = st[DF_K] = st[DF_ND] = st[DF_NROT] = 0;
    st[DF_TOL] = as_type<int>(tol);
    st[DF_RHO] = as_type<int>(rho);
  }
}

/* Deflate (xLAED2), its decisions in the merged order, a value a step,
   then a step that ends them: the values set aside and the rotations,
   which dc_rotate, dc_rank and dc_gather then apply; a merge keeps k
   values and sets N − k aside. A thread a merge, the units (matrix,
   merge) from [first] to [count] − 1, at most [q] steps a launch, [du]
   the divisor word of a matrix's merges. */
static inline void dc_deflate(grp g, long gx, long ngx, device uchar *wsp,
                              int n, int lev, ulong per, int q, ulong du,
                              long first, long count) {
  const int span = LEAF << lev, merges = (n + span - 1) / span;
  const bool small = is_small(count);
  for (long u = first + gx * g.nthr + g.tid; u < count; u += ngx * g.nthr) {
    long b = u;
    const int at = int(split(b, merges, du, small)) * span;
    int a, m, e;
    if (!merge_of(at, lev, n, a, m, e)) continue;
    const dc_ws ws = dc_of(wsp, b, n, per);
    device int *st = ws.sec + ulong(a) * NX_SP_SECULAR;
    int mode = st[DF_MODE];
    if (mode == DF_DONE) continue;
    const int N = e - a;
    device float *z = ws.z + a, *d = ws.d + a, *dl = ws.dl + a, *w = ws.w + a,
                 *dd = ws.dd + a, *rc = ws.rc + a, *rs = ws.rs + a;
    device int *cols = ws.cols + a, *rp = ws.rp + a, *rn = ws.rn + a;
    device const int *ix = ws.kq + a;
    const float tol = as_type<float>(st[DF_TOL]);
    const float rho = as_type<float>(st[DF_RHO]);
    int j = st[DF_J], pj = st[DF_PJ], k = st[DF_K], nd = st[DF_ND],
        nrot = st[DF_NROT];
    for (int t = 0; t < q && mode != DF_DONE; t++) {
      if (j == N) {
        if (mode != DF_ALL) {
          dl[k] = d[pj];
          w[k] = z[pj];
          cols[k++] = pj;
        }
        ws.kq[a] = k;
        ws.nr[a] = nrot;
        /* ρ for the roots, in the tear's slot of e, unused after. */
        ws.e[m - 1] = rho;
        mode = DF_DONE;
        break;
      }
      const int nj = ix[j++];
      if (mode == DF_ALL || rho * fabs_(z[nj]) <= tol) {
        dd[nd] = d[nj];
        cols[N - 1 - nd++] = nj;
        continue;
      }
      if (mode == DF_FIND) {
        pj = nj;
        mode = DF_PAIR;
        continue;
      }
      /* Two close values: a rotation zeroes pj's part of z where it moves
         no value by more than tol. */
      float sn = z[pj], cs = z[nj], tau = hypot_(cs, sn), t2 = d[nj] - d[pj];
      cs /= tau;
      sn = -sn / tau;
      if (fabs_(t2 * cs * sn) > tol) {
        dl[k] = d[pj];
        w[k] = z[pj];
        cols[k++] = pj;
        pj = nj;
        continue;
      }
      z[nj] = tau;
      z[pj] = 0;
      rp[nrot] = pj;
      rn[nrot] = nj;
      rc[nrot] = cs;
      rs[nrot++] = sn;
      t2 = d[pj] * (cs * cs) + d[nj] * (sn * sn);
      d[nj] = d[pj] * (sn * sn) + d[nj] * (cs * cs);
      d[pj] = t2;
      dd[nd] = d[pj];
      cols[N - 1 - nd++] = pj;
      pj = nj;
    }
    st[DF_MODE] = mode;
    st[DF_J] = j;
    st[DF_PJ] = pj;
    st[DF_K] = k;
    st[DF_ND] = nd;
    st[DF_NROT] = nrot;
  }
}

/* Deflate's rotations, in order, on a row of its merge's columns: a
   thread a row, the units (matrix, row) from [first] to [count] − 1. */
static inline void dc_rotate(grp g, long gx, long ngx, device uchar *wsp,
                             int n, int lev, ulong per, long first,
                             long count) {
  const bool small = is_small(count);
  for (long x = first + gx * g.nthr + g.tid; x < count; x += ngx * g.nthr) {
    long b = x;
    const int r = int(split(b, n, small));
    int a, m, e;
    if (!merge_of(r, lev, n, a, m, e)) continue;
    const dc_ws ws = dc_of(wsp, b, n, per);
    const int nrot = ws.nr[a];
    device const int *rp = ws.rp + a, *rn = ws.rn + a;
    device const float *rc = ws.rc + a, *rs = ws.rs + a;
    device float *q = ws.q;
    for (int t = 0; t < nrot; t++) {
      device float *x = &Q(q, r, a + rp[t]), *y = &Q(q, r, a + rn[t]);
      const float xr = *x, yr = *y;
      *x = rc[t] * xr + rs[t] * yr;
      *y = rc[t] * yr - rs[t] * xr;
    }
  }
}

/* Deflate's set-aside values ascending, ties by order set aside: a
   thread an index i of a merge, the units (matrix, index) from [first]
   to [count] − 1, the rank of dd[i] where i < N − k, its value into
   λ[k + rank] and its column, cols[N − 1 − i], into rn[k + rank], after
   dc_rotate has read the rotations there. NaN, from a failed leaf, ranks
   last: the ranks stay a permutation, so every column read is the
   merge's. */
static inline void dc_rank(grp g, long gx, long ngx, device uchar *wsp, int n,
                           int lev, ulong per, long first, long count) {
  const bool small = is_small(count);
  for (long x = first + gx * g.nthr + g.tid; x < count; x += ngx * g.nthr) {
    long b = x;
    const int y = int(split(b, n, small));
    int a, m, e;
    if (!merge_of(y, lev, n, a, m, e)) continue;
    const dc_ws ws = dc_of(wsp, b, n, per);
    const int N = e - a, k = ws.kq[a], nd = N - k, i = y - a;
    if (i >= nd) continue;
    device const float *dd = ws.dd + a;
    device const int *cols = ws.cols + a;
    int rank = 0;
    const bool nan_i = dd[i] != dd[i];
    for (int j = 0; j < nd; j++)
      rank += dd[j] < dd[i] || (dd[j] == dd[i] && j < i) ||
              (nan_i && (dd[j] == dd[j] || j < i));
    ws.lam[a + k + rank] = dd[i];
    ws.rn[a + k + rank] = cols[N - 1 - i];
  }
}

/* The element of unit [x] of dc_gather and dc_place: (matrix, column
   c, row o of c's merge), min(span, n) rows a column, by [du] and [dr],
   the divisor words of n min(span, n) and of min(span, n); false where c
   has no merge at this level or row a + o is past it. */
static inline bool merge_element(long x, int n, int lev, bool small, ulong du,
                                 ulong dr, thread long &b, thread int &r,
                                 thread int &c, thread int &a) {
  const int rows = metal::min(LEAF << lev, n);
  b = x;
  long oc = long(split(b, long(n) * rows, du, small));
  const int o = int(split(oc, rows, dr, is_small(long(n) * rows)));
  c = int(oc);
  int m, e;
  if (!merge_of(c, lev, n, a, m, e)) return false;
  r = a + o;
  return r < e;
}

/* Deflate's set-aside columns into Qt[:, k…] in their ranks' order: a
   thread an element (r, c) of a matrix within its merge, the units
   (matrix, column, row of its merge) from [first] to [count] − 1, where
   column c is past its merge's k kept ones; [du] and [dr] as
   merge_element takes them. */
static inline void dc_gather(grp g, long gx, long ngx, device uchar *wsp,
                             int n, int lev, ulong per, ulong du,
                             ulong dr, long first, long count) {
  const bool small = is_small(count);
  for (long x = first + gx * g.nthr + g.tid; x < count; x += ngx * g.nthr) {
    long b;
    int r, c, a;
    if (!merge_element(x, n, lev, small, du, dr, b, r, c, a)) continue;
    const dc_ws ws = dc_of(wsp, b, n, per);
    if (c - a < ws.kq[a]) continue;
    Q(ws.qt, r, c) = Q(ws.q, r, a + ws.rn[c]);
  }
}

/* The root [i] of the secular equation of two values, d[0] < d[1]
   (xLAED5): λ into [dlam], the normalised d_j − λ into delta. */
static inline void laed5(int i, device const float *d, device const float *z,
                         device float *delta, ulong ds, float rho,
                         device float *dlam) {
  const float del = d[1] - d[0];
  float tau, d0, d1;
  float b = -del + rho * (z[0] * z[0] + z[1] * z[1]),
        c = rho * z[1] * z[1] * del;
  if (i == 0 && 1 + 2 * rho * (z[1] * z[1] - z[0] * z[0]) / del > 0) {
    b = del + rho * (z[0] * z[0] + z[1] * z[1]);
    c = rho * z[0] * z[0] * del;
    tau = 2 * c / (b + sqrt_(fabs_(b * b - 4 * c)));
    *dlam = d[0] + tau;
    d0 = -z[0] / tau;
    d1 = z[1] / (del - tau);
  } else {
    if (i == 0)
      tau = b > 0 ? -2 * c / (b + sqrt_(b * b + 4 * c))
                  : (b - sqrt_(b * b + 4 * c)) / 2;
    else
      tau = b > 0 ? (b + sqrt_(b * b + 4 * c)) / 2
                  : 2 * c / (-b + sqrt_(b * b + 4 * c));
    *dlam = d[1] + tau;
    d0 = -z[0] / (del + tau);
    d1 = -z[1] / tau;
  }
  const float t = sqrt_(d0 * d0 + d1 * d1);
  delta[0] = d0 / t;
  delta[ds] = d1 / t;
}

/* Three values in registers, read by index. */
struct three {
  float v0, v1, v2;
  float operator[](int i) const thread {
    return i == 0 ? v0 : i == 1 ? v1 : v2;
  }
  void set(int i, float x) thread {
    if (i == 0)
      v0 = x;
    else if (i == 1)
      v1 = x;
    else
      v2 = x;
  }
};

/* The root τ near zero of f(x) = rho + Σ z_j / (d_j − x) over three
   poles (xLAED6). Returns false after 40 steps. */
static inline bool laed6(int kniter, bool orgati, float rho, three d, three z,
                         float finit, thread float &tau_) {
  const float eps = lim::eps();
  float lbd = orgati ? d[1] : d[0], ubd = orgati ? d[2] : d[1], tau = 0;
  if (finit < 0)
    lbd = 0;
  else
    ubd = 0;
  if (kniter == 2) {
    float a, b, c;
    if (orgati) {
      const float t = (d[2] - d[1]) / 2;
      c = rho + z[0] / ((d[0] - d[1]) - t);
      a = c * (d[1] + d[2]) + z[1] + z[2];
      b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1];
    } else {
      const float t = (d[0] - d[1]) / 2;
      c = rho + z[2] / ((d[2] - d[1]) - t);
      a = c * (d[0] + d[1]) + z[0] + z[1];
      b = c * d[0] * d[1] + z[0] * d[1] + z[1] * d[0];
    }
    const float t = fmax_(fabs_(a), fmax_(fabs_(b), fabs_(c)));
    a /= t, b /= t, c /= t;
    tau = c == 0   ? b / a
          : a <= 0 ? (a - sqrt_(fabs_(a * a - 4 * b * c))) / (2 * c)
                   : 2 * b / (a + sqrt_(fabs_(a * a - 4 * b * c)));
    if (tau < lbd || tau > ubd) tau = (lbd + ubd) / 2;
    if (d[0] == tau || d[1] == tau || d[2] == tau)
      tau = 0;
    else {
      const float f = finit + tau * z[0] / (d[0] * (d[0] - tau)) +
                      tau * z[1] / (d[1] * (d[1] - tau)) +
                      tau * z[2] / (d[2] * (d[2] - tau));
      if (f <= 0)
        lbd = tau;
      else
        ubd = tau;
      if (fabs_(finit) <= fabs_(f)) tau = 0;
    }
  }
  const float small1 = 0x1p-42f, small2 = small1 * small1;
  float t = orgati ? d[1] - tau : d[0] - tau,
        t2 = orgati ? d[2] - tau : d[1] - tau;
  t = fmin_(fabs_(t), fabs_(t2));
  float sclfac = 1, sclinv = 1;
  if (t <= small1) {
    sclfac = t <= small2 ? 1 / small2 : 1 / small1;
    sclinv = t <= small2 ? small2 : small1;
  }
  three ds, zs;
  for (int i = 0; i < 3; i++)
    ds.set(i, d[i] * sclfac), zs.set(i, z[i] * sclfac);
  tau *= sclfac, lbd *= sclfac, ubd *= sclfac;
  float fc = 0, df = 0, ddf = 0;
  for (int i = 0; i < 3; i++) {
    const float r = 1 / (ds[i] - tau), t1 = zs[i] * r, q2 = t1 * r;
    fc += t1 / ds[i];
    df += q2;
    ddf += q2 * r;
  }
  float f = finit + tau * fc;
  bool ok = fabs_(f) <= 0;
  if (f <= 0)
    lbd = tau;
  else
    ubd = tau;
  for (int it = 2; it <= 40 && !ok; it++) {
    const float t1 = orgati ? ds[1] - tau : ds[0] - tau;
    const float q2 = orgati ? ds[2] - tau : ds[1] - tau;
    float a = (t1 + q2) * f - t1 * q2 * df, b = t1 * q2 * f;
    float c = f - (t1 + q2) * df + t1 * q2 * ddf;
    const float mm = fmax_(fabs_(a), fmax_(fabs_(b), fabs_(c)));
    a /= mm, b /= mm, c /= mm;
    float eta = c == 0   ? b / a
                : a <= 0 ? (a - sqrt_(fabs_(a * a - 4 * b * c))) / (2 * c)
                         : 2 * b / (a + sqrt_(fabs_(a * a - 4 * b * c)));
    if (f * eta >= 0) eta = -f / df;
    tau += eta;
    if (tau < lbd || tau > ubd) tau = (lbd + ubd) / 2;
    float err = 0;
    fc = df = ddf = 0;
    int i = 0;
    for (; i < 3 && ds[i] - tau != 0; i++) {
      const float r = 1 / (ds[i] - tau), p1 = zs[i] * r, p2 = p1 * r,
                  p4 = p1 / ds[i];
      fc += p4;
      err += fabs_(p4);
      df += p2;
      ddf += p2 * r;
    }
    if (i < 3) {
      ok = true;
      break;
    }
    f = finit + tau * fc;
    err = 8 * (fabs_(finit) + fabs_(tau) * err) + fabs_(tau) * df;
    ok = fabs_(f) <= 4 * eps * err || ubd - lbd <= 4 * eps * fabs_(tau);
    if (f <= 0)
      lbd = tau;
    else
      ubd = tau;
  }
  tau_ = tau * sclinv;
  return ok;
}

/* Root [i] of 1 + ρ Σ z_j² / (d_j − λ) = 0, d ascending, z without
   zeros, ρ > 0 (xLAED4): λ into [dlam], d_j − λ into delta[j ds], each
   from the pole nearer the root, as a start and then steps, each step
   one of xLAED4's iterations, that resume from a state saved between
   launches. A root that none of [steps] steps, 30 but in tests,
   converges fails.

   The state: τ, the bounds on it, the pole it is measured from and the
   last step's w; the step; the pole's side (orgati) and the switches
   between xLAED4's two- and three-pole models; whether the root is done
   and whether it failed. */
enum {
  SEC_ORGATI = 1,
  SEC_SWTCH = 2,
  SEC_SWTCH3 = 4,
  SEC_DONE = 8,
  SEC_FAIL = 16,
};

struct secular {
  float tau, lb, ub, org, prew;
  int step, flags;
};

static inline secular secular_of(device const int *v) {
  secular st;
  st.tau = as_type<float>(v[0]);
  st.lb = as_type<float>(v[1]);
  st.ub = as_type<float>(v[2]);
  st.org = as_type<float>(v[3]);
  st.prew = as_type<float>(v[4]);
  st.step = v[5];
  st.flags = v[6];
  return st;
}

static inline void secular_to(device int *v, thread const secular &st) {
  v[0] = as_type<int>(st.tau);
  v[1] = as_type<int>(st.lb);
  v[2] = as_type<int>(st.ub);
  v[3] = as_type<int>(st.org);
  v[4] = as_type<int>(st.prew);
  v[5] = st.step;
  v[6] = st.flags;
}

#define DL(j) delta[(ulong)(j) * ds]

static inline void laed4_start(int n, int i, device const float *d,
                               device const float *z, device float *delta,
                               ulong ds, float rho, device float *dlam,
                               thread secular &st) {
  st.prew = 0;
  st.step = 0;
  if (n == 1) {
    *dlam = d[0] + rho * z[0] * z[0];
    delta[0] = 1;
    st.flags = SEC_DONE;
    return;
  }
  if (n == 2) {
    laed5(i, d, z, delta, ds, rho, dlam);
    st.flags = SEC_DONE;
    return;
  }
  const float rhoinv = 1 / rho;
  const bool last = i == n - 1;
  bool orgati = true;
  float tau, lb, ub, org;
  if (last) {
    const float midpt = rho / 2, del = d[n - 1] - d[n - 2];
    float psi = 0;
    for (int j = 0; j < n; j++) DL(j) = (d[j] - d[n - 1]) - midpt;
    for (int j = 0; j < n - 2; j++) psi += z[j] * z[j] / DL(j);
    const float c = rhoinv + psi;
    const float w =
        c + z[n - 2] * z[n - 2] / DL(n - 2) + z[n - 1] * z[n - 1] / DL(n - 1);
    const float a = -c * del + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
    const float b = z[n - 1] * z[n - 1] * del;
    tau = a < 0 ? 2 * b / (sqrt_(a * a + 4 * b * c) - a)
                : (a + sqrt_(a * a + 4 * b * c)) / (2 * c);
    lb = 0;
    ub = midpt;
    if (w <= 0) {
      lb = midpt;
      ub = rho;
      if (c <= z[n - 2] * z[n - 2] / (del + rho) + z[n - 1] * z[n - 1] / rho)
        tau = rho;
    }
    org = d[n - 1];
  } else {
    const float del = d[i + 1] - d[i], midpt = del / 2;
    float psi = 0, phi = 0;
    for (int j = 0; j < n; j++) DL(j) = (d[j] - d[i]) - midpt;
    for (int j = 0; j < i; j++) psi += z[j] * z[j] / DL(j);
    for (int j = n - 1; j > i + 1; j--) phi += z[j] * z[j] / DL(j);
    const float c = rhoinv + psi + phi, zi = z[i] * z[i],
                zj = z[i + 1] * z[i + 1];
    const float w = c + zi / DL(i) + zj / DL(i + 1);
    orgati = w > 0;
    if (orgati) {
      const float a = c * del + zi + zj, b = zi * del;
      tau = a > 0 ? 2 * b / (a + sqrt_(fabs_(a * a - 4 * b * c)))
                  : (a - sqrt_(fabs_(a * a - 4 * b * c))) / (2 * c);
      lb = 0;
      ub = midpt;
    } else {
      const float a = c * del - zi - zj, b = zj * del;
      tau = a < 0 ? 2 * b / (a - sqrt_(fabs_(a * a + 4 * b * c)))
                  : -(a + sqrt_(fabs_(a * a + 4 * b * c))) / (2 * c);
      lb = -midpt;
      ub = 0;
    }
    org = orgati ? d[i] : d[i + 1];
  }
  for (int j = 0; j < n; j++) DL(j) = (d[j] - org) - tau;
  st.tau = tau;
  st.lb = lb;
  st.ub = ub;
  st.org = org;
  st.flags = orgati ? SEC_ORGATI : 0;
}

static inline void laed4_step(int n, int i, device const float *d,
                              device const float *z, device float *delta,
                              ulong ds, float rho, device float *dlam,
                              int steps, thread secular &st) {
  const float eps = lim::eps(), rhoinv = 1 / rho;
  const bool last = i == n - 1, orgati = st.flags & SEC_ORGATI;
  bool swtch3 = st.flags & SEC_SWTCH3, swtch = st.flags & SEC_SWTCH;
  const int ii = last ? n - 1 : orgati ? i : i + 1, ip = last ? n - 2 : i;
  const int step = st.step;
  float tau = st.tau, lb = st.lb, ub = st.ub;
  const float org = st.org, prew = st.prew;
  float w, dw, psi = 0, dpsi = 0, phi = 0, dphi = 0, err = 0;
  for (int j = 0; j < ii; j++) {
    const float t = z[j] / DL(j);
    psi += z[j] * t;
    dpsi += t * t;
    err += psi;
  }
  err = fabs_(err);
  for (int j = n - 1; j > ii; j--) {
    const float t = z[j] / DL(j);
    phi += z[j] * t;
    dphi += t * t;
    err += phi;
  }
  float t = z[ii] / DL(ii);
  if (last) {
    phi = z[ii] * t;
    dphi = t * t;
    w = rhoinv + phi + psi;
    dw = dpsi + dphi;
    err = 8 * (-phi - psi) + err - phi + rhoinv + fabs_(tau) * dw;
  } else {
    w = rhoinv + phi + psi;
    if (step == 0)
      swtch3 = (orgati ? w < 0 : w > 0) && ii != 0 && ii != n - 1;
    dw = dpsi + dphi + t * t;
    t *= z[ii];
    w += t;
    err = 8 * (phi - psi) + err + 2 * rhoinv + 3 * fabs_(t) + fabs_(tau) * dw;
    if (step == 1)
      swtch = orgati ? -w > fabs_(prew) / 10 : w > fabs_(prew) / 10;
    else if (step > 1 && w * prew > 0 && fabs_(w) > fabs_(prew) / 10)
      swtch = !swtch;
  }
  if (fabs_(w) <= eps * err) {
    *dlam = org + tau;
    st.flags |= SEC_DONE;
    return;
  }
  if (step + 1 >= steps) {
    *dlam = org + tau;
    st.flags |= SEC_DONE | SEC_FAIL;
    return;
  }
  if (w <= 0)
    lb = fmax_(lb, tau);
  else
    ub = fmin_(ub, tau);
  float eta, a, b, c;
  const float di = DL(ip), dj = DL(ip + 1);
  if (last) {
    c = w - di * dpsi - dj * dphi;
    a = (di + dj) * w - di * dj * dw;
    b = di * dj * w;
    if (step == 0 && c < 0) c = fabs_(c);
    if (step == 0 && c == 0)
      eta = -w / dw;
    else if (a >= 0)
      eta = (a + sqrt_(fabs_(a * a - 4 * b * c))) / (2 * c);
    else
      eta = 2 * b / (a - sqrt_(fabs_(a * a - 4 * b * c)));
    if (w * eta > 0) eta = -w / dw;
  } else if (!swtch3) {
    if (swtch) {
      const float q = z[ii] / DL(ii);
      if (orgati)
        dpsi += q * q;
      else
        dphi += q * q;
      c = w - di * dpsi - dj * dphi;
    } else {
      const float q = orgati ? z[i] / di : z[i + 1] / dj;
      c = orgati ? w - dj * dw - (d[i] - d[i + 1]) * (q * q)
                 : w - di * dw - (d[i + 1] - d[i]) * (q * q);
    }
    a = (di + dj) * w - di * dj * dw;
    b = di * dj * w;
    if (c == 0) {
      if (a == 0)
        a = swtch    ? di * di * dpsi + dj * dj * dphi
            : orgati ? z[i] * z[i] + dj * dj * (dpsi + dphi)
                     : z[i + 1] * z[i + 1] + di * di * (dpsi + dphi);
      eta = b / a;
    } else if (a <= 0)
      eta = (a - sqrt_(fabs_(a * a - 4 * b * c))) / (2 * c);
    else
      eta = 2 * b / (a + sqrt_(fabs_(a * a - 4 * b * c)));
    if (w * eta >= 0) eta = -w / dw;
  } else {
    three zz, dv;
    const float s = rhoinv + psi + phi, dm = DL(ii - 1), dp = DL(ii + 1);
    if (swtch) {
      c = s - dm * dpsi - dp * dphi;
      zz.v0 = dm * dm * dpsi;
      zz.v2 = dp * dp * dphi;
    } else if (orgati) {
      float q = z[ii - 1] / dm;
      q *= q;
      c = s - dp * (dpsi + dphi) - (d[ii - 1] - d[ii + 1]) * q;
      zz.v0 = z[ii - 1] * z[ii - 1];
      zz.v2 = dp * dp * ((dpsi - q) + dphi);
    } else {
      float q = z[ii + 1] / dp;
      q *= q;
      c = s - dm * (dpsi + dphi) - (d[ii + 1] - d[ii - 1]) * q;
      zz.v0 = dm * dm * (dpsi + (dphi - q));
      zz.v2 = z[ii + 1] * z[ii + 1];
    }
    zz.v1 = z[ii] * z[ii];
    dv.v0 = dm;
    dv.v1 = DL(ii);
    dv.v2 = dp;
    if (!laed6(step + 2, orgati, c, dv, zz, w, eta)) {
      st.flags |= SEC_DONE | SEC_FAIL;
      return;
    }
    if (w * eta >= 0) eta = -w / dw;
  }
  if (tau + eta > ub || tau + eta < lb) {
    const float mid = w < 0 ? (ub - tau) / 2 : (lb - tau) / 2,
                newton = -w / dw;
    const bool mean = (last ? step == 0 : step > 0) && lb <= tau + newton &&
                      tau + newton <= ub;
    eta = mean ? copysign_(1.0f, newton) * sqrt_(fabs_(newton)) *
                     sqrt_(fabs_(mid))
               : mid;
  }
  for (int j = 0; j < n; j++) DL(j) -= eta;
  st.tau = tau + eta;
  st.lb = lb;
  st.ub = ub;
  st.prew = w;
  st.step = step + 1;
  st.flags = (orgati ? SEC_ORGATI : 0) | (swtch ? SEC_SWTCH : 0) |
             (swtch3 ? SEC_SWTCH3 : 0);
}

#undef DL

/* Roots, weights and vectors: one thread an index x of a matrix, root
   x − a of its merge where it is below k, the matrices' indices from
   [first] to [count] − 1. [phase] 0 roots, 1 weights, 2 vectors. A
   launch of the roots runs at most [q] steps of each, the start where
   [start], and saves its state. */
static inline void dc_secular(grp g, long gx, long ngx, device uchar *wsp,
                              int n, int lev, int phase, int steps, ulong per,
                              int q, bool start, long first, long count) {
  const bool small = is_small(count);
  for (long y = first + gx * g.nthr + g.tid; y < count; y += ngx * g.nthr) {
    long b = y;
    const int x = int(split(b, n, small));
    int a, m, e;
    if (!merge_of(x, lev, n, a, m, e)) continue;
    const dc_ws ws = dc_of(wsp, b, n, per);
    const int k = ws.kq[a], i = x - a;
    if (i >= k) continue;
    device float *s = &Q(ws.s, a, a), *dl = ws.dl + a, *w = ws.w + a;
    if (phase == 0) {
      device int *sv = ws.sec + ulong(x) * NX_SP_SECULAR;
      device float *col = s + ulong(i) * n, *lam = &ws.lam[a + i];
      const float rho = ws.e[m - 1];
      secular st;
      int t = 0;
      if (start) {
        laed4_start(k, i, dl, w, col, 1, rho, lam, st);
        t = 1;
      } else
        st = secular_of(sv);
      for (; t < q && !(st.flags & SEC_DONE); t++)
        laed4_step(k, i, dl, w, col, 1, rho, lam, steps, st);
      secular_to(sv, st);
      if (st.flags & SEC_FAIL) ws.fail[0] = 1;
      continue;
    }
    if (k <= 2) continue;
    if (phase == 1) {
      /* Gu and Eisenstat: |z_i| from the roots, z's sign kept. */
      float p = s[i + ulong(i) * n];
      for (int c = 0; c < k; c++)
        if (c != i) p *= s[i + ulong(c) * n] / (dl[i] - dl[c]);
      ws.z[a + i] = copysign_(sqrt_(-p), w[i]);
      continue;
    }
    device float *col = s + ulong(i) * n;
    float big = 0;
    device const float *zn = ws.z + a;
    for (int r = 0; r < k; r++) {
      col[r] = zn[r] / col[r];
      big = fmax_(big, fabs_(col[r]));
    }
    /* A column has a few large elements near its pole and many below the
       last bit of their running sum: a compensated sum. */
    sumsq sum;
    for (int r = 0; r < k; r++) sum.sq(col[r] / big);
    const float norm = big * sqrt_(sum.get());
    for (int r = 0; r < k; r++) col[r] /= norm;
  }
}

/* Qt[:, 0:k] = Q[:, kept] S over each merge's rows: 64 x 64 tiles of
   (row, root), one group each along x and y from tile [x0] and [y0], the
   units (matrix, merge) along z from [first] to [count] − 1; tiles past
   a merge's extents exit. Threadgroup memory: As and Bs, 16 x 65 floats
   each. */
static inline void dc_product(grp g, uint3 gid, uint3 ngrid,
                              threadgroup float *sm, device uchar *wsp, int n,
                              int lev, ulong per, int x0, int y0, long first,
                              long count) {
  const int span = LEAF << lev, merges = (n + span - 1) / span;
  const bool small = is_small(count);
  threadgroup float(*As)[65] = (threadgroup float(*)[65])sm;
  threadgroup float(*Bs)[65] = (threadgroup float(*)[65])(sm + 16 * 65);
  const int tx = g.tid % 16, ty = g.tid / 16;
  for (long u = first + gid.z; u < count; u += ngrid.z) {
    long b = u;
    const int at = int(split(b, merges, small)) * span;
    int a, m, e;
    if (!merge_of(at, lev, n, a, m, e)) continue;
    const dc_ws ws = dc_of(wsp, b, n, per);
    const int N = e - a, k = ws.kq[a];
    const int r0 = (x0 + int(gid.x)) * 64, c0 = (y0 + int(gid.y)) * 64;
    if (r0 >= N || c0 >= k) continue;
    device const int *cols = ws.cols + a;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < k; k0 += 16) {
      barrier();
      for (int t = g.tid; t < 16 * 64; t += 256) {
        const int r = t % 64, j = t / 64;
        As[j][r] = r0 + r < N && k0 + j < k
                       ? Q(ws.q, a + r0 + r, a + cols[k0 + j])
                       : 0.0f;
        const int jj = t % 16, c = t / 16;
        Bs[jj][c] = c0 + c < k && k0 + jj < k
                        ? Q(ws.s, a + k0 + jj, a + c0 + c)
                        : 0.0f;
      }
      barrier();
      for (int j = 0; j < 16; j++)
        for (int p = 0; p < 4; p++)
          for (int q = 0; q < 4; q++)
            acc[p][q] += As[j][tx + 16 * p] * Bs[j][ty + 16 * q];
    }
    for (int p = 0; p < 4; p++)
      for (int q = 0; q < 4; q++) {
        const int r = r0 + tx + 16 * p, c = c0 + ty + 16 * q;
        if (r < N && c < k) Q(ws.qt, a + r, a + c) = acc[p][q];
      }
  }
}

/* Sort: the roots (λ ascending) and the set-aside values (ascending, at
   λ[k…]) merged ascending, roots first among equals, into d, each
   output's source into cols. A thread a merge, the units (matrix,
   merge) from [first] to [count] − 1, [du] the divisor word of a
   matrix's merges. */
static inline void dc_sort(grp g, long gx, long ngx, device uchar *wsp, int n,
                           int lev, ulong per, ulong du, long first,
                           long count) {
  const int span = LEAF << lev, merges = (n + span - 1) / span;
  const bool small = is_small(count);
  for (long u = first + gx * g.nthr + g.tid; u < count; u += ngx * g.nthr) {
    long b = u;
    const int at = int(split(b, merges, du, small)) * span;
    int a, m, e;
    if (!merge_of(at, lev, n, a, m, e)) continue;
    const dc_ws ws = dc_of(wsp, b, n, per);
    const int N = e - a, k = ws.kq[a];
    device float *lam = ws.lam + a;
    device int *src = ws.cols + a;
    for (int o = 0, i = 0, j = k; o < N; o++) {
      const bool root = j >= N || (i < k && lam[i] <= lam[j]);
      src[o] = root ? i++ : j++;
    }
    for (int o = 0; o < N; o++) ws.d[a + o] = lam[src[o]];
  }
}

/* Place: Qt's columns into Q's in the order dc_sort left in cols. A
   thread an element (r, c) of a matrix within its merge, the units
   (matrix, column, row of its merge) from [first] to [count] − 1; [du]
   and [dr] as merge_element takes them. */
static inline void dc_place(grp g, long gx, long ngx, device uchar *wsp,
                            int n, int lev, ulong per, ulong du,
                            ulong dr, long first, long count) {
  const bool small = is_small(count);
  for (long x = first + gx * g.nthr + g.tid; x < count; x += ngx * g.nthr) {
    long b;
    int r, c, a;
    if (!merge_element(x, n, lev, small, du, dr, b, r, c, a)) continue;
    const dc_ws ws = dc_of(wsp, b, n, per);
    Q(ws.q, r, c) = Q(ws.qt, r, a + ws.cols[c]);
  }
}

/* The bounds bisection starts from, as xSTEBZ: each matrix's
   Gershgorin interval, widened by 2u n of its magnitude and 2 pivmin,
   then pivmin, into [bd], NX_SP_BOUNDS floats a matrix. A thread a
   matrix, the matrices from [first] to [count] − 1. */
static inline void tri_bounds(grp g, long gx, long ngx,
                              device const float *dsrc,
                              device const float *esrc, ulong stride,
                              device float *bd, int n, long first,
                              long count) {
  const float eps = lim::eps(), tiny = lim::tiny();
  for (long b = first + gx * g.nthr + g.tid; b < count; b += ngx * g.nthr) {
    device const float *d = dsrc + ulong(b) * stride,
                       *e = esrc + ulong(b) * stride;
    float lo = d[0], hi = d[0], emax = 0;
    for (int j = 0; j < n; j++) {
      const float r =
          (j > 0 ? fabs_(e[j - 1]) : 0.0f) + (j < n - 1 ? fabs_(e[j]) : 0.0f);
      lo = fmin_(lo, d[j] - r);
      hi = fmax_(hi, d[j] + r);
      if (j < n - 1) emax = fmax_(emax, e[j] * e[j]);
    }
    const float pivmin = tiny * fmax_(1.0f, emax);
    const float span = fmax_(fabs_(lo), fabs_(hi));
    device float *o = bd + ulong(b) * NX_SP_BOUNDS;
    o[0] = lo - (2 * eps * span * n + 2 * pivmin);
    o[1] = hi + (2 * eps * span * n + 2 * pivmin);
    o[2] = pivmin;
  }
}

/* Bisection: eigenvalues [i0, i0 + cnt) of each matrix's T into w,
   [cnt] a matrix, one thread each, the matrices' values from [first] to
   [count] − 1, by Sturm counts (the negative pivots of T − mid I, a
   pivot below pivmin in magnitude taken as −pivmin), from [bd]'s bounds,
   for at most NX_SP_HALVINGS halvings or until the interval is within
   2u of its magnitude or pivmin.

   A step is a halving's test or one row of its count, and a launch runs
   at most [q] steps of each value, then saves its state, NX_SP_BISECT
   words a value at [st]: lo, hi, the halving (−1 once w is written),
   the row (−1 before the test), the pivot and the count. Where [start],
   a value starts from its bounds. */
static inline void tri_bisect(grp g, long gx, long ngx,
                              device const float *dsrc,
                              device const float *esrc, ulong stride,
                              device const float *bd, device int *st,
                              device float *w, int n, int i0, int cnt, int q,
                              bool start, long first, long count) {
  const float eps = lim::eps();
  const bool small = is_small(count);
  for (long x = first + gx * g.nthr + g.tid; x < count; x += ngx * g.nthr) {
    long b = x;
    const int i = i0 + int(split(b, cnt, small));
    device const float *d = dsrc + ulong(b) * stride,
                       *e = esrc + ulong(b) * stride,
                       *o = bd + ulong(b) * NX_SP_BOUNDS;
    device int *s = st + ulong(x) * NX_SP_BISECT;
    const float pivmin = o[2];
    float lo = start ? o[0] : as_type<float>(s[0]);
    float hi = start ? o[1] : as_type<float>(s[1]);
    int it = start ? 0 : s[2], j = start ? -1 : s[3];
    float p = start ? 0.0f : as_type<float>(s[4]);
    int c = start ? 0 : s[5];
    if (it < 0) continue;
    for (int k = 0; k < q && it >= 0; k++) {
      const float mid = lo + (hi - lo) / 2;
      if (j < 0) {
        if (it == NX_SP_HALVINGS ||
            hi - lo <= 2 * eps * fmax_(fabs_(lo), fabs_(hi)) + pivmin ||
            mid == lo || mid == hi) {
          w[ulong(b) * cnt + i - i0] = mid;
          it = -1;
        } else {
          j = 0;
          c = 0;
        }
        continue;
      }
      p = j == 0 ? d[0] - mid : d[j] - mid - e[j - 1] * e[j - 1] / p;
      if (fabs_(p) < pivmin) p = -pivmin;
      c += p < 0;
      if (++j == n) {
        if (c > i)
          hi = mid;
        else
          lo = mid;
        it++;
        j = -1;
      }
    }
    s[0] = as_type<int>(lo);
    s[1] = as_type<int>(hi);
    s[2] = it;
    s[3] = j;
    s[4] = as_type<int>(p);
    s[5] = c;
  }
}

#undef LEAF
#undef Q

} // namespace sp

#endif
