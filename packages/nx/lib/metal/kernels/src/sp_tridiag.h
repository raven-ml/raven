/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* eigh past the small path: a = Q T Qᴴ with T real tridiagonal, as
   LAPACK's blocked xHETRD computes it (lower), then T's eigensystem
   (sp_stedc.h), then v = Q Z by the reflectors' blocks (xUNMTR's xLARFB).

   The reduction takes panels of NB columns. For each column c of a
   panel, td_step (one group a matrix) brings the column up to date with
   the panel's earlier reflectors through W (xLATRD), makes its
   reflector, and finishes W's previous column; td_hemv (a SIMD-group a
   row, every matrix) is W's product with the trailing matrix, the step
   that reads the matrix once a column. The panel's trailing update is
   two products, A22 −= V Wᴴ + W Vᴴ, both triangles, so that the trailing
   matrix stays whole and td_hemv reads columns, contiguous.

   Matrices are column-major here, element (i, j) at x[i + j n]. Every
   sum runs in a fixed order: a thread's in index order, a SIMD-group's
   by one shuffle tree.

   This follows nx.amd's sp_tridiag.h. */

#ifndef NX_METAL_SP_TRIDIAG_H
#define NX_METAL_SP_TRIDIAG_H

#include "sp_block.h"

namespace sp {

#define NB NX_SP_TNB

/* Workspace a matrix, [per] bytes apart: A (n²), W (n x NB), Vc
   (n x NB, row-major), T and G (NB²), τ (n); d, e (float); flags ok and
   the scale's exponent. */
template <class T> struct td_ws {
  device T *a, *w, *vc, *t, *g, *tau;
  device float *d, *e;
  device int *ok, *ex;
  device metal::atomic_uint *scan;
};

/* Matrix [b]'s workspace, [tb] an element's bytes: a step of parts alone
   takes its format's workspace as T = float. */
template <class T>
static inline td_ws<T> td_of(device uchar *base, long b, int n, ulong per,
                             int tb = sizeof(T)) {
  td_ws<T> ws;
  const ulong nn = ulong(n) * n;
  device uchar *p = base + ulong(b) * per;
  ws.a = (device T *)p;
  ws.w = (device T *)(p += nn * tb);
  ws.vc = (device T *)(p += ulong(n) * NB * tb);
  ws.t = (device T *)(p += ulong(n) * NB * tb);
  ws.g = (device T *)(p += NB * NB * tb);
  ws.tau = (device T *)(p += NB * NB * tb);
  ws.d = (device float *)(p + ulong(n) * tb);
  ws.e = ws.d + n;
  ws.ok = (device int *)(ws.e + n);
  ws.ex = ws.ok + 1;
  ws.scan = (device metal::atomic_uint *)(ws.ex + 1);
  return ws;
}

/* Stage: the Hermitian matrix eigh reads (lower triangle, real diagonal),
   flushed, whole into A, scaled by a power of 2 into nx.cpu's range; a
   matrix that is not finite where read is staged as 0 and flagged. [P]
   parts an element. Three phases over the units from [first] to
   [count] − 1: 0 clears a matrix's scan words, a thread a matrix; 1
   scans a row of the lower triangle for a value that is not finite and
   for the largest magnitude, a thread a row (matrix, row); 2 stages an
   element, a thread an element (matrix, element), element 0 writing the
   flag and the exponent. A maximum and a flag do not depend on the order
   the rows reach them. [du] and [dn] are the divisor words of a phase's
   units a matrix and of n. */
static inline void td_stage(grp g, long gx, long ngx, device const float *src,
                            device uchar *wsp, int n, int P, ulong per,
                            int phase, ulong du, ulong dn, long first,
                            long count) {
  const ulong nn = ulong(n) * n;
  const float eps = lim::eps(), tiny = lim::tiny();
  const float rmin = sqrt_(tiny / (2 * eps)), rmax = 1 / rmin;
  const bool small = is_small(count), inner = is_small(long(nn));
  constexpr metal::memory_order relaxed = metal::memory_order_relaxed;
  for (long u = first + gx * g.nthr + g.tid; u < count; u += ngx * g.nthr) {
    long b = u;
    const ulong k = phase == 0 ? 0
                    : phase == 1 ? ulong(split(b, n, du, small))
                                 : ulong(split(b, long(nn), du, small));
    const td_ws<float> ws = td_of<float>(wsp, b, n, per, P * sizeof(float));
    device const float *x = src + ulong(b) * nn * P;
    if (phase == 0) {
      metal::atomic_store_explicit(&ws.scan[0], 0u, relaxed);
      metal::atomic_store_explicit(&ws.scan[1], 0u, relaxed);
      continue;
    }
    if (phase == 1) {
      const int i = int(k);
      bool bad = false;
      float big = 0;
      for (int j = 0; j <= i; j++) {
        const ulong e = ulong(i) * n + j;
        const float re_ = flush(x[e * P]);
        const float im_ = i == j || P == 1 ? 0 : flush(x[e * P + 1]);
        bad = bad || (re_ - re_) + (im_ - im_) != 0;
        big = fmax_(big, fmax_(fabs_(re_), fabs_(im_)));
      }
      if (bad) metal::atomic_store_explicit(&ws.scan[0], 1u, relaxed);
      metal::atomic_fetch_max_explicit(&ws.scan[1], as_type<uint>(big),
                                       relaxed);
      continue;
    }
    const bool ok = metal::atomic_load_explicit(&ws.scan[0], relaxed) == 0;
    const float big =
        as_type<float>(metal::atomic_load_explicit(&ws.scan[1], relaxed));
    const int ex = scale_exp(big, rmin, rmax);
    long col = long(k); /* k = i + n j, column-major */
    const int i = int(split(col, n, dn, inner)), j = int(col);
    device const float *v =
        x + (i >= j ? ulong(i) * n + j : ulong(j) * n + i) * P;
    device float *a = ws.a + k * P;
    a[0] = ok ? scaled(flush(v[0]), ex) : 0.0f;
    if (P == 2) {
      const float vi = flush(v[1]);
      a[1] = ok && i != j ? scaled(i > j ? vi : -vi, ex) : 0.0f;
    }
    if (k == 0) {
      ws.ok[0] = ok;
      ws.ex[0] = ex;
    }
  }
}

/* A SIMD-group's sum of [x], by one tree: lane l adds lane l + o's, o
   from 16 down to 1, a lane past the group taking its own, and every
   lane takes lane 0's. Every lane of the SIMD-group calls it. */
static inline float warp_sum(float x) {
  for (ushort o = 16; o > 0; o >>= 1) x += metal::simd_shuffle_down(x, o);
  return metal::simd_broadcast(x, ushort(0));
}
static inline cf warp_sum(cf x) { return C(warp_sum(x.re), warp_sum(x.im)); }

/* Step: for panel [j0] of [nbk] columns, finish W's column i − 1 (its
   product with the trailing matrix is in place), then bring column
   c = j0 + i up to date and make its reflector, where i < nbk. One group
   of NX_SP_THREADS threads a matrix. Threadgroup memory: a float a
   thread from byte 0, then t1 and t2, NB elements each, from bytes
   NX_SP_TD_T1 and NX_SP_TD_T2. */
template <class T>
static inline void td_step(grp g, long gx, long ngx, threadgroup uchar *lds,
                           device uchar *wsp, int n, int j0, int nbk, int i,
                           ulong per, long first, long count) {
  threadgroup float *sh = (threadgroup float *)lds;
  threadgroup T *t1 = (threadgroup T *)(lds + NX_SP_TD_T1),
                *t2 = (threadgroup T *)(lds + NX_SP_TD_T2);
  const int t = g.tid, nt = g.nthr, lane = t % 32, warp = t / 32,
            nw = nt / 32;
  for (long b = first + gx; b < count; b += ngx) {
    const td_ws<T> ws = td_of<T>(wsp, b, n, per);
    device T *a = ws.a, *w = ws.w;
    if (i > 0) {
      /* W's column p = i − 1 of reflector cp = j0 + p, v = A(cp+1:n, cp):
         y −= A V₁ (W₁ᴴ v) and W₁ (V₁ᴴ v) over the panel's earlier
         columns, then y τ and the rank-one correction. */
      const int p = i - 1, cp = j0 + p, r0 = cp + 1;
      const T tau = ws.tau[cp];
      for (int q = warp; q < p; q += nw) {
        T s1 = elt<T>(0), s2 = elt<T>(0);
        for (int r = r0 + lane; r < n; r += 32) {
          const T v = M(a, r, cp);
          s1 += conj(T(M(w, r, q))) * v;
          s2 += conj(T(M(a, r, j0 + q))) * v;
        }
        s1 = warp_sum(s1);
        s2 = warp_sum(s2);
        if (lane == 0) t1[q] = s1, t2[q] = s2;
      }
      barrier();
      for (int r = r0 + t; r < n; r += nt) {
        T y = M(w, r, p);
        for (int q = 0; q < p; q++)
          y -= T(M(a, r, j0 + q)) * T(t1[q]) + T(M(w, r, q)) * T(t2[q]);
        M(w, r, p) = tau * y;
      }
      barrier();
      T s = elt<T>(0);
      for (int r = r0 + t; r < n; r += nt)
        s += conj(T(M(w, r, p))) * T(M(a, r, cp));
      T al = elt<T>(bsum(g, re(s), sh),
                    real_of<T>::cplx() ? bsum(g, im(s), sh) : 0.0f);
      al = elt<T>(-0.5f) * tau * al;
      for (int r = r0 + t; r < n; r += nt)
        M(w, r, p) = T(M(w, r, p)) + al * T(M(a, r, cp));
      barrier();
    }
    if (i == nbk) continue;
    const int c = j0 + i;
    /* A(c:n, c) −= A(c:n, j0:c) conj(W(c, 0:i))ᵀ
                   + W(c:n, 0:i) conj(A(c, j0:c))ᵀ */
    for (int r = c + t; r < n && i > 0; r += nt) {
      T y = M(a, r, c);
      for (int q = 0; q < i; q++)
        y -= T(M(a, r, j0 + q)) * conj(T(M(w, c, q))) +
             T(M(w, r, q)) * conj(T(M(a, c, j0 + q)));
      M(a, r, c) = r == c ? elt<T>(re(y)) : y;
    }
    barrier();
    if (c < n - 1) {
      device T *alpha = &M(a, c + 1, c);
      const T tau = reflector<T>(g, alpha, alpha + 1, n - c - 2, 1, sh);
      if (t == 0) {
        ws.e[c] = re(T(*alpha));
        ws.tau[c] = tau;
        *alpha = elt<T>(1);
      }
    } else if (t == 0)
      ws.tau[c] = elt<T>(0);
    barrier();
  }
}

/* W(c+1:n, i) = A(c+1:n, c+1:n) v, v = A(c+1:n, c): a SIMD-group a row
   r, reading column r, conj(A(q, r)) = A(r, q). A launch takes the units
   (matrix, rows of a group) from [first] to [count] − 1. */
template <class T>
static inline void td_hemv(grp g, long gx, long ngx, device uchar *wsp, int n,
                           int c, int i, ulong per, long first, long count) {
  const int lane = g.tid % 32, wpb = g.nthr / 32;
  const int rows = n - c - 1, units = (rows + wpb - 1) / wpb;
  const bool small = is_small(count);
  for (long u = first + gx; u < count; u += ngx) {
    long b = u;
    const int r = c + 1 + int(split(b, units, small)) * wpb + g.tid / 32;
    if (r >= n) continue;
    const td_ws<T> ws = td_of<T>(wsp, b, n, per);
    device const T *col = &M(ws.a, 0, r), *v = &M(ws.a, 0, c);
    T s = elt<T>(0);
    for (int q = c + 1 + lane; q < n; q += 32) s += conj(T(col[q])) * T(v[q]);
    s = warp_sum(s);
    if (lane == 0) M(ws.w, r, i) = s;
  }
}

/* d from A's diagonal, once every update is in: a thread an element of
   the matrices' diagonals, from [first] to [count] − 1. */
static inline void td_diag(grp g, long gx, long ngx, device uchar *wsp, int n,
                           int P, ulong per, long first, long count) {
  const bool small = is_small(count);
  for (long x = first + gx * g.nthr + g.tid; x < count; x += ngx * g.nthr) {
    long b = x;
    const int c = int(split(b, n, small));
    const td_ws<float> ws = td_of<float>(wsp, b, n, per, P * sizeof(float));
    ws.d[c] = ws.a[(c + ulong(c) * n) * P];
  }
}

/* Vc (row-major (n − j0 − 1) x nbk) the panel's reflectors, zeros above
   their units; then G = Vcᴴ Vc runs as a product, then larft
   (sp_block.h). A thread an element of the matrices' Vc. */
static inline void td_vcopy(grp g, long gx, long ngx, device uchar *wsp, int n,
                            int j0, int nbk, int P, ulong per, long first,
                            long count) {
  const int m = n - j0 - 1;
  const long els = long(m) * nbk;
  const bool small = is_small(count);
  for (long x = first + gx * g.nthr + g.tid; x < count; x += ngx * g.nthr) {
    long b = x;
    const int k = int(split(b, els, small));
    long row = k;
    const int p = int(split(row, nbk, small)), r = j0 + 1 + int(row);
    const td_ws<float> ws = td_of<float>(wsp, b, n, per, P * sizeof(float));
    const int c = j0 + p;
    for (int q = 0; q < P; q++)
      ws.vc[ulong(k) * P + q] =
          r < c + 1 ? 0.0f : ws.a[(r + ulong(c) * n) * P + q];
  }
}

/* Z (column-major, T) from the eigenvectors of T (real): a thread an
   element of the matrices' Z. */
static inline void td_zinit(grp g, long gx, long ngx, device const float *q,
                            ulong qs, device float *z, int n, int P,
                            long first, long count) {
  const ulong nn = ulong(n) * n;
  const bool small = is_small(count);
  for (long x = first + gx * g.nthr + g.tid; x < count; x += ngx * g.nthr) {
    long b = x;
    const long k = split(b, long(nn), small);
    z[x * P] = q[ulong(b) * qs + k];
    if (P == 2) z[x * P + 1] = 0;
  }
}

/* After the back-transformation, eigh's results: w = d scaled back,
   v = Z row-major; NaN for a flagged matrix. A thread an element of the
   matrices' v. [dcx] is the solver's failure flag then its exponent. */
static inline void td_out(grp g, long gx, long ngx, device uchar *wsp,
                          device const float *d, ulong ds,
                          device const int *dcx, ulong es,
                          device const float *z, device float *w,
                          device float *v, int n, int P, ulong per,
                          long first, long count) {
  const ulong nn = ulong(n) * n;
  const bool small = is_small(count);
  for (long x = first + gx * g.nthr + g.tid; x < count; x += ngx * g.nthr) {
    long b = x;
    const ulong k = ulong(split(b, long(nn), small));
    const td_ws<float> ws = td_of<float>(wsp, b, n, per, P * sizeof(float));
    const bool ok = ws.ok[0] && !(dcx && dcx[ulong(b) * es]);
    const int ex = ws.ex[0] + (dcx ? dcx[ulong(b) * es + 1] : 0);
    if (k < ulong(n))
      w[ulong(b) * n + k] =
          ok ? ldexp_(d[ulong(b) * ds + k], -ex) + 0.0f : nan_();
    if (!v) continue;
    long row = long(k);
    const int j = int(split(row, n, small)), i = int(row);
    for (int q = 0; q < P; q++)
      v[ulong(x) * P + q] =
          ok ? z[(ulong(b) * nn + i + ulong(j) * n) * P + q] : nan_();
  }
}

#undef NB

} // namespace sp

#endif
