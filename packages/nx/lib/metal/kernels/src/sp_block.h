/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Group-wide helpers shared by the one-group-a-matrix steps: fixed-tree
   reductions, the 2-norm, Householder reflectors and their application,
   as LAPACK's xNRM2, xLARFG and xLARF. Every thread of the group calls
   each helper with the same arguments; matrices are column-major, element
   (i, j) at x[i + j n]. A helper's data lie in device or threadgroup
   memory, as its pointer's type [P] says; its partials in threadgroup
   memory, a float a thread.

   This follows nx.amd's sp_block.h. */

#ifndef NX_METAL_SP_BLOCK_H
#define NX_METAL_SP_BLOCK_H

#include "sp_common.h"

#ifndef M
#define M(x, i, j) (x)[(ulong)(i) + (ulong)(j) * n]
#endif

namespace sp {

/* Group reductions in one fixed tree, every thread getting the result.
   g.nthr is a power of two. [sh] holds g.nthr values. */
static inline float bsum(grp g, float x, threadgroup float *sh) {
  const int t = g.tid;
  barrier();
  sh[t] = x;
  barrier();
  for (int s = g.nthr / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    barrier();
  }
  return sh[0];
}
static inline float bmax(grp g, float x, threadgroup float *sh) {
  const int t = g.tid;
  barrier();
  sh[t] = x;
  barrier();
  for (int s = g.nthr / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] = fmax_(sh[t], sh[t + s]);
    barrier();
  }
  return sh[0];
}
/* The first index of the largest value; [x] is -1 where a thread has
   none. */
static inline int bargmax(grp g, float x, int i, threadgroup float *sh,
                          threadgroup int *si) {
  const int t = g.tid;
  barrier();
  sh[t] = x;
  si[t] = i;
  barrier();
  for (int s = g.nthr / 2; s > 0; s >>= 1) {
    if (t < s &&
        (sh[t + s] > sh[t] || (sh[t + s] == sh[t] && si[t + s] < si[t]))) {
      sh[t] = sh[t + s];
      si[t] = si[t + s];
    }
    barrier();
  }
  return si[0];
}

static inline float safmin() { return lim::tiny() / lim::eps(); }

/* sqrt(a² + b² + c²), scaled (xLAPY3). */
static inline float lapy3(float a, float b, float c) {
  float xa = fabs_(a), xb = fabs_(b), xc = fabs_(c);
  const float w = fmax_(xa, fmax_(xb, xc));
  if (w == 0 || w != w) return xa + xb + xc;
  xa /= w, xb /= w, xc /= w;
  return w * sqrt_(xa * xa + xb * xb + xc * xc);
}

/* The 2-norm of the [len] elements of T at [x], [s] apart: the sum of
   squares, or scaled by the largest part where that sum leaves the safe
   range. */
template <class T, class P>
static inline float bnorm(grp g, P x, long len, ulong s,
                          threadgroup float *sh) {
  float q = 0;
  for (long i = g.tid; i < len; i += g.nthr) q += abs2(T(x[i * s]));
  q = bsum(g, q, sh);
  const float sm = safmin();
  if ((q >= sm && q <= 1 / sm) || q != q) return sqrt_(q);
  float big = 0;
  for (long i = g.tid; i < len; i += g.nthr) {
    const T xi = x[i * s];
    big = fmax_(big, fmax_(fabs_(re(xi)), fabs_(im(xi))));
  }
  big = bmax(g, big, sh);
  if (big == 0) return sqrt_(q);
  float t = 0;
  for (long i = g.tid; i < len; i += g.nthr)
    t += abs2(cdiv(T(x[i * s]), elt<T>(big)));
  return big * sqrt_(bsum(g, t, sh));
}

/* The reflector H = I − τ v vᴴ with Hᴴ [α; x] = [β; 0], β real (xLARFG):
   v = [1; x'], x' over the [len] elements at [x], [s] apart, β over
   [*alpha] (written by thread 0, after a barrier). Returns τ. */
template <class T, class P>
static inline T reflector(grp g, P alpha, P x, long len, ulong s,
                          threadgroup float *sh) {
  const float sm = safmin();
  float xnorm = bnorm<T>(g, x, len, s, sh);
  const T a0 = *alpha;
  float ar = re(a0), ai = im(a0);
  if (xnorm == 0 && ai == 0) return elt<T>(0);
  float beta = ar >= 0 ? -lapy3(ar, ai, xnorm) : lapy3(ar, ai, xnorm);
  int knt = 0;
  if (fabs_(beta) < sm) {
    do {
      knt++;
      barrier();
      for (long i = g.tid; i < len; i += g.nthr)
        x[i * s] = T(x[i * s]) * elt<T>(1 / sm);
      beta *= 1 / sm;
      ar *= 1 / sm;
      ai *= 1 / sm;
    } while (fabs_(beta) < sm && knt < 20);
    xnorm = bnorm<T>(g, x, len, s, sh);
    const float an = lapy3(ar, ai, xnorm);
    beta = ar >= 0 ? -an : an;
  }
  const T tau = elt<T>((beta - ar) / beta, -(ai / beta));
  const T sc = elt<T>(ar - beta, ai);
  barrier();
  for (long i = g.tid; i < len; i += g.nthr) x[i * s] = cdiv(T(x[i * s]), sc);
  for (int j = 0; j < knt; j++) beta *= sm;
  barrier();
  if (g.tid == 0) *alpha = elt<T>(beta);
  barrier();
  return tau;
}

/* c ← (I − τ v vᴴ) c, c the [m] x [k] block at [c], columns [n] apart:
   each thread a column (xLARF from the left). */
template <class T, class P, class Q>
static inline void larf_left(grp g, P v, T tau, int m, int k, Q c, int n) {
  if (tau == elt<T>(0)) return;
  barrier();
  for (int j = g.tid; j < k; j += g.nthr) {
    T s = elt<T>(0);
    for (int i = 0; i < m; i++) s += conj(T(v[i])) * T(M(c, i, j));
    s = tau * s;
    for (int i = 0; i < m; i++) M(c, i, j) = T(M(c, i, j)) - T(v[i]) * s;
  }
  barrier();
}

/* c ← c (I − τ v vᴴ), each thread a row (xLARF from the right). */
template <class T, class P, class Q>
static inline void larf_right(grp g, P v, T tau, int m, int k, Q c, int n) {
  if (tau == elt<T>(0)) return;
  barrier();
  for (int i = g.tid; i < m; i += g.nthr) {
    T s = elt<T>(0);
    for (int j = 0; j < k; j++) s += T(M(c, i, j)) * T(v[j]);
    for (int j = 0; j < k; j++)
      M(c, i, j) = T(M(c, i, j)) - s * (tau * conj(T(v[j])));
  }
  barrier();
}

/* T of the block reflector I − V T Vᴴ = H(0) ⋯ H(nbk − 1) (xLARFT,
   forward, columnwise) of the matrices [first, count), from G = Vᴴ V,
   nbk x nbk: T(i, i) = τ_i, T(0:i, i) = −τ_i T(0:i, 0:i) G(0:i, i), T's
   rows NX_SP_TNB apart. [tm], [gm] and [tau] are matrix 0's, each next
   one's [per] bytes on. One group of at least [nbk] threads a matrix:
   eigh's and svd's panels alike. */
template <class T>
static inline void larft(grp g, long gx, long ngx, device uchar *tm,
                         device const uchar *gm, device const uchar *tau,
                         long first, long count, int nbk, ulong per) {
  for (long b = first + gx; b < count; b += ngx) {
    device T *t = (device T *)(tm + ulong(b) * per);
    device const T *gb = (device const T *)(gm + ulong(b) * per);
    device const T *taub = (device const T *)(tau + ulong(b) * per);
    for (int i = 0; i < nbk; i++) {
      const int j = g.tid;
      T y = elt<T>(0);
      if (j < i) {
        for (int p = j; p < i; p++)
          y += T(t[j * NX_SP_TNB + p]) * T(gb[p * nbk + i]);
        y = -(T(taub[i]) * y);
      }
      barrier();
      if (j < i) t[j * NX_SP_TNB + i] = y;
      if (j == i) t[i * NX_SP_TNB + i] = taub[i];
      if (j > i && j < nbk) t[j * NX_SP_TNB + i] = elt<T>(0);
      barrier();
    }
  }
}

} // namespace sp

#endif
