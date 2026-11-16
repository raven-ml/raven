/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Block-wide helpers shared by the one-block-per-matrix routines: fixed-
   tree reductions, the 2-norm, Householder reflectors and their
   application, as LAPACK's xNRM2, xLARFG and xLARF. Every thread of the
   block calls each helper with the same arguments; matrices are
   column-major, element (i, j) at x[i + j·n]. */

#ifndef NX_AMD_SP_BLOCK_H
#define NX_AMD_SP_BLOCK_H

#include "sp_common.h"

#ifndef M
#define M(x, i, j) (x)[(size_t)(i) + (size_t)(j) * n]
#endif

/* Block reductions in one fixed tree, every thread getting the result.
   nthr() is a power of two. [sh] holds nthr() values. */
template <class R> D R bsum(R x, R *sh) {
  const int t = tid();
  barrier();
  sh[t] = x;
  barrier();
  #pragma unroll 1
  for (int s = nthr() / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    barrier();
  }
  return sh[0];
}
template <class R> D R bmax(R x, R *sh) {
  const int t = tid();
  barrier();
  sh[t] = x;
  barrier();
  #pragma unroll 1
  for (int s = nthr() / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] = fmax(sh[t], sh[t + s]);
    barrier();
  }
  return sh[0];
}
/* The first index of the largest value; [x] is -1 where a thread has
   none. */
template <class R> D int bargmax(R x, int i, R *sh, int *si) {
  const int t = tid();
  barrier();
  sh[t] = x;
  si[t] = i;
  barrier();
  #pragma unroll 1
  for (int s = nthr() / 2; s > 0; s >>= 1) {
    if (t < s && (sh[t + s] > sh[t] || (sh[t + s] == sh[t] && si[t + s] < si[t]))) {
      sh[t] = sh[t + s];
      si[t] = si[t + s];
    }
    barrier();
  }
  return si[0];
}

template <class R> D R safmin() { return lim<R>::tiny() / lim<R>::eps(); }

/* sqrt(a² + b² + c²), scaled (xLAPY3). */
template <class R> D R lapy3(R a, R b, R c) {
  R xa = fabs(a), xb = fabs(b), xc = fabs(c);
  R w = fmax(xa, fmax(xb, xc));
  if (w == 0 || w != w) return xa + xb + xc;
  xa /= w, xb /= w, xc /= w;
  return w * sqrt(xa * xa + xb * xb + xc * xc);
}

/* The 2-norm of the [len] elements at [x], [s] apart: the sum of
   squares, or scaled by the largest part where that sum leaves the safe
   range. */
template <class T>
D typename real_of<T>::R bnorm(const T *x, int64_t len, size_t s,
                               typename real_of<T>::R *sh) {
  typedef typename real_of<T>::R R;
  R q = 0;
  #pragma unroll 1
  for (int64_t i = tid(); i < len; i += nthr()) q += abs2(x[i * s]);
  q = bsum(q, sh);
  const R sm = safmin<R>();
  if ((q >= sm && q <= 1 / sm) || q != q) return sqrt(q);
  R big = 0;
  #pragma unroll 1
  for (int64_t i = tid(); i < len; i += nthr())
    big = fmax(big, fmax(fabs(re(x[i * s])), fabs(im(x[i * s]))));
  big = bmax(big, sh);
  if (big == 0) return sqrt(q);
  R t = 0;
  #pragma unroll 1
  for (int64_t i = tid(); i < len; i += nthr())
    t += abs2(x[i * s] / elt<T>(big));
  return big * sqrt(bsum(t, sh));
}

/* The reflector H = I − τ v vᴴ with Hᴴ [α; x] = [β; 0], β real (xLARFG):
   v = [1; x'], x' over the [len] elements at [x], [s] apart, β over
   [*alpha] (written by thread 0, after a barrier). Returns τ. */
template <class T>
D T reflector(T *alpha, T *x, int64_t len, size_t s, typename real_of<T>::R *sh) {
  typedef typename real_of<T>::R R;
  const R sm = safmin<R>();
  R xnorm = bnorm(x, len, s, sh);
  T a0 = *alpha;
  R ar = re(a0), ai = im(a0);
  if (xnorm == 0 && ai == 0) return elt<T>((R)0);
  R beta = ar >= 0 ? -lapy3(ar, ai, xnorm) : lapy3(ar, ai, xnorm);
  int knt = 0;
  if (fabs(beta) < sm) {
    do {
      knt++;
      barrier();
      #pragma unroll 1
      for (int64_t i = tid(); i < len; i += nthr())
        x[i * s] = x[i * s] * elt<T>(1 / sm);
      beta *= 1 / sm;
      ar *= 1 / sm;
      ai *= 1 / sm;
    } while (fabs(beta) < sm && knt < 20);
    xnorm = bnorm(x, len, s, sh);
    R an = lapy3(ar, ai, xnorm);
    beta = ar >= 0 ? -an : an;
  }
  T tau = elt<T>((beta - ar) / beta, -(ai / beta));
  T sc = elt<T>(ar - beta, ai);
  barrier();
  #pragma unroll 1
  for (int64_t i = tid(); i < len; i += nthr())
    x[i * s] = cdiv(x[i * s], sc);
  #pragma unroll 1
  for (int j = 0; j < knt; j++) beta *= sm;
  barrier();
  if (tid() == 0) *alpha = elt<T>(beta);
  barrier();
  return tau;
}

/* c ← (I − τ v vᴴ) c, c the [m] × [k] block at [c], columns [n] apart:
   each thread a column (xLARF from the left). */
template <class T>
D void larf_left(const T *v, T tau, int m, int k, T *c, int n) {
  typedef typename real_of<T>::R R;
  if (tau == elt<T>((R)0)) return;
  barrier();
  #pragma unroll 1
  for (int j = tid(); j < k; j += nthr()) {
    T s = elt<T>((R)0);
    #pragma unroll 1
    for (int i = 0; i < m; i++) s += conj(v[i]) * M(c, i, j);
    s = tau * s;
    #pragma unroll 1
    for (int i = 0; i < m; i++) M(c, i, j) -= v[i] * s;
  }
  barrier();
}

/* c ← c (I − τ v vᴴ), each thread a row (xLARF from the right). */
template <class T>
D void larf_right(const T *v, T tau, int m, int k, T *c, int n) {
  typedef typename real_of<T>::R R;
  if (tau == elt<T>((R)0)) return;
  barrier();
  #pragma unroll 1
  for (int i = tid(); i < m; i += nthr()) {
    T s = elt<T>((R)0);
    #pragma unroll 1
    for (int j = 0; j < k; j++) s += M(c, i, j) * v[j];
    #pragma unroll 1
    for (int j = 0; j < k; j++) M(c, i, j) -= s * (tau * conj(v[j]));
  }
  barrier();
}

/* T of the block reflector I − V T Vᴴ = H(0) ⋯ H(nbk − 1) (xLARFT,
   forward, columnwise) of [nb] matrices, from G = Vᴴ V, nbk × nbk:
   T(i, i) = τ_i, T(0:i, i) = −τ_i T(0:i, 0:i) G(0:i, i), T's rows
   NX_SP_TNB apart. [tm], [g] and [tau] are the first matrix's, each next
   one's NX_SP_PER bytes on. One group of at least [nbk] work-items a
   matrix: eigh's and svd's panels alike. */
template <class T>
D void larft(T *tm, const T *g, const T *tau, int64_t nb, int nbk) {
  typedef typename real_of<T>::R R;
  const size_t per = (size_t)uniform(staged()[NX_SP_PER]);
  #pragma unroll 1
  for (int64_t b = bid_x(); b < nb; b += ngx()) {
    T *t = (T *)((char *)tm + (size_t)b * per);
    const T *gb = (const T *)((const char *)g + (size_t)b * per);
    const T *taub = (const T *)((const char *)tau + (size_t)b * per);
    #pragma unroll 1
    for (int i = 0; i < nbk; i++) {
      int j = tid();
      T y = elt<T>((R)0);
      if (j < i) {
        #pragma unroll 1
        for (int p = j; p < i; p++) y += t[j * NX_SP_TNB + p] * gb[p * nbk + i];
        y = -(taub[i] * y);
      }
      barrier();
      if (j < i) t[j * NX_SP_TNB + i] = y;
      if (j == i) t[i * NX_SP_TNB + i] = taub[i];
      if (j > i && j < nbk) t[j * NX_SP_TNB + i] = elt<T>((R)0);
      barrier();
    }
  }
}

#endif
