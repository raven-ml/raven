/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Block-wide helpers shared by the one-block-per-matrix routines: fixed-
   tree reductions, the 2-norm, Householder reflectors and their
   application, as LAPACK's xNRM2, xLARFG and xLARF. Every thread of the
   block calls each helper with the same arguments; matrices are
   column-major, element (i, j) at x[i + j·n]. */

#ifndef SPECTRAL_BLOCK_CUH
#define SPECTRAL_BLOCK_CUH

#include "sp_common.cuh"

#ifndef M
#define M(x, i, j) (x)[(size_t)(i) + (size_t)(j) * n]
#endif

/* Block reductions in one fixed tree, every thread getting the result.
   blockDim.x is a power of two. [sh] holds blockDim.x values. */
template <class R> D R bsum(R x, R *sh) {
  const int t = threadIdx.x;
  __syncthreads();
  sh[t] = x;
  __syncthreads();
  #pragma unroll 1
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    __syncthreads();
  }
  return sh[0];
}
template <class R> D R bmax(R x, R *sh) {
  const int t = threadIdx.x;
  __syncthreads();
  sh[t] = x;
  __syncthreads();
  #pragma unroll 1
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] = fmax(sh[t], sh[t + s]);
    __syncthreads();
  }
  return sh[0];
}
/* The first index of the largest value; [x] is -1 where a thread has
   none. */
template <class R> D int bargmax(R x, int i, R *sh, int *si) {
  const int t = threadIdx.x;
  __syncthreads();
  sh[t] = x;
  si[t] = i;
  __syncthreads();
  #pragma unroll 1
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s && (sh[t + s] > sh[t] || (sh[t + s] == sh[t] && si[t + s] < si[t]))) {
      sh[t] = sh[t + s];
      si[t] = si[t + s];
    }
    __syncthreads();
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
__device__ __noinline__ typename real_of<T>::R bnorm(const T *x, int64_t len, size_t s,
                               typename real_of<T>::R *sh) {
  typedef typename real_of<T>::R R;
  R q = 0;
  #pragma unroll 1
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x) q += abs2(x[i * s]);
  q = bsum(q, sh);
  const R sm = safmin<R>();
  if ((q >= sm && q <= 1 / sm) || q != q) return sqrt(q);
  R big = 0;
  #pragma unroll 1
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x)
    big = fmax(big, fmax(fabs(re(x[i * s])), fabs(im(x[i * s]))));
  big = bmax(big, sh);
  if (big == 0) return sqrt(q);
  R t = 0;
  #pragma unroll 1
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x)
    t += abs2(x[i * s] / elt<T>(big));
  return big * sqrt(bsum(t, sh));
}

/* The reflector H = I − τ v vᴴ with Hᴴ [α; x] = [β; 0], β real (xLARFG):
   v = [1; x'], x' over the [len] elements at [x], [s] apart, β over
   [*alpha] (written by thread 0, after a barrier). Returns τ. */
template <class T>
__device__ __noinline__ T reflector(T *alpha, T *x, int64_t len, size_t s, typename real_of<T>::R *sh) {
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
      __syncthreads();
      #pragma unroll 1
      for (int64_t i = threadIdx.x; i < len; i += blockDim.x)
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
  __syncthreads();
  #pragma unroll 1
  for (int64_t i = threadIdx.x; i < len; i += blockDim.x)
    x[i * s] = cdiv(x[i * s], sc);
  #pragma unroll 1
  for (int j = 0; j < knt; j++) beta *= sm;
  __syncthreads();
  if (threadIdx.x == 0) *alpha = elt<T>(beta);
  __syncthreads();
  return tau;
}

/* c ← (I − τ v vᴴ) c, c the [m] × [k] block at [c], columns [n] apart:
   each thread a column (xLARF from the left). */
template <class T>
__device__ __noinline__ void larf_left(const T *v, T tau, int m, int k, T *c, int n) {
  typedef typename real_of<T>::R R;
  if (tau == elt<T>((R)0)) return;
  __syncthreads();
  #pragma unroll 1
  for (int j = threadIdx.x; j < k; j += blockDim.x) {
    T s = elt<T>((R)0);
    #pragma unroll 1
    for (int i = 0; i < m; i++) s += conj(v[i]) * M(c, i, j);
    s = tau * s;
    #pragma unroll 1
    for (int i = 0; i < m; i++) M(c, i, j) -= v[i] * s;
  }
  __syncthreads();
}

/* c ← c (I − τ v vᴴ), each thread a row (xLARF from the right). */
template <class T>
__device__ __noinline__ void larf_right(const T *v, T tau, int m, int k, T *c, int n) {
  typedef typename real_of<T>::R R;
  if (tau == elt<T>((R)0)) return;
  __syncthreads();
  #pragma unroll 1
  for (int i = threadIdx.x; i < m; i += blockDim.x) {
    T s = elt<T>((R)0);
    #pragma unroll 1
    for (int j = 0; j < k; j++) s += M(c, i, j) * v[j];
    #pragma unroll 1
    for (int j = 0; j < k; j++) M(c, i, j) -= s * (tau * conj(v[j]));
  }
  __syncthreads();
}


#endif
