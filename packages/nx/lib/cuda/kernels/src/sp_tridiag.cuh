/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* eigh past the small path: a = Q T Qᴴ with T real tridiagonal, as
   LAPACK's blocked xHETRD computes it (lower), then T's eigensystem
   (sp_stedc.cuh), then v = Q Z by the reflectors' blocks (xUNMTR's xLARFB).

   The reduction takes panels of NB columns. For each column c of a
   panel, td_step (one block per matrix) brings the column up to date
   with the panel's earlier reflectors through W (xLATRD), makes its
   reflector, and finishes W's previous column; td_hemv (a warp per row,
   every matrix) is W's product with the trailing matrix, the step that
   reads the matrix once per column. The panel's trailing update is two
   products, A22 −= V Wᴴ + W Vᴴ, both triangles, so that the trailing
   matrix stays whole and td_hemv reads columns, contiguous.

   Matrices are column-major here, element (i, j) at x[i + j·n]. Every
   sum runs in a fixed order: a thread's in index order, a warp's by one
   shuffle tree. */

#include "sp_block.cuh"

#define NB NX_SP_TNB

/* Workspace per matrix: A (n²), W (n × NB), Vc (n × NB, row-major), T
   and G (NB²), τ (n); d, e (R); flags ok and the scale's exponent. */
template <class T> struct td_ws {
  typedef typename real_of<T>::R R;
  T *a, *w, *vc, *t, *g, *tau;
  R *d, *e;
  int *ok, *ex;
  /* [tb] an element's bytes: a step of parts alone takes its format's
     workspace as T = R. */
  __device__ td_ws(void *base, int64_t b, int n, int tb = sizeof(T)) {
    size_t nn = (size_t)n * n;
    size_t per = nx_sp_td_per(n, tb, sizeof(R));
    char *p = (char *)base + (size_t)b * per;
    a = (T *)p;
    w = (T *)(p += nn * tb);
    vc = (T *)(p += (size_t)n * NB * tb);
    t = (T *)(p += (size_t)n * NB * tb);
    g = (T *)(p += NB * NB * tb);
    tau = (T *)(p += NB * NB * tb);
    d = (R *)(p + (size_t)n * tb);
    e = d + n;
    ok = (int *)(e + n);
    ex = ok + 1;
  }
};

/* Stage: the Hermitian matrix eigh reads (lower triangle, real diagonal)
   whole into A, scaled by a power of 2 into nx.cpu's range; a matrix
   that is not finite where read is staged as 0 and flagged. One block
   per matrix. */
template <class R>
__device__ __noinline__ void td_stage(const R *src, void *wsp, int64_t nb, int n, int P) {
  __shared__ R sh[256];
  const size_t nn = (size_t)n * n;
  const R eps = lim<R>::eps(), tiny = lim<R>::tiny();
  const R rmin = sqrt(tiny / (2 * eps)), rmax = 1 / rmin;
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    td_ws<R> ws(wsp, b, n, P * sizeof(R));
    const R *x = src + b * nn * P;
    R nan = 0, big = 0;
    #pragma unroll 1
    for (size_t k = threadIdx.x; k < nn; k += blockDim.x) {
      int i = (int)(k / n), j = (int)(k % n);
      if (j > i) continue;
      R re_ = x[k * P], im_ = i == j || P == 1 ? 0 : x[k * P + 1];
      nan += (re_ - re_) + (im_ - im_);
      big = fmax(big, fmax(fabs(re_), fabs(im_)));
    }
    bool ok = bsum(nan, sh) == 0;
    big = bmax(big, sh);
    int ex = scale_exp(big, rmin, rmax);
    #pragma unroll 1
    for (size_t k = threadIdx.x; k < nn; k += blockDim.x) {
      int i = (int)(k % n), j = (int)(k / n); /* column-major (i, j) */
      const R *v = x + (i >= j ? (size_t)i * n + j : (size_t)j * n + i) * P;
      R *a = ws.a + k * P;
      a[0] = ok ? scaled(v[0], ex) : (R)0;
      if (P == 2) a[1] = ok && i != j ? scaled(i > j ? v[1] : -v[1], ex) : (R)0;
    }
    if (threadIdx.x == 0) {
      ws.ok[0] = ok;
      ws.ex[0] = ex;
    }
  }
}

/* A warp's sum of [x], by one shuffle tree. */
template <class R> D R warp_sum(R x) {
  #pragma unroll 1
  for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
  return __shfl_sync(0xffffffffu, x, 0);
}
template <class R> D cx<R> warp_sum(cx<R> x) {
  return {warp_sum(x.re), warp_sum(x.im)};
}

/* Step: for panel [j0] of [nbk] columns, finish W's column i − 1 (its
   product with the trailing matrix is in place), then bring column
   c = j0 + i up to date and make its reflector, where i < nbk. One block
   of 256 threads per matrix. */
template <class T>
D void td_step(void *wsp, int64_t nb, int n, int j0, int nbk, int i) {
  typedef typename real_of<T>::R R;
  __shared__ R sh[256];
  __shared__ T t1[NB], t2[NB];
  const int t = threadIdx.x, nt = blockDim.x, lane = t % 32, warp = t / 32,
            nw = nt / 32;
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    td_ws<T> ws(wsp, b, n);
    T *a = ws.a, *w = ws.w;
    if (i > 0) {
      /* W's column p = i − 1 of reflector cp = j0 + p, v = A(cp+1:n, cp):
         y −= A V₁ (W₁ᴴ v) and W₁ (V₁ᴴ v) over the panel's earlier
         columns, then y τ and the rank-one correction. */
      const int p = i - 1, cp = j0 + p, r0 = cp + 1;
      const T tau = ws.tau[cp];
      #pragma unroll 1
      for (int q = warp; q < p; q += nw) {
        T s1 = elt<T>((R)0), s2 = elt<T>((R)0);
        #pragma unroll 1
        for (int r = r0 + lane; r < n; r += 32) {
          T v = M(a, r, cp);
          s1 += conj(M(w, r, q)) * v;
          s2 += conj(M(a, r, j0 + q)) * v;
        }
        s1 = warp_sum(s1);
        s2 = warp_sum(s2);
        if (lane == 0) t1[q] = s1, t2[q] = s2;
      }
      __syncthreads();
      #pragma unroll 1
      for (int r = r0 + t; r < n; r += nt) {
        T y = M(w, r, p);
        #pragma unroll 1
        for (int q = 0; q < p; q++)
          y -= M(a, r, j0 + q) * t1[q] + M(w, r, q) * t2[q];
        M(w, r, p) = tau * y;
      }
      __syncthreads();
      T s = elt<T>((R)0);
      #pragma unroll 1
      for (int r = r0 + t; r < n; r += nt) s += conj(M(w, r, p)) * M(a, r, cp);
      T al = elt<T>(bsum(re(s), sh), real_of<T>::cplx ? bsum(im(s), sh) : (R)0);
      al = elt<T>((R)-0.5) * tau * al;
      #pragma unroll 1
      for (int r = r0 + t; r < n; r += nt) M(w, r, p) += al * M(a, r, cp);
      __syncthreads();
    }
    if (i == nbk) continue;
    const int c = j0 + i;
    /* A(c:n, c) −= A(c:n, j0:c) conj(W(c, 0:i))ᵀ + W(c:n, 0:i) conj(A(c, j0:c))ᵀ */
    #pragma unroll 1
    for (int r = c + t; r < n && i > 0; r += nt) {
      T y = M(a, r, c);
      #pragma unroll 1
      for (int q = 0; q < i; q++)
        y -= M(a, r, j0 + q) * conj(M(w, c, q)) + M(w, r, q) * conj(M(a, c, j0 + q));
      M(a, r, c) = r == c ? elt<T>(re(y)) : y;
    }
    __syncthreads();
    if (c < n - 1) {
      T *alpha = &M(a, c + 1, c);
      T tau = reflector(alpha, alpha + 1, n - c - 2, 1, sh);
      if (t == 0) {
        ws.e[c] = re(*alpha);
        ws.tau[c] = tau;
        *alpha = elt<T>((R)1);
      }
    } else if (t == 0)
      ws.tau[c] = elt<T>((R)0);
    __syncthreads();
  }
}

/* W(c+1:n, i) = A(c+1:n, c+1:n) v, v = A(c+1:n, c): a warp per row r,
   reading column r, conj(A(q, r)) = A(r, q). */
template <class T>
D void td_hemv(void *wsp, int64_t nb, int n, int c, int i) {
  typedef typename real_of<T>::R R;
  const int lane = threadIdx.x % 32, wpb = blockDim.x / 32;
  const int rows = n - c - 1, per = (rows + wpb - 1) / wpb;
  #pragma unroll 1
  for (int64_t g = blockIdx.x; g < nb * per; g += gridDim.x) {
    int64_t b = g / per;
    int r = c + 1 + (int)(g % per) * wpb + threadIdx.x / 32;
    if (r >= n) continue;
    td_ws<T> ws(wsp, b, n);
    const T *col = &M(ws.a, 0, r), *v = &M(ws.a, 0, c);
    T s = elt<T>((R)0);
    #pragma unroll 1
    for (int q = c + 1 + lane; q < n; q += 32) s += conj(col[q]) * v[q];
    s = warp_sum(s);
    if (lane == 0) M(ws.w, r, i) = s;
  }
}

/* d from A's diagonal, once every update is in. */
template <class R> __device__ __noinline__ void td_diag(void *wsp, int64_t nb, int n, int P) {
  #pragma unroll 1
  for (int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       g < nb * (int64_t)n; g += (int64_t)gridDim.x * blockDim.x) {
    td_ws<R> ws(wsp, g / n, n, P * sizeof(R));
    int c = (int)(g % n);
    ws.d[c] = ws.a[(c + (size_t)c * n) * P];
  }
}

/* Vc (row-major (n − j0 − 1) × nbk) the panel's reflectors, zeros above
   their units; then G = Vcᴴ Vc runs as a product, then td_larft. */
template <class R>
__device__ __noinline__ void td_vcopy(void *wsp, int64_t nb, int n, int j0, int nbk, int P) {
  const int m = n - j0 - 1;
  #pragma unroll 1
  for (int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       g < nb * (int64_t)m * nbk; g += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = g / ((int64_t)m * nbk);
    int k = (int)(g % ((int64_t)m * nbk)), r = j0 + 1 + k / nbk, p = k % nbk;
    td_ws<R> ws(wsp, b, n, P * sizeof(R));
    int c = j0 + p;
    for (int q = 0; q < P; q++)
      ws.vc[(size_t)k * P + q] = r < c + 1 ? (R)0 : ws.a[(r + (size_t)c * n) * P + q];
  }
}

/* T of the block reflector I − V T Vᴴ = H(j0) ⋯ H(j0+nbk−1) (xLARFT,
   forward, columnwise), from G = Vᴴ V: T(i, i) = τ_i, T(0:i, i) =
   −τ_i T(0:i, 0:i) G(0:i, i). One block per matrix. */
template <class T>
D void td_larft(void *wsp, int64_t nb, int n, int j0, int nbk) {
  typedef typename real_of<T>::R R;
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    td_ws<T> ws(wsp, b, n);
    T *tm = ws.t, *g = ws.g;
    #pragma unroll 1
    for (int i = 0; i < nbk; i++) {
      T tau = ws.tau[j0 + i];
      int j = threadIdx.x;
      T y = elt<T>((R)0);
      if (j < i) {
        #pragma unroll 1
        for (int p = j; p < i; p++) y += tm[j * NB + p] * g[p * nbk + i];
        y = -(tau * y);
      }
      __syncthreads();
      if (j < i) tm[j * NB + i] = y;
      if (j == i) tm[i * NB + i] = tau;
      if (j > i && j < nbk) tm[j * NB + i] = elt<T>((R)0);
      __syncthreads();
    }
  }
}

/* Z (column-major, T) from the eigenvectors of T (real); then, after the
   back-transformation, eigh's results. */
template <class R>
__device__ __noinline__ void td_zinit(const R *q, size_t qs, R *z, int64_t nb, int n, int P) {
  const size_t nn = (size_t)n * n;
  #pragma unroll 1
  for (int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       g < nb * (int64_t)nn; g += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = g / nn;
    z[g * P] = q[b * qs + g % nn];
    if (P == 2) z[g * P + 1] = 0;
  }
}

/* w = d scaled back, v = Z row-major; NaN for a flagged matrix. */
template <class R>
__device__ __noinline__ void td_out(void *wsp, const R *d, size_t ds, const int *dcx, size_t es,
                                    const R *z, R *w, R *v, int64_t nb, int n, int P) {
  const size_t nn = (size_t)n * n;
  #pragma unroll 1
  for (int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       g < nb * (int64_t)nn; g += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = g / nn;
    size_t k = g % nn;
    td_ws<R> ws(wsp, b, n, P * sizeof(R));
    /* [dcx] is the solver's failure flag then its exponent. */
    bool ok = ws.ok[0] && !(dcx && dcx[b * es]);
    int ex = ws.ex[0] + (dcx ? dcx[b * es + 1] : 0);
    if (k < (size_t)n)
      w[b * n + k] = ok ? ldexp(d[b * ds + k], -ex) + (R)0 : (R)NAN;
    if (!v) continue;
    int i = (int)(k / n), j = (int)(k % n);
    for (int q = 0; q < P; q++)
      v[g * P + q] = ok ? z[(b * nn + i + (size_t)j * n) * P + q] : (R)NAN;
  }
}

