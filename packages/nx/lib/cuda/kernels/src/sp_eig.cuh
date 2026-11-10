/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* eig: a v = v diag w for general square matrices, as LAPACK's xGEEV
   computes it, one block per matrix.

   a is balanced (xGEBAL), scaled into a safe range by a power of 2,
   reduced to the upper Hessenberg H = Qᴴ a Q (xGEHD2), Z = Q formed
   (xORGHR), then H = Z S Zᴴ in Schur form by the implicit QR iteration
   (xLAHQR): single shifts on complex numbers, Francis double shifts on
   reals, where each complex conjugate pair of eigenvalues stays a 2 × 2
   block in standard form (xLANV2), so that the pair is exactly conjugate.
   The eigenvectors X of S come by back substitution (xTREVC3), one thread
   per eigenvalue (eig_vectors), v = Z X by the contraction's product,
   then eig_finish undoes the balancing and gives each column unit norm,
   its element of largest magnitude real.

   Every thread of a block runs the same control flow and computes the
   same scalars from the same memory; loops over rows or columns split
   among the threads, and the block's reductions combine in one fixed
   tree. So the bits depend on the matrix and the block size alone, which
   the dispatcher fixes per n.

   Workspace per matrix, column-major (element (i, j) at x[i + j·n]):
   H, Z and X of n² elements, a complex scratch of n² for the vectors'
   back substitution, and per index τ, scale, wr, wi, column norms; a
   flag per matrix. */

#include "sp_block.cuh"

#define M(x, i, j) (x)[(size_t)(i) + (size_t)(j) * n]
#define KEXSH 10

template <class T> struct eig_ws {
  typedef typename real_of<T>::R R;
  T *h, *z, *x;
  cx<R> *xs;
  T *tau;
  R *scale, *wr, *wi, *cn;
  int *ok;
  /* [tb] an element's bytes: a step of parts alone takes its format's
     workspace as T = R. */
  __device__ eig_ws(void *base, int64_t b, int n, int tb = sizeof(T)) {
    size_t nn = (size_t)n * n;
    size_t per = nx_sp_eig_per(n, tb, sizeof(R));
    char *p = (char *)base + (size_t)b * per;
    h = (T *)p;
    z = (T *)(p += nn * tb);
    x = (T *)(p += nn * tb);
    xs = (cx<R> *)(p += nn * tb);
    tau = (T *)(xs + nn);
    scale = (R *)((char *)tau + (size_t)n * tb);
    wr = scale + n;
    wi = wr + n;
    cn = wi + n;
    ok = (int *)(cn + n);
  }
};

/* Balancing (xGEBAL 'B'): rows and columns that isolate an eigenvalue
   swapped to the ends, then rows and columns [lo, hi] scaled by powers
   of 2 until each row's and column's norms are within 5% of balanced.
   scale takes, outside [lo, hi], the index each was swapped with, inside,
   its scale. */
template <class T>
__device__ __noinline__ void gebal(T *a, int n, int *plo, int *phi, typename real_of<T>::R *scale,
             typename real_of<T>::R *sh, int *si) {
  typedef typename real_of<T>::R R;
  const int t = threadIdx.x, nt = blockDim.x;
  int k = 0, l = n - 1;
  auto swap_cols = [&](int i, int j, int rows) {
    #pragma unroll 1
    for (int p = t; p < rows; p += nt) {
      T x = M(a, p, i);
      M(a, p, i) = M(a, p, j);
      M(a, p, j) = x;
    }
  };
  auto swap_rows = [&](int i, int j, int c0) {
    #pragma unroll 1
    for (int p = c0 + t; p < n; p += nt) {
      T x = M(a, i, p);
      M(a, i, p) = M(a, j, p);
      M(a, j, p) = x;
    }
  };
  /* A row alone in columns [0, l] goes to row l; the last such row
     first, as xGEBAL searches from l down. */
  #pragma unroll 1
  for (bool again = true; again && n > 1;) {
    again = false;
    int found = -1;
    #pragma unroll 1
    for (int i = t; i <= l; i += nt) {
      bool alone = true;
      #pragma unroll 1
      for (int j = 0; j <= l && alone; j++)
        alone = i == j || M(a, i, j) == elt<T>((R)0);
      if (alone) found = max(found, i);
    }
    found = (int)bmax((R)found, sh);
    if (found < 0) break;
    if (t == 0) scale[l] = (R)found;
    if (found != l) {
      __syncthreads();
      swap_cols(found, l, l + 1);
      __syncthreads();
      swap_rows(found, l, k);
    }
    __syncthreads();
    if (l == 0) {
      *plo = 0;
      *phi = 0;
      return;
    }
    l--;
    again = true;
  }
  /* A column alone in rows [k, l] goes to column k. */
  #pragma unroll 1
  for (bool again = true; again && n > 1;) {
    again = false;
    int found = n;
    #pragma unroll 1
    for (int j = k + t; j <= l; j += nt) {
      bool alone = true;
      #pragma unroll 1
      for (int i = k; i <= l && alone; i++)
        alone = i == j || M(a, i, j) == elt<T>((R)0);
      if (alone) found = min(found, j);
    }
    found = -(int)bmax((R)-found, sh);
    if (found > l) break;
    if (t == 0) scale[k] = (R)found;
    if (found != k) {
      __syncthreads();
      swap_cols(found, k, l + 1);
      __syncthreads();
      swap_rows(found, k, k);
    }
    __syncthreads();
    k++;
    again = true;
  }
  *plo = k;
  *phi = l;
  #pragma unroll 1
  for (int i = k + t; i <= l; i += nt) scale[i] = 1;
  __syncthreads();
  const R ulp = 2 * lim<R>::eps();
  const R sfmin1 = lim<R>::tiny() / ulp, sfmax1 = 1 / sfmin1,
          sfmin2 = sfmin1 * 2, sfmax2 = 1 / sfmin2;
  #pragma unroll 1
  for (bool noconv = true; noconv;) {
    noconv = false;
    #pragma unroll 1
    for (int i = k; i <= l; i++) {
      R c = bnorm(&M(a, k, i), l - k + 1, 1, sh);
      R r = bnorm(&M(a, i, k), l - k + 1, n, sh);
      R best = -1;
      int bi = 0;
      #pragma unroll 1
      for (int p = t; p <= l; p += nt)
        if (mag1(M(a, p, i)) > best) best = mag1(M(a, p, i)), bi = p;
      bi = bargmax(best, bi, sh, si);
      T x = M(a, bi, i);
      best = -1;
      bi = 0;
      #pragma unroll 1
      for (int p = k + t; p < n; p += nt)
        if (mag1(M(a, i, p)) > best) best = mag1(M(a, i, p)), bi = p;
      bi = bargmax(best, bi, sh, si);
      T y = M(a, i, bi);
      R ca = lapy3(re(x), im(x), (R)0), ra = lapy3(re(y), im(y), (R)0);
      if (c == 0 || r == 0) continue;
      R g = r / 2, f = 1, s = c + r;
      while (c < g && fmax(f, c) < sfmax2 && fmax(ca, c) < sfmax2 &&
             fmin(r, g) > sfmin2 && ra > sfmin2)
        f *= 2, c *= 2, ca *= 2, r /= 2, g /= 2, ra /= 2;
      g = c / 2;
      while (g >= r && fmax(r, ra) < sfmax2 && fmin(f, c) > sfmin2 &&
             fmin(g, ca) > sfmin2)
        f /= 2, c /= 2, g /= 2, ca /= 2, r *= 2, ra *= 2;
      if (c + r >= (R)0.95 * s) continue;
      R si_ = scale[i];
      if (f < 1 && si_ < 1 && f * si_ <= sfmin1) continue;
      if (f > 1 && si_ > 1 && si_ >= sfmax1 / f) continue;
      noconv = true;
      __syncthreads();
      if (t == 0) scale[i] = si_ * f;
      #pragma unroll 1
      for (int j = k + t; j < n; j += nt) M(a, i, j) = M(a, i, j) * elt<T>(1 / f);
      __syncthreads();
      #pragma unroll 1
      for (int j = t; j <= l; j += nt) M(a, j, i) = M(a, j, i) * elt<T>(f);
      __syncthreads();
    }
  }
}

/* Whether H's subdiagonal element (k, k − 1) is negligible in the active
   block [lo, hi] (Ahues and Tisseur's test, as xLAHQR). */
template <class T>
__device__ __noinline__ bool negligible(const T *h, int n, int k, int lo, int hi,
                  typename real_of<T>::R smlnum) {
  typedef typename real_of<T>::R R;
  const R ulp = 2 * lim<R>::eps();
  if (mag1(M(h, k, k - 1)) <= smlnum) return true;
  R tst = mag1(M(h, k - 1, k - 1)) + mag1(M(h, k, k));
  if (tst == 0) {
    if (k - 2 >= lo) tst += fabs(re(M(h, k - 1, k - 2)));
    if (k + 1 <= hi) tst += fabs(re(M(h, k + 1, k)));
  }
  if (!(fabs(re(M(h, k, k - 1))) <= ulp * tst)) return false;
  R x = mag1(M(h, k, k - 1)), y = mag1(M(h, k - 1, k));
  R ab = fmax(x, y), ba = fmin(x, y);
  x = mag1(M(h, k, k));
  y = mag1(M(h, k - 1, k - 1) - M(h, k, k));
  R aa = fmax(x, y), bb = fmin(x, y), s = aa + ab;
  R tt = ulp * (bb * (aa / s));
  return ba * (ab / s) <= fmax(smlnum, tt);
}

/* sqrt of a complex number, principal branch. */
template <class R> D cx<R> csqrt(cx<R> a) {
  R m = hypot(a.re, a.im);
  if (m == 0) return {0, a.im};
  R r = sqrt((m + fabs(a.re)) / 2);
  if (a.re >= 0) return {r, a.im / (2 * r)};
  return {fabs(a.im) / (2 * r), copysign(r, a.im)};
}

/* Complex: single-shift QR sweeps (xLAHQR). H made upper triangular in
   columns [i1, i2] under [wantt], Z's rows [0, n) rotated where [z].
   Returns false if an eigenvalue takes more than 30 max(10, hi − lo + 1)
   iterations. */
template <class R>
__device__ __noinline__ bool lahqr(bool wantt, int n, int lo, int hi, cx<R> *h, R *wr, R *wi,
             cx<R> *z, R *sh) {
  typedef cx<R> T;
  const int t = threadIdx.x, nt = blockDim.x;
  const R ulp = 2 * lim<R>::eps();
  if (lo == hi) {
    wr[lo] = re(M(h, lo, lo));
    wi[lo] = im(M(h, lo, lo));
    return true;
  }
  if (t == 0) {
    #pragma unroll 1
    for (int j = lo; j <= hi - 3; j++) M(h, j + 2, j) = M(h, j + 3, j) = T{0, 0};
    if (lo <= hi - 2) M(h, hi, hi - 2) = T{0, 0};
  }
  int jlo = wantt ? 0 : lo, jhi = wantt ? n - 1 : hi;
  /* The subdiagonal made real by a diagonal similarity. */
  #pragma unroll 1
  for (int i = lo + 1; i <= hi; i++) {
    __syncthreads();
    T x = M(h, i, i - 1);
    if (x.im == 0) continue;
    T sc = x / mag1(x);
    sc = conj(sc) / lapy3(sc.re, sc.im, (R)0);
    R ax = lapy3(x.re, x.im, (R)0);
    __syncthreads();
    #pragma unroll 1
    for (int j = i + t; j <= jhi; j += nt) M(h, i, j) = M(h, i, j) * sc;
    __syncthreads();
    #pragma unroll 1
    for (int j = jlo + t; j <= min(jhi, i + 1); j += nt)
      M(h, j, i) = M(h, j, i) * conj(sc);
    #pragma unroll 1
    for (int j = t; z && j < n; j += nt) M(z, j, i) = M(z, j, i) * conj(sc);
    __syncthreads();
    if (t == 0) M(h, i, i - 1) = T{ax, 0};
  }
  __syncthreads();
  const R smlnum = lim<R>::tiny() * ((R)(hi - lo + 1) / ulp);
  const int itmax = 30 * max(10, hi - lo + 1);
  int i1 = 0, i2 = n - 1, kdefl = 0;
  #pragma unroll 1
  for (int i = hi; i >= lo;) {
    int l = lo, its = 0;
    #pragma unroll 1
    for (; its <= itmax; its++) {
      __syncthreads();
      int k = i;
      while (k > l && !negligible(h, n, k, lo, hi, smlnum)) k--;
      l = k;
      __syncthreads();
      if (l > lo && t == 0) M(h, l, l - 1) = T{0, 0};
      if (l >= i) break;
      kdefl++;
      if (!wantt) i1 = l, i2 = i;
      T s;
      if (kdefl % (2 * KEXSH) == 0)
        s = T{(R)0.75 * fabs(M(h, i, i - 1).re), 0} + M(h, i, i);
      else if (kdefl % KEXSH == 0)
        s = T{(R)0.75 * fabs(M(h, l + 1, l).re), 0} + M(h, l, l);
      else {
        s = M(h, i, i);
        T u = csqrt(M(h, i - 1, i)) * csqrt(M(h, i, i - 1));
        R su = mag1(u);
        if (su != 0) {
          T x = (M(h, i - 1, i - 1) - s) / (R)2;
          R sx = mag1(x);
          su = fmax(su, sx);
          T y = su * csqrt((x / su) * (x / su) + (u / su) * (u / su));
          if (sx > 0) {
            T xs = x / sx;
            if (xs.re * y.re + xs.im * y.im < 0) y = -y;
          }
          s = s - u * cdiv(u, x + y);
        }
      }
      int m = i - 1;
      T v0, v1;
      #pragma unroll 1
      for (;; m--) {
        T h11 = M(h, m, m), h22 = M(h, m + 1, m + 1), h11s = h11 - s;
        R h21 = M(h, m + 1, m).re, ss = mag1(h11s) + fabs(h21);
        h11s = h11s / ss;
        h21 /= ss;
        v0 = h11s;
        v1 = T{h21, 0};
        if (m == l) break;
        if (fabs(M(h, m, m - 1).re) * fabs(h21) <=
            ulp * (mag1(h11s) * (mag1(h11) + mag1(h22))))
          break;
      }
      #pragma unroll 1
      for (int k2 = m; k2 < i; k2++) {
        __syncthreads();
        if (k2 > m) v0 = M(h, k2, k2 - 1), v1 = M(h, k2 + 1, k2 - 1);
        /* The 2-vector's reflector, every thread alike. */
        T t1;
        {
          R xn = mag(v1), ar = v0.re, ai = v0.im;
          if (xn == 0 && ai == 0)
            t1 = T{0, 0};
          else {
            R beta = ar >= 0 ? -lapy3(ar, ai, xn) : lapy3(ar, ai, xn);
            const R sm = safmin<R>();
            int knt = 0;
            T vv = v1;
            while (fabs(beta) < sm && knt < 20) {
              knt++;
              vv = vv * (1 / sm);
              beta *= 1 / sm, ar *= 1 / sm, ai *= 1 / sm;
            }
            if (knt) {
              R an = lapy3(ar, ai, mag(vv));
              beta = ar >= 0 ? -an : an;
            }
            t1 = T{(beta - ar) / beta, -(ai / beta)};
            v1 = cdiv(vv, T{ar - beta, ai});
            #pragma unroll 1
            for (int j = 0; j < knt; j++) beta *= sm;
            v0 = T{beta, 0};
          }
        }
        __syncthreads();
        if (k2 > m && t == 0)
          M(h, k2, k2 - 1) = v0, M(h, k2 + 1, k2 - 1) = T{0, 0};
        T v2 = v1;
        R t2 = (t1 * v2).re;
        #pragma unroll 1
        for (int j = k2 + t; j <= i2; j += nt) {
          T sum = conj(t1) * M(h, k2, j) + t2 * M(h, k2 + 1, j);
          M(h, k2, j) -= sum;
          M(h, k2 + 1, j) -= sum * v2;
        }
        __syncthreads();
        #pragma unroll 1
        for (int j = i1 + t; j <= min(k2 + 2, i); j += nt) {
          T sum = t1 * M(h, j, k2) + t2 * M(h, j, k2 + 1);
          M(h, j, k2) -= sum;
          M(h, j, k2 + 1) -= sum * conj(v2);
        }
        #pragma unroll 1
        for (int j = t; z && j < n; j += nt) {
          T sum = t1 * M(z, j, k2) + t2 * M(z, j, k2 + 1);
          M(z, j, k2) -= sum;
          M(z, j, k2 + 1) -= sum * conj(v2);
        }
        if (k2 == m && m > l) {
          /* H(m + 1, m) made real again, as the sweep started below l. */
          T x = T{1, 0} - t1;
          x = x / lapy3(x.re, x.im, (R)0);
          __syncthreads();
          if (t == 0) {
            M(h, m + 1, m) = M(h, m + 1, m) * conj(x);
            if (m + 2 <= i) M(h, m + 2, m + 1) = M(h, m + 2, m + 1) * x;
          }
          #pragma unroll 1
          for (int j = m; j <= i; j++) {
            if (j == m + 1) continue;
            __syncthreads();
            #pragma unroll 1
            for (int p = j + 1 + t; p <= i2; p += nt) M(h, j, p) = M(h, j, p) * x;
            __syncthreads();
            #pragma unroll 1
            for (int p = i1 + t; p < j; p += nt) M(h, p, j) = M(h, p, j) * conj(x);
            #pragma unroll 1
            for (int p = t; z && p < n; p += nt) M(z, p, j) = M(z, p, j) * conj(x);
          }
        }
      }
      __syncthreads();
      T x = M(h, i, i - 1);
      if (x.im != 0) {
        R r = lapy3(x.re, x.im, (R)0);
        x = x / r;
        __syncthreads();
        if (t == 0) M(h, i, i - 1) = T{r, 0};
        #pragma unroll 1
        for (int p = i + 1 + t; p <= i2; p += nt) M(h, i, p) = M(h, i, p) * conj(x);
        __syncthreads();
        #pragma unroll 1
        for (int p = i1 + t; p < i; p += nt) M(h, p, i) = M(h, p, i) * x;
        #pragma unroll 1
        for (int p = t; z && p < n; p += nt) M(z, p, i) = M(z, p, i) * x;
      }
    }
    if (its > itmax) return false;
    __syncthreads();
    if (t == 0) {
      wr[i] = M(h, i, i).re;
      wi[i] = M(h, i, i).im;
    }
    kdefl = 0;
    i = l - 1;
  }
  __syncthreads();
  return true;
}

/* The Schur factorisation of the real 2 × 2 [a b; c d] (xLANV2). */
template <class R>
__device__ __noinline__ void lanv2(R *a, R *b, R *c, R *d, R *rt1r, R *rt1i, R *rt2r, R *rt2i,
             R *cs, R *sn) {
  const R ulp = 2 * lim<R>::eps();
  const R safmn2 = sizeof(R) == 4 ? (R)0x1p-51 : (R)0x1p-485,
          safmx2 = 1 / safmn2;
  if (*c == 0) {
    *cs = 1;
    *sn = 0;
  } else if (*b == 0) {
    *cs = 0;
    *sn = 1;
    R tt = *d;
    *d = *a;
    *a = tt;
    *b = -*c;
    *c = 0;
  } else if (*a - *d == 0 && copysign((R)1, *b) != copysign((R)1, *c)) {
    *cs = 1;
    *sn = 0;
  } else {
    R temp = *a - *d, p = temp / 2;
    R bcmax = fmax(fabs(*b), fabs(*c));
    R bcmis = fmin(fabs(*b), fabs(*c)) * copysign((R)1, *b) * copysign((R)1, *c);
    R scale = fmax(fabs(p), bcmax);
    R zz = (p / scale) * p + (bcmax / scale) * bcmis;
    if (zz >= 4 * ulp) {
      zz = p + copysign(sqrt(scale) * sqrt(zz), p);
      *a = *d + zz;
      *d = *d - (bcmax / zz) * bcmis;
      R tau = lapy3(*c, zz, (R)0);
      *cs = zz / tau;
      *sn = *c / tau;
      *b = *b - *c;
      *c = 0;
    } else {
      R sigma = *b + *c;
      #pragma unroll 1
      for (int count = 1;; count++) {
        R sc = fmax(fabs(temp), fabs(sigma));
        if (sc >= safmx2) {
          sigma *= safmn2;
          temp *= safmn2;
          if (count <= 20) continue;
        }
        if (sc <= safmn2) {
          sigma *= safmx2;
          temp *= safmx2;
          if (count <= 20) continue;
        }
        break;
      }
      p = temp / 2;
      R tau = lapy3(sigma, temp, (R)0);
      *cs = sqrt((1 + fabs(sigma) / tau) / 2);
      *sn = -(p / (tau * *cs)) * copysign((R)1, sigma);
      R aa = *a * *cs + *b * *sn, bb = -*a * *sn + *b * *cs;
      R cc = *c * *cs + *d * *sn, dd = -*c * *sn + *d * *cs;
      *a = aa * *cs + cc * *sn;
      *b = bb * *cs + dd * *sn;
      *c = -aa * *sn + cc * *cs;
      *d = -bb * *sn + dd * *cs;
      temp = (*a + *d) / 2;
      *a = *d = temp;
      if (*c != 0 && *b == 0) {
        *b = -*c;
        *c = 0;
        temp = *cs;
        *cs = -*sn;
        *sn = temp;
      } else if (*c != 0 && copysign((R)1, *b) == copysign((R)1, *c)) {
        R sab = sqrt(fabs(*b)), sac = sqrt(fabs(*c));
        p = copysign(sab * sac, *c);
        tau = 1 / sqrt(fabs(*b + *c));
        *a = temp + p;
        *d = temp - p;
        *b = *b - *c;
        *c = 0;
        R cs1 = sab * tau, sn1 = sac * tau;
        temp = *cs * cs1 - *sn * sn1;
        *sn = *cs * sn1 + *sn * cs1;
        *cs = temp;
      }
    }
  }
  *rt1r = *a;
  *rt2r = *d;
  *rt1i = *c == 0 ? 0 : sqrt(fabs(*b)) * sqrt(fabs(*c));
  *rt2i = -*rt1i;
}

/* Real: Francis double-shift QR sweeps (xLAHQR), each complex pair a 2 ×
   2 block in xLANV2's form under [wantt]. */
template <class R>
__device__ __noinline__ bool lahqr(bool wantt, int n, int lo, int hi, R *h, R *wr, R *wi, R *z,
             R *sh) {
  const int t = threadIdx.x, nt = blockDim.x;
  const R ulp = 2 * lim<R>::eps();
  if (lo == hi) {
    if (t == 0) wr[lo] = M(h, lo, lo), wi[lo] = 0;
    __syncthreads();
    return true;
  }
  if (t == 0) {
    #pragma unroll 1
    for (int j = lo; j <= hi - 3; j++) M(h, j + 2, j) = M(h, j + 3, j) = 0;
    if (lo <= hi - 2) M(h, hi, hi - 2) = 0;
  }
  const R smlnum = lim<R>::tiny() * ((R)(hi - lo + 1) / ulp);
  const int itmax = 30 * max(10, hi - lo + 1);
  int i1 = 0, i2 = n - 1, kdefl = 0;
  #pragma unroll 1
  for (int i = hi; i >= lo;) {
    int l = lo, its = 0;
    #pragma unroll 1
    for (; its <= itmax; its++) {
      __syncthreads();
      int k = i;
      while (k > l && !negligible(h, n, k, lo, hi, smlnum)) k--;
      l = k;
      __syncthreads();
      if (l > lo && t == 0) M(h, l, l - 1) = 0;
      if (l >= i - 1) break;
      kdefl++;
      if (!wantt) i1 = l, i2 = i;
      R h11, h12, h21, h22, s;
      if (kdefl % (2 * KEXSH) == 0 || kdefl % KEXSH == 0) {
        int p = kdefl % (2 * KEXSH) == 0 ? i : l;
        s = kdefl % (2 * KEXSH) == 0
                ? fabs(M(h, i, i - 1)) + fabs(M(h, i - 1, i - 2))
                : fabs(M(h, l + 1, l)) + fabs(M(h, l + 2, l + 1));
        h11 = h22 = (R)0.75 * s + M(h, p, p);
        h12 = (R)-0.4375 * s;
        h21 = s;
      } else {
        h11 = M(h, i - 1, i - 1);
        h21 = M(h, i, i - 1);
        h12 = M(h, i - 1, i);
        h22 = M(h, i, i);
      }
      s = fabs(h11) + fabs(h12) + fabs(h21) + fabs(h22);
      R rt1r = 0, rt1i = 0, rt2r = 0, rt2i = 0;
      if (s != 0) {
        h11 /= s, h21 /= s, h12 /= s, h22 /= s;
        R tr = (h11 + h22) / 2, det = (h11 - tr) * (h22 - tr) - h12 * h21;
        R rtdisc = sqrt(fabs(det));
        if (det >= 0) {
          rt1r = rt2r = tr * s;
          rt1i = rtdisc * s;
          rt2i = -rt1i;
        } else {
          rt1r = tr + rtdisc;
          rt2r = tr - rtdisc;
          rt1r = rt2r =
              (fabs(rt1r - h22) <= fabs(rt2r - h22) ? rt1r : rt2r) * s;
        }
      }
      int m = i - 2;
      R v[3];
      #pragma unroll 1
      for (;; m--) {
        R h21s = M(h, m + 1, m), hmm = M(h, m, m);
        s = fabs(hmm - rt2r) + fabs(rt2i) + fabs(h21s);
        h21s = M(h, m + 1, m) / s;
        v[0] = h21s * M(h, m, m + 1) + (hmm - rt1r) * ((hmm - rt2r) / s) -
               rt1i * (rt2i / s);
        v[1] = h21s * (hmm + M(h, m + 1, m + 1) - rt1r - rt2r);
        v[2] = h21s * M(h, m + 2, m + 1);
        s = fabs(v[0]) + fabs(v[1]) + fabs(v[2]);
        v[0] /= s, v[1] /= s, v[2] /= s;
        if (m == l) break;
        if (fabs(M(h, m, m - 1)) * (fabs(v[1]) + fabs(v[2])) <=
            ulp * fabs(v[0]) *
                (fabs(M(h, m - 1, m - 1)) + fabs(hmm) +
                 fabs(M(h, m + 1, m + 1))))
          break;
      }
      #pragma unroll 1
      for (int k2 = m; k2 < i; k2++) {
        __syncthreads();
        int nr = min(3, i - k2 + 1);
        if (k2 > m)
          #pragma unroll 1
          for (int p = 0; p < nr; p++) v[p] = M(h, k2 + p, k2 - 1);
        /* The reflector of [v0; v1; v2], every thread alike. */
        R t1 = 0;
        {
          R xn = nr == 3 ? lapy3(v[1], v[2], (R)0) : fabs(v[1]);
          R ar = v[0];
          if (xn != 0) {
            const R sm = safmin<R>();
            R beta = ar >= 0 ? -lapy3(ar, (R)0, xn) : lapy3(ar, (R)0, xn);
            int knt = 0;
            R x1 = v[1], x2 = nr == 3 ? v[2] : 0;
            while (fabs(beta) < sm && knt < 20) {
              knt++;
              x1 *= 1 / sm, x2 *= 1 / sm, beta *= 1 / sm, ar *= 1 / sm;
            }
            if (knt) {
              R an = lapy3(ar, (R)0, lapy3(x1, x2, (R)0));
              beta = ar >= 0 ? -an : an;
            }
            t1 = (beta - ar) / beta;
            v[1] = x1 / (ar - beta);
            if (nr == 3) v[2] = x2 / (ar - beta);
            #pragma unroll 1
            for (int j = 0; j < knt; j++) beta *= sm;
            v[0] = beta;
          }
        }
        __syncthreads();
        if (t == 0) {
          if (k2 > m) {
            M(h, k2, k2 - 1) = v[0];
            M(h, k2 + 1, k2 - 1) = 0;
            if (k2 < i - 1) M(h, k2 + 2, k2 - 1) = 0;
          } else if (m > l)
            M(h, k2, k2 - 1) *= 1 - t1;
        }
        R v2 = v[1], t2 = t1 * v2, v3 = nr == 3 ? v[2] : 0, t3 = t1 * v3;
        #pragma unroll 1
        for (int j = k2 + t; j <= i2; j += nt) {
          R sum = M(h, k2, j) + v2 * M(h, k2 + 1, j) +
                  (nr == 3 ? v3 * M(h, k2 + 2, j) : 0);
          M(h, k2, j) -= sum * t1;
          M(h, k2 + 1, j) -= sum * t2;
          if (nr == 3) M(h, k2 + 2, j) -= sum * t3;
        }
        __syncthreads();
        int e = nr == 3 ? min(k2 + 3, i) : i;
        #pragma unroll 1
        for (int j = i1 + t; j <= e; j += nt) {
          R sum = M(h, j, k2) + v2 * M(h, j, k2 + 1) +
                  (nr == 3 ? v3 * M(h, j, k2 + 2) : 0);
          M(h, j, k2) -= sum * t1;
          M(h, j, k2 + 1) -= sum * t2;
          if (nr == 3) M(h, j, k2 + 2) -= sum * t3;
        }
        #pragma unroll 1
        for (int j = t; z && j < n; j += nt) {
          R sum = M(z, j, k2) + v2 * M(z, j, k2 + 1) +
                  (nr == 3 ? v3 * M(z, j, k2 + 2) : 0);
          M(z, j, k2) -= sum * t1;
          M(z, j, k2 + 1) -= sum * t2;
          if (nr == 3) M(z, j, k2 + 2) -= sum * t3;
        }
      }
    }
    if (its > itmax) return false;
    __syncthreads();
    if (l == i) {
      if (t == 0) wr[i] = M(h, i, i), wi[i] = 0;
    } else {
      R a = M(h, i - 1, i - 1), b = M(h, i - 1, i), c = M(h, i, i - 1),
        d = M(h, i, i), r1r, r1i, r2r, r2i, cs, sn;
      lanv2(&a, &b, &c, &d, &r1r, &r1i, &r2r, &r2i, &cs, &sn);
      __syncthreads();
      if (t == 0) {
        M(h, i - 1, i - 1) = a, M(h, i - 1, i) = b, M(h, i, i - 1) = c,
                     M(h, i, i) = d;
        wr[i - 1] = r1r, wi[i - 1] = r1i, wr[i] = r2r, wi[i] = r2i;
      }
      if (wantt) {
        #pragma unroll 1
        for (int j = i + 1 + t; j <= i2; j += nt) {
          R x = M(h, i - 1, j), y = M(h, i, j);
          M(h, i - 1, j) = cs * x + sn * y;
          M(h, i, j) = cs * y - sn * x;
        }
        #pragma unroll 1
        for (int j = i1 + t; j < i - 1; j += nt) {
          R x = M(h, j, i - 1), y = M(h, j, i);
          M(h, j, i - 1) = cs * x + sn * y;
          M(h, j, i) = cs * y - sn * x;
        }
      }
      #pragma unroll 1
      for (int j = t; z && j < n; j += nt) {
        R x = M(z, j, i - 1), y = M(z, j, i);
        M(z, j, i - 1) = cs * x + sn * y;
        M(z, j, i) = cs * y - sn * x;
      }
    }
    kdefl = 0;
    i = l - 1;
  }
  __syncthreads();
  return true;
}

/* Schur: w, and S and Z for the vectors, of matrix b. Launch a power of
   two threads, one block per matrix (looping), shared memory 2 ×
   blockDim.x words of R and int. */
template <class T>
D void eig_schur(const T *a, typename real_of<T>::R *w, void *wsp, int vectors,
                 int64_t nb, int n) {
  typedef typename real_of<T>::R R;
  extern __shared__ __align__(16) unsigned char smem[];
  R *sh = (R *)smem;
  int *si = (int *)(sh + blockDim.x);
  const int t = threadIdx.x, nt = blockDim.x;
  const size_t nn = (size_t)n * n;
  const R ulp = 2 * lim<R>::eps();
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    eig_ws<T> ws(wsp, b, n);
    T *h = ws.h, *z = ws.z;
    /* a, row-major, into H, column-major; its scan. */
    R big = 0, nan = 0;
    #pragma unroll 1
    for (size_t k = t; k < nn; k += nt) {
      T x = a[b * nn + k];
      nan += (re(x) - re(x)) + (im(x) - im(x));
      big = fmax(big, fmax(fabs(re(x)), fabs(im(x))));
      h[(k % n) * n + k / n] = x;
    }
    bool ok = bsum(nan, sh) == 0;
    int lo = 0, hi = n - 1, e = 0;
    if (ok) {
      /* Balanced first, then scaled: xGEEV scales first, which flushes
         entries balancing would bring near the others. */
      gebal(h, n, &lo, &hi, ws.scale, sh, si);
      big = 0;
      #pragma unroll 1
      for (size_t k = t; k < nn; k += nt)
        big = fmax(big, fmax(fabs(re(h[k])), fabs(im(h[k]))));
      big = bmax(big, sh);
      const R small = sqrt(lim<R>::tiny()) / ulp;
      e = scale_exp(big, small, 1 / small);
      if (e != 0)
        #pragma unroll 1
        for (size_t k = t; k < nn; k += nt) h[k] = scaled(h[k], e);
      /* Hessenberg (xGEHD2): reflector i in column i below the
         subdiagonal. */
      #pragma unroll 1
      for (int i = t; i < n; i += nt) ws.tau[i] = elt<T>((R)0);
      __syncthreads();
      T *vcol = ws.x;
      #pragma unroll 1
      for (int i = lo; i < hi; i++) {
        T *alpha = &M(h, i + 1, i);
        T tau = reflector(alpha, alpha + 1, hi - i - 1, 1, sh);
        if (t == 0) ws.tau[i] = tau;
        /* v = [1; column i below], staged so the updates read it. */
        T al = *alpha;
        __syncthreads();
        #pragma unroll 1
        for (int p = t; p < hi - i; p += nt)
          vcol[p] = p == 0 ? elt<T>((R)1) : M(h, i + 1 + p, i);
        __syncthreads();
        larf_right(vcol, tau, hi + 1, hi - i, &M(h, 0, i + 1), n);
        larf_left(vcol, conj(tau), hi - i, n - i - 1, &M(h, i + 1, i + 1), n);
        (void)al;
      }
      __syncthreads();
      /* Z = Q (xORGHR): the reflectors applied to I, last first, before
         H's subdiagonal is cleared. */
      if (vectors) {
        #pragma unroll 1
        for (size_t k = t; k < nn; k += nt) z[k] = elt<T>((R)(k % (n + 1) == 0));
        __syncthreads();
        T *v = ws.x + nn - n;
        #pragma unroll 1
        for (int i = hi - 1; i >= lo; i--) {
          #pragma unroll 1
          for (int p = t; p < hi - i; p += nt)
            v[p] = p == 0 ? elt<T>((R)1) : M(h, i + 1 + p, i);
          __syncthreads();
          larf_left(v, ws.tau[i], hi - i, hi - i, &M(z, i + 1, i + 1), n);
        }
      }
      #pragma unroll 1
      for (size_t k = t; k < nn; k += nt)
        if ((int)(k % n) > (int)(k / n) + 1) h[k] = elt<T>((R)0);
      __syncthreads();
      ok = lahqr(vectors != 0, n, lo, hi, h, ws.wr, ws.wi, vectors ? z : NULL,
                 sh);
    }
    __syncthreads();
    if (t == 0) {
      ws.ok[0] = ok;
      ws.ok[1] = e;
      ws.ok[2] = lo;
      ws.ok[3] = hi;
    }
    /* Eigenvalues outside [lo, hi] are the isolated diagonal; lahqr set
       the others. */
    #pragma unroll 1
    for (int j = t; ok && j < n; j += nt)
      if (j < lo || j > hi) ws.wr[j] = re(M(h, j, j)), ws.wi[j] = im(M(h, j, j));
    __syncthreads();
    #pragma unroll 1
    for (int j = t; j < n; j += nt) {
      R x = ok ? scaled(ws.wr[j], -e) + (R)0 : (R)NAN;
      R y = ok ? scaled(ws.wi[j], -e) + (R)0 : (R)NAN;
      /* A pair's imaginary part that rounds to 0 scaled back is η, its
         sign kept: the values stay a pair, as their vectors are. */
      if (!real_of<T>::cplx && ok && ws.wi[j] != 0 && y == 0)
        y = copysign(lim<R>::eta(), ws.wi[j]);
      w[2 * (b * n + j)] = x;
      w[2 * (b * n + j) + 1] = y;
    }
    __syncthreads();
  }
}

template <class R> D R cabs1(cx<R> x) { return fabs(x.re) + fabs(x.im); }
template <class R> D void cscale(cx<R> *x, int k, R f) {
  #pragma unroll 1
  for (int i = 0; i <= k; i++) x[i] = x[i] * f;
}

/* Vectors: X's column for eigenvalue [ki] of matrix [b] by back
   substitution (xTREVC3), one thread each; on reals a pair's thread is
   its second's, filling both columns (real and imaginary parts). */
template <class T>
D void eig_vectors(void *wsp, int64_t nb, int n) {
  typedef typename real_of<T>::R R;
  typedef cx<R> C;
  const bool cplx = real_of<T>::cplx;
  const R ulp = 2 * lim<R>::eps();
  const R smlnum = lim<R>::tiny() * ((R)n / ulp), bignum = (1 - ulp) / smlnum;
  const int64_t total = nb * n;
  #pragma unroll 1
  for (int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; g < total;
       g += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = g / n;
    int ki = (int)(g % n);
    eig_ws<T> ws(wsp, b, n);
    if (!ws.ok[0]) continue;
    const T *s = ws.h;
    const R *wr = ws.wr, *wi = ws.wi;
    T *xm = ws.x;
    C *x = ws.xs + (size_t)ki * n;
    if (!cplx && wi[ki] > 0) continue; /* the pair's second does it */
    bool pair = !cplx && wi[ki] != 0;
    int top = pair ? ki - 1 : ki;
    C lambda = {wr[top], wi[top]};
    R smin = fmax(ulp * cabs1(lambda), smlnum);
    if (!pair) {
      x[ki] = {1, 0};
      #pragma unroll 1
      for (int k = 0; k < ki; k++) x[k] = C{-re(M(s, k, ki)), -im(M(s, k, ki))};
    } else {
      R bb = re(M(s, ki - 1, ki)), cc = re(M(s, ki, ki - 1));
      if (fabs(bb) >= fabs(cc)) {
        x[ki - 1] = {1, 0};
        x[ki] = {0, wi[top] / bb};
      } else {
        x[ki - 1] = {-wi[top] / cc, 0};
        x[ki] = {0, 1};
      }
      #pragma unroll 1
      for (int k = 0; k < top; k++)
        x[k] = (-x[ki - 1]) * C{re(M(s, k, ki - 1)), 0} -
               x[ki] * C{re(M(s, k, ki)), 0};
    }
    #pragma unroll 1
    for (int j = top - 1; j >= 0;) {
      bool block = !cplx && j > 0 && re(M(s, j, j - 1)) != 0;
      int j0 = block ? j - 1 : j;
      R xmax = 0;
      if (!block) {
        C d = C{re(M(s, j, j)), im(M(s, j, j))} - lambda;
        if (cabs1(d) < smin) d = {smin, 0};
        R dm = cabs1(d), xj = cabs1(x[j]);
        if (dm < 1 && xj > 1 && xj > bignum * dm) cscale(x, ki, 1 / xj);
        x[j] = cdiv(x[j], d);
        xmax = cabs1(x[j]);
      } else {
        /* The 2 × 2 block by elimination with complete pivoting
           (xLALN2). */
        C e4[4] = {C{re(M(s, j0, j0)), 0} - lambda, C{re(M(s, j0, j)), 0},
                   C{re(M(s, j, j0)), 0}, C{re(M(s, j, j)), 0} - lambda};
        int p = 0;
        #pragma unroll 1
        for (int q = 1; q < 4; q++)
          if (cabs1(e4[q]) > cabs1(e4[p])) p = q;
        if (cabs1(e4[p]) < smin) p = 0;
        int pr = p / 2, pc = p % 2;
        C u11 = e4[p], l21, u12, u22;
        if (cabs1(u11) < smin) {
          u11 = {smin, 0};
          l21 = u12 = {0, 0};
          u22 = {smin, 0};
        } else {
          l21 = cdiv(e4[(1 - pr) * 2 + pc], u11);
          u12 = e4[pr * 2 + 1 - pc];
          u22 = e4[(1 - pr) * 2 + 1 - pc] - l21 * u12;
          if (cabs1(u22) < smin) u22 = {smin, 0};
        }
        C b1 = x[j0 + pr], b2 = x[j0 + 1 - pr] - l21 * b1;
        R bm = fmax(cabs1(b1), cabs1(b2)), um = fmin(cabs1(u22), cabs1(u11));
        if (um < 1 && bm > 1 && bm > bignum * um) {
          cscale(x, ki, 1 / bm);
          b1 = b1 / bm;
          b2 = b2 / bm;
        }
        C y2 = cdiv(b2, u22), y1 = cdiv(b1 - u12 * y2, u11);
        x[j0 + 1 - pc] = y2;
        x[j0 + pc] = y1;
        xmax = fmax(cabs1(y1), cabs1(y2));
        R emax = 0;
        #pragma unroll 1
        for (int q = 0; q < 4; q++) emax = fmax(emax, cabs1(e4[q]));
        if (xmax > 1 && emax > 1 && xmax > bignum / emax) {
          cscale(x, ki, emax / bignum);
          xmax *= emax / bignum;
        }
      }
      R cmax = fmax(ws.cn[j], ws.cn[j0]);
      if (xmax > 1 && cmax > bignum / xmax) {
        cscale(x, ki, 1 / xmax);
        xmax = 1;
      }
      #pragma unroll 1
      for (int k = 0; k < j0; k++) {
        x[k] -= x[j] * C{re(M(s, k, j)), im(M(s, k, j))};
        if (block) x[k] -= x[j0] * C{re(M(s, k, j0)), 0};
      }
      j = j0 - 1;
    }
    #pragma unroll 1
    for (int k = 0; k < n; k++) {
      C xk = k <= ki ? x[k] : C{0, 0};
      if (cplx)
        M(xm, k, ki) = elt<T>(xk.re, xk.im);
      else {
        M(xm, k, top) = elt<T>(xk.re);
        if (pair) M(xm, k, ki) = elt<T>(xk.im);
      }
    }
  }
}

/* The column norms of S's strict upper part, which back substitution
   reads to avoid overflow: one thread per column. */
template <class T> D void eig_norms(void *wsp, int64_t nb, int n) {
  typedef typename real_of<T>::R R;
  const int64_t total = nb * n;
  #pragma unroll 1
  for (int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; g < total;
       g += (int64_t)gridDim.x * blockDim.x) {
    eig_ws<T> ws(wsp, g / n, n);
    int j = (int)(g % n);
    const T *s = ws.h;
    R c = 0;
    #pragma unroll 1
    for (int i = 0; i < j; i++) c += mag1(M(s, i, j));
    ws.cn[j] = c;
  }
}

/* Finish: [v], row-major Z X in, columns made of largest |re| + |im| 1,
   the balancing undone, unit norm and largest element real; NaN where
   the matrix failed. One block per matrix. v holds complex numbers of
   R's precision; on reals Z X is first read from [zx], real. */
template <class R>
__device__ __noinline__ void eig_finish(const R *zx, R *vo, void *wsp, int64_t nb, int n,
                                        bool cplx) {
  const int t = threadIdx.x, nt = blockDim.x;
  const size_t nn = (size_t)n * n;
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    eig_ws<R> ws(wsp, b, n, (cplx ? 2 : 1) * sizeof(R));
    R *v = vo + 2 * b * nn;
    if (!ws.ok[0]) {
      #pragma unroll 1
      for (size_t k = t; k < 2 * nn; k += nt) v[k] = (R)NAN;
      continue;
    }
    const R *y = zx + b * nn * (cplx ? 2 : 1);
    const int lo = ws.ok[2], hi = ws.ok[3];
    const R *wi = ws.wi, *scale = ws.scale;
    #pragma unroll 1
    for (int j = t; j < n; j += nt) {
      if (!cplx && wi[j] < 0) continue;
      bool pair = !cplx && wi[j] > 0;
      /* Columns j, and j + 1 for a pair, as complex numbers: the pair's
         real and imaginary parts. */
      auto get = [&](int i) -> cx<R> {
        if (cplx) return cx<R>{y[2 * ((size_t)i * n + j)], y[2 * ((size_t)i * n + j) + 1]};
        return cx<R>{y[(size_t)i * n + j], pair ? y[(size_t)i * n + j + 1] : (R)0};
      };
      R emax = 0;
      #pragma unroll 1
      for (int i = 0; i < n; i++) emax = fmax(emax, cabs1(get(i)));
      R f = emax > 0 ? 1 / emax : (R)1;
      /* Balancing undone: rows [lo, hi] scaled, the others swapped back
         in the order they were swapped out. */
      #pragma unroll 1
      for (int i = 0; i < n; i++) {
        cx<R> c = get(i) * f;
        if (i >= lo && i <= hi && lo < hi) c = c * scale[i];
        v[2 * ((size_t)i * n + j)] = c.re;
        v[2 * ((size_t)i * n + j) + 1] = c.im;
      }
      #pragma unroll 1
      for (int p = 0; p < n; p++) {
        int i = p < lo ? lo - 1 - p : p;
        if (i >= lo && i <= hi) continue;
        int k = (int)scale[i];
        if (k == i) continue;
        R *x = v + 2 * ((size_t)i * n + j), *z = v + 2 * ((size_t)k * n + j);
        R a0 = x[0], a1 = x[1];
        x[0] = z[0], x[1] = z[1], z[0] = a0, z[1] = a1;
      }
      /* Unit norm, scaled by the largest part, and the element of largest
         magnitude, the first of equals, made real and positive. */
      R big = 0;
      #pragma unroll 1
      for (int i = 0; i < n; i++)
        big = fmax(big, fmax(fabs(v[2 * ((size_t)i * n + j)]),
                             fabs(v[2 * ((size_t)i * n + j) + 1])));
      R q = 0;
      #pragma unroll 1
      for (int i = 0; i < n; i++) {
        R a0 = v[2 * ((size_t)i * n + j)] / big,
          a1 = v[2 * ((size_t)i * n + j) + 1] / big;
        q += a0 * a0 + a1 * a1;
      }
      R nrm = big * sqrt(q);
      int kb = 0;
      R best = -1;
      #pragma unroll 1
      for (int i = 0; i < n; i++) {
        R *x = v + 2 * ((size_t)i * n + j);
        x[0] /= nrm;
        x[1] /= nrm;
        R m2 = x[0] * x[0] + x[1] * x[1];
        if (m2 > best) best = m2, kb = i;
      }
      R *xk = v + 2 * ((size_t)kb * n + j);
      R am = hypot(xk[0], xk[1]);
      cx<R> ph = {xk[0] / am, -xk[1] / am};
      #pragma unroll 1
      for (int i = 0; i < n; i++) {
        R *x = v + 2 * ((size_t)i * n + j);
        cx<R> c = cx<R>{x[0], x[1]} * ph;
        x[0] = c.re;
        /* A real value's vector is real: +0 imaginary parts. */
        x[1] = i == kb || (!cplx && !pair) ? (R)0 : c.im;
        if (i == kb) x[0] = am;
      }
      /* A real pair's second column is the first's conjugate. */
      if (pair)
        #pragma unroll 1
        for (int i = 0; i < n; i++) {
          R *x = v + 2 * ((size_t)i * n + j);
          x[2] = x[0];
          x[3] = -x[1];
        }
    }
  }
}

