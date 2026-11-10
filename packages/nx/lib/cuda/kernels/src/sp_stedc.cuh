/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The eigenvalues of real symmetric tridiagonal matrices T, ascending,
   and with them Q, T = Q diag λ Qᵀ, by divide and conquer as LAPACK's
   xSTEDC computes them (Cuppen 1981; Gu and Eisenstat 1995); or by
   bisection on Sturm counts where only the values are asked.

   Divide and conquer: T is torn at every multiple of L = 32 into leaves,
   T = diag(T1, …) + Σ ρ u uᵀ, each leaf solved by Jacobi (sp_jacobi.cuh),
   then neighbouring blocks merged level by level, all the merges of a
   level, of every matrix of the batch, in one launch per phase:

     deflate   one block per merge: z from the halves' vectors, the values
               whose part of z is negligible or that a rotation makes
               equal to a neighbour set aside (xLAED2); the rotations
               applied to Q's columns, each thread a row
     roots     one thread per root of 1 + ρ Σ z_j² / (d_j − λ) = 0
               (xLAED4), its d_j − λ into S's column
     weights   one thread per root: z recomputed from the roots (Gu and
               Eisenstat), so that the vectors are orthogonal
     vectors   one thread per root: S's column, normalised
     product   Q' = Q[:, kept] S over the merge's rows (the O(n³))
     sort      one block per merge: roots and set-aside values merged
               ascending, Q's columns with them

   Every value is computed in a fixed order by one thread, so the bits do
   not depend on scheduling. Matrices are column-major here, element
   (i, j) of an n × n Q at q[i + j·n].

   Workspace per matrix (dc_ws): Q, Qt (the product's output), S, n²
   each; per index d, e, z, λ, w, set-aside values; cols, k and rotation
   lists. */

#include "sp_common.cuh"

#define LEAF NX_SP_LEAF

template <class R> struct dc_ws {
  R *d, *e, *q, *qt, *s, *z, *dl, *w, *lam, *dd, *rc, *rs;
  int *cols, *kq, *rp, *rn, *nr, *fail;
  int *ex;
  __device__ dc_ws(void *base, int64_t b, int n) {
    size_t nn = (size_t)n * n;
    size_t per = nx_sp_dc_per(n, sizeof(R));
    char *p = (char *)base + (size_t)b * per;
    q = (R *)p;
    qt = q + nn;
    s = qt + nn;
    d = s + nn;
    e = d + n;
    z = e + n;
    dl = z + n;
    w = dl + n;
    lam = w + n;
    dd = lam + n;
    rc = dd + n;
    rs = rc + n;
    cols = (int *)(rs + 2 * n);
    kq = cols + n;
    rp = kq + n;
    rn = rp + n;
    nr = rn + n;
    fail = nr + n;
    ex = fail + 1;
  }
};

#define Q(x, i, j) (x)[(size_t)(i) + (size_t)(j) * n]

/* The merge of level [lev] (1 = pairs of leaves) that index [x] of a
   matrix of order n falls in: its rows [a, b), split at [m]; false where
   that block has no right half at this level. */
D bool merge_of(int x, int lev, int n, int *a, int *m, int *b) {
  int span = LEAF << lev;
  *a = x / span * span;
  *m = *a + (span >> 1);
  *b = min(*a + span, n);
  return *m < *b;
}

/* Scale: T's largest magnitude brought into [1, 2) by a power of 2, the
   exponent kept; the tears at every leaf boundary; the leaves as dense
   L × L blocks (row-major) for Jacobi, the last one, of [last] values,
   apart. One thread per matrix for the scale, then per element. */
template <class R>
D void dc_leaves(void *wsp, const R *dsrc, const R *esrc, size_t stride,
                 R *full, R *tail, int64_t nb, int n) {
  const int nl = (n + LEAF - 1) / LEAF, last = n - (nl - 1) * LEAF;
  #pragma unroll 1
  for (int64_t b = (int64_t)blockIdx.x; b < nb; b += gridDim.x) {
    dc_ws<R> ws(wsp, b, n);
    __shared__ R big;
    #pragma unroll 1
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      ws.d[i] = dsrc[b * stride + i];
      ws.e[i] = i < n - 1 ? esrc[b * stride + i] : (R)0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      R x = 0;
      #pragma unroll 1
      for (int i = 0; i < n; i++) {
        x = fmax(x, fabs(ws.d[i]));
        if (i < n - 1) x = fmax(x, fabs(ws.e[i]));
      }
      big = x;
      ws.fail[0] = 0;
    }
    __syncthreads();
    int ex = big > 0 ? -ilogb(big) : 0;
    if (threadIdx.x == 0) ws.ex[0] = ex;
    #pragma unroll 1
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      ws.d[i] = ldexp(ws.d[i], ex);
      if (i < n - 1) ws.e[i] = ldexp(ws.e[i], ex);
    }
    __syncthreads();
    /* Each leaf's diagonal, less |ρ| at a torn end. */
    #pragma unroll 1
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      int leaf = i / LEAF, li = i % LEAF, size = leaf == nl - 1 ? last : LEAF;
      R di = ws.d[i];
      if (li == 0 && leaf > 0) di -= fabs(ws.e[i - 1]);
      if (li == size - 1 && leaf < nl - 1) di -= fabs(ws.e[i]);
      R *blk = leaf == nl - 1 ? tail + (size_t)b * last * last
                              : full + ((size_t)b * (nl - 1) + leaf) * LEAF * LEAF;
      #pragma unroll 1
      for (int j = 0; j < size; j++) {
        R x = 0;
        if (j == li) x = di;
        if (j == li + 1) x = ws.e[i];
        if (j == li - 1) x = ws.e[i - 1];
        blk[li * size + j] = x;
      }
    }
    __syncthreads();
  }
}

/* The leaves' eigenvalues into d and vectors into Q's diagonal blocks,
   Q zero elsewhere; a leaf whose Jacobi failed (NaN, its input finite)
   fails the matrix. One thread per element of Q. */
template <class R>
D void dc_scatter(void *wsp, const R *wf, const R *vf, const R *wt,
                  const R *vt, int64_t nb, int n) {
  const int nl = (n + LEAF - 1) / LEAF, last = n - (nl - 1) * LEAF;
  const size_t nn = (size_t)n * n;
  #pragma unroll 1
  for (int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       g < nb * (int64_t)nn; g += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = g / nn;
    dc_ws<R> ws(wsp, b, n);
    size_t k = g % nn;
    int i = (int)(k % n), j = (int)(k / n);
    int leaf = i / LEAF, size = leaf == nl - 1 ? last : LEAF;
    R x = 0;
    if (j / LEAF == leaf) {
      int li = i % LEAF, lj = j % LEAF;
      x = leaf == nl - 1
              ? vt[(size_t)b * last * last + li * size + lj]
              : vf[((size_t)b * (nl - 1) + leaf) * LEAF * LEAF + li * size + lj];
      if (i == j) {
        const R w = leaf == nl - 1 ? wt[(size_t)b * last + li]
                                   : wf[((size_t)b * (nl - 1) + leaf) * LEAF + li];
        ws.d[i] = w;
        if (w != w) ws.fail[0] = 1;
      }
    }
    ws.q[k] = x;
  }
}

/* Deflate (xLAED2): one block per merge. Thread 0 decides; the block
   rotates Q's rows and copies the set-aside columns to Qt[:, k…]. */
template <class R> D void dc_deflate(void *wsp, int64_t nb, int n, int lev) {
  const int span = LEAF << lev, per = (n + span - 1) / span;
  const R eps = lim<R>::eps();
  __shared__ int sk, snd, snr;
  #pragma unroll 1
  for (int64_t g = blockIdx.x; g < nb * per; g += gridDim.x) {
    int64_t b = g / per;
    int a, m, e;
    if (!merge_of((int)(g % per) * span, lev, n, &a, &m, &e)) continue;
    dc_ws<R> ws(wsp, b, n);
    const int N = e - a, n1 = m - a;
    R *z = ws.z + a, *d = ws.d + a, *dl = ws.dl + a, *w = ws.w + a,
      *dd = ws.dd + a, *rc = ws.rc + a, *rs = ws.rs + a;
    int *cols = ws.cols + a, *rp = ws.rp + a, *rn = ws.rn + a;
    R *q = ws.q;
    if (threadIdx.x == 0) {
      R rho = ws.e[m - 1];
      #pragma unroll 1
      for (int j = 0; j < N; j++) {
        R x = j < n1 ? Q(q, m - 1, a + j) : Q(q, m, a + j);
        if (rho < 0 && j >= n1) x = -x;
        z[j] = x * (1 / sqrt((R)2));
      }
      rho = fabs(2 * rho);
      /* The halves merged ascending, the first of equals from the left;
         indices into cols as scratch, then the kept ones. */
      int *ix = ws.kq + a;
      #pragma unroll 1
      for (int i = 0, l = 0, r = n1; i < N; i++)
        ix[i] = (r >= N || (l < n1 && d[l] <= d[r])) ? l++ : r++;
      R zmax = 0, dmax = 0;
      #pragma unroll 1
      for (int i = 0; i < N; i++) {
        zmax = fmax(zmax, fabs(z[i]));
        dmax = fmax(dmax, fabs(d[i]));
      }
      R tol = 8 * eps * fmax(dmax, zmax);
      int k = 0, nd = 0, nrot = 0;
      if (rho * zmax <= tol) {
        #pragma unroll 1
        for (int i = 0; i < N; i++) {
          dd[nd] = d[ix[i]];
          cols[N - 1 - nd++] = ix[i];
        }
      } else {
        int j = 0, pj = 0;
        #pragma unroll 1
        for (; j < N; j++) {
          int nj = ix[j];
          if (rho * fabs(z[nj]) > tol) {
            pj = nj;
            break;
          }
          dd[nd] = d[nj];
          cols[N - 1 - nd++] = nj;
        }
        #pragma unroll 1
        for (j++; j < N; j++) {
          int nj = ix[j];
          if (rho * fabs(z[nj]) <= tol) {
            dd[nd] = d[nj];
            cols[N - 1 - nd++] = nj;
            continue;
          }
          /* Two close values: a rotation zeroes pj's part of z where it
             moves no value by more than tol. */
          R sn = z[pj], cs = z[nj], tau = hypot(cs, sn), t = d[nj] - d[pj];
          cs /= tau;
          sn = -sn / tau;
          if (fabs(t * cs * sn) > tol) {
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
          t = d[pj] * (cs * cs) + d[nj] * (sn * sn);
          d[nj] = d[pj] * (sn * sn) + d[nj] * (cs * cs);
          d[pj] = t;
          dd[nd] = d[pj];
          cols[N - 1 - nd++] = pj;
          pj = nj;
        }
        dl[k] = d[pj];
        w[k] = z[pj];
        cols[k++] = pj;
      }
      ws.kq[a] = k;
      ws.nr[a] = nrot;
      sk = k;
      snd = nd;
      snr = nrot;
      /* ρ for the roots, in the tear's slot of e, unused after. */
      ws.e[m - 1] = rho;
    }
    __syncthreads();
    const int k = sk, nd = snd, nrot = snr;
    /* The rotations, in order, on each row of the merge's columns. */
    #pragma unroll 1
    for (int r = a + threadIdx.x; r < e; r += blockDim.x)
      #pragma unroll 1
      for (int t = 0; t < nrot; t++) {
        R *x = &Q(q, r, a + rp[t]), *y = &Q(q, r, a + rn[t]);
        R xr = *x, yr = *y;
        *x = rc[t] * xr + rs[t] * yr;
        *y = rc[t] * yr - rs[t] * xr;
      }
    __syncthreads();
    /* The set-aside values ascending, ties by order set aside: their
       ranks, then their columns into Qt[:, k + rank]. cols[N − 1 − i]
       holds the column of dd[i]. */
    #pragma unroll 1
    for (int i = threadIdx.x; i < nd; i += blockDim.x) {
      int rank = 0;
      #pragma unroll 1
      for (int j = 0; j < nd; j++)
        rank += dd[j] < dd[i] || (dd[j] == dd[i] && j < i);
      ws.lam[a + k + rank] = dd[i];
      ws.rn[a + k + rank] = cols[N - 1 - i];
    }
    __syncthreads();
    #pragma unroll 1
    for (size_t x = threadIdx.x; x < (size_t)nd * N; x += blockDim.x) {
      int j = (int)(x / N), r = (int)(x % N);
      Q(ws.qt, a + r, a + k + j) = Q(q, a + r, a + ws.rn[a + k + j]);
    }
    __syncthreads();
  }
}

/* The root [i] of the secular equation of two values, d[0] < d[1]
   (xLAED5): λ into [dlam], the normalised d_j − λ into delta. */
template <class R>
D void laed5(int i, const R *d, const R *z, R *delta, size_t ds, R rho,
             R *dlam) {
  R del = d[1] - d[0], tau, d0, d1;
  R b = -del + rho * (z[0] * z[0] + z[1] * z[1]), c = rho * z[1] * z[1] * del;
  if (i == 0 && 1 + 2 * rho * (z[1] * z[1] - z[0] * z[0]) / del > 0) {
    b = del + rho * (z[0] * z[0] + z[1] * z[1]);
    c = rho * z[0] * z[0] * del;
    tau = 2 * c / (b + sqrt(fabs(b * b - 4 * c)));
    *dlam = d[0] + tau;
    d0 = -z[0] / tau;
    d1 = z[1] / (del - tau);
  } else {
    if (i == 0)
      tau = b > 0 ? -2 * c / (b + sqrt(b * b + 4 * c))
                  : (b - sqrt(b * b + 4 * c)) / 2;
    else
      tau = b > 0 ? (b + sqrt(b * b + 4 * c)) / 2
                  : 2 * c / (-b + sqrt(b * b + 4 * c));
    *dlam = d[1] + tau;
    d0 = -z[0] / (del + tau);
    d1 = -z[1] / tau;
  }
  R t = sqrt(d0 * d0 + d1 * d1);
  delta[0] = d0 / t;
  delta[ds] = d1 / t;
}

/* The root τ near zero of f(x) = rho + Σ z_j / (d_j − x) over three
   poles (xLAED6). Returns false after 40 steps. */
template <class R>
D bool laed6(int kniter, bool orgati, R rho, const R *d, const R *z, R finit,
             R *tau_) {
  const R eps = lim<R>::eps();
  R lbd = orgati ? d[1] : d[0], ubd = orgati ? d[2] : d[1], tau = 0;
  if (finit < 0)
    lbd = 0;
  else
    ubd = 0;
  if (kniter == 2) {
    R a, b, c;
    if (orgati) {
      R t = (d[2] - d[1]) / 2;
      c = rho + z[0] / ((d[0] - d[1]) - t);
      a = c * (d[1] + d[2]) + z[1] + z[2];
      b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1];
    } else {
      R t = (d[0] - d[1]) / 2;
      c = rho + z[2] / ((d[2] - d[1]) - t);
      a = c * (d[0] + d[1]) + z[0] + z[1];
      b = c * d[0] * d[1] + z[0] * d[1] + z[1] * d[0];
    }
    R t = fmax(fabs(a), fmax(fabs(b), fabs(c)));
    a /= t, b /= t, c /= t;
    tau = c == 0   ? b / a
          : a <= 0 ? (a - sqrt(fabs(a * a - 4 * b * c))) / (2 * c)
                   : 2 * b / (a + sqrt(fabs(a * a - 4 * b * c)));
    if (tau < lbd || tau > ubd) tau = (lbd + ubd) / 2;
    if (d[0] == tau || d[1] == tau || d[2] == tau)
      tau = 0;
    else {
      R f = finit + tau * z[0] / (d[0] * (d[0] - tau)) +
            tau * z[1] / (d[1] * (d[1] - tau)) +
            tau * z[2] / (d[2] * (d[2] - tau));
      if (f <= 0)
        lbd = tau;
      else
        ubd = tau;
      if (fabs(finit) <= fabs(f)) tau = 0;
    }
  }
  const R small1 = sizeof(R) == 4 ? (R)0x1p-42 : (R)0x1p-340,
          small2 = small1 * small1;
  R t = orgati ? d[1] - tau : d[0] - tau, t2 = orgati ? d[2] - tau : d[1] - tau;
  t = fmin(fabs(t), fabs(t2));
  R sclfac = 1, sclinv = 1;
  if (t <= small1) {
    sclfac = t <= small2 ? 1 / small2 : 1 / small1;
    sclinv = t <= small2 ? small2 : small1;
  }
  R ds[3], zs[3];
  #pragma unroll 1
  for (int i = 0; i < 3; i++) ds[i] = d[i] * sclfac, zs[i] = z[i] * sclfac;
  tau *= sclfac, lbd *= sclfac, ubd *= sclfac;
  R fc = 0, df = 0, ddf = 0;
  #pragma unroll 1
  for (int i = 0; i < 3; i++) {
    R r = 1 / (ds[i] - tau), t1 = zs[i] * r, q2 = t1 * r;
    fc += t1 / ds[i];
    df += q2;
    ddf += q2 * r;
  }
  R f = finit + tau * fc;
  bool ok = fabs(f) <= 0;
  if (f <= 0)
    lbd = tau;
  else
    ubd = tau;
  #pragma unroll 1
  for (int it = 2; it <= 40 && !ok; it++) {
    R t1 = orgati ? ds[1] - tau : ds[0] - tau;
    R q2 = orgati ? ds[2] - tau : ds[1] - tau;
    R a = (t1 + q2) * f - t1 * q2 * df, b = t1 * q2 * f;
    R c = f - (t1 + q2) * df + t1 * q2 * ddf;
    R mm = fmax(fabs(a), fmax(fabs(b), fabs(c)));
    a /= mm, b /= mm, c /= mm;
    R eta = c == 0   ? b / a
            : a <= 0 ? (a - sqrt(fabs(a * a - 4 * b * c))) / (2 * c)
                     : 2 * b / (a + sqrt(fabs(a * a - 4 * b * c)));
    if (f * eta >= 0) eta = -f / df;
    tau += eta;
    if (tau < lbd || tau > ubd) tau = (lbd + ubd) / 2;
    R err = 0;
    fc = df = ddf = 0;
    int i = 0;
    #pragma unroll 1
    for (; i < 3 && ds[i] - tau != 0; i++) {
      R r = 1 / (ds[i] - tau), p1 = zs[i] * r, p2 = p1 * r, p4 = p1 / ds[i];
      fc += p4;
      err += fabs(p4);
      df += p2;
      ddf += p2 * r;
    }
    if (i < 3) {
      ok = true;
      break;
    }
    f = finit + tau * fc;
    err = 8 * (fabs(finit) + fabs(tau) * err) + fabs(tau) * df;
    ok = fabs(f) <= 4 * eps * err || ubd - lbd <= 4 * eps * fabs(tau);
    if (f <= 0)
      lbd = tau;
    else
      ubd = tau;
  }
  *tau_ = tau * sclinv;
  return ok;
}

/* Root [i] of 1 + ρ Σ z_j² / (d_j − λ) = 0, d ascending, z without
   zeros, ρ > 0 (xLAED4): λ into [dlam], d_j − λ into delta[j·ds], each
   from the pole nearer the root. Returns false after 30 steps. */
template <class R>
D bool laed4(int n, int i, const R *d, const R *z, R *delta, size_t ds, R rho,
             R *dlam) {
  const R eps = lim<R>::eps();
  if (n == 1) {
    *dlam = d[0] + rho * z[0] * z[0];
    delta[0] = 1;
    return true;
  }
  if (n == 2) {
    laed5(i, d, z, delta, ds, rho, dlam);
    return true;
  }
#define DL(j) delta[(size_t)(j) * ds]
  const R rhoinv = 1 / rho;
  const bool last = i == n - 1;
  bool orgati = true;
  R tau, lb, ub, org;
  if (last) {
    R midpt = rho / 2, psi = 0, del = d[n - 1] - d[n - 2];
    #pragma unroll 1
    for (int j = 0; j < n; j++) DL(j) = (d[j] - d[n - 1]) - midpt;
    #pragma unroll 1
    for (int j = 0; j < n - 2; j++) psi += z[j] * z[j] / DL(j);
    R c = rhoinv + psi;
    R w = c + z[n - 2] * z[n - 2] / DL(n - 2) + z[n - 1] * z[n - 1] / DL(n - 1);
    R a = -c * del + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
    R b = z[n - 1] * z[n - 1] * del;
    tau = a < 0 ? 2 * b / (sqrt(a * a + 4 * b * c) - a)
                : (a + sqrt(a * a + 4 * b * c)) / (2 * c);
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
    R del = d[i + 1] - d[i], midpt = del / 2, psi = 0, phi = 0;
    #pragma unroll 1
    for (int j = 0; j < n; j++) DL(j) = (d[j] - d[i]) - midpt;
    #pragma unroll 1
    for (int j = 0; j < i; j++) psi += z[j] * z[j] / DL(j);
    #pragma unroll 1
    for (int j = n - 1; j > i + 1; j--) phi += z[j] * z[j] / DL(j);
    R c = rhoinv + psi + phi, zi = z[i] * z[i], zj = z[i + 1] * z[i + 1];
    R w = c + zi / DL(i) + zj / DL(i + 1);
    orgati = w > 0;
    if (orgati) {
      R a = c * del + zi + zj, b = zi * del;
      tau = a > 0 ? 2 * b / (a + sqrt(fabs(a * a - 4 * b * c)))
                  : (a - sqrt(fabs(a * a - 4 * b * c))) / (2 * c);
      lb = 0;
      ub = midpt;
    } else {
      R a = c * del - zi - zj, b = zj * del;
      tau = a < 0 ? 2 * b / (a - sqrt(fabs(a * a + 4 * b * c)))
                  : -(a + sqrt(fabs(a * a + 4 * b * c))) / (2 * c);
      lb = -midpt;
      ub = 0;
    }
    org = orgati ? d[i] : d[i + 1];
  }
  #pragma unroll 1
  for (int j = 0; j < n; j++) DL(j) = (d[j] - org) - tau;
  const int ii = last ? n - 1 : orgati ? i : i + 1, ip = last ? n - 2 : i;
  bool swtch3 = false, swtch = false;
  R w = 0, dw = 0, psi = 0, dpsi = 0, phi = 0, dphi = 0, err = 0, prew = 0;
  #pragma unroll 1
  for (int step = 0;; step++) {
    psi = dpsi = err = phi = dphi = 0;
    #pragma unroll 1
    for (int j = 0; j < ii; j++) {
      R t = z[j] / DL(j);
      psi += z[j] * t;
      dpsi += t * t;
      err += psi;
    }
    err = fabs(err);
    #pragma unroll 1
    for (int j = n - 1; j > ii; j--) {
      R t = z[j] / DL(j);
      phi += z[j] * t;
      dphi += t * t;
      err += phi;
    }
    R t = z[ii] / DL(ii);
    if (last) {
      phi = z[ii] * t;
      dphi = t * t;
      w = rhoinv + phi + psi;
      dw = dpsi + dphi;
      err = 8 * (-phi - psi) + err - phi + rhoinv + fabs(tau) * dw;
    } else {
      w = rhoinv + phi + psi;
      if (step == 0) swtch3 = (orgati ? w < 0 : w > 0) && ii != 0 && ii != n - 1;
      dw = dpsi + dphi + t * t;
      t *= z[ii];
      w += t;
      err = 8 * (phi - psi) + err + 2 * rhoinv + 3 * fabs(t) + fabs(tau) * dw;
      if (step == 1)
        swtch = orgati ? -w > fabs(prew) / 10 : w > fabs(prew) / 10;
      else if (step > 1 && w * prew > 0 && fabs(w) > fabs(prew) / 10)
        swtch = !swtch;
    }
    if (step == 29) break;
    if (fabs(w) <= eps * err) {
      *dlam = org + tau;
      return true;
    }
    if (w <= 0)
      lb = fmax(lb, tau);
    else
      ub = fmin(ub, tau);
    R eta, a, b, c;
    R di = DL(ip), dj = DL(ip + 1);
    if (last) {
      c = w - di * dpsi - dj * dphi;
      a = (di + dj) * w - di * dj * dw;
      b = di * dj * w;
      if (step == 0 && c < 0) c = fabs(c);
      if (step == 0 && c == 0)
        eta = -w / dw;
      else if (a >= 0)
        eta = (a + sqrt(fabs(a * a - 4 * b * c))) / (2 * c);
      else
        eta = 2 * b / (a - sqrt(fabs(a * a - 4 * b * c)));
      if (w * eta > 0) eta = -w / dw;
    } else if (!swtch3) {
      if (swtch) {
        R q = z[ii] / DL(ii);
        if (orgati)
          dpsi += q * q;
        else
          dphi += q * q;
        c = w - di * dpsi - dj * dphi;
      } else {
        R q = orgati ? z[i] / di : z[i + 1] / dj;
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
        eta = (a - sqrt(fabs(a * a - 4 * b * c))) / (2 * c);
      else
        eta = 2 * b / (a + sqrt(fabs(a * a - 4 * b * c)));
      if (w * eta >= 0) eta = -w / dw;
    } else {
      R zz[3], dv[3], s = rhoinv + psi + phi, dm = DL(ii - 1), dp = DL(ii + 1);
      if (swtch) {
        c = s - dm * dpsi - dp * dphi;
        zz[0] = dm * dm * dpsi;
        zz[2] = dp * dp * dphi;
      } else if (orgati) {
        R q = z[ii - 1] / dm;
        q *= q;
        c = s - dp * (dpsi + dphi) - (d[ii - 1] - d[ii + 1]) * q;
        zz[0] = z[ii - 1] * z[ii - 1];
        zz[2] = dp * dp * ((dpsi - q) + dphi);
      } else {
        R q = z[ii + 1] / dp;
        q *= q;
        c = s - dm * (dpsi + dphi) - (d[ii + 1] - d[ii - 1]) * q;
        zz[0] = dm * dm * (dpsi + (dphi - q));
        zz[2] = z[ii + 1] * z[ii + 1];
      }
      zz[1] = z[ii] * z[ii];
      dv[0] = dm;
      dv[1] = DL(ii);
      dv[2] = dp;
      if (!laed6(step + 2, orgati, c, dv, zz, w, &eta)) return false;
      if (w * eta >= 0) eta = -w / dw;
    }
    if (tau + eta > ub || tau + eta < lb) {
      R half = w < 0 ? (ub - tau) / 2 : (lb - tau) / 2, newton = -w / dw;
      bool mean = (last ? step == 0 : step > 0) && lb <= tau + newton &&
                  tau + newton <= ub;
      eta = mean ? copysign((R)1, newton) * sqrt(fabs(newton)) * sqrt(fabs(half))
                 : half;
    }
    prew = w;
    #pragma unroll 1
    for (int j = 0; j < n; j++) DL(j) -= eta;
    tau += eta;
  }
#undef DL
  *dlam = org + tau;
  return false;
}

/* Roots, weights and vectors: one thread per index x of a matrix, root
   x − a of its merge where it is below k. [phase] 0 roots, 1 weights, 2
   vectors. */
template <class R> D void dc_secular(void *wsp, int64_t nb, int n, int lev,
                                     int phase) {
  #pragma unroll 1
  for (int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       g < nb * (int64_t)n; g += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = g / n;
    int x = (int)(g % n), a, m, e;
    if (!merge_of(x, lev, n, &a, &m, &e)) continue;
    dc_ws<R> ws(wsp, b, n);
    const int k = ws.kq[a], i = x - a;
    if (i >= k) continue;
    R *s = &Q(ws.s, a, a), *dl = ws.dl + a, *w = ws.w + a;
    if (phase == 0) {
      if (!laed4(k, i, dl, w, s + (size_t)i * n, 1, ws.e[m - 1], &ws.lam[a + i]))
        ws.fail[0] = 1;
      continue;
    }
    if (k <= 2) continue;
    if (phase == 1) {
      /* Gu and Eisenstat: |z_i| from the roots, z's sign kept. */
      R p = s[i + (size_t)i * n];
      #pragma unroll 1
      for (int c = 0; c < k; c++)
        if (c != i) p *= s[i + (size_t)c * n] / (dl[i] - dl[c]);
      ws.z[a + i] = copysign(sqrt(-p), w[i]);
      continue;
    }
    R *col = s + (size_t)i * n, big = 0;
    const R *zn = ws.z + a;
    #pragma unroll 1
    for (int r = 0; r < k; r++) {
      col[r] = zn[r] / col[r];
      big = fmax(big, fabs(col[r]));
    }
    /* A column has a few large elements near its pole and many below the
       last bit of their running sum: a compensated sum. */
    sumsq<R> sum;
    #pragma unroll 1
    for (int r = 0; r < k; r++) sum.sq(col[r] / big);
    R norm = big * sqrt(sum.get());
    #pragma unroll 1
    for (int r = 0; r < k; r++) col[r] /= norm;
  }
}

/* Qt[:, 0:k] = Q[:, kept] S over each merge's rows: 64 × 64 tiles of
   (row, root), one block each, z the merge; tiles past a merge's extents
   exit. */
template <class R> D void dc_product(void *wsp, int64_t nb, int n, int lev) {
  const int span = LEAF << lev, per = (n + span - 1) / span;
  __shared__ R As[16][65], Bs[16][65];
  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  #pragma unroll 1
  for (int64_t g = blockIdx.z; g < nb * per; g += gridDim.z) {
    int64_t b = g / per;
    int a, m, e;
    if (!merge_of((int)(g % per) * span, lev, n, &a, &m, &e)) continue;
    dc_ws<R> ws(wsp, b, n);
    const int N = e - a, k = ws.kq[a];
    const int r0 = blockIdx.x * 64, c0 = blockIdx.y * 64;
    if (r0 >= N || c0 >= k) continue;
    const int *cols = ws.cols + a;
    R acc[4][4] = {};
    #pragma unroll 1
    for (int k0 = 0; k0 < k; k0 += 16) {
      __syncthreads();
      #pragma unroll 1
      for (int t = threadIdx.x; t < 16 * 64; t += 256) {
        int r = t % 64, j = t / 64;
        As[j][r] = r0 + r < N && k0 + j < k
                       ? Q(ws.q, a + r0 + r, a + cols[k0 + j]) : (R)0;
        int jj = t % 16, c = t / 16;
        Bs[jj][c] = c0 + c < k && k0 + jj < k
                        ? Q(ws.s, a + k0 + jj, a + c0 + c) : (R)0;
      }
      __syncthreads();
#pragma unroll 4
      for (int j = 0; j < 16; j++)
#pragma unroll
        for (int p = 0; p < 4; p++)
#pragma unroll
          for (int q = 0; q < 4; q++)
            acc[p][q] += As[j][tx + 16 * p] * Bs[j][ty + 16 * q];
    }
#pragma unroll
    for (int p = 0; p < 4; p++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        int r = r0 + tx + 16 * p, c = c0 + ty + 16 * q;
        if (r < N && c < k) Q(ws.qt, a + r, a + c) = acc[p][q];
      }
  }
}

/* Sort: the roots (λ ascending) and the set-aside values (ascending, at
   λ[k…]) merged ascending, roots first among equals, into d, and Qt's
   columns into Q's in that order. One block per merge. */
template <class R> D void dc_sort(void *wsp, int64_t nb, int n, int lev) {
  const int span = LEAF << lev, per = (n + span - 1) / span;
  #pragma unroll 1
  for (int64_t g = blockIdx.x; g < nb * per; g += gridDim.x) {
    int64_t b = g / per;
    int a, m, e;
    if (!merge_of((int)(g % per) * span, lev, n, &a, &m, &e)) continue;
    dc_ws<R> ws(wsp, b, n);
    const int N = e - a, k = ws.kq[a];
    R *lam = ws.lam + a;
    int *src = ws.cols + a;
    if (threadIdx.x == 0)
      #pragma unroll 1
      for (int o = 0, i = 0, j = k; o < N; o++) {
        bool root = j >= N || (i < k && lam[i] <= lam[j]);
        src[o] = root ? i++ : j++;
      }
    __syncthreads();
    #pragma unroll 1
    for (int o = threadIdx.x; o < N; o += blockDim.x) ws.d[a + o] = lam[src[o]];
    #pragma unroll 1
    for (size_t t = threadIdx.x; t < (size_t)N * N; t += blockDim.x) {
      int o = (int)(t / N), r = (int)(t % N);
      Q(ws.q, a + r, a + o) = Q(ws.qt, a + r, a + src[o]);
    }
    __syncthreads();
  }
}

/* Bisection: eigenvalues [i0, i0 + cnt) of each matrix's T into w,
   [cnt] per matrix, one thread each, by Sturm
   counts (the negative pivots of T − x I, a pivot below pivmin in
   magnitude taken as −pivmin, as xSTEBZ), until the interval is within
   2u of its magnitude or pivmin. Gershgorin's bounds start it. */
template <class R>
D void tri_bisect(const R *dsrc, const R *esrc, size_t stride, R *w,
                  int64_t nb, int n, int i0, int cnt) {
  const R eps = lim<R>::eps(), tiny = lim<R>::tiny();
  #pragma unroll 1
  for (int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       g < nb * (int64_t)cnt; g += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = g / cnt;
    int i = i0 + (int)(g % cnt);
    const R *d = dsrc + b * stride, *e = esrc + b * stride;
    R lo = d[0], hi = d[0], emax = 0;
    #pragma unroll 1
    for (int j = 0; j < n; j++) {
      R r = (j > 0 ? fabs(e[j - 1]) : 0) + (j < n - 1 ? fabs(e[j]) : 0);
      lo = fmin(lo, d[j] - r);
      hi = fmax(hi, d[j] + r);
      if (j < n - 1) emax = fmax(emax, e[j] * e[j]);
    }
    R pivmin = tiny * fmax((R)1, emax);
    R span = fmax(fabs(lo), fabs(hi));
    lo -= 2 * eps * span * n + 2 * pivmin;
    hi += 2 * eps * span * n + 2 * pivmin;
    /* The count of eigenvalues below x. */
    auto count = [&](R x) {
      int c = 0;
      R q = d[0] - x;
      if (fabs(q) < pivmin) q = -pivmin;
      c += q < 0;
      #pragma unroll 1
      for (int j = 1; j < n; j++) {
        q = d[j] - x - e[j - 1] * e[j - 1] / q;
        if (fabs(q) < pivmin) q = -pivmin;
        c += q < 0;
      }
      return c;
    };
    #pragma unroll 1
    for (int it = 0; it < 4 * (int)sizeof(R) * 8; it++) {
      R mid = lo + (hi - lo) / 2;
      if (hi - lo <= 2 * eps * fmax(fabs(lo), fabs(hi)) + pivmin ||
          mid == lo || mid == hi)
        break;
      if (count(mid) > i)
        hi = mid;
      else
        lo = mid;
    }
    w[b * cnt + i - i0] = lo + (hi - lo) / 2;
  }
}

