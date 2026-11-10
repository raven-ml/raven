/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* svd past the small path. g, a or aᴴ where a is wide, M × K with
   M ≥ K, is reduced to g = Q B Pᴴ, B real upper bidiagonal, by
   Householder reflectors from both sides (xGEBD2), each a few launches
   over every matrix: the reflector (one block per matrix), its product
   with the trailing matrix (a warp per column, or a thread per row),
   the rank-one update (a thread per element).

   B's singular values alone come by bisection on the Golub-Kahan matrix
   (sp_stedc.cuh's tri_bisect), the symmetric tridiagonal of zero diagonal
   whose eigenvalues are ±σ. With vectors, B = U_B Σ V_Bᵀ by implicit QR
   sweeps (xBDSQR, every bulge chased down), the sweeps' rotations made
   by one thread per matrix (bd_gen) and recorded, then applied to U_B's
   and V_B's rows by a thread per row, S sweeps per pass in a wavefront
   (bd_apply: Van Zee, van de Geijn and Quintana-Ortí, "Restructuring the
   tridiagonal and bidiagonal QR algorithms for performance", ACM TOMS
   40(3), 2014), so that each pass reads and writes a row once. Then
   u = Q U_B and v = P V_B by the reflectors' blocks, as in eigh.

   Matrices are column-major here, element (i, j) of g at g[i + j·M]. */

#include "sp_block.cuh"

#define S 8

template <class T> struct bd_ws {
  typedef typename real_of<T>::R R;
  T *g, *x, *y, *vc, *tm, *gg, *wv, *tauq, *taup;
  R *d, *e, *ub, *vb, *rec, *sc;
  int *sw, *st, cap;
  /* [t] an element's bytes: a step of reals alone takes its format's
     workspace as T = R. */
  __device__ bd_ws(void *base, int64_t b, int64_t M, int K, int gc, int t = sizeof(T)) {
    nx_sp_bd l = nx_sp_bd_layout(M, K, gc, t, sizeof(R));
    char *p = (char *)base + (size_t)b * l.per;
    g = (T *)(p + l.g);
    x = (T *)(p + l.x);
    y = (T *)(p + l.y);
    vc = (T *)(p + l.vc);
    tm = (T *)(p + l.tm);
    gg = (T *)(p + l.gg);
    wv = (T *)(p + l.wv);
    tauq = (T *)(p + l.tauq);
    taup = (T *)(p + l.taup);
    d = (R *)(p + l.d);
    e = (R *)(p + l.e);
    ub = (R *)(p + l.ub);
    vb = (R *)(p + l.vb);
    rec = (R *)(p + l.rec);
    sc = (R *)(p + l.sc);
    sw = (int *)(p + l.sw);
    st = (int *)(p + l.st);
    cap = l.cap;
  }
};

#define G(i, j) ws.g[(size_t)(i) + (size_t)(j) * M]
#define ARGS void *wsp, int64_t nb, int64_t M, int K, int gc

/* Stage: g from a (m × n, row-major), aᴴ where a is wide, scaled by a
   power of 2 into nx.cpu's range; 0 and flagged where a is not finite. */
template <class R>
__device__ __noinline__ void bd_stage(const R *a, ARGS, int64_t m, int64_t n, int P) {
  __shared__ R sh[256];
  const R small = sqrt(lim<R>::tiny()) / (2 * lim<R>::eps());
  const size_t mn = (size_t)m * n;
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    bd_ws<R> ws(wsp, b, M, K, gc, P * sizeof(R));
    const R *src = a + b * mn * P;
    R nan = 0, big = 0;
    #pragma unroll 1
    for (size_t k = threadIdx.x; k < mn * P; k += blockDim.x) {
      R v = src[k];
      nan += v - v;
      big = fmax(big, fabs(v));
    }
    bool ok = bsum(nan, sh) == 0;
    big = bmax(big, sh);
    int ex = scale_exp(big, small, 1 / small);
    #pragma unroll 1
    for (size_t k = threadIdx.x; k < mn; k += blockDim.x) {
      int64_t i = k % M, j = k / M;
      const R *v = src + (m >= n ? (size_t)i * n + j : (size_t)j * n + i) * P;
      R *g = ws.g + k * P;
      g[0] = ok ? scaled(v[0], ex) : (R)0;
      if (P == 2) g[1] = ok ? scaled(m >= n ? v[1] : -v[1], ex) : (R)0;
    }
    #pragma unroll 1
    for (int j = threadIdx.x; j < K * P; j += blockDim.x)
      ws.taup[j] = ws.tauq[j] = (R)0;
    if (threadIdx.x == 0) ws.st[NX_SP_ST_OK] = ok, ws.st[NX_SP_ST_EX] = ex;
  }
}

/* The left reflector of column c, H = I − τ v vᴴ, v = g(c:M, c) with
   its unit stored; d[c] = β. One block per matrix. */
template <class T> D void bd_left(ARGS, int c) {
  typedef typename real_of<T>::R R;
  __shared__ R sh[256];
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    bd_ws<T> ws(wsp, b, M, K, gc);
    T *alpha = &G(c, c);
    T tau = reflector(alpha, alpha + 1, M - c - 1, 1, sh);
    if (threadIdx.x == 0) {
      ws.d[c] = re(*alpha);
      ws.tauq[c] = tau;
      *alpha = elt<T>((R)1);
    }
    __syncthreads();
  }
}

/* w_j = vᴴ g(c:M, j), j in (c, K): a block per column, its sum by the
   block's fixed tree. */
template <class T> D void bd_wh(ARGS, int c) {
  typedef typename real_of<T>::R R;
  __shared__ R sh[256];
  const int cols = K - c - 1;
  #pragma unroll 1
  for (int64_t q = blockIdx.x; q < nb * cols; q += gridDim.x) {
    int64_t b = q / cols;
    int j = c + 1 + (int)(q % cols);
    bd_ws<T> ws(wsp, b, M, K, gc);
    T s = elt<T>((R)0);
    #pragma unroll 1
    for (int64_t r = c + threadIdx.x; r < M; r += blockDim.x)
      s += conj(G(r, c)) * G(r, j);
    R sr = bsum(re(s), sh), si = real_of<T>::cplx ? bsum(im(s), sh) : (R)0;
    if (threadIdx.x == 0) ws.wv[j] = elt<T>(sr, si);
  }
}

/* g(c:M, c+1:K) −= conj(τ) v wᵀ: Hᴴ from the left. */
template <class T> D void bd_updl(ARGS, int c) {
  const int64_t rows = M - c, per = rows * (K - c - 1);
  #pragma unroll 1
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; q < nb * per;
       q += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = q / per, k = q % per;
    int64_t r = c + k % rows;
    int j = c + 1 + (int)(k / rows);
    bd_ws<T> ws(wsp, b, M, K, gc);
    G(r, j) -= conj(ws.tauq[c]) * G(r, c) * ws.wv[j];
  }
}

/* The right reflector of row c, G = I − τ u uᴴ, u the conjugated row
   g(c, c+1:K) with its unit, stored as u; e[c] = β. One block per
   matrix. */
template <class T> D void bd_right(ARGS, int c) {
  typedef typename real_of<T>::R R;
  __shared__ R sh[256];
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    bd_ws<T> ws(wsp, b, M, K, gc);
    #pragma unroll 1
    for (int j = c + 1 + threadIdx.x; j < K; j += blockDim.x)
      G(c, j) = conj(G(c, j));
    __syncthreads();
    T *alpha = &G(c, c + 1);
    T tau = reflector(alpha, alpha + M, K - c - 2, (size_t)M, sh);
    if (threadIdx.x == 0) {
      ws.e[c] = re(*alpha);
      ws.taup[c] = tau;
      *alpha = elt<T>((R)1);
    }
    __syncthreads();
  }
}

/* y_r = g(r, c+1:K) u, r in (c, M): a thread per row. */
template <class T> D void bd_wn(ARGS, int c) {
  typedef typename real_of<T>::R R;
  const int64_t rows = M - c - 1;
  #pragma unroll 1
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < nb * rows; q += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = q / rows;
    int64_t r = c + 1 + q % rows;
    bd_ws<T> ws(wsp, b, M, K, gc);
    T s = elt<T>((R)0);
    #pragma unroll 1
    for (int j = c + 1; j < K; j++) s += G(r, j) * G(c, j);
    ws.wv[r] = s;
  }
}

/* g(c+1:M, c+1:K) −= τ y uᴴ: G from the right. */
template <class T> D void bd_updr(ARGS, int c) {
  const int64_t rows = M - c - 1, per = rows * (K - c - 1);
  #pragma unroll 1
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; q < nb * per;
       q += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = q / per, k = q % per;
    int64_t r = c + 1 + k % rows;
    int j = c + 1 + (int)(k / rows);
    bd_ws<T> ws(wsp, b, M, K, gc);
    G(r, j) -= ws.taup[c] * ws.wv[r] * conj(G(c, j));
  }
}

/* The Golub-Kahan tridiagonal of B for bisection: zero diagonal and
   d0, e0, d1, … below it, 2K of each per matrix. */
template <class T>
__device__ __noinline__ void bd_tgk(ARGS, typename real_of<T>::R *dt, typename real_of<T>::R *et,
                                    int t) {
  const int n2 = 2 * K;
  #pragma unroll 1
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < nb * n2; q += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = q / n2;
    int i = (int)(q % n2);
    bd_ws<T> ws(wsp, b, M, K, gc, t);
    dt[q] = 0;
    et[q] = i % 2 == 0 ? ws.d[i / 2] : i / 2 < K - 1 ? ws.e[i / 2] : 0;
  }
}

/* The values alone: σ_j the eigenvalue K − 1 − j above K of the
   Golub-Kahan matrix, from [lam] (ascending, K per matrix). */
template <class T>
__device__ __noinline__ void bd_values(ARGS, const typename real_of<T>::R *lam,
                 typename real_of<T>::R *s, int t) {
  typedef typename real_of<T>::R R;
  #pragma unroll 1
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; q < nb * K;
       q += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = q / K;
    int j = (int)(q % K);
    bd_ws<T> ws(wsp, b, M, K, gc, t);
    R x = fmax(lam[b * K + K - 1 - j], (R)0);
    s[q] = ws.st[NX_SP_ST_OK] ? ldexp(x, -ws.st[NX_SP_ST_EX]) + (R)0 : (R)NAN;
  }
}

/* xBDSQR's tolerance factor: u^(-1/8) clamped to [10, 100]. */
template <class R> D R tolmul() {
  return sizeof(R) == 4 ? (R)10 : (R)0x1.8ace5422aa0dbp+6;
}

/* The singular values [smin] ≤ [smax] of [f g; 0 h] (xLAS2). */
template <class R> D void las2(R f, R g, R h, R *smin, R *smax) {
  R fa = fabs(f), ga = fabs(g), ha = fabs(h);
  R mn = fmin(fa, ha), mx = fmax(fa, ha);
  if (mn == 0) {
    R lo = fmin(mx, ga), hi = fmax(mx, ga);
    *smin = 0;
    *smax = mx == 0 ? ga : hi * sqrt(1 + (lo / hi) * (lo / hi));
    return;
  }
  if (ga < mx) {
    R as = 1 + mn / mx, at = (mx - mn) / mx, au = (ga / mx) * (ga / mx);
    R c = 2 / (sqrt(as * as + au) + sqrt(at * at + au));
    *smin = mn * c;
    *smax = mx / c;
    return;
  }
  R au = mx / ga;
  if (au == 0) {
    *smin = (mn * mx) / ga;
    *smax = ga;
    return;
  }
  R as = 1 + mn / mx, at = (mx - mn) / mx;
  R c = 1 / (sqrt(1 + (as * au) * (as * au)) + sqrt(1 + (at * au) * (at * au)));
  *smin = 2 * ((mn * c) * au);
  *smax = ga / (c + c);
}

/* The SVD of [f g; 0 h] and its rotations (xLASV2). */
template <class R>
D void lasv2(R f, R g, R h, R *smin, R *smax, R *sr, R *cr, R *sl, R *cl) {
  const R eps = lim<R>::eps();
  R ft = f, fa = fabs(f), ht = h, ha = fabs(h), gt = g, ga = fabs(g);
  int pmax = 1;
  bool swap = ha > fa;
  if (swap) {
    pmax = 3;
    ft = h;
    ht = f;
    fa = fabs(h);
    ha = fabs(f);
  }
  R mn = ha, mx = fa, clt = 1, crt = 1, slt = 0, srt = 0;
  if (ga != 0 && ga > fa && fa / ga < eps) {
    pmax = 2;
    mx = ga;
    mn = ha > 1 ? fa / (ga / ha) : (fa / ga) * ha;
    slt = ht / gt;
    srt = 1;
    crt = ft / gt;
  } else if (ga != 0) {
    if (ga > fa) pmax = 2;
    R dd = fa - ha, l = dd == fa ? 1 : dd / fa, m = gt / ft, t = 2 - l;
    R mm = m * m, s = sqrt(t * t + mm);
    R r = l == 0 ? fabs(m) : sqrt(l * l + mm), a = (s + r) / 2;
    mn = ha / a;
    mx = fa * a;
    if (mm != 0)
      t = (m / (s + t) + m / (r + l)) * (1 + a);
    else if (l == 0)
      t = copysign((R)2, ft) * copysign((R)1, gt);
    else
      t = gt / copysign(dd, ft) + m / t;
    l = sqrt(t * t + 4);
    crt = 2 / l;
    srt = t / l;
    clt = (crt + srt * m) / a;
    slt = (ht / ft) * srt / a;
  }
  *cl = swap ? srt : clt;
  *sl = swap ? crt : slt;
  *cr = swap ? slt : crt;
  *sr = swap ? clt : srt;
  R sg = pmax == 1   ? copysign((R)1, *cr) * copysign((R)1, *cl) * copysign((R)1, f)
         : pmax == 2 ? copysign((R)1, *sr) * copysign((R)1, *cl) * copysign((R)1, g)
                     : copysign((R)1, *sr) * copysign((R)1, *sl) * copysign((R)1, h);
  *smax = copysign(mx, sg);
  *smin = copysign(mn, sg * copysign((R)1, f) * copysign((R)1, h));
}

/* c, s, r with [c s; −s c] [f; g] = [r; 0] (xLARTG). */
template <class R> D void lartg(R f, R g, R *c, R *s, R *r) {
  const R tiny = lim<R>::tiny();
  const R lo = sqrt(tiny), hi = sqrt((1 / tiny) / 2);
  R af = fabs(f), ag = fabs(g);
  if (g == 0) {
    *c = 1;
    *s = 0;
    *r = f;
    return;
  }
  if (f == 0) {
    *c = 0;
    *s = g < 0 ? -1 : 1;
    *r = ag;
    return;
  }
  R u = 1;
  if (!(af > lo && af < hi && ag > lo && ag < hi)) {
    u = fmin(fmax(fmax(af, ag), tiny), 1 / tiny);
    f /= u;
    g /= u;
  }
  R h = sqrt(f * f + g * g), rr = f < 0 ? -h : h;
  *c = fabs(f) / h;
  *s = g / rr;
  *r = rr * u;
}

/* The QR iteration's start: U_B = V_B = I, the threshold below which an
   e is negligible (xBDSQR). One block per matrix. */
template <class T> __device__ __noinline__ void bd_ginit(ARGS, int t) {
  typedef typename real_of<T>::R R;
  const R eps = lim<R>::eps(), tiny = lim<R>::tiny();
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    bd_ws<T> ws(wsp, b, M, K, gc, t);
    #pragma unroll 1
    for (size_t k = threadIdx.x; k < (size_t)K * K; k += blockDim.x)
      ws.ub[k] = ws.vb[k] = (R)(k % (K + 1) == 0);
    if (threadIdx.x) continue;
    const R *d = ws.d, *e = ws.e;
    R tol = tolmul<R>() * eps, sminoa = fabs(d[0]), mu = sminoa;
    #pragma unroll 1
    for (int i = 1; i < K && sminoa != 0; i++) {
      mu = fabs(d[i]) * (mu / (mu + fabs(e[i - 1])));
      sminoa = fmin(mu, sminoa);
    }
    sminoa /= sqrt((R)K);
    R thresh = fmax(tol * sminoa, 6 * ((R)K * ((R)K * tiny)));
    ws.sc[0] = thresh;
    ws.sc[1] = tol;
    int *st = ws.st;
    st[NX_SP_ST_M] = K - 1;
    st[NX_SP_ST_ITER] = -1;
    st[NX_SP_ST_SWEEPS] = 0;
    st[NX_SP_ST_OLDLL] = -1;
    st[NX_SP_ST_OLDM] = -1;
    st[NX_SP_ST_DONE] = K <= 1 || !st[NX_SP_ST_OK];
    st[NX_SP_ST_FAIL] = 0;
    st[NX_SP_ST_NSW] = 0;
  }
}

/* Generate: xBDSQR's sweeps on B, each chasing its bulge down, recorded
   (rotations at k in [ll, m) of the right, then left, factor), until the
   records are full or B is diagonal; a sweep records a rotation at least,
   so the sweeps fit as the records do. One block per matrix, its thread 0
   working; the state carries over to the next launch. */
template <class T> D void bd_gen(ARGS) {
  typedef typename real_of<T>::R R;
  const R eps = lim<R>::eps();
  /* d and e in shared memory where they fit: the iteration is one
     dependent chain, and each step reads them. */
  extern __shared__ __align__(16) unsigned char smem[];
  const bool fits = 2 * (size_t)K * sizeof(R) <= 48 * 1024;
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    if (threadIdx.x) continue;
    bd_ws<T> ws(wsp, b, M, K, gc);
    int *st = ws.st;
    st[NX_SP_ST_NSW] = 0;
    if (st[NX_SP_ST_DONE]) continue;
    R *d = ws.d, *e = ws.e, *rec = ws.rec;
    if (fits) {
      d = (R *)smem;
      e = d + K;
      for (int i = 0; i < K; i++) d[i] = ws.d[i], e[i] = ws.e[i];
    }
    const R thresh = ws.sc[0], tol = ws.sc[1];
    int m = st[NX_SP_ST_M], iter = st[NX_SP_ST_ITER], sweeps = st[NX_SP_ST_SWEEPS],
        oldll = st[NX_SP_ST_OLDLL], oldm = st[NX_SP_ST_OLDM];
    int nsw = 0, nrec = 0;
    const int maxit = 6 * K;
    bool full = false, fail = false;
    while (m > 0) {
      if (iter >= K) {
        iter -= K;
        if (++sweeps >= maxit) {
          fail = true;
          break;
        }
      }
      R smax = fabs(d[m]);
      int ll = m - 1;
      #pragma unroll 1
      for (; ll >= 0; ll--) {
        R abss = fabs(d[ll]), abse = fabs(e[ll]);
        if (abse <= thresh) break;
        smax = fmax(smax, fmax(abss, abse));
      }
      if (ll >= 0) {
        e[ll] = 0;
        if (ll == m - 1) {
          m--;
          continue;
        }
      }
      ll++;
      if (nrec + (m - ll) > ws.cap) {
        full = true;
        break;
      }
      int *sw = ws.sw + 3 * nsw;
      if (ll == m - 1) {
        R smin, sx, sr, cr, sl, cl;
        lasv2(d[m - 1], e[m - 1], d[m], &smin, &sx, &sr, &cr, &sl, &cl);
        d[m - 1] = sx;
        e[m - 1] = 0;
        d[m] = smin;
        sw[0] = m - 1, sw[1] = m, sw[2] = nrec;
        R *r4 = rec + 4 * (size_t)nrec;
        r4[0] = cr, r4[1] = sr, r4[2] = cl, r4[3] = sl;
        nrec++, nsw++;
        m -= 2;
        continue;
      }
      if (fabs(e[m - 1]) <= tol * fabs(d[m])) {
        e[m - 1] = 0;
        continue;
      }
      bool split = false;
      R mu = fabs(d[ll]), smin = mu;
      #pragma unroll 1
      for (int l = ll; l < m; l++) {
        if (fabs(e[l]) <= tol * mu) {
          e[l] = 0;
          split = true;
          break;
        }
        mu = fabs(d[l + 1]) * (mu / (mu + fabs(e[l])));
        smin = fmin(mu, smin);
      }
      if (split) continue;
      oldll = ll;
      oldm = m;
      R shift = 0, lm = fmax(eps, (R)0.01 * tol);
      if (!((R)K * tol * (smin / smax) <= lm)) {
        R sll = fabs(d[ll]), r;
        las2(d[m - 1], e[m - 1], d[m], &shift, &r);
        if (sll > 0 && (shift / sll) * (shift / sll) < eps) shift = 0;
      }
      iter += m - ll;
      sw[0] = ll, sw[1] = m, sw[2] = nrec;
      R *r4 = rec + 4 * (size_t)nrec;
      R cs = 1, sn = 0, ocs = 1, osn = 0, r, f, g;
      if (shift == 0) {
        #pragma unroll 1
        for (int i = ll; i < m; i++, r4 += 4) {
          lartg(d[i] * cs, e[i], &cs, &sn, &r);
          if (i > ll) e[i - 1] = osn * r;
          lartg(ocs * r, d[i + 1] * sn, &ocs, &osn, &d[i]);
          r4[0] = cs, r4[1] = sn, r4[2] = ocs, r4[3] = osn;
        }
        R h = d[m] * cs;
        d[m] = h * ocs;
        e[m - 1] = h * osn;
      } else {
        f = (fabs(d[ll]) - shift) * (copysign((R)1, d[ll]) + shift / d[ll]);
        g = e[ll];
        #pragma unroll 1
        for (int i = ll; i < m; i++, r4 += 4) {
          lartg(f, g, &cs, &sn, &r);
          if (i > ll) e[i - 1] = r;
          f = cs * d[i] + sn * e[i];
          e[i] = cs * e[i] - sn * d[i];
          g = sn * d[i + 1];
          d[i + 1] = cs * d[i + 1];
          lartg(f, g, &ocs, &osn, &r);
          d[i] = r;
          f = ocs * e[i] + osn * d[i + 1];
          d[i + 1] = ocs * d[i + 1] - osn * e[i];
          if (i < m - 1) {
            g = osn * e[i + 1];
            e[i + 1] = ocs * e[i + 1];
          }
          r4[0] = cs, r4[1] = sn, r4[2] = ocs, r4[3] = osn;
        }
        e[m - 1] = f;
      }
      nrec += m - ll;
      nsw++;
      if (fabs(e[m - 1]) <= thresh) e[m - 1] = 0;
    }
    if (fits)
      for (int i = 0; i < K; i++) ws.d[i] = d[i], ws.e[i] = e[i];
    st[NX_SP_ST_M] = m, st[NX_SP_ST_ITER] = iter, st[NX_SP_ST_SWEEPS] = sweeps;
    st[NX_SP_ST_OLDLL] = oldll, st[NX_SP_ST_OLDM] = oldm;
    st[NX_SP_ST_NSW] = nsw;
    st[NX_SP_ST_FAIL] |= fail;
    st[NX_SP_ST_DONE] = fail || !full;
  }
}

/* Apply: the recorded sweeps to U_B's rows (left rotations) and V_B's
   (right), a thread per row of either, S sweeps per pass in a wavefront:
   at stage t sweep s rotates columns t − s and t − s + 1, so that a row's
   S + 1 columns in flight stay in registers and each column is read and
   written once a pass. */
template <class T> __device__ __noinline__ void bd_apply(ARGS, int t) {
  typedef typename real_of<T>::R R;
  const int64_t rows = 2 * (int64_t)K;
  #pragma unroll 1
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       q < nb * rows; q += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = q / rows;
    int r = (int)(q % rows), side = r >= K;
    r -= side * K;
    bd_ws<T> ws(wsp, b, M, K, gc, t);
    const int nsw = ws.st[NX_SP_ST_NSW];
    R *x = (side ? ws.vb : ws.ub) + r;
    const R *rec = ws.rec + (side ? 0 : 2);
    #pragma unroll 1
    for (int g0 = 0; g0 < nsw; g0 += S) {
      const int ns = min(S, nsw - g0);
      int ll[S], mm[S], off[S], lo = K, hi = 0;
#pragma unroll
      for (int s = 0; s < S; s++) {
        ll[s] = s < ns ? ws.sw[3 * (g0 + s)] : 0;
        mm[s] = s < ns ? ws.sw[3 * (g0 + s) + 1] : 0;
        off[s] = s < ns ? ws.sw[3 * (g0 + s) + 2] : 0;
        if (s < ns) lo = min(lo, ll[s]), hi = max(hi, mm[s]);
      }
      R w[S + 1];
#pragma unroll
      for (int j = 0; j <= S; j++) w[j] = 0;
      w[S - 1] = x[(size_t)lo * K];
      #pragma unroll 1
      for (int t = lo; t <= hi + S - 1; t++) {
        w[S] = t + 1 <= hi ? x[(size_t)(t + 1) * K] : (R)0;
#pragma unroll
        for (int s = 0; s < S; s++) {
          int k = t - s;
          if (s < ns && k >= ll[s] && k < mm[s]) {
            const R *c2 = rec + 4 * (size_t)(off[s] + k - ll[s]);
            R c = c2[0], sn = c2[1], a0 = w[S - 1 - s], a1 = w[S - s];
            w[S - 1 - s] = c * a0 + sn * a1;
            w[S - s] = c * a1 - sn * a0;
          }
        }
        int col = t - S + 1;
        if (col >= lo) x[(size_t)col * K] = w[0];
#pragma unroll
        for (int j = 0; j < S; j++) w[j] = w[j + 1];
      }
    }
  }
}

/* X = [U_B 0; 0 I] (M × gc) and Y = V_B, in T, for the reflectors. */
template <class R> __device__ __noinline__ void bd_xinit(ARGS, int P) {
  const int64_t per = (int64_t)M * gc + (int64_t)K * K;
  #pragma unroll 1
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; q < nb * per;
       q += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = q / per, k = q % per;
    bd_ws<R> ws(wsp, b, M, K, gc, P * sizeof(R));
    R *x;
    if (k < (int64_t)M * gc) {
      int64_t i = k % M;
      int j = (int)(k / M);
      x = ws.x + k * P;
      x[0] = i < K && j < K ? ws.ub[i + (size_t)j * K] : (R)(i == j);
    } else {
      k -= (int64_t)M * gc;
      x = ws.y + k * P;
      x[0] = ws.vb[k];
    }
    if (P == 2) x[1] = 0;
  }
}

/* The panel's reflectors, zeros above their units, row-major into Vc:
   side 0 the left ones (rows j0…M), side 1 the right ones (rows
   j0+1…K). */
template <class R> __device__ __noinline__ void bd_vcopy(ARGS, int j0, int nbk, int side, int P) {
  const int64_t rows = side ? K - j0 - 1 : M - j0;
  const int64_t per = rows * nbk;
  #pragma unroll 1
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; q < nb * per;
       q += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = q / per;
    const int64_t k = q % per;
    const int c = j0 + (int)(k % nbk);
    bd_ws<R> ws(wsp, b, M, K, gc, P * sizeof(R));
    /* The reflector's element (r, c) and where it lies in g. */
    int64_t r, at;
    if (!side) {
      r = j0 + k / nbk;
      at = r + (size_t)c * M;
    } else {
      r = j0 + k / nbk;
      at = c + (size_t)(r + 1) * M;
    }
    R *v = ws.vc + k * P;
    v[0] = r < c ? (R)0 : r == c ? (R)1 : ws.g[at * P];
    if (P == 2) v[1] = r <= c ? (R)0 : ws.g[at * P + 1];
  }
}

/* T of the panel's block reflector from G = Vᴴ V (xLARFT, forward,
   columnwise). One block of NB threads per matrix. */
template <class T> D void bd_larft(ARGS, int j0, int nbk, int side) {
  typedef typename real_of<T>::R R;
  #pragma unroll 1
  for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
    bd_ws<T> ws(wsp, b, M, K, gc);
    T *tm = ws.tm, *g = ws.gg, *tau = side ? ws.taup : ws.tauq;
    #pragma unroll 1
    for (int i = 0; i < nbk; i++) {
      int j = threadIdx.x;
      T y = elt<T>((R)0);
      if (j < i) {
        #pragma unroll 1
        for (int p = j; p < i; p++) y += tm[j * NX_SP_BD_NB + p] * g[p * nbk + i];
        y = -(tau[j0 + i] * y);
      }
      __syncthreads();
      if (j < i) tm[j * NX_SP_BD_NB + i] = y;
      if (j == i) tm[i * NX_SP_BD_NB + i] = tau[j0 + i];
      if (j > i && j < nbk) tm[j * NX_SP_BD_NB + i] = elt<T>((R)0);
      __syncthreads();
    }
  }
}

/* The order: singular values |d| descending, ties by index, rank and its
   inverse into the records (spent by now), as ints. */
template <class T> __device__ __noinline__ void bd_rank(ARGS, int t) {
  #pragma unroll 1
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; q < nb * K;
       q += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = q / K;
    int j = (int)(q % K);
    bd_ws<T> ws(wsp, b, M, K, gc, t);
    int rank = 0;
    #pragma unroll 1
    for (int i = 0; i < K; i++)
      rank += fabs(ws.d[i]) > fabs(ws.d[j]) ||
              (fabs(ws.d[i]) == fabs(ws.d[j]) && i < j);
    int *ix = (int *)ws.rec;
    ix[j] = rank;
    ix[K + rank] = j;
  }
}

/* s, u and vh: tall, u = X and vh = Yᴴ; wide, u = Y and vh = Xᴴ, columns
   of X and Y in the order, Y's signed by d's; NaN for a flagged or
   failed matrix. */
template <class R>
__device__ __noinline__ void bd_out(ARGS, R *s, R *u, R *vh, int64_t m, int64_t n, int uc, int vr,
                                    int P) {
  const bool wide = m < n;
  const int64_t nu = (int64_t)m * uc, nv = (int64_t)vr * n, per = K + nu + nv;
  #pragma unroll 1
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; q < nb * per;
       q += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = q / per, k = q % per;
    bd_ws<R> ws(wsp, b, M, K, gc, P * sizeof(R));
    const int *ix = (const int *)ws.rec, *inv = ix + K;
    const bool ok = ws.st[NX_SP_ST_OK] && !ws.st[NX_SP_ST_FAIL];
    auto xcol = [&](int j) { return j < K ? inv[j] : j; };
    auto sgn = [&](int j) { return ws.d[j] < 0 ? (R)-1 : (R)1; };
    if (k < K) {
      s[b * K + ix[k]] =
          ok ? ldexp(fabs(ws.d[k]), -ws.st[NX_SP_ST_EX]) + (R)0 : (R)NAN;
      continue;
    }
    k -= K;
    /* The element of X or Y at [at], conjugated under [cj], times [sg]
       as the product of two complex numbers is, sg's imaginary part 0. */
    const R *src;
    R *dst;
    bool cj, signed_;
    R sg = 1;
    if (k < nu) {
      int64_t i = k / uc;
      int j = (int)(k % uc);
      dst = u + (b * nu + k) * P;
      cj = false;
      signed_ = wide;
      if (!wide) src = ws.x + (i + (size_t)xcol(j) * M) * P;
      else src = ws.y + (i + (size_t)inv[j] * K) * P, sg = sgn(inv[j]);
    } else {
      k -= nu;
      int i = (int)(k / n);
      int64_t j = k % n;
      dst = vh + (b * nv + k) * P;
      cj = true;
      signed_ = !wide;
      if (!wide) src = ws.y + (j + (size_t)inv[i] * K) * P, sg = sgn(inv[i]);
      else src = ws.x + (j + (size_t)xcol(i) * M) * P;
    }
    if (!ok) {
      for (int p = 0; p < P; p++) dst[p] = (R)NAN;
      continue;
    }
    const R a = src[0], c = P == 2 ? (cj ? -src[1] : src[1]) : (R)0;
    if (!signed_) {
      dst[0] = a;
      if (P == 2) dst[1] = c;
    } else if (P == 1) {
      dst[0] = a * sg;
    } else {
      dst[0] = a * sg - c * (R)0;
      dst[1] = a * (R)0 + c * sg;
    }
  }
}

