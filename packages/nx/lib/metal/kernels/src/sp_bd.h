/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* svd past the small path. g, a or aᴴ where a is wide, M x K with
   M >= K, is reduced to g = Q B Pᴴ, B real upper bidiagonal, by
   Householder reflectors from both sides (xGEBD2), each a few launches
   over every matrix: the reflector (one group a matrix), its product
   with the trailing matrix (a group a column, or a thread a row), the
   rank-one update (a thread an element).

   B's singular values alone come by bisection on the Golub-Kahan matrix
   (sp_stedc.h's tri_bisect), the symmetric tridiagonal of zero diagonal
   whose eigenvalues are ±σ. With vectors, B = U_B Σ V_Bᵀ by implicit QR
   sweeps (xBDSQR, every bulge chased down), the sweeps' rotations made
   by one thread a matrix (bd_gen) and recorded, then applied to U_B's
   and V_B's rows by a thread a row, S sweeps a pass in a wavefront
   (bd_apply: Van Zee, van de Geijn and Quintana-Ortí, "Restructuring the
   tridiagonal and bidiagonal QR algorithms for performance", ACM TOMS
   40(3), 2014), so that each pass reads and writes a row once. Then
   u = Q U_B and v = P V_B by the reflectors' blocks, as in eigh.

   Matrices are column-major here, element (i, j) of g at g[i + j M]. A
   matrix's workspace lies at the byte offsets its layout gives
   (bd_layout), which the host lays out for the call's format; a step of
   parts alone reads its format's g, x, y and vc as [P] floats an
   element. A launch splits by its units, from [first] to [count] − 1,
   the bits not.

   This follows nx.amd's sp_bd.h. */

#ifndef NX_METAL_SP_BD_H
#define NX_METAL_SP_BD_H

#include "sp_block.h"

/* A large svd's workspace layout: the bytes of a matrix's workspace, the
   byte offsets of its arrays in them, and the records a pass holds. */
struct bd_layout {
  long per, g, x, y, vc, wv, tauq, taup, d, e, ub, vb, rec, sc, sw, st, cap,
      pt;
};

namespace sp {

#define S 8

template <class T> struct bd_ws {
  device T *g, *x, *y, *vc, *wv, *tauq, *taup, *pt;
  device float *d, *e, *ub, *vb, *rec, *sc;
  device int *sw, *st;
  int cap;
};

/* The views of matrix [b]'s workspace. */
template <class T>
static inline bd_ws<T> bd_of(device uchar *base, long b,
                             constant bd_layout &l) {
  bd_ws<T> ws;
  device uchar *p = base + ulong(b) * ulong(l.per);
  ws.g = (device T *)(p + l.g);
  ws.x = (device T *)(p + l.x);
  ws.y = (device T *)(p + l.y);
  ws.vc = (device T *)(p + l.vc);
  ws.wv = (device T *)(p + l.wv);
  ws.tauq = (device T *)(p + l.tauq);
  ws.taup = (device T *)(p + l.taup);
  ws.pt = (device T *)(p + l.pt);
  ws.d = (device float *)(p + l.d);
  ws.e = (device float *)(p + l.e);
  ws.ub = (device float *)(p + l.ub);
  ws.vb = (device float *)(p + l.vb);
  ws.rec = (device float *)(p + l.rec);
  ws.sc = (device float *)(p + l.sc);
  ws.sw = (device int *)(p + l.sw);
  ws.st = (device int *)(p + l.st);
  ws.cap = int(l.cap);
  return ws;
}

#define G(i, j) ws.g[(ulong)(i) + (ulong)(j) * M]

/* A thread an element: [q] over the launch's elements. */
#define ELEMENTS(q)                                                           \
  for (long q = first + gx * g.nthr + g.tid; q < count; q += ngx * g.nthr)

/* Stage: g from a (m x n, row-major), aᴴ where a is wide, read flushed,
   scaled by a power of 2 into nx.cpu's range; 0 and flagged where a is
   not finite. Three phases over the units from [first] to [count] − 1:
   0 clears a matrix's scan words, a thread a matrix; 1 scans a run of
   NX_SP_SCAN_RUN of a's floats for one that is not finite and for the
   largest magnitude, a thread a run (matrix, run); 2 stages an element
   and clears the reflectors' τ, a thread an element (matrix, element),
   element 0 writing the flag and the exponent. A maximum and a flag do
   not depend on the order the runs reach them. [du] and [dm] are the
   divisor words of a phase's units a matrix and of M. */
static inline void bd_stage(grp g, long gx, long ngx, device const float *a,
                            device uchar *wsp, long M, int K, long m, long n,
                            int P, int phase, ulong du, ulong dm,
                            constant bd_layout &l, long first, long count) {
  const float lo = sqrt_(lim::tiny()) / (2 * lim::eps());
  const ulong mn = ulong(m) * n;
  const long runs = long((mn * P + NX_SP_SCAN_RUN - 1) / NX_SP_SCAN_RUN);
  const bool small = is_small(count), inner = is_small(long(mn));
  constexpr metal::memory_order relaxed = metal::memory_order_relaxed;
  for (long u = first + gx * g.nthr + g.tid; u < count; u += ngx * g.nthr) {
    long b = u;
    const ulong k = phase == 0 ? 0
                    : phase == 1 ? ulong(split(b, runs, du, small))
                                 : ulong(split(b, long(mn), du, small));
    const bd_ws<float> ws = bd_of<float>(wsp, b, l);
    device metal::atomic_uint *scan =
        (device metal::atomic_uint *)(ws.st + NX_SP_ST_BAD);
    device const float *src = a + ulong(b) * mn * P;
    if (phase == 0) {
      metal::atomic_store_explicit(&scan[0], 0u, relaxed);
      metal::atomic_store_explicit(&scan[1], 0u, relaxed);
      continue;
    }
    if (phase == 1) {
      const ulong e = metal::min(mn * P, (k + 1) * NX_SP_SCAN_RUN);
      bool bad = false;
      float big = 0;
      for (ulong q = k * NX_SP_SCAN_RUN; q < e; q++) {
        const float v = flush(src[q]);
        bad = bad || v - v != 0;
        big = fmax_(big, fabs_(v));
      }
      if (bad) metal::atomic_store_explicit(&scan[0], 1u, relaxed);
      metal::atomic_fetch_max_explicit(&scan[1], as_type<uint>(big), relaxed);
      continue;
    }
    const bool ok = metal::atomic_load_explicit(&scan[0], relaxed) == 0;
    const float big =
        as_type<float>(metal::atomic_load_explicit(&scan[1], relaxed));
    const int ex = scale_exp(big, lo, 1 / lo);
    long j = long(k); /* k = i + M j, column-major */
    const long i = split(j, M, dm, inner);
    device const float *v =
        src + (m >= n ? ulong(i) * n + j : ulong(j) * n + i) * P;
    device float *gk = ws.g + k * P;
    gk[0] = ok ? scaled(flush(v[0]), ex) : 0.0f;
    if (P == 2) {
      const float vi = flush(v[1]);
      gk[1] = ok ? scaled(m >= n ? vi : -vi, ex) : 0.0f;
    }
    if (k < ulong(K))
      for (int p = 0; p < P; p++) ws.taup[k * P + p] = ws.tauq[k * P + p] = 0;
    if (k == 0) ws.st[NX_SP_ST_OK] = ok, ws.st[NX_SP_ST_EX] = ex;
  }
}

/* The left reflector of column c, H = I − τ v vᴴ, v = g(c:M, c) with
   its unit stored; d[c] = β. One group a matrix. */
template <class T>
static inline void bd_left(grp g, long gx, long ngx, threadgroup float *sh,
                           device uchar *wsp, long M, int c,
                           constant bd_layout &l, long first, long count) {
  for (long b = first + gx; b < count; b += ngx) {
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    device T *alpha = &G(c, c);
    const T tau = reflector<T>(g, alpha, alpha + 1, M - c - 1, 1, sh);
    if (g.tid == 0) {
      ws.d[c] = re(T(*alpha));
      ws.tauq[c] = tau;
      *alpha = elt<T>(1);
    }
    barrier();
  }
}

/* w_j = vᴴ g(c:M, j), j in (c, K): a group a column, its sum by the
   group's fixed tree. */
template <class T>
static inline void bd_wh(grp g, long gx, long ngx, threadgroup float *sh,
                         device uchar *wsp, long M, int K, int c,
                         constant bd_layout &l, long first, long count) {
  const int cols = K - c - 1;
  const bool small = is_small(count);
  for (long q = first + gx; q < count; q += ngx) {
    long b = q;
    const int j = c + 1 + int(split(b, cols, small));
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    T s = elt<T>(0);
    for (long r = c + g.tid; r < M; r += g.nthr)
      s += conj(T(G(r, c))) * T(G(r, j));
    const float sr = bsum(g, re(s), sh),
                si = real_of<T>::cplx() ? bsum(g, im(s), sh) : 0.0f;
    if (g.tid == 0) ws.wv[j] = elt<T>(sr, si);
  }
}

/* Columns past a chunk

   A column whose x (bd_left's) or v (bd_wh's) passes NX_SP_BD_CHUNK rows
   sums in chunks of that many rows, x's numbered from c + 1 and v's from
   c, each chunk a group, its thread t taking rows t, t + 256, … of it and
   the group's fixed tree closing them; then one group sums the chunks'
   partials in chunk order, its thread t taking chunks t, t + 256, …, and
   its tree closing them. The partials lie at the layout's [pt]. */

/* The rows [lo, hi) of chunk [q] of [len] rows. */
static inline void chunk_rows(long q, long len, thread long &lo,
                              thread long &hi) {
  lo = q * NX_SP_BD_CHUNK;
  hi = metal::min(len, lo + NX_SP_BD_CHUNK);
}

/* The reflector of column c past a chunk, a group a (matrix, chunk) of
   its x = g(c+1:M, c): in [phase] 0 x's partial sum of squares and
   largest part; in phase 1, of a matrix whose sum left the safe range
   (st's SCALED), the sum of squares of x over the largest part. */
template <class T>
static inline void bd_lsum(grp g, long gx, long ngx, threadgroup float *sh,
                           device uchar *wsp, long M, int c, int phase,
                           long chunks, ulong dchunks, constant bd_layout &l,
                           long first, long count) {
  const long len = M - c - 1;
  const bool small = is_small(count);
  for (long u = first + gx; u < count; u += ngx) {
    long b = u;
    const long q = split(b, chunks, dchunks, small);
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    device float *pt = (device float *)ws.pt;
    if (phase == 1 && ws.st[NX_SP_ST_SCALED] == 0) continue;
    const float big = as_type<float>(ws.st[NX_SP_ST_XBIG]);
    long lo, hi;
    chunk_rows(q, len, lo, hi);
    device T *x = &G(c + 1, c);
    float s = 0, m = 0;
    for (long i = lo + g.tid; i < hi; i += g.nthr) {
      const T xi = x[i];
      if (phase == 0) {
        s += abs2(xi);
        m = fmax_(m, fmax_(fabs_(re(xi)), fabs_(im(xi))));
      } else {
        s += abs2(cdiv(xi, elt<T>(big)));
      }
    }
    s = bsum(g, s, sh);
    if (phase == 0) m = bmax(g, m, sh);
    if (g.tid == 0) {
      pt[2 * q] = s;
      if (phase == 0) pt[2 * q + 1] = m;
    }
  }
}

/* The reflector's scalar part (xLARFG) from x's norm [xnorm]: β, τ and
   the scale x takes, τ = 0 where x and α's imaginary part are zero. A
   β below the safe minimum is found again at α and x times 2^103, the
   one rescaling a float32 needs, x's norm then [xnorm] times 2^103.
   Thread 0. */
template <class T>
static inline void bd_lfinish(bd_ws<T> ws, long M, int c, float xnorm) {
  const float sm = safmin();
  const T a0 = G(c, c);
  float ar = re(a0), ai = im(a0);
  int knt = 0;
  T tau = elt<T>(0), sc = elt<T>(1);
  float beta = ar;
  if (xnorm != 0 || ai != 0) {
    beta = ar >= 0 ? -lapy3(ar, ai, xnorm) : lapy3(ar, ai, xnorm);
    if (fabs_(beta) < sm) {
      knt = 1;
      ar *= 1 / sm, ai *= 1 / sm, xnorm *= 1 / sm;
      const float an = lapy3(ar, ai, xnorm);
      beta = ar >= 0 ? -an : an;
    }
    tau = elt<T>((beta - ar) / beta, -(ai / beta));
    sc = elt<T>(ar - beta, ai);
    if (knt) beta *= sm;
  }
  ws.d[c] = beta;
  ws.tauq[c] = tau;
  G(c, c) = elt<T>(1);
  ws.st[NX_SP_ST_KNT] = knt;
  ws.st[NX_SP_ST_SCRE] = as_type<int>(re(sc));
  ws.st[NX_SP_ST_SCIM] = as_type<int>(im(sc));
}

/* The reflector's partials closed, a group a matrix: in [phase] 0 the
   sum of squares q; where q lies in [safmin, 1/safmin], or is NaN, or x's
   largest part is 0, x's norm is √q and the reflector is finished, and
   otherwise st's SCALED is set with the largest part; in phase 1, of a
   SCALED matrix, the norm is the largest part times the root of the
   scaled sum. */
template <class T>
static inline void bd_lfin(grp g, long gx, long ngx, threadgroup float *sh,
                           device uchar *wsp, long M, int c, int phase,
                           long chunks, constant bd_layout &l, long first,
                           long count) {
  const float sm = safmin();
  for (long b = first + gx; b < count; b += ngx) {
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    device const float *pt = (device const float *)ws.pt;
    if (phase == 1 && ws.st[NX_SP_ST_SCALED] == 0) continue;
    float s = 0, m = 0;
    for (long q = g.tid; q < chunks; q += g.nthr) {
      s += pt[2 * q];
      if (phase == 0) m = fmax_(m, pt[2 * q + 1]);
    }
    s = bsum(g, s, sh);
    if (phase == 0) m = bmax(g, m, sh);
    if (g.tid == 0) {
      if (phase == 1) {
        bd_lfinish(ws, M, c, as_type<float>(ws.st[NX_SP_ST_XBIG]) * sqrt_(s));
      } else if ((s >= sm && s <= 1 / sm) || s != s || m == 0) {
        ws.st[NX_SP_ST_SCALED] = 0;
        bd_lfinish(ws, M, c, sqrt_(s));
      } else {
        ws.st[NX_SP_ST_SCALED] = 1;
        ws.st[NX_SP_ST_XBIG] = as_type<int>(m);
      }
    }
    barrier();
  }
}

/* x ← x 2^(103 knt) / sc, the reflector's v below its unit, a group a
   (matrix, chunk); x stays where τ = 0. */
template <class T>
static inline void bd_lscale(grp g, long gx, long ngx, device uchar *wsp,
                             long M, int c, long chunks, ulong dchunks,
                             constant bd_layout &l, long first, long count) {
  const long len = M - c - 1;
  const bool small = is_small(count);
  for (long u = first + gx; u < count; u += ngx) {
    long b = u;
    const long q = split(b, chunks, dchunks, small);
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    const T sc = elt<T>(as_type<float>(ws.st[NX_SP_ST_SCRE]),
                        as_type<float>(ws.st[NX_SP_ST_SCIM]));
    const bool knt = ws.st[NX_SP_ST_KNT] != 0;
    if (T(ws.tauq[c]) == elt<T>(0)) continue;
    long lo, hi;
    chunk_rows(q, len, lo, hi);
    device T *x = &G(c + 1, c);
    for (long i = lo + g.tid; i < hi; i += g.nthr) {
      const T xi = knt ? T(x[i]) * elt<T>(1 / safmin()) : T(x[i]);
      x[i] = cdiv(xi, sc);
    }
  }
}

/* w_j's partials past a chunk, a group a (matrix, column j in (c, K),
   chunk of v = g(c:M, c)), into pt[(j − c − 1) chunks + chunk]. */
template <class T>
static inline void bd_whc(grp g, long gx, long ngx, threadgroup float *sh,
                          device uchar *wsp, long M, int K, int c, long chunks,
                          ulong dchunks, ulong dcols, constant bd_layout &l,
                          long first, long count) {
  const long len = M - c;
  const int cols = K - c - 1;
  const bool small = is_small(count);
  for (long u = first + gx; u < count; u += ngx) {
    long b = u;
    const long q = split(b, chunks, dchunks, small);
    const int jc = int(split(b, cols, dcols, small));
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    long lo, hi;
    chunk_rows(q, len, lo, hi);
    T s = elt<T>(0);
    for (long r = c + lo + g.tid; r < c + hi; r += g.nthr)
      s += conj(T(G(r, c))) * T(G(r, c + 1 + jc));
    const float sr = bsum(g, re(s), sh),
                si = real_of<T>::cplx() ? bsum(g, im(s), sh) : 0.0f;
    if (g.tid == 0) ws.pt[long(jc) * chunks + q] = elt<T>(sr, si);
  }
}

/* w_j from its column's partials in chunk order, a group a (matrix,
   column j in (c, K)). */
template <class T>
static inline void bd_whfin(grp g, long gx, long ngx, threadgroup float *sh,
                            device uchar *wsp, int K, int c, long chunks,
                            ulong dcols, constant bd_layout &l, long first,
                            long count) {
  const int cols = K - c - 1;
  const bool small = is_small(count);
  for (long u = first + gx; u < count; u += ngx) {
    long b = u;
    const int jc = int(split(b, cols, dcols, small));
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    T s = elt<T>(0);
    for (long q = g.tid; q < chunks; q += g.nthr)
      s += T(ws.pt[long(jc) * chunks + q]);
    const float sr = bsum(g, re(s), sh),
                si = real_of<T>::cplx() ? bsum(g, im(s), sh) : 0.0f;
    if (g.tid == 0) ws.wv[c + 1 + jc] = elt<T>(sr, si);
  }
}

/* g(c:M, c+1:K) −= conj(τ) v wᵀ: Hᴴ from the left. */
template <class T>
static inline void bd_updl(grp g, long gx, long ngx, device uchar *wsp, long M,
                           int K, int c, constant bd_layout &l, long first,
                           long count) {
  const long rows = M - c, els = rows * (K - c - 1);
  const bool small = is_small(count);
  ELEMENTS(q) {
    long b = q;
    long k = split(b, els, small);
    const long r = c + split(k, rows, small);
    const int j = c + 1 + int(k);
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    G(r, j) = T(G(r, j)) - conj(T(ws.tauq[c])) * T(G(r, c)) * T(ws.wv[j]);
  }
}

/* The right reflector of row c, G = I − τ u uᴴ, u the conjugated row
   g(c, c+1:K) with its unit, stored as u; e[c] = β. One group a
   matrix. */
template <class T>
static inline void bd_right(grp g, long gx, long ngx, threadgroup float *sh,
                            device uchar *wsp, long M, int K, int c,
                            constant bd_layout &l, long first, long count) {
  for (long b = first + gx; b < count; b += ngx) {
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    for (int j = c + 1 + g.tid; j < K; j += g.nthr) G(c, j) = conj(T(G(c, j)));
    barrier();
    device T *alpha = &G(c, c + 1);
    const T tau = reflector<T>(g, alpha, alpha + M, K - c - 2, ulong(M), sh);
    if (g.tid == 0) {
      ws.e[c] = re(T(*alpha));
      ws.taup[c] = tau;
      *alpha = elt<T>(1);
    }
    barrier();
  }
}

/* y_r = g(r, c+1:K) u, r in (c, M): a thread a row. */
template <class T>
static inline void bd_wn(grp g, long gx, long ngx, device uchar *wsp, long M,
                         int K, int c, constant bd_layout &l, long first,
                         long count) {
  const long rows = M - c - 1;
  const bool small = is_small(count);
  ELEMENTS(q) {
    long b = q;
    const long r = c + 1 + split(b, rows, small);
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    T s = elt<T>(0);
    for (int j = c + 1; j < K; j++) s += T(G(r, j)) * T(G(c, j));
    ws.wv[r] = s;
  }
}

/* g(c+1:M, c+1:K) −= τ y uᴴ: G from the right. */
template <class T>
static inline void bd_updr(grp g, long gx, long ngx, device uchar *wsp, long M,
                           int K, int c, constant bd_layout &l, long first,
                           long count) {
  const long rows = M - c - 1, els = rows * (K - c - 1);
  const bool small = is_small(count);
  ELEMENTS(q) {
    long b = q;
    long k = split(b, els, small);
    const long r = c + 1 + split(k, rows, small);
    const int j = c + 1 + int(k);
    const bd_ws<T> ws = bd_of<T>(wsp, b, l);
    G(r, j) = T(G(r, j)) - T(ws.taup[c]) * T(ws.wv[r]) * conj(T(G(c, j)));
  }
}

/* The Golub-Kahan tridiagonal of B for bisection: zero diagonal and
   d0, e0, d1, … below it, 2K of each a matrix. */
static inline void bd_tgk(grp g, long gx, long ngx, device uchar *wsp, int K,
                          device float *dt, device float *et,
                          constant bd_layout &l, long first, long count) {
  const int n2 = 2 * K;
  const bool small = is_small(count);
  ELEMENTS(q) {
    long b = q;
    const int i = int(split(b, n2, small));
    const bd_ws<float> ws = bd_of<float>(wsp, b, l);
    dt[q] = 0;
    et[q] = i % 2 == 0 ? ws.d[i / 2] : i / 2 < K - 1 ? ws.e[i / 2] : 0.0f;
  }
}

/* The values alone: σ_j the eigenvalue K − 1 − j above K of the
   Golub-Kahan matrix, from [lam] (ascending, K a matrix). */
static inline void bd_values(grp g, long gx, long ngx, device uchar *wsp,
                             int K, device const float *lam, device float *s,
                             constant bd_layout &l, long first, long count) {
  ELEMENTS(q) {
    const long b = q / K;
    const int j = int(q % K);
    const bd_ws<float> ws = bd_of<float>(wsp, b, l);
    const float x = fmax_(lam[b * K + K - 1 - j], 0.0f);
    s[q] = ws.st[NX_SP_ST_OK] ? ldexp_(x, -ws.st[NX_SP_ST_EX]) + 0.0f : nan_();
  }
}

/* xBDSQR's tolerance factor: u^(-1/8) clamped to [10, 100]. */
static inline float tolmul() { return 10; }

/* The singular values [smin] <= [smax] of [f g; 0 h] (xLAS2). */
static inline void las2(float f, float g, float h, thread float &smin,
                        thread float &smax) {
  const float fa = fabs_(f), ga = fabs_(g), ha = fabs_(h);
  const float mn = fmin_(fa, ha), mx = fmax_(fa, ha);
  if (mn == 0) {
    const float lo = fmin_(mx, ga), hi = fmax_(mx, ga);
    smin = 0;
    smax = mx == 0 ? ga : hi * sqrt_(1 + (lo / hi) * (lo / hi));
    return;
  }
  if (ga < mx) {
    const float as = 1 + mn / mx, at = (mx - mn) / mx,
                au = (ga / mx) * (ga / mx);
    const float c = 2 / (sqrt_(as * as + au) + sqrt_(at * at + au));
    smin = mn * c;
    smax = mx / c;
    return;
  }
  const float au = mx / ga;
  if (au == 0) {
    smin = (mn * mx) / ga;
    smax = ga;
    return;
  }
  const float as = 1 + mn / mx, at = (mx - mn) / mx;
  const float c =
      1 / (sqrt_(1 + (as * au) * (as * au)) + sqrt_(1 + (at * au) * (at * au)));
  smin = 2 * ((mn * c) * au);
  smax = ga / (c + c);
}

/* The SVD of [f g; 0 h] and its rotations (xLASV2). */
static inline void lasv2(float f, float g, float h, thread float &smin,
                         thread float &smax, thread float &sr,
                         thread float &cr, thread float &sl,
                         thread float &cl) {
  const float eps = lim::eps();
  float ft = f, fa = fabs_(f), ht = h, ha = fabs_(h);
  const float gt = g, ga = fabs_(g);
  int pmax = 1;
  const bool swap = ha > fa;
  if (swap) {
    pmax = 3;
    ft = h;
    ht = f;
    fa = fabs_(h);
    ha = fabs_(f);
  }
  float mn = ha, mx = fa, clt = 1, crt = 1, slt = 0, srt = 0;
  if (ga != 0 && ga > fa && fa / ga < eps) {
    pmax = 2;
    mx = ga;
    mn = ha > 1 ? fa / (ga / ha) : (fa / ga) * ha;
    slt = ht / gt;
    srt = 1;
    crt = ft / gt;
  } else if (ga != 0) {
    if (ga > fa) pmax = 2;
    const float dd = fa - ha;
    float l = dd == fa ? 1 : dd / fa;
    const float m = gt / ft;
    float t = 2 - l;
    const float mm = m * m, s = sqrt_(t * t + mm);
    const float r = l == 0 ? fabs_(m) : sqrt_(l * l + mm), a = (s + r) / 2;
    mn = ha / a;
    mx = fa * a;
    if (mm != 0)
      t = (m / (s + t) + m / (r + l)) * (1 + a);
    else if (l == 0)
      t = copysign_(2.0f, ft) * copysign_(1.0f, gt);
    else
      t = gt / copysign_(dd, ft) + m / t;
    l = sqrt_(t * t + 4);
    crt = 2 / l;
    srt = t / l;
    clt = (crt + srt * m) / a;
    slt = (ht / ft) * srt / a;
  }
  cl = swap ? srt : clt;
  sl = swap ? crt : slt;
  cr = swap ? slt : crt;
  sr = swap ? clt : srt;
  const float sg =
      pmax == 1
          ? copysign_(1.0f, cr) * copysign_(1.0f, cl) * copysign_(1.0f, f)
      : pmax == 2
          ? copysign_(1.0f, sr) * copysign_(1.0f, cl) * copysign_(1.0f, g)
          : copysign_(1.0f, sr) * copysign_(1.0f, sl) * copysign_(1.0f, h);
  smax = copysign_(mx, sg);
  smin = copysign_(mn, sg * copysign_(1.0f, f) * copysign_(1.0f, h));
}

/* c, s, r with [c s; −s c] [f; g] = [r; 0] (xLARTG). */
static inline void lartg(float f, float g, thread float &c, thread float &s,
                         thread float &r) {
  const float tiny = lim::tiny();
  const float lo = sqrt_(tiny), hi = sqrt_((1 / tiny) / 2);
  const float af = fabs_(f), ag = fabs_(g);
  if (g == 0) {
    c = 1;
    s = 0;
    r = f;
    return;
  }
  if (f == 0) {
    c = 0;
    s = g < 0 ? -1 : 1;
    r = ag;
    return;
  }
  float u = 1;
  if (!(af > lo && af < hi && ag > lo && ag < hi)) {
    u = fmin_(fmax_(fmax_(af, ag), tiny), 1 / tiny);
    f /= u;
    g /= u;
  }
  const float h = sqrt_(f * f + g * g), rr = f < 0 ? -h : h;
  c = fabs_(f) / h;
  s = g / rr;
  r = rr * u;
}

/* The QR iteration's start: U_B = V_B = I, the threshold below which an
   e is negligible (xBDSQR). One group a matrix. */
static inline void bd_ginit(grp g, long gx, long ngx, device uchar *wsp, int K,
                            constant bd_layout &l, long first, long count) {
  const float eps = lim::eps(), tiny = lim::tiny();
  for (long b = first + gx; b < count; b += ngx) {
    const bd_ws<float> ws = bd_of<float>(wsp, b, l);
    for (ulong k = g.tid; k < ulong(K) * K; k += g.nthr)
      ws.ub[k] = ws.vb[k] = float(k % (K + 1) == 0);
    if (g.tid) continue;
    device const float *d = ws.d, *e = ws.e;
    const float tol = tolmul() * eps;
    float sminoa = fabs_(d[0]), mu = sminoa;
    for (int i = 1; i < K && sminoa != 0; i++) {
      mu = fabs_(d[i]) * (mu / (mu + fabs_(e[i - 1])));
      sminoa = fmin_(mu, sminoa);
    }
    sminoa /= sqrt_(float(K));
    const float thresh =
        fmax_(tol * sminoa, 6 * (float(K) * (float(K) * tiny)));
    ws.sc[0] = thresh;
    ws.sc[1] = tol;
    device int *st = ws.st;
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

/* xBDSQR's sweeps on B of [d] and [e], each chasing its bulge down,
   recorded into [ws]'s records, from the state [st] carries. */
template <class P>
static inline void bd_sweeps(bd_ws<float> ws, P d, P e, int K, int maxit) {
  const float eps = lim::eps();
  device int *st = ws.st;
  device float *rec = ws.rec;
  const float thresh = ws.sc[0], tol = ws.sc[1];
  int m = st[NX_SP_ST_M], iter = st[NX_SP_ST_ITER],
      sweeps = st[NX_SP_ST_SWEEPS], oldll = st[NX_SP_ST_OLDLL],
      oldm = st[NX_SP_ST_OLDM];
  int nsw = 0, nrec = 0;
  bool full = false, fail = false;
  while (m > 0) {
    if (iter >= K) {
      iter -= K;
      if (++sweeps >= maxit) {
        fail = true;
        break;
      }
    }
    float smax = fabs_(d[m]);
    int ll = m - 1;
    for (; ll >= 0; ll--) {
      const float abss = fabs_(d[ll]), abse = fabs_(e[ll]);
      if (abse <= thresh) break;
      smax = fmax_(smax, fmax_(abss, abse));
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
    device int *sw = ws.sw + 3 * nsw;
    if (ll == m - 1) {
      float smin, sx, sr, cr, sl, cl;
      lasv2(d[m - 1], e[m - 1], d[m], smin, sx, sr, cr, sl, cl);
      d[m - 1] = sx;
      e[m - 1] = 0;
      d[m] = smin;
      sw[0] = m - 1, sw[1] = m, sw[2] = nrec;
      device float *r4 = rec + 4 * ulong(nrec);
      r4[0] = cr, r4[1] = sr, r4[2] = cl, r4[3] = sl;
      nrec++, nsw++;
      m -= 2;
      continue;
    }
    if (fabs_(e[m - 1]) <= tol * fabs_(d[m])) {
      e[m - 1] = 0;
      continue;
    }
    bool split = false;
    float mu = fabs_(d[ll]), smin = mu;
    for (int l = ll; l < m; l++) {
      if (fabs_(e[l]) <= tol * mu) {
        e[l] = 0;
        split = true;
        break;
      }
      mu = fabs_(d[l + 1]) * (mu / (mu + fabs_(e[l])));
      smin = fmin_(mu, smin);
    }
    if (split) continue;
    oldll = ll;
    oldm = m;
    float shift = 0;
    const float lm = fmax_(eps, 0.01f * tol);
    if (!(float(K) * tol * (smin / smax) <= lm)) {
      const float sll = fabs_(d[ll]);
      float r;
      las2(d[m - 1], e[m - 1], d[m], shift, r);
      if (sll > 0 && (shift / sll) * (shift / sll) < eps) shift = 0;
    }
    iter += m - ll;
    sw[0] = ll, sw[1] = m, sw[2] = nrec;
    device float *r4 = rec + 4 * ulong(nrec);
    float cs = 1, sn = 0, ocs = 1, osn = 0, r, f, gg;
    if (shift == 0) {
      for (int i = ll; i < m; i++, r4 += 4) {
        lartg(d[i] * cs, e[i], cs, sn, r);
        if (i > ll) e[i - 1] = osn * r;
        float di;
        lartg(ocs * r, d[i + 1] * sn, ocs, osn, di);
        d[i] = di;
        r4[0] = cs, r4[1] = sn, r4[2] = ocs, r4[3] = osn;
      }
      const float h = d[m] * cs;
      d[m] = h * ocs;
      e[m - 1] = h * osn;
    } else {
      f = (fabs_(d[ll]) - shift) * (copysign_(1.0f, d[ll]) + shift / d[ll]);
      gg = e[ll];
      for (int i = ll; i < m; i++, r4 += 4) {
        lartg(f, gg, cs, sn, r);
        if (i > ll) e[i - 1] = r;
        f = cs * d[i] + sn * e[i];
        e[i] = cs * e[i] - sn * d[i];
        gg = sn * d[i + 1];
        d[i + 1] = cs * d[i + 1];
        lartg(f, gg, ocs, osn, r);
        d[i] = r;
        f = ocs * e[i] + osn * d[i + 1];
        d[i + 1] = ocs * d[i + 1] - osn * e[i];
        if (i < m - 1) {
          gg = osn * e[i + 1];
          e[i + 1] = ocs * e[i + 1];
        }
        r4[0] = cs, r4[1] = sn, r4[2] = ocs, r4[3] = osn;
      }
      e[m - 1] = f;
    }
    nrec += m - ll;
    nsw++;
    if (fabs_(e[m - 1]) <= thresh) e[m - 1] = 0;
  }
  st[NX_SP_ST_M] = m, st[NX_SP_ST_ITER] = iter, st[NX_SP_ST_SWEEPS] = sweeps;
  st[NX_SP_ST_OLDLL] = oldll, st[NX_SP_ST_OLDM] = oldm;
  st[NX_SP_ST_NSW] = nsw;
  st[NX_SP_ST_FAIL] |= fail;
  st[NX_SP_ST_DONE] = fail || !full;
}

/* Generate: xBDSQR's sweeps on B, each chasing its bulge down, recorded
   (rotations at k in [ll, m) of the right, then left, factor), until the
   records are full or B is diagonal; a sweep records a rotation at least,
   so the sweeps fit as the records do. One group a matrix, its thread 0
   working; the state carries over to the next launch. d and e lie in
   threadgroup memory, 2K floats, where they fit NX_SP_LDS_MOST bytes:
   the iteration is one dependent chain, and each step reads them.
   [maxit] is the rotations' limit over K², times K. */
static inline void bd_gen(grp g, long gx, long ngx, threadgroup float *sm,
                          device uchar *wsp, int K, int rotations,
                          constant bd_layout &l, long first, long count) {
  const bool fits = 2 * ulong(K) * sizeof(float) <= NX_SP_LDS_MOST;
  const int maxit = rotations * K;
  for (long b = first + gx; b < count; b += ngx) {
    if (g.tid) continue;
    const bd_ws<float> ws = bd_of<float>(wsp, b, l);
    ws.st[NX_SP_ST_NSW] = 0;
    if (ws.st[NX_SP_ST_DONE]) continue;
    if (!fits) {
      bd_sweeps(ws, ws.d, ws.e, K, maxit);
      continue;
    }
    threadgroup float *d = sm, *e = sm + K;
    for (int i = 0; i < K; i++) d[i] = ws.d[i], e[i] = ws.e[i];
    bd_sweeps(ws, d, e, K, maxit);
    for (int i = 0; i < K; i++) ws.d[i] = d[i], ws.e[i] = e[i];
  }
}

/* Apply: the recorded sweeps to U_B's rows (left rotations) and V_B's
   (right), a thread a row of either, S sweeps a pass in a wavefront: at
   stage t sweep s rotates columns t − s and t − s + 1, so that a row's
   S + 1 columns in flight stay in registers and each column is read and
   written once a pass. */
static inline void bd_apply(grp g, long gx, long ngx, device uchar *wsp, int K,
                            constant bd_layout &l, long first, long count) {
  const long rows = 2 * long(K);
  const bool small = is_small(count);
  ELEMENTS(q) {
    long b = q;
    int r = int(split(b, rows, small));
    const int side = r >= K;
    r -= side * K;
    const bd_ws<float> ws = bd_of<float>(wsp, b, l);
    const int nsw = ws.st[NX_SP_ST_NSW];
    device float *x = (side ? ws.vb : ws.ub) + r;
    device const float *rec = ws.rec + (side ? 0 : 2);
    for (int g0 = 0; g0 < nsw; g0 += S) {
      const int ns = metal::min(S, nsw - g0);
      int ll[S], mm[S], off[S], lo = K, hi = 0;
      #pragma unroll
      for (int s = 0; s < S; s++) {
        ll[s] = s < ns ? ws.sw[3 * (g0 + s)] : 0;
        mm[s] = s < ns ? ws.sw[3 * (g0 + s) + 1] : 0;
        off[s] = s < ns ? ws.sw[3 * (g0 + s) + 2] : 0;
        if (s < ns) lo = metal::min(lo, ll[s]), hi = metal::max(hi, mm[s]);
      }
      float w[S + 1];
      #pragma unroll
      for (int j = 0; j <= S; j++) w[j] = 0;
      w[S - 1] = x[ulong(lo) * K];
      for (int t = lo; t <= hi + S - 1; t++) {
        w[S] = t + 1 <= hi ? x[ulong(t + 1) * K] : 0.0f;
        #pragma unroll
        for (int s = 0; s < S; s++) {
          const int k = t - s;
          if (s < ns && k >= ll[s] && k < mm[s]) {
            device const float *c2 = rec + 4 * ulong(off[s] + k - ll[s]);
            const float c = c2[0], sn = c2[1], a0 = w[S - 1 - s],
                        a1 = w[S - s];
            w[S - 1 - s] = c * a0 + sn * a1;
            w[S - s] = c * a1 - sn * a0;
          }
        }
        const int col = t - S + 1;
        if (col >= lo) x[ulong(col) * K] = w[0];
        #pragma unroll
        for (int j = 0; j < S; j++) w[j] = w[j + 1];
      }
    }
  }
}

/* X = [U_B 0; 0 I] (M x gc) and Y = V_B, in [P] parts, for the
   reflectors. */
static inline void bd_xinit(grp g, long gx, long ngx, device uchar *wsp,
                            long M, int K, int gc, int P,
                            constant bd_layout &l, long first, long count) {
  const long els = M * gc + long(K) * K;
  const bool small = is_small(count);
  ELEMENTS(q) {
    long b = q;
    long k = split(b, els, small);
    const bd_ws<float> ws = bd_of<float>(wsp, b, l);
    device float *x;
    if (k < M * gc) {
      long col = k;
      const long i = split(col, M, small);
      const int j = int(col);
      x = ws.x + k * P;
      x[0] = i < K && j < K ? ws.ub[i + ulong(j) * K] : float(i == j);
    } else {
      k -= M * gc;
      x = ws.y + k * P;
      x[0] = ws.vb[k];
    }
    if (P == 2) x[1] = 0;
  }
}

/* The panel's reflectors, zeros above their units, row-major into Vc:
   side 0 the left ones (rows j0…M), side 1 the right ones (rows
   j0+1…K). */
static inline void bd_vcopy(grp g, long gx, long ngx, device uchar *wsp,
                            long M, int K, int j0, int nbk, int side, int P,
                            constant bd_layout &l, long first, long count) {
  const long rows = side ? K - j0 - 1 : M - j0;
  const long els = rows * nbk;
  const bool small = is_small(count);
  ELEMENTS(q) {
    long b = q;
    const long k = split(b, els, small);
    long row = k;
    const int c = j0 + int(split(row, nbk, small));
    const bd_ws<float> ws = bd_of<float>(wsp, b, l);
    /* The reflector's element (r, c) and where it lies in g. */
    const long r = j0 + row;
    const long at = side ? c + ulong(r + 1) * M : r + ulong(c) * M;
    device float *v = ws.vc + k * P;
    v[0] = r < c ? 0.0f : r == c ? 1.0f : ws.g[at * P];
    if (P == 2) v[1] = r <= c ? 0.0f : ws.g[at * P + 1];
  }
}

/* The order: singular values |d| descending, ties by index, rank and its
   inverse into the records (spent by now), as ints. */
static inline void bd_rank(grp g, long gx, long ngx, device uchar *wsp, int K,
                           constant bd_layout &l, long first, long count) {
  const bool small = is_small(count);
  ELEMENTS(q) {
    long b = q;
    const int j = int(split(b, K, small));
    const bd_ws<float> ws = bd_of<float>(wsp, b, l);
    /* NaN ranks last: the ranks stay a permutation, so bd_out writes
       every s of a failed matrix. */
    int rank = 0;
    const float dj = fabs_(ws.d[j]);
    const bool nan_j = dj != dj;
    for (int i = 0; i < K; i++) {
      const float di = fabs_(ws.d[i]);
      rank += di > dj || (di == dj && i < j) || (nan_j && (di == di || i < j));
    }
    device int *ix = (device int *)ws.rec;
    ix[j] = rank;
    ix[K + rank] = j;
  }
}

/* s, u and vh: tall, u = X and vh = Yᴴ; wide, u = Y and vh = Xᴴ, columns
   of X and Y in the order, Y's signed by d's; NaN for a flagged or
   failed matrix. */
static inline void bd_out(grp g, long gx, long ngx, device uchar *wsp, long M,
                          int K, device float *s, device float *u,
                          device float *vh, long m, long n, int uc, int vr,
                          int P, constant bd_layout &l, long first,
                          long count) {
  const bool wide = m < n;
  const long nu = m * uc, nv = long(vr) * n, els = K + nu + nv;
  const bool small = is_small(count);
  ELEMENTS(q) {
    long b = q;
    long k = split(b, els, small);
    const bd_ws<float> ws = bd_of<float>(wsp, b, l);
    device const int *ix = (device const int *)ws.rec, *inv = ix + K;
    const bool ok =
        ws.st[NX_SP_ST_OK] && !ws.st[NX_SP_ST_FAIL] && ws.st[NX_SP_ST_DONE];
    if (k < K) {
      s[b * K + ix[k]] =
          ok ? ldexp_(fabs_(ws.d[k]), -ws.st[NX_SP_ST_EX]) + 0.0f : nan_();
      continue;
    }
    k -= K;
    /* The element of X or Y at [src], conjugated under [cj], times [sg]
       as the product of two complex numbers is, sg's imaginary part 0.
       X's column j is U_B's inv[j] where j < K. */
    device const float *src;
    device float *dst;
    bool cj, signed_;
    float sg = 1;
    if (k < nu) {
      long i = k;
      const int j = int(split(i, uc, small));
      dst = u + (b * nu + k) * P;
      cj = false;
      signed_ = wide;
      if (!wide) {
        src = ws.x + (i + ulong(j < K ? inv[j] : j) * M) * P;
      } else {
        src = ws.y + (i + ulong(inv[j]) * K) * P;
        sg = ws.d[inv[j]] < 0 ? -1.0f : 1.0f;
      }
    } else {
      k -= nu;
      long row = k;
      const long j = split(row, n, small);
      const int i = int(row);
      dst = vh + (b * nv + k) * P;
      cj = true;
      signed_ = !wide;
      if (!wide) {
        src = ws.y + (j + ulong(inv[i]) * K) * P;
        sg = ws.d[inv[i]] < 0 ? -1.0f : 1.0f;
      } else {
        src = ws.x + (j + ulong(i < K ? inv[i] : i) * M) * P;
      }
    }
    if (!ok) {
      for (int p = 0; p < P; p++) dst[p] = nan_();
      continue;
    }
    const float a = src[0], c = P == 2 ? (cj ? -src[1] : src[1]) : 0.0f;
    if (!signed_) {
      dst[0] = a;
      if (P == 2) dst[1] = c;
    } else if (P == 1) {
      dst[0] = a * sg;
    } else {
      dst[0] = a * sg - c * 0.0f;
      dst[1] = a * 0.0f + c * sg;
    }
  }
}

#undef S
#undef G
#undef ELEMENTS

} // namespace sp

#endif
