/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fourier transforms (kernels.h, Fourier transforms): fft runs every step
   of a transform, fft_real a long real line's split or pre-pass, one
   kernel of each per width.

   A launch of fft is one Stockham step: virtual line v is batch line
   l = v / M and butterfly b = v mod M; it loads the L points b + r M,
   twiddles point r by w_N^((b mod Ns) r N / (Ns L)), runs an L-point FFT
   in LDS and stores output k at (b / Ns) Ns L + b mod Ns + k Ns. Outputs
   come in natural order, and a launch writes only points it read (the
   last step) or points of another buffer, so no group reads what another
   wrote.

   A group first copies the words its launch uses into LDS (stage) and
   reads them there: each crosses from the kernel arguments once a group,
   never in a loop. A group has NX_FFT_THREADS
   work-items, of which the step's g lines of tl take the first g tl; the
   others leave once the words are copied.

   The inner FFT holds a work-item's butterflies (at most 16 complex64 or 8
   complex128 points) in one register array that every radix shares, with
   the barrier between a pass's reads and writes in the dispatcher, so the
   radices take one register allocation. LDS is padded one slot in 16 (8),
   so strides of 2 to 16 points fall on distinct banks. Each product by an
   irrational constant fuses into the addition after it, an explicit fma,
   since the unit compiles without contraction; fft.ml's comments derive
   each butterfly's error from these operations. */

#include "device.h"

namespace nx_fft {

/* Complex arithmetic */

template <class S> struct cx {
  S r, i;
};

DEVICE float fmaS(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
DEVICE double fmaS(double a, double b, double c) {
  return __builtin_fma(a, b, c);
}

template <class S> DEVICE cx<S> cadd(cx<S> a, cx<S> b) {
  return {a.r + b.r, a.i + b.i};
}
template <class S> DEVICE cx<S> csub(cx<S> a, cx<S> b) {
  return {a.r - b.r, a.i - b.i};
}
template <class S> DEVICE cx<S> cconj(cx<S> a) { return {a.r, -a.i}; }

/* a w: two products and two fused multiply-adds. */
template <class S> DEVICE cx<S> cmul(cx<S> a, cx<S> w) {
  return {fmaS(a.r, w.r, -(a.i * w.i)), fmaS(a.r, w.i, a.i * w.r)};
}

/* Butterflies: forward DFTs of 2, 3, 4, 5, 7, 8 and 16 points in natural
   order, in place. */

template <class S>
DEVICE void dft4(cx<S> &a0, cx<S> &a1, cx<S> &a2, cx<S> &a3) {
  cx<S> t0 = cadd(a0, a2), t1 = csub(a0, a2), t2 = cadd(a1, a3),
        t3 = csub(a1, a3);
  a0 = cadd(t0, t2);
  a2 = csub(t0, t2);
  /* -i t3 = (t3.i, -t3.r) */
  a1 = {t1.r + t3.i, t1.i - t3.r};
  a3 = {t1.r - t3.i, t1.i + t3.r};
}

template <class S, int R> struct dft;

template <class S> struct dft<S, 2> {
  DEVICE void run(cx<S> *v) {
    cx<S> a = v[0], b = v[1];
    v[0] = cadd(a, b);
    v[1] = csub(a, b);
  }
};

template <class S> struct dft<S, 4> {
  DEVICE void run(cx<S> *v) { dft4(v[0], v[1], v[2], v[3]); }
};

/* DFT8: radix 2 across stride 4, then DFT4 of the sums and of the
   differences times w8^k, w8 = (1 - i)/sqrt 2, the products by 1/sqrt 2
   fused into the odd outputs. */
template <class S> struct dft<S, 8> {
  DEVICE void run(cx<S> *v) {
    const S c = S(0.70710678118654752440);
    cx<S> b0 = cadd(v[0], v[4]), b1 = cadd(v[1], v[5]), b2 = cadd(v[2], v[6]),
          b3 = cadd(v[3], v[7]);
    cx<S> d0 = csub(v[0], v[4]), d1 = csub(v[1], v[5]), d2 = csub(v[2], v[6]),
          d3 = csub(v[3], v[7]);
    dft4(b0, b1, b2, b3);
    v[0] = b0;
    v[2] = b1;
    v[4] = b2;
    v[6] = b3;
    /* e2 = -i d2; t0 = d0 + e2, t1 = d0 - e2 */
    cx<S> t0 = {d0.r + d2.i, d0.i - d2.r}, t1 = {d0.r - d2.i, d0.i + d2.r};
    /* e1 = c (p1, q1), e3 = c (p3, q3) */
    S p1 = d1.r + d1.i, q1 = d1.i - d1.r;
    S p3 = d3.i - d3.r, q3 = -(d3.r + d3.i);
    S sr = p1 + p3, si = q1 + q3, dr = p1 - p3, di = q1 - q3;
    v[1] = {fmaS(c, sr, t0.r), fmaS(c, si, t0.i)};
    v[5] = {fmaS(-c, sr, t0.r), fmaS(-c, si, t0.i)};
    v[3] = {fmaS(c, di, t1.r), fmaS(-c, dr, t1.i)};
    v[7] = {fmaS(-c, di, t1.r), fmaS(c, dr, t1.i)};
  }
};

/* DFT16 as 4 x 4: DFT4 over n2 of v[n1 + 4 n2], twiddles w16^(n1 k2),
   DFT4 over n1 into y[k2 + 4 k1]. */
template <class S> struct dft<S, 16> {
  DEVICE cx<S> w1(cx<S> x, S C, S Sn) {
    return {fmaS(x.r, C, x.i * Sn), fmaS(x.i, C, -(x.r * Sn))};
  }
  DEVICE void run(cx<S> *v) {
    const S c = S(0.70710678118654752440);
    const S C = S(0.92387953251128675613), Sn = S(0.38268343236508977173);
#pragma unroll
    for (int n1 = 0; n1 < 4; n1++)
      dft4(v[n1], v[n1 + 4], v[n1 + 8], v[n1 + 12]);
    /* u[n1][k2] is at v[n1 + 4 k2] */
    v[5] = w1(v[5], C, Sn);                                /* w^1 */
    v[9] = {c * (v[9].r + v[9].i), c * (v[9].i - v[9].r)}; /* w^2 */
    v[13] = w1(v[13], Sn, C);                              /* w^3 = (S, -C) */
    v[6] = {c * (v[6].r + v[6].i), c * (v[6].i - v[6].r)}; /* w^2 */
    v[10] = {v[10].i, -v[10].r};                           /* w^4 = -i */
    v[14] = {c * (v[14].i - v[14].r), -(c * (v[14].r + v[14].i))}; /* w^6 */
    v[7] = w1(v[7], Sn, C);                                        /* w^3 */
    v[11] = {c * (v[11].i - v[11].r), -(c * (v[11].r + v[11].i))}; /* w^6 */
    {
      cx<S> t = w1(v[15], C, Sn); /* w^9 = -w^1 */
      v[15] = {-t.r, -t.i};
    }
    cx<S> u[16];
#pragma unroll
    for (int k = 0; k < 16; k++) u[k] = v[k];
#pragma unroll
    for (int k2 = 0; k2 < 4; k2++) {
      cx<S> a0 = u[4 * k2], a1 = u[4 * k2 + 1], a2 = u[4 * k2 + 2],
            a3 = u[4 * k2 + 3];
      dft4(a0, a1, a2, a3);
      v[k2] = a0;
      v[k2 + 4] = a1;
      v[k2 + 8] = a2;
      v[k2 + 12] = a3;
    }
  }
};

/* An odd prime P's constants cos(2 pi m / P) and sin(2 pi m / P), m < P,
   as constant expressions: every index is known once the butterfly's
   loops unroll. */
template <int P> struct oddc;
template <> struct oddc<3> {
  static constexpr double c(int m) { return m == 0 ? 1.0 : -0.5; }
  static constexpr double s(int m) {
    return m == 0   ? 0.0
           : m == 1 ? 0.86602540378443864676
                    : -0.86602540378443864676;
  }
};
template <> struct oddc<5> {
  static constexpr double c(int m) {
    return m == 0              ? 1.0
           : (m == 1 || m == 4) ? 0.30901699437494742410
                                : -0.80901699437494742410;
  }
  static constexpr double s(int m) {
    return m == 0   ? 0.0
           : m == 1 ? 0.95105651629515357212
           : m == 2 ? 0.58778525229247312917
           : m == 3 ? -0.58778525229247312917
                    : -0.95105651629515357212;
  }
};
template <> struct oddc<7> {
  static constexpr double c(int m) {
    return m == 0              ? 1.0
           : (m == 1 || m == 6) ? 0.62348980185873353053
           : (m == 2 || m == 5) ? -0.22252093395631440429
                                : -0.90096886790241912624;
  }
  static constexpr double s(int m) {
    return m == 0   ? 0.0
           : m == 1 ? 0.78183148246802980871
           : m == 2 ? 0.97492791218182360702
           : m == 3 ? 0.43388373911755812048
           : m == 4 ? -0.43388373911755812048
           : m == 5 ? -0.97492791218182360702
                    : -0.78183148246802980871;
  }
};

/* An odd prime P: t_j = a_j + a_(P-j), d_j = a_j - a_(P-j);
   y_k = a0 + sum_j cos(2 pi jk/P) t_j - i sum_j sin(2 pi jk/P) d_j and
   y_(P-k) its conjugate pair. */
template <class S, int P> struct dft_odd {
  DEVICE void run(cx<S> *v) {
    constexpr int H = (P - 1) / 2;
    cx<S> t[H + 1], d[H + 1];
#pragma unroll
    for (int j = 1; j <= H; j++) {
      t[j] = cadd(v[j], v[P - j]);
      d[j] = csub(v[j], v[P - j]);
    }
    cx<S> a0 = v[0], y0 = v[0];
#pragma unroll
    for (int j = 1; j <= H; j++) y0 = cadd(y0, t[j]);
    v[0] = y0;
#pragma unroll
    for (int k = 1; k <= H; k++) {
      cx<S> m = a0;
#pragma unroll
      for (int j = 1; j <= H; j++) {
        const S cj = S(oddc<P>::c((j * k) % P));
        m = {fmaS(cj, t[j].r, m.r), fmaS(cj, t[j].i, m.i)};
      }
      const S s1 = S(oddc<P>::s(k % P));
      cx<S> n = {s1 * d[1].r, s1 * d[1].i};
#pragma unroll
      for (int j = 2; j <= H; j++) {
        const S sj = S(oddc<P>::s((j * k) % P));
        n = {fmaS(sj, d[j].r, n.r), fmaS(sj, d[j].i, n.i)};
      }
      /* y_k = m - i n, y_(P-k) = m + i n */
      v[k] = {m.r + n.i, m.i - n.r};
      v[P - k] = {m.r - n.i, m.i + n.r};
    }
  }
};
template <class S> struct dft<S, 3> : dft_odd<S, 3> {};
template <class S> struct dft<S, 5> : dft_odd<S, 5> {};
template <class S> struct dft<S, 7> : dft_odd<S, 7> {};

/* Memory */

/* Points a work-item holds in a pass, and LDS's padding: one slot after
   every 2^PS. */
template <class S> struct prec;
template <> struct prec<float> {
  static constexpr int PM = 16, PS = 4;
};
template <> struct prec<double> {
  static constexpr int PM = 8, PS = 3;
};

template <int PS> DEVICE int padi(int x) { return x + (x >> PS); }

typedef float float2_ __attribute__((ext_vector_type(2)));
typedef double double2_ __attribute__((ext_vector_type(2)));

/* An operand's or a result's point. An array is promised its parts'
   alignment alone, and every point's alignment is its first byte's, the
   side's address's: a side on twice a part's alignment takes one access a
   point, any other two. The branch is uniform across a launch. */
template <class S> DEVICE bool paired(const void *p) {
  return ((uintptr_t)p & (2 * sizeof(S) - 1)) == 0;
}
DEVICE cx<float> gld(const cx<float> *p, bool pair) {
  if (pair) {
    const float2_ v = *(const float2_ *)p;
    return {v.x, v.y};
  }
  return {p->r, p->i};
}
DEVICE cx<double> gld(const cx<double> *p, bool pair) {
  if (pair) {
    const double2_ v = *(const double2_ *)p;
    return {v.x, v.y};
  }
  return {p->r, p->i};
}
DEVICE void gst(cx<float> *p, cx<float> v, bool pair) {
  if (pair)
    *(float2_ *)p = float2_{v.r, v.i};
  else {
    p->r = v.r;
    p->i = v.i;
  }
}
DEVICE void gst(cx<double> *p, cx<double> v, bool pair) {
  if (pair)
    *(double2_ *)p = double2_{v.r, v.i};
  else {
    p->r = v.r;
    p->i = v.i;
  }
}

/* The layouts of a launch's sides, the source's from word [in] of the
   copied words [p] and the destination's after it, of one rank. */
struct sides {
  const int64_t *in, *out;
};

DEVICE sides sides_of(const int64_t *p, int in) {
  const int out =
      in + NX_LAYOUT_AXES + NX_LAYOUT_AXIS * wi(p, in + NX_LAYOUT_RANK);
  return {p + in, p + out};
}

/* The plan's table [off] bytes from its first. */
template <class T> DEVICE const T *table(const void *plan, int64_t off) {
  return (const T *)((const char *)plan + off);
}

/* w_L^m from the inner table: the whole table, or its first half and
   w_L^(m + L/2) = -w_L^m, exactly. */
template <class S>
DEVICE cx<S> twq(const cx<S> *t, int m, int L, int half) {
  int h2 = half ? L >> 1 : L;
  bool up = m >= h2;
  cx<S> w = t[up ? m - h2 : m];
  if (up) w = {-w.r, -w.i};
  return w;
}

/* w_N^m = lo[m mod 2^bits] hi[m >> bits]: one product. */
template <class S>
DEVICE cx<S> tw2(const cx<S> *lo, const cx<S> *hi, int bits, int64_t m) {
  return cmul(lo[m & ((1LL << bits) - 1)], hi[m >> bits]);
}

/* The inner FFT */

/* A pass: radix r after ns points, q = l / r butterflies, twiddle step
   ts = l / (ns r), nsm ns's divisor word. */
struct pass {
  int r, ns, q, ts;
  uint64_t nsm;
};

/* Pass [i] of those whose words start at [first]. */
DEVICE pass pass_of(const int64_t *p, int first, int i) {
  const int at = first + i * NX_FFT_PASS_WORDS;
  return {wi(p, at + NX_FFT_PASS_R), wi(p, at + NX_FFT_PASS_NS),
          wi(p, at + NX_FFT_PASS_Q), wi(p, at + NX_FFT_PASS_TS),
          (uint64_t)wd(p, at + NX_FFT_PASS_NS_DIV)};
}

/* One Stockham pass of radix R over a line in LDS: butterfly
   b = j + Tl k reads points b + r Q, twiddles them by w_(Ns R)^((b mod Ns) r)
   and writes output r at (b / Ns) Ns R + b mod Ns + r Ns. */
template <class S, int R>
DEVICE void pass_in(cx<S> *v, const cx<S> *line, int j, int Tl,
                    const pass &p, const cx<S> *tw, int L, int half) {
  constexpr int PS = prec<S>::PS, K = prec<S>::PM / R;
  const int Q = p.q, Ns = p.ns;
#pragma unroll
  for (int k = 0; k < K; k++) {
    int b = j + Tl * k;
    if (b < Q) {
#pragma unroll
      for (int r = 0; r < R; r++) v[k * R + r] = line[padi<PS>(b + r * Q)];
      if (Ns > 1) {
        int m = (b - (int)divide((uint32_t)b, p.nsm) * Ns) * p.ts;
#pragma unroll
        for (int r = 1; r < R; r++)
          v[k * R + r] = cmul(v[k * R + r], twq(tw, m * r, L, half));
      }
      dft<S, R>::run(v + k * R);
    }
  }
}

template <class S, int R>
DEVICE void pass_out(const cx<S> *v, cx<S> *line, int j, int Tl,
                     const pass &p) {
  constexpr int PS = prec<S>::PS, K = prec<S>::PM / R;
  const int Q = p.q, Ns = p.ns;
#pragma unroll
  for (int k = 0; k < K; k++) {
    int b = j + Tl * k;
    if (b < Q) {
      int base = (int)divide((uint32_t)b, p.nsm) * Ns * (R - 1) + b;
#pragma unroll
      for (int r = 0; r < R; r++) line[padi<PS>(base + r * Ns)] = v[k * R + r];
    }
  }
}

#define NX_FFT_RADICES(X) X(2) X(3) X(4) X(5) X(7) X(8)

template <class S>
DEVICE void run_fft(cx<S> *line, int j, const int64_t *P, const void *plan) {
  const cx<S> *tw = table<cx<S>>(plan, wd(P, NX_FFT_TW));
  const int np = wi(P, NX_FFT_NP), tl = wi(P, NX_FFT_TL), l = wi(P, NX_FFT_L),
            half = wi(P, NX_FFT_HALF), first = NX_FFT_OP + wi(P, NX_FFT_NOPS);
  cx<S> v[prec<S>::PM];
  for (int q = 0; q < np; q++) {
    const pass p = pass_of(P, first, q);
    switch (p.r) {
#define NX_FFT_IN(R)                                                           \
  case R: pass_in<S, R>(v, line, j, tl, p, tw, l, half); break;
      NX_FFT_RADICES(NX_FFT_IN)
#undef NX_FFT_IN
    default:
      if constexpr (prec<S>::PM >= 16)
        pass_in<S, 16>(v, line, j, tl, p, tw, l, half);
      break;
    }
    barrier();
    switch (p.r) {
#define NX_FFT_OUT(R)                                                          \
  case R: pass_out<S, R>(v, line, j, tl, p); break;
      NX_FFT_RADICES(NX_FFT_OUT)
#undef NX_FFT_OUT
    default:
      if constexpr (prec<S>::PM >= 16) pass_out<S, 16>(v, line, j, tl, p);
      break;
    }
    barrier();
  }
}

#undef NX_FFT_RADICES

/* Real lines */

/* A real line's bins X_k, X_(h-k) from its half-length transform's
   Z_k, Z_(h-k) (Z_h = Z_0), w = w_2h^k:
   X_k = (E + U, E.i + V) / 2, X_(h-k) = (E.r - U, V - E.i) / 2, where
   E = Z_k + conj Z_(h-k), D = Z_k - conj Z_(h-k), U = c D.i + s D.r,
   V = s D.i - c D.r. */
template <class S>
DEVICE void split_pair(cx<S> a, cx<S> b, cx<S> w, cx<S> &xk, cx<S> &xh) {
  cx<S> bc = cconj(b);
  cx<S> E = cadd(a, bc), D = csub(a, bc);
  S U = fmaS(w.r, D.i, w.i * D.r), V = fmaS(-w.r, D.r, w.i * D.i);
  const S half = S(0.5);
  xk = {half * (E.r + U), half * (E.i + V)};
  xh = {half * (E.r - U), half * (V - E.i)};
}

/* The half-length points of a real line from its bins X_k, X_(h-k), the
   imaginary parts of X_0 and X_h dropped by the caller, w = w_2h^k:
   Z_k = (E.r - U, E.i + V), Z_(h-k) = (E.r + U, V - E.i), where
   E = X_k + conj X_(h-k), D = X_k - conj X_(h-k), with w' = conj w,
   U = c D.i + s' D.r, V = c D.r - s' D.i. Returns them conjugated. */
template <class S>
DEVICE void hpre_pair(cx<S> a, cx<S> b, cx<S> w, cx<S> &zk, cx<S> &zh) {
  cx<S> bc = cconj(b);
  cx<S> E = cadd(a, bc), D = csub(a, bc);
  S sp = -w.i;
  S U = fmaS(w.r, D.i, sp * D.r), V = fmaS(w.r, D.r, -(sp * D.i));
  zk = {E.r - U, -(E.i + V)};
  zh = {E.r + U, -(V - E.i)};
}

/* A step */

/* Point r of line i of the block, the operand's point j = b + r M, its
   side's point step [ps]. The words are read from [P] where used: held in
   registers across the block loop, they spill. */
template <class S>
DEVICE cx<S> fetch(const void *src, bool pin, int64_t ps, const void *plan,
                   const int64_t *P, const int64_t *s_ain, const int64_t *s_b,
                   const int64_t *s_bm, int i, int r) {
  const int64_t b = s_b[i], j = b + (int64_t)r * wd(P, NX_FFT_M), a = s_ain[i];
  const int ld = wi(P, NX_FFT_LD);
  cx<S> x = {S(0), S(0)};
  if (b < 0 || j >= wd(P, NX_FFT_NSRC)) return x;
  if (ld == NX_FFT_LD_C || ld == NX_FFT_LD_HC) {
    /* LD_HC completes the bins: X_j = conj X_(n-j) above n / 2. */
    const int64_t nh = wd(P, NX_FFT_NH);
    const bool up = ld == NX_FFT_LD_HC && j > nh / 2;
    x = gld((const cx<S> *)src + a + (up ? nh - j : j) * ps, pin);
    if (up) x.i = -x.i;
    if (ld == NX_FFT_LD_HC && j == 0) x.i = S(0);
  } else {
    const S *xr = (const S *)src + a;
    const int64_t m = ld == NX_FFT_LD_RP ? 2 * j : j;
    x.r = xr[m * ps];
    if (ld == NX_FFT_LD_RP) x.i = xr[(m + 1) * ps];
  }
  if (wi(P, NX_FFT_CONJ_IN)) x.i = -x.i;
  if (wi(P, NX_FFT_CHIRP_IN))
    x = cmul(x, table<cx<S>>(plan, wd(P, NX_FFT_CHIRP))[j]);
  const int64_t gts = wd(P, NX_FFT_GTS);
  if (gts)
    x = cmul(x, tw2<S>(table<cx<S>>(plan, wd(P, NX_FFT_GLO)),
                       table<cx<S>>(plan, wd(P, NX_FFT_GHI)),
                       wi(P, NX_FFT_G_BITS), s_bm[i] * r * gts));
  return x;
}

/* Stores output k of line i of the block at the result's point
   (b / Ns) Ns L + b mod Ns + k Ns. */
template <class S>
DEVICE void put(void *dst, bool pout, int64_t ps, const void *plan,
                const int64_t *P, const cx<S> *sh, int lp,
                const int64_t *s_aout, const int64_t *s_b,
                const int64_t *s_bm, const int64_t *s_bd, int i, int k) {
  constexpr int PS = prec<S>::PS;
  const int64_t b = s_b[i];
  if (b < 0) return;
  const int64_t ns = wd(P, NX_FFT_NS);
  const int l = wi(P, NX_FFT_L);
  const int64_t ko = s_bd[i] * ns * l + s_bm[i] + (int64_t)k * ns,
                a = s_aout[i];
  if (ko >= wd(P, NX_FFT_LIMIT)) return;
  const cx<S> *ln = sh + i * lp;
  cx<S> x;
  if (!wi(P, NX_FFT_RADER))
    x = ln[padi<PS>(k)];
  else {
    /* y_k = x_0 + c_q, k = g^(-q); y_0 from slot L + 1. */
    const int32_t *rlog = table<int32_t>(plan, wd(P, NX_FFT_RLOG));
    const int q = k ? l - rlog[k] : 0;
    x = k ? cadd(cconj(ln[padi<PS>(q == l ? 0 : q)]), ln[padi<PS>(l)])
          : ln[padi<PS>(l + 1)];
  }
  const int post = wi(P, NX_FFT_POST);
  if (post == NX_FFT_POST_BMUL)
    x = cconj(cmul(x, table<cx<S>>(plan, wd(P, NX_FFT_BSPEC))[ko]));
  else if (post == NX_FFT_POST_BFIN)
    x = cmul(cconj(x), table<cx<S>>(plan, wd(P, NX_FFT_CHIRP))[ko]);
  if (wi(P, NX_FFT_CONJ_OUT)) x.i = -x.i;
  const int st = wi(P, NX_FFT_ST);
  if (st == NX_FFT_ST_C) {
    gst((cx<S> *)dst + a + ko * ps, x, pout);
  } else {
    S *yr = (S *)dst + a;
    const int64_t m = st == NX_FFT_ST_RP ? 2 * ko : ko;
    yr[m * ps] = x.r;
    if (st == NX_FFT_ST_RP) yr[(m + 1) * ps] = x.i;
  }
}

/* The ops in LDS on line [line], [jt] the work-item's place among its
   line's [tl]. */
template <class S>
DEVICE void ops(cx<S> *line, int jt, const int64_t *P, const void *plan) {
  constexpr int PS = prec<S>::PS;
  const int nops = wi(P, NX_FFT_NOPS), tl = wi(P, NX_FFT_TL),
            l = wi(P, NX_FFT_L), nh = (int)wd(P, NX_FFT_NH);
  for (int o = 0; o < nops; o++) {
    switch (wi(P, NX_FFT_OP + o)) {
    case NX_FFT_OP_FFT: run_fft<S>(line, jt, P, plan); break;
    case NX_FFT_OP_CHIRP: {
      const cx<S> *chirp = table<cx<S>>(plan, wd(P, NX_FFT_CHIRP));
      for (int k = jt; k < l; k += tl) {
        cx<S> x = {S(0), S(0)};
        if (k < nh) x = cmul(line[padi<PS>(k)], chirp[k]);
        line[padi<PS>(k)] = x;
      }
      barrier();
    } break;
    case NX_FFT_OP_BMUL: {
      const cx<S> *bspec = table<cx<S>>(plan, wd(P, NX_FFT_BSPEC));
      for (int k = jt; k < l; k += tl)
        line[padi<PS>(k)] = cconj(cmul(line[padi<PS>(k)], bspec[k]));
      barrier();
    } break;
    case NX_FFT_OP_BFIN: {
      const cx<S> *chirp = table<cx<S>>(plan, wd(P, NX_FFT_CHIRP));
      for (int k = jt; k < nh; k += tl)
        line[padi<PS>(k)] = cmul(cconj(line[padi<PS>(k)]), chirp[k]);
      barrier();
    } break;
    case NX_FFT_OP_RMUL: {
      /* Rader: y_0 = A_0 + x_0 into slot L + 1, then conj(A B'). */
      const cx<S> *bspec = table<cx<S>>(plan, wd(P, NX_FFT_BSPEC));
      for (int k = jt; k < l; k += tl) {
        cx<S> a = line[padi<PS>(k)];
        if (k == 0) line[padi<PS>(l + 1)] = cadd(a, line[padi<PS>(l)]);
        line[padi<PS>(k)] = cconj(cmul(a, bspec[k]));
      }
      barrier();
    } break;
    case NX_FFT_OP_SPLIT: {
      const cx<S> *rlo = table<cx<S>>(plan, wd(P, NX_FFT_RLO)),
                  *rhi = table<cx<S>>(plan, wd(P, NX_FFT_RHI));
      const int bits = wi(P, NX_FFT_R_BITS);
      for (int k = jt; k <= nh / 2; k += tl) {
        int k2 = k ? nh - k : 0;
        cx<S> xk, xh, w = tw2<S>(rlo, rhi, bits, k);
        split_pair(line[padi<PS>(k)], line[padi<PS>(k2)], w, xk, xh);
        line[padi<PS>(k)] = xk;
        line[padi<PS>(nh - k)] = xh;
      }
      barrier();
    } break;
    case NX_FFT_OP_HPRE: {
      const cx<S> *rlo = table<cx<S>>(plan, wd(P, NX_FFT_RLO)),
                  *rhi = table<cx<S>>(plan, wd(P, NX_FFT_RHI));
      const int bits = wi(P, NX_FFT_R_BITS);
      for (int k = jt; k <= nh / 2; k += tl) {
        cx<S> a = line[padi<PS>(k)], b = line[padi<PS>(nh - k)], zk, zh;
        if (k == 0) a.i = b.i = S(0);
        hpre_pair(a, b, tw2<S>(rlo, rhi, bits, k), zk, zh);
        line[padi<PS>(k)] = zk;
        if (k && 2 * k != nh) line[padi<PS>(nh - k)] = zh;
      }
      barrier();
    } break;
    }
  }
}

template <class S> DEVICE void step(void) {
  constexpr int PS = prec<S>::PS;
  static SHARED int64_t P[NX_FFT_WORDS];
  extern SHARED __attribute__((aligned(16))) unsigned char smem[];
  stage(P, (int)words()[NX_FFT_USED], NX_FFT_THREADS);

  const int G = wi(P, NX_FFT_G), Tl = wi(P, NX_FFT_TL), T = G * Tl;
  const int t = (int)item();
  if (t >= T) return;
  const int it = t / Tl, jt = t - it * Tl, lp = wi(P, NX_FFT_LP),
            gshift = wi(P, NX_FFT_GSHIFT);
  /* Each line's first element on each side and its butterfly b, b mod Ns
     and b / Ns, then the lines' points. */
  int64_t *s_ain = (int64_t *)smem, *s_aout = s_ain + G, *s_b = s_aout + G,
          *s_bm = s_b + G, *s_bd = s_bm + G;
  cx<S> *sh = (cx<S> *)(smem + ((5 * G * 8 + 15) & ~15));
  cx<S> *line = sh + it * lp;
  const void *src = address<const void>(NX_FFT_SRC);
  void *dst = address<void>(NX_FFT_DST);
  const void *plan = address<const void>(NX_FFT_PLAN);
  const int64_t nv = wd(P, NX_FFT_NV);
  const int64_t nblk = (nv + G - 1) >> gshift;
  const int64_t groups = wd(P, NX_FFT_GROUPS);
  const sides sd = sides_of(P, NX_FFT_OP + wi(P, NX_FFT_NOPS) +
                                   wi(P, NX_FFT_NP) * NX_FFT_PASS_WORDS);
  const bool pin = paired<S>(src), pout = paired<S>(dst);

  for (int64_t blk = group_x(); blk < nblk; blk += groups) {
    if (t < G) {
      const int64_t v = blk * G + t;
      if (v < nv) {
        const bool small = wd(P, NX_FFT_SMALL);
        int64_t l = v;
        const int64_t b =
            split(l, wd(P, NX_FFT_M), (uint64_t)wd(P, NX_FFT_M_DIV), small);
        int64_t bd = b;
        const int64_t bm =
            split(bd, wd(P, NX_FFT_NS), (uint64_t)wd(P, NX_FFT_NS_DIV), small);
        s_ain[t] = position(sd.in, l);
        s_aout[t] = position(sd.out, l);
        s_b[t] = b;
        s_bd[t] = bd;
        s_bm[t] = bm;
      } else
        s_b[t] = -1;
    }
    barrier();

    {
      /* Lines side by side (lf) or each line's points in turn. Rader: point
         j of a prime line goes to slot log_g j, point 0 to L. */
      const bool lf = wi(P, NX_FFT_LF_LD);
      const int lld = wi(P, NX_FFT_LLD);
      const int ne = lf ? G * lld : lld, stride = lf ? T : Tl;
      const int32_t *rlog = table<int32_t>(plan, wd(P, NX_FFT_RLOG));
      const bool rader = wi(P, NX_FFT_RADER);
      const int l = wi(P, NX_FFT_L);
      const int64_t ps = wd(P, NX_FFT_PS_IN);
      for (int k = lf ? t : jt; k < ne; k += stride) {
        const int i = lf ? k & (G - 1) : it, r = lf ? k >> gshift : k;
        const int at = rader ? (r ? rlog[r] : l) : r;
        sh[i * lp + padi<PS>(at)] =
            fetch<S>(src, pin, ps, plan, P, s_ain, s_b, s_bm, i, r);
      }
    }
    barrier();

    ops<S>(line, jt, P, plan);

    {
      const bool lf = wi(P, NX_FFT_LF_ST);
      const int lst = wi(P, NX_FFT_LST);
      const int ne = lf ? G * lst : lst, stride = lf ? T : Tl;
      const int64_t ps = wd(P, NX_FFT_PS_OUT);
      for (int k = lf ? t : jt; k < ne; k += stride)
        put<S>(dst, pout, ps, plan, P, sh, lp, s_aout, s_b, s_bm, s_bd,
               lf ? k & (G - 1) : it, lf ? k >> gshift : k);
    }
    barrier();
  }
}

/* A long real line: split (r2c, in place on the destination) or the
   pre-pass (c2r, from the operand into the destination, h points a line,
   conjugated). Pair k < h/2 + 1 of line l. */
template <class S> DEVICE void real_step(void) {
  static SHARED int64_t P[NX_FFT_REAL_WORDS];
  stage(P, (int)words()[NX_FFT_REAL_USED], NX_FFT_REAL_THREADS);
  const sides sd = sides_of(P, NX_FFT_REAL_IN);
  const bool splits = wi(P, NX_FFT_REAL_SPLIT);
  const int64_t h = wd(P, NX_FFT_REAL_H), pairs = wd(P, NX_FFT_REAL_PAIRS),
                items = wd(P, NX_FFT_REAL_NL) * pairs;
  const uint64_t pairs_div = (uint64_t)wd(P, NX_FFT_REAL_PAIRS_DIV);
  const bool small = wd(P, NX_FFT_REAL_SMALL);
  const void *plan = address<const void>(NX_FFT_REAL_PLAN);
  const cx<S> *rlo = table<cx<S>>(plan, wd(P, NX_FFT_REAL_RLO)),
              *rhi = table<cx<S>>(plan, wd(P, NX_FFT_REAL_RHI));
  const int bits = wi(P, NX_FFT_REAL_R_BITS);
  const cx<S> *x = address<const cx<S>>(NX_FFT_REAL_SRC);
  cx<S> *y = address<cx<S>>(NX_FFT_REAL_DST);
  const bool pin = paired<S>(x), pout = paired<S>(y);
  const int64_t psi = wd(P, NX_FFT_REAL_PS_IN),
                pso = wd(P, NX_FFT_REAL_PS_OUT);
  const int64_t stride = wd(P, NX_FFT_REAL_GROUPS) * NX_FFT_REAL_THREADS;
  for (int64_t e = (int64_t)group_x() * NX_FFT_REAL_THREADS + item();
       e < items; e += stride) {
    int64_t l = e;
    const int64_t k = split(l, pairs, pairs_div, small);
    const cx<S> w = tw2<S>(rlo, rhi, bits, k);
    const int64_t o = position(sd.out, l);
    if (splits) {
      cx<S> xk, xh;
      split_pair(gld(y + o + k * pso, pout),
                 gld(y + o + (k ? h - k : 0) * pso, pout), w, xk, xh);
      gst(y + o + k * pso, xk, pout);
      gst(y + o + (h - k) * pso, xh, pout);
    } else {
      const int64_t a = position(sd.in, l);
      cx<S> xa = gld(x + a + k * psi, pin), xb = gld(x + a + (h - k) * psi, pin),
            zk, zh;
      if (k == 0) xa.i = xb.i = S(0);
      hpre_pair(xa, xb, w, zk, zh);
      gst(y + o + k * pso, zk, pout);
      if (k && 2 * k != h) gst(y + o + (h - k) * pso, zh, pout);
    }
  }
}

} // namespace nx_fft

KERNEL SIZED(NX_FFT_THREADS) void fft_f32(Words<NX_FFT_WORDS>) {
  nx_fft::step<float>();
}
KERNEL SIZED(NX_FFT_THREADS) void fft_f64(Words<NX_FFT_WORDS>) {
  nx_fft::step<double>();
}
KERNEL SIZED(NX_FFT_REAL_THREADS) void fft_real_f32(Words<NX_FFT_REAL_WORDS>) {
  nx_fft::real_step<float>();
}
KERNEL SIZED(NX_FFT_REAL_THREADS) void fft_real_f64(Words<NX_FFT_REAL_WORDS>) {
  nx_fft::real_step<double>();
}
