/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fourier transforms (kernels.h, Fourier transforms): fft runs every step
   of a transform, fft_real a long real line's split or pre-pass, one
   instance of each per width. Part of kernels.cu's translation unit.

   A launch of fft is one Stockham step: virtual line v is batch line
   l = v / M and butterfly b = v mod M; it loads the L points b + r M,
   twiddles point r by w_N^((b mod Ns) r N / (Ns L)), runs an L-point FFT
   in shared memory and stores output k at (b / Ns) Ns L + b mod Ns + k Ns.
   Outputs come in natural order, and a launch writes only points it read
   (the last step) or points of another buffer, so no group reads what
   another wrote.

   The inner FFT holds a thread's butterflies (at most 16 complex64 or 8
   complex128 points) in one register array that every radix shares, with
   the barrier between a pass's reads and writes in the dispatcher, so the
   radices take one register allocation. Shared memory is padded one slot
   in 16 (8), so strides of 2 to 16 points fall on distinct banks. Each
   product by an irrational constant fuses into the addition after it;
   fft.ml's comments derive each butterfly's error from these operations. */

#ifndef NX_CUDA_FFT_CU
#define NX_CUDA_FFT_CU

namespace nx_fft {

/* Complex arithmetic */

template <class S> struct cx {
  S r, i;
};

__device__ __forceinline__ float fmaS(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ double fmaS(double a, double b, double c) { return __fma_rn(a, b, c); }

template <class S> __device__ __forceinline__ cx<S> cadd(cx<S> a, cx<S> b) { return {a.r + b.r, a.i + b.i}; }
template <class S> __device__ __forceinline__ cx<S> csub(cx<S> a, cx<S> b) { return {a.r - b.r, a.i - b.i}; }
template <class S> __device__ __forceinline__ cx<S> cconj(cx<S> a) { return {a.r, -a.i}; }

/* a w: two products and two fused multiply-adds. */
template <class S> __device__ __forceinline__ cx<S> cmul(cx<S> a, cx<S> w) {
  return {fmaS(a.r, w.r, -(a.i * w.i)), fmaS(a.r, w.i, a.i * w.r)};
}

/* Butterflies: forward DFTs of 2, 3, 4, 5, 7, 8 and 16 points in natural
   order, in place. */

template <class S> __device__ __forceinline__ void dft4(cx<S> &a0, cx<S> &a1, cx<S> &a2, cx<S> &a3) {
  cx<S> t0 = cadd(a0, a2), t1 = csub(a0, a2), t2 = cadd(a1, a3), t3 = csub(a1, a3);
  a0 = cadd(t0, t2);
  a2 = csub(t0, t2);
  /* -i t3 = (t3.i, -t3.r) */
  a1 = {t1.r + t3.i, t1.i - t3.r};
  a3 = {t1.r - t3.i, t1.i + t3.r};
}

template <class S, int R> struct dft;

template <class S> struct dft<S, 2> {
  static __device__ __forceinline__ void run(cx<S> *v) {
    cx<S> a = v[0], b = v[1];
    v[0] = cadd(a, b);
    v[1] = csub(a, b);
  }
};

template <class S> struct dft<S, 4> {
  static __device__ __forceinline__ void run(cx<S> *v) { dft4(v[0], v[1], v[2], v[3]); }
};

/* DFT8: radix 2 across stride 4, then DFT4 of the sums and of the
   differences times w8^k, w8 = (1 - i)/sqrt 2, the products by 1/sqrt 2
   fused into the odd outputs. */
template <class S> struct dft<S, 8> {
  static __device__ __forceinline__ void run(cx<S> *v) {
    const S c = S(0.70710678118654752440);
    cx<S> b0 = cadd(v[0], v[4]), b1 = cadd(v[1], v[5]), b2 = cadd(v[2], v[6]), b3 = cadd(v[3], v[7]);
    cx<S> d0 = csub(v[0], v[4]), d1 = csub(v[1], v[5]), d2 = csub(v[2], v[6]), d3 = csub(v[3], v[7]);
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
  static __device__ __forceinline__ cx<S> w1(cx<S> x, S C, S Sn) {
    return {fmaS(x.r, C, x.i * Sn), fmaS(x.i, C, -(x.r * Sn))};
  }
  static __device__ __forceinline__ void run(cx<S> *v) {
    const S c = S(0.70710678118654752440);
    const S C = S(0.92387953251128675613), Sn = S(0.38268343236508977173);
#pragma unroll
    for (int n1 = 0; n1 < 4; n1++) dft4(v[n1], v[n1 + 4], v[n1 + 8], v[n1 + 12]);
    /* u[n1][k2] is at v[n1 + 4 k2] */
    v[5] = w1(v[5], C, Sn);                                        /* w^1 */
    v[9] = {c * (v[9].r + v[9].i), c * (v[9].i - v[9].r)};         /* w^2 */
    v[13] = w1(v[13], Sn, C);                                      /* w^3 = (S, -C) */
    v[6] = {c * (v[6].r + v[6].i), c * (v[6].i - v[6].r)};         /* w^2 */
    v[10] = {v[10].i, -v[10].r};                                   /* w^4 = -i */
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
      cx<S> a0 = u[4 * k2], a1 = u[4 * k2 + 1], a2 = u[4 * k2 + 2], a3 = u[4 * k2 + 3];
      dft4(a0, a1, a2, a3);
      v[k2] = a0;
      v[k2 + 4] = a1;
      v[k2 + 8] = a2;
      v[k2 + 12] = a3;
    }
  }
};

/* An odd prime P's constants cos(2 pi m / P) and sin(2 pi m / P). */
template <int P> struct oddc;
template <> struct oddc<3> {
  static __device__ __forceinline__ double c(int m) { return m == 0 ? 1.0 : -0.5; }
  static __device__ __forceinline__ double s(int m) {
    return m == 0 ? 0.0 : (m == 1 ? 0.86602540378443864676 : -0.86602540378443864676);
  }
};
template <> struct oddc<5> {
  static __device__ __forceinline__ double c(int m) {
    const double t[5] = {1.0, 0.30901699437494742410, -0.80901699437494742410, -0.80901699437494742410,
                         0.30901699437494742410};
    return t[m];
  }
  static __device__ __forceinline__ double s(int m) {
    const double t[5] = {0.0, 0.95105651629515357212, 0.58778525229247312917, -0.58778525229247312917,
                         -0.95105651629515357212};
    return t[m];
  }
};
template <> struct oddc<7> {
  static __device__ __forceinline__ double c(int m) {
    const double t[7] = {1.0,
                         0.62348980185873353053,
                         -0.22252093395631440429,
                         -0.90096886790241912624,
                         -0.90096886790241912624,
                         -0.22252093395631440429,
                         0.62348980185873353053};
    return t[m];
  }
  static __device__ __forceinline__ double s(int m) {
    const double t[7] = {0.0,
                         0.78183148246802980871,
                         0.97492791218182360702,
                         0.43388373911755812048,
                         -0.43388373911755812048,
                         -0.97492791218182360702,
                         -0.78183148246802980871};
    return t[m];
  }
};

/* An odd prime P: t_j = a_j + a_(P-j), d_j = a_j - a_(P-j);
   y_k = a0 + sum_j cos(2 pi jk/P) t_j - i sum_j sin(2 pi jk/P) d_j and
   y_(P-k) its conjugate pair. */
template <class S, int P> struct dft_odd {
  static __device__ __forceinline__ void run(cx<S> *v) {
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

/* Points a thread holds in a pass, and the shared-memory padding: one
   slot after every 2^PS. */
template <class S> struct prec;
template <> struct prec<float> {
  static constexpr int PM = 16, PS = 4;
};
template <> struct prec<double> {
  static constexpr int PM = 8, PS = 3;
};

template <int PS> __device__ __forceinline__ int padi(int x) { return x + (x >> PS); }

/* x / d for x, d < 2^16, m = floor(2^32 / d) + 1, or m = 0 for d = 1. */
__device__ __forceinline__ int fdiv(int x, uint32_t m) { return m ? (int)__umulhi((uint32_t)x, m) : x; }

/* Loads through the non-coherent path, so that they run ahead of shared
   memory's writes, and stores past L1. */
__device__ __forceinline__ cx<float> ldg(const cx<float> *p) {
  float2 v = __ldg((const float2 *)p);
  return {v.x, v.y};
}
__device__ __forceinline__ cx<double> ldg(const cx<double> *p) {
  double2 v = __ldg((const double2 *)p);
  return {v.x, v.y};
}
__device__ __forceinline__ float ldgr(const float *p) { return __ldg(p); }
__device__ __forceinline__ double ldgr(const double *p) { return __ldg(p); }

/* An operand's or a result's point. An array is promised its parts'
   alignment alone, and every point's alignment is its first byte's, the
   side's pointer's: a side on twice a part's alignment takes one access a
   point, any other two. The branch is uniform across a launch. */
template <class S> __device__ __forceinline__ bool paired(const void *p) {
  return ((uintptr_t)p & (2 * sizeof(S) - 1)) == 0;
}
template <class S> __device__ __forceinline__ cx<S> gld(const cx<S> *p, bool pair) {
  if (pair) return ldg(p);
  return {__ldg(&p->r), __ldg(&p->i)};
}
/* The same through the coherent path, for a launch that reads points it
   then writes. */
__device__ __forceinline__ cx<float> pld(const cx<float> *p, bool pair) {
  if (pair) {
    float2 v = *(const float2 *)p;
    return {v.x, v.y};
  }
  return {p->r, p->i};
}
__device__ __forceinline__ cx<double> pld(const cx<double> *p, bool pair) {
  if (pair) {
    double2 v = *(const double2 *)p;
    return {v.x, v.y};
  }
  return {p->r, p->i};
}
__device__ __forceinline__ void gst(cx<float> *p, cx<float> v, bool pair) {
  if (pair)
    __stcg((float2 *)p, make_float2(v.r, v.i));
  else {
    __stcg(&p->r, v.r);
    __stcg(&p->i, v.i);
  }
}
__device__ __forceinline__ void gst(cx<double> *p, cx<double> v, bool pair) {
  if (pair)
    __stcg((double2 *)p, make_double2(v.r, v.i));
  else {
    __stcg(&p->r, v.r);
    __stcg(&p->i, v.i);
  }
}
__device__ __forceinline__ void gstr(float *p, float v) { __stcg(p, v); }
__device__ __forceinline__ void gstr(double *p, double v) { __stcg(p, v); }

/* a / b for a, b >= 0, out of line: one copy of the 64-bit division. */
__device__ __noinline__ int64_t div64(int64_t a, int64_t b) { return (int64_t)((uint64_t)a / (uint64_t)b); }

/* Line l's first element on side s. */
__device__ __forceinline__ int64_t side_at(const fft_side &s, int64_t l) {
  int64_t at = s.off;
  for (int d = s.rank - 1; d >= 0; d--) {
    int64_t q = div64(l, s.dim[d]);
    at += (l - q * s.dim[d]) * s.str[d];
    l = q;
  }
  return at;
}

/* The plan's table [off] bytes from its first. */
template <class T> __device__ __forceinline__ const T *table(const void *plan, int64_t off) {
  return (const T *)((const char *)plan + off);
}

/* w_L^m from the inner table: the whole table, or its first half and
   w_L^(m + L/2) = -w_L^m, exactly. */
template <class S> __device__ __forceinline__ cx<S> twq(const cx<S> *t, int m, int L, int half) {
  int h2 = half ? L >> 1 : L;
  bool up = m >= h2;
  cx<S> w = ldg(t + (up ? m - h2 : m));
  if (up) w = {-w.r, -w.i};
  return w;
}

/* w_N^m = lo[m mod 2^bits] hi[m >> bits]: one product. */
template <class S> __device__ __forceinline__ cx<S> tw2(const cx<S> *lo, const cx<S> *hi, int bits, int64_t m) {
  return cmul(ldg(lo + (m & ((1LL << bits) - 1))), ldg(hi + (m >> bits)));
}

/* The inner FFT */

/* One Stockham pass of radix R over a line in shared memory: butterfly
   b = j + Tl k reads points b + r Q, twiddles them by w_(Ns R)^((b mod Ns) r)
   and writes output r at (b / Ns) Ns R + b mod Ns + r Ns. */
template <class S, int R>
__device__ __forceinline__ void pass_in(cx<S> *v, const cx<S> *line, int j, int Tl, const fft_pass &p,
                                        const cx<S> *tw, int L, int half) {
  constexpr int PS = prec<S>::PS, K = prec<S>::PM / R;
  const int Q = p.q, Ns = p.ns;
#pragma unroll
  for (int k = 0; k < K; k++) {
    int b = j + Tl * k;
    if (b < Q) {
#pragma unroll
      for (int r = 0; r < R; r++) v[k * R + r] = line[padi<PS>(b + r * Q)];
      if (Ns > 1) {
        int m = (b - fdiv(b, p.nsm) * Ns) * p.ts;
#pragma unroll
        for (int r = 1; r < R; r++) v[k * R + r] = cmul(v[k * R + r], twq(tw, m * r, L, half));
      }
      dft<S, R>::run(v + k * R);
    }
  }
}

template <class S, int R>
__device__ __forceinline__ void pass_out(const cx<S> *v, cx<S> *line, int j, int Tl, const fft_pass &p) {
  constexpr int PS = prec<S>::PS, K = prec<S>::PM / R;
  const int Q = p.q, Ns = p.ns;
#pragma unroll
  for (int k = 0; k < K; k++) {
    int b = j + Tl * k;
    if (b < Q) {
      int base = fdiv(b, p.nsm) * Ns * (R - 1) + b;
#pragma unroll
      for (int r = 0; r < R; r++) line[padi<PS>(base + r * Ns)] = v[k * R + r];
    }
  }
}

#define NX_FFT_RADICES(X) X(2) X(3) X(4) X(5) X(7) X(8)

template <class S> __device__ void run_fft(cx<S> *line, int j, const fft_params &P) {
  const cx<S> *tw = table<cx<S>>(P.plan, P.tw);
  cx<S> v[prec<S>::PM];
  for (int q = 0; q < P.np; q++) {
    const fft_pass &p = P.pass[q];
    switch (p.r) {
#define NX_FFT_IN(R) \
  case R: pass_in<S, R>(v, line, j, P.tl, p, tw, P.l, P.half); break;
      NX_FFT_RADICES(NX_FFT_IN)
#undef NX_FFT_IN
    default:
      if constexpr (prec<S>::PM >= 16) pass_in<S, 16>(v, line, j, P.tl, p, tw, P.l, P.half);
      break;
    }
    __syncthreads();
    switch (p.r) {
#define NX_FFT_OUT(R) \
  case R: pass_out<S, R>(v, line, j, P.tl, p); break;
      NX_FFT_RADICES(NX_FFT_OUT)
#undef NX_FFT_OUT
    default:
      if constexpr (prec<S>::PM >= 16) pass_out<S, 16>(v, line, j, P.tl, p);
      break;
    }
    __syncthreads();
  }
}

#undef NX_FFT_RADICES

/* Real lines */

/* A real line's bins X_k, X_(h-k) from its half-length transform's
   Z_k, Z_(h-k) (Z_h = Z_0), w = w_2h^k:
   X_k = (E + U, E.i + V) / 2, X_(h-k) = (E.r - U, V - E.i) / 2, where
   E = Z_k + conj Z_(h-k), D = Z_k - conj Z_(h-k), U = c D.i + s D.r,
   V = s D.i - c D.r. */
template <class S> __device__ __forceinline__ void split_pair(cx<S> a, cx<S> b, cx<S> w, cx<S> &xk, cx<S> &xh) {
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
template <class S> __device__ __forceinline__ void hpre_pair(cx<S> a, cx<S> b, cx<S> w, cx<S> &zk, cx<S> &zh) {
  cx<S> bc = cconj(b);
  cx<S> E = cadd(a, bc), D = csub(a, bc);
  S sp = -w.i;
  S U = fmaS(w.r, D.i, sp * D.r), V = fmaS(w.r, D.r, -(sp * D.i));
  zk = {E.r - U, -(E.i + V)};
  zh = {E.r + U, -(V - E.i)};
}

/* A step */

template <class S> __device__ __forceinline__ void fft_body(const fft_params &P) {
  constexpr int PS = prec<S>::PS;
  extern __shared__ __align__(16) unsigned char smem[];
  const int G = P.g, Tl = P.tl, T = G * Tl;
  /* Each line's first element on each side and its butterfly b, b mod Ns
     and b / Ns, then the lines' points. */
  int64_t *s_ain = (int64_t *)smem, *s_aout = s_ain + G, *s_b = s_aout + G, *s_bm = s_b + G, *s_bd = s_bm + G;
  cx<S> *sh = (cx<S> *)(smem + ((5 * G * 8 + 15) & ~15));
  const int t = threadIdx.x, it = t / Tl, jt = t - it * Tl;
  cx<S> *line = sh + it * P.lp;
  /* The plan's tables, each addressed where it is read, from the
     parameters: no register holds them across the step. */
#define NX_FFT_TAB(f) table<cx<S>>(P.plan, P.f)
#define NX_FFT_RLOG table<int32_t>(P.plan, P.rlog)
  const int64_t nblk = (P.nv + G - 1) >> P.gshift;
  const int64_t psi = P.in.ps, pso = P.out.ps;
  const bool pin = paired<S>(P.in.ptr), pout = paired<S>(P.out.ptr);

  for (int64_t blk = blockIdx.x; blk < nblk; blk += gridDim.x) {
    if (t < G) {
      int64_t v = blk * G + t;
      if (v < P.nv) {
        int64_t l = div64(v, P.m), b = v - l * P.m, bd = div64(b, P.ns);
        s_ain[t] = side_at(P.in, l);
        s_aout[t] = side_at(P.out, l);
        s_b[t] = b;
        s_bd[t] = bd;
        s_bm[t] = b - bd * P.ns;
      } else
        s_b[t] = -1;
    }
    __syncthreads();

    /* Load: point r of line i is the operand's point j = b + r M. */
    auto fetch = [&](int i, int r) -> cx<S> {
      int64_t b = s_b[i], j = b + (int64_t)r * P.m, a = s_ain[i];
      cx<S> x = {S(0), S(0)};
      if (b < 0 || j >= P.nsrc) return x;
      if (P.ld == NX_FFT_LD_C || P.ld == NX_FFT_LD_HC) {
        /* LD_HC completes the bins: X_j = conj X_(n-j) above n / 2. */
        bool up = P.ld == NX_FFT_LD_HC && j > P.nh / 2;
        x = gld((const cx<S> *)P.in.ptr + a + (up ? P.nh - j : j) * psi, pin);
        if (up) x.i = -x.i;
        if (P.ld == NX_FFT_LD_HC && j == 0) x.i = S(0);
      } else {
        const S *xr = (const S *)P.in.ptr + a;
        int64_t m = P.ld == NX_FFT_LD_RP ? 2 * j : j;
        x.r = ldgr(xr + m * psi);
        if (P.ld == NX_FFT_LD_RP) x.i = ldgr(xr + (m + 1) * psi);
      }
      if (P.conj_in) x.i = -x.i;
      if (P.chirp_in) x = cmul(x, ldg(NX_FFT_TAB(chirp) + j));
      if (P.gts) x = cmul(x, tw2<S>(NX_FFT_TAB(glo), NX_FFT_TAB(ghi), P.g_bits, s_bm[i] * r * P.gts));
      return x;
    };
    {
      /* Lines side by side (lf) or each line's points in turn. */
      const bool lf = P.lf_ld;
      const int ne = lf ? G * P.lld : P.lld, step = lf ? T : Tl;
#pragma unroll 8
      for (int e = lf ? t : jt; e < ne; e += step) {
        int i = lf ? e & (G - 1) : it, r = lf ? e >> P.gshift : e;
        /* Rader: point j of a prime line goes to slot log_g j, point 0 to L. */
        int at = P.rader ? (r ? __ldg(NX_FFT_RLOG + r) : P.l) : r;
        sh[i * P.lp + padi<PS>(at)] = fetch(i, r);
      }
    }
    __syncthreads();

    for (int o = 0; o < P.nops; o++) {
      switch (P.ops[o]) {
      case NX_FFT_OP_FFT: run_fft<S>(line, jt, P); break;
      case NX_FFT_OP_CHIRP:
        for (int k = jt; k < P.l; k += Tl) {
          cx<S> x = {S(0), S(0)};
          if (k < P.nh) x = cmul(line[padi<PS>(k)], ldg(NX_FFT_TAB(chirp) + k));
          line[padi<PS>(k)] = x;
        }
        __syncthreads();
        break;
      case NX_FFT_OP_BMUL:
        for (int k = jt; k < P.l; k += Tl) line[padi<PS>(k)] = cconj(cmul(line[padi<PS>(k)], ldg(NX_FFT_TAB(bspec) + k)));
        __syncthreads();
        break;
      case NX_FFT_OP_BFIN:
        for (int k = jt; k < P.nh; k += Tl) line[padi<PS>(k)] = cmul(cconj(line[padi<PS>(k)]), ldg(NX_FFT_TAB(chirp) + k));
        __syncthreads();
        break;
      case NX_FFT_OP_RMUL:
        /* Rader: y_0 = A_0 + x_0 into slot L + 1, then conj(A B'). */
        for (int k = jt; k < P.l; k += Tl) {
          cx<S> a = line[padi<PS>(k)];
          if (k == 0) line[padi<PS>(P.l + 1)] = cadd(a, line[padi<PS>(P.l)]);
          line[padi<PS>(k)] = cconj(cmul(a, ldg(NX_FFT_TAB(bspec) + k)));
        }
        __syncthreads();
        break;
      case NX_FFT_OP_SPLIT: {
        int h = (int)P.nh;
        for (int k = jt; k <= h / 2; k += Tl) {
          int k2 = k ? h - k : 0;
          cx<S> xk, xh, w = tw2<S>(NX_FFT_TAB(rlo), NX_FFT_TAB(rhi), P.r_bits, k);
          split_pair(line[padi<PS>(k)], line[padi<PS>(k2)], w, xk, xh);
          line[padi<PS>(k)] = xk;
          line[padi<PS>(h - k)] = xh;
        }
        __syncthreads();
      } break;
      case NX_FFT_OP_HPRE: {
        int h = (int)P.nh;
        for (int k = jt; k <= h / 2; k += Tl) {
          cx<S> a = line[padi<PS>(k)], b = line[padi<PS>(h - k)], zk, zh;
          if (k == 0) a.i = b.i = S(0);
          hpre_pair(a, b, tw2<S>(NX_FFT_TAB(rlo), NX_FFT_TAB(rhi), P.r_bits, k), zk, zh);
          line[padi<PS>(k)] = zk;
          if (k && 2 * k != h) line[padi<PS>(h - k)] = zh;
        }
        __syncthreads();
      } break;
      }
    }

    /* Store: output k of line i is the result's point
       (b / Ns) Ns L + b mod Ns + k Ns. */
    auto put = [&](int i, int k) {
      int64_t b = s_b[i];
      if (b < 0) return;
      int64_t ko = s_bd[i] * P.ns * P.l + s_bm[i] + (int64_t)k * P.ns, a = s_aout[i];
      if (ko >= P.limit) return;
      cx<S> x;
      if (!P.rader)
        x = sh[i * P.lp + padi<PS>(k)];
      else {
        /* y_k = x_0 + c_q, k = g^(-q); y_0 from slot L + 1. */
        const cx<S> *ln = sh + i * P.lp;
        int q = k ? P.l - __ldg(NX_FFT_RLOG + k) : 0;
        x = k ? cadd(cconj(ln[padi<PS>(q == P.l ? 0 : q)]), ln[padi<PS>(P.l)]) : ln[padi<PS>(P.l + 1)];
      }
      if (P.post == NX_FFT_POST_BMUL)
        x = cconj(cmul(x, ldg(NX_FFT_TAB(bspec) + ko)));
      else if (P.post == NX_FFT_POST_BFIN)
        x = cmul(cconj(x), ldg(NX_FFT_TAB(chirp) + ko));
      if (P.conj_out) x.i = -x.i;
      if (P.st == NX_FFT_ST_C) {
        gst((cx<S> *)P.out.ptr + a + ko * pso, x, pout);
      } else {
        S *yr = (S *)P.out.ptr + a;
        int64_t m = P.st == NX_FFT_ST_RP ? 2 * ko : ko;
        gstr(yr + m * pso, x.r);
        if (P.st == NX_FFT_ST_RP) gstr(yr + (m + 1) * pso, x.i);
      }
    };
    {
      const bool lf = P.lf_st;
      const int ne = lf ? G * P.lst : P.lst, step = lf ? T : Tl;
#pragma unroll 4
      for (int e = lf ? t : jt; e < ne; e += step) put(lf ? e & (G - 1) : it, lf ? e >> P.gshift : e);
    }
    __syncthreads();
  }
}

#undef NX_FFT_TAB
#undef NX_FFT_RLOG

/* A long real line: split (r2c, in place on out) or the pre-pass (c2r,
   from in into out, h points a line, conjugated). Pair k < h/2 + 1 of
   line l. */
template <class S> __device__ __forceinline__ void real_body(const fft_real_params &P) {
  const bool split = P.split;
  const int64_t pairs = P.h / 2 + 1, items = P.nl * pairs, h = P.h;
  const cx<S> *rlo = table<cx<S>>(P.plan, P.rlo), *rhi = table<cx<S>>(P.plan, P.rhi);
  const bool pin = paired<S>(P.in.ptr), pout = paired<S>(P.out.ptr);
  cx<S> *y = (cx<S> *)P.out.ptr;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < items;
       e += (int64_t)gridDim.x * blockDim.x) {
    int64_t l = div64(e, pairs), k = e - l * pairs;
    cx<S> w = tw2<S>(rlo, rhi, P.r_bits, k);
    if (split) {
      int64_t a = side_at(P.out, l), ps = P.out.ps;
      cx<S> xk, xh;
      split_pair(pld(y + a + k * ps, pout), pld(y + a + (k ? h - k : 0) * ps, pout), w, xk, xh);
      gst(y + a + k * ps, xk, pout);
      gst(y + a + (h - k) * ps, xh, pout);
    } else {
      const cx<S> *x = (const cx<S> *)P.in.ptr;
      int64_t a = side_at(P.in, l), ps = P.in.ps, o = side_at(P.out, l);
      cx<S> xa = gld(x + a + k * ps, pin), xb = gld(x + a + (h - k) * ps, pin), zk, zh;
      if (k == 0) xa.i = xb.i = S(0);
      hpre_pair(xa, xb, w, zk, zh);
      gst(y + o + k * P.out.ps, zk, pout);
      if (k && 2 * k != h) gst(y + o + (h - k) * P.out.ps, zh, pout);
    }
  }
}

} // namespace nx_fft

/* Each family's definition of its rows of NX_CUDA_KERNELS. A step's group
   has at most NX_FFT_THREADS threads, and an SM holds one at least. */
#define FFT_TYPE_f32 float
#define FFT_TYPE_f64 double
#define FFT(name, p)                                                           \
  extern "C" __global__ void __launch_bounds__(NX_FFT_THREADS, 1)              \
      name(const __grid_constant__ fft_params P) {                             \
    nx_fft::fft_body<FFT_TYPE_##p>(P);                                         \
  }
#define FFT_REAL(name, p)                                                      \
  extern "C" __global__ void name(const __grid_constant__ fft_real_params P) { \
    nx_fft::real_body<FFT_TYPE_##p>(P);                                        \
  }

#endif
