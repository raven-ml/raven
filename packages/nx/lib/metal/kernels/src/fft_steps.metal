/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Specialised steps (kernels.h, A specialised step): one body for every
   step of up to 4096 points, its length, radices, threads and lines a
   group function constants, so that each key's pipeline folds every pass's
   radix, butterfly count, twiddle stride and plane slot to constants.

   A thread keeps its points in registers between passes, [v], indexed by
   constants alone once the loops unroll; a pass exchanges them through its
   line's complex plane, one write and one read, or, for lines side by
   side, through the group's float plane, a part at a time. Points a
   thread is an array's size, which no constant sizes, so each count is a
   function of its own. What its sides hold, a real line's floats or bins
   and a halved line's split or pre-pass, and its line's convolution,
   Rader's or Bluestein's, are constants too. */

#include "fft.h"

constant uint fc_l [[function_constant(NX_FFT_FC_L)]];
constant ushort fc_tl [[function_constant(NX_FFT_FC_TL)]];
constant ushort fc_lines [[function_constant(NX_FFT_FC_LINES)]];
constant uchar fc_passes [[function_constant(NX_FFT_FC_PASSES)]];
constant uchar fc_r0 [[function_constant(NX_FFT_FC_R0)]];
constant uchar fc_r1 [[function_constant(NX_FFT_FC_R0 + 1)]];
constant uchar fc_r2 [[function_constant(NX_FFT_FC_R0 + 2)]];
constant uchar fc_r3 [[function_constant(NX_FFT_FC_R0 + 3)]];
constant uchar fc_r4 [[function_constant(NX_FFT_FC_R0 + 4)]];
constant uchar fc_r5 [[function_constant(NX_FFT_FC_R0 + 5)]];
constant uchar fc_r6 [[function_constant(NX_FFT_FC_R0 + 6)]];
constant uchar fc_halved [[function_constant(NX_FFT_FC_HALVED)]];
constant uchar fc_input [[function_constant(NX_FFT_FC_INPUT)]];
constant uchar fc_output [[function_constant(NX_FFT_FC_OUTPUT)]];
constant uchar fc_conv [[function_constant(NX_FFT_FC_CONV)]];

static_assert(NX_FFT_FC_RADICES == 7, "a constant for each radix");

struct fft_key_words {
  device uchar *src;
  device uchar *dst;
  constant float2 *lo;
  constant float2 *hi;
  constant long *tail;
  ulong first;
  ulong mode;
  ulong lines;
  constant float2 *kern;
  constant float2 *chirp;
  constant int *logs;
  ulong n;
  fft_side_words in;
  fft_side_words out;
};

NX_FIELD(fft_key_words, src, NX_FFT_KEY_SRC);
NX_FIELD(fft_key_words, dst, NX_FFT_KEY_DST);
NX_FIELD(fft_key_words, lo, NX_FFT_KEY_LO);
NX_FIELD(fft_key_words, hi, NX_FFT_KEY_HI);
NX_FIELD(fft_key_words, tail, NX_FFT_KEY_TAIL);
NX_FIELD(fft_key_words, first, NX_FFT_KEY_FIRST);
NX_FIELD(fft_key_words, mode, NX_FFT_KEY_MODE);
NX_FIELD(fft_key_words, lines, NX_FFT_KEY_LINES);
NX_FIELD(fft_key_words, kern, NX_FFT_KEY_KERNEL);
NX_FIELD(fft_key_words, chirp, NX_FFT_KEY_CHIRP);
NX_FIELD(fft_key_words, logs, NX_FFT_KEY_LOGS);
NX_FIELD(fft_key_words, n, NX_FFT_KEY_N);
NX_FIELD(fft_key_words, in, NX_FFT_KEY_IN);
NX_FIELD(fft_key_words, out, NX_FFT_KEY_OUT);
NX_SIZE(fft_key_words, NX_FFT_KEY_WORDS);

/* fft.ml sets a key's buffers, tables and tail at a transpose's words. */
static_assert(NX_FFT_KEY_SRC == NX_FFT_MOVE_SRC &&
                  NX_FFT_KEY_DST == NX_FFT_MOVE_DST &&
                  NX_FFT_KEY_LO == NX_FFT_MOVE_LO &&
                  NX_FFT_KEY_HI == NX_FFT_MOVE_HI &&
                  NX_FFT_KEY_TAIL == NX_FFT_MOVE_TAIL,
              "a key's words as a transpose's");

/* Butterflies of 2, 8 and the odd primes, fft.hip's forms */

#define NX_FFT_ALWAYS inline __attribute__((always_inline))
#define NX_FFT_INLINE static NX_FFT_ALWAYS

NX_FFT_INLINE void fft_dft2(thread float2 *v) {
  const float2 a = v[0], b = v[1];
  v[0] = a + b;
  v[1] = a - b;
}

/* A DFT of 4 of the sums a_k + a_(k+4) into the even outputs; the
   differences d_k times w_8^k, the products by c = 1/√2 fused into the
   odd outputs: e2 = -i d2, t0 = d0 + e2, t1 = d0 - e2, e1 = c (p1, q1),
   e3 = c (p3, q3). */
NX_FFT_INLINE void fft_dft8(thread float2 *v) {
  const float c = 0.70710678118654752440f;
  float2 b0 = v[0] + v[4], b1 = v[1] + v[5], b2 = v[2] + v[6],
         b3 = v[3] + v[7];
  const float2 d0 = v[0] - v[4], d1 = v[1] - v[5], d2 = v[2] - v[6],
               d3 = v[3] - v[7];
  fft_dft4(b0, b1, b2, b3);
  v[0] = b0;
  v[2] = b1;
  v[4] = b2;
  v[6] = b3;
  const float2 t0 = float2(d0.x + d2.y, d0.y - d2.x),
               t1 = float2(d0.x - d2.y, d0.y + d2.x);
  const float p1 = d1.x + d1.y, q1 = d1.y - d1.x;
  const float p3 = d3.y - d3.x, q3 = -(d3.x + d3.y);
  const float sr = p1 + p3, si = q1 + q3, dr = p1 - p3, di = q1 - q3;
  v[1] = float2(fma(c, sr, t0.x), fma(c, si, t0.y));
  v[5] = float2(fma(-c, sr, t0.x), fma(-c, si, t0.y));
  v[3] = float2(fma(c, di, t1.x), fma(-c, dr, t1.y));
  v[7] = float2(fma(-c, di, t1.x), fma(c, dr, t1.y));
}

/* cos(2π m / P) and sin(2π m / P), 0 < m < P, as float32: constant
   expressions, so that every one folds once the loops unroll. */
template <uint P> constexpr float fft_cos(uint m);
template <uint P> constexpr float fft_sin(uint m);

template <> constexpr float fft_cos<3>(uint) { return -0.5f; }
template <> constexpr float fft_sin<3>(uint m) {
  return m == 1 ? 0.86602540378443864676f : -0.86602540378443864676f;
}
template <> constexpr float fft_cos<5>(uint m) {
  return m == 1 || m == 4 ? 0.30901699437494742410f
                          : -0.80901699437494742410f;
}
template <> constexpr float fft_sin<5>(uint m) {
  return m == 1   ? 0.95105651629515357212f
         : m == 2 ? 0.58778525229247312917f
         : m == 3 ? -0.58778525229247312917f
                  : -0.95105651629515357212f;
}
template <> constexpr float fft_cos<7>(uint m) {
  return m == 1 || m == 6   ? 0.62348980185873353053f
         : m == 2 || m == 5 ? -0.22252093395631440429f
                            : -0.90096886790241912624f;
}
template <> constexpr float fft_sin<7>(uint m) {
  return m == 1   ? 0.78183148246802980871f
         : m == 2 ? 0.97492791218182360702f
         : m == 3 ? 0.43388373911755812048f
         : m == 4 ? -0.43388373911755812048f
         : m == 5 ? -0.97492791218182360702f
                  : -0.78183148246802980871f;
}

/* An odd prime P: t_j = a_j + a_(P-j), d_j = a_j - a_(P-j); y_k = a_0 +
   Σ_j cos(2π jk/P) t_j - i Σ_j sin(2π jk/P) d_j and y_(P-k) its pair. */
template <uint P> NX_FFT_INLINE void fft_dft_odd(thread float2 *v) {
  constexpr uint H = (P - 1) / 2;
  float2 t[H + 1], d[H + 1];
  NX_FFT_UNROLL for (uint j = 1; j <= H; j++) {
    t[j] = v[j] + v[P - j];
    d[j] = v[j] - v[P - j];
  }
  const float2 a0 = v[0];
  float2 y0 = v[0];
  NX_FFT_UNROLL for (uint j = 1; j <= H; j++) { y0 = y0 + t[j]; }
  v[0] = y0;
  NX_FFT_UNROLL for (uint k = 1; k <= H; k++) {
    float2 m = a0;
    NX_FFT_UNROLL for (uint j = 1; j <= H; j++) {
      const float cj = fft_cos<P>((j * k) % P);
      m = float2(fma(cj, t[j].x, m.x), fma(cj, t[j].y, m.y));
    }
    const float s1 = fft_sin<P>(k % P);
    float2 n = float2(s1 * d[1].x, s1 * d[1].y);
    NX_FFT_UNROLL for (uint j = 2; j <= H; j++) {
      const float sj = fft_sin<P>((j * k) % P);
      n = float2(fma(sj, d[j].x, n.x), fma(sj, d[j].y, n.y));
    }
    /* y_k = m - i n, y_(P-k) = m + i n */
    v[k] = float2(m.x + n.y, m.y - n.x);
    v[P - k] = float2(m.x - n.y, m.y + n.x);
  }
}

template <uint R> NX_FFT_INLINE void fft_dft(thread float2 *v) {
  fft_dft_odd<R>(v);
}
template <> NX_FFT_ALWAYS void fft_dft<2>(thread float2 *v) { fft_dft2(v); }
template <> NX_FFT_ALWAYS void fft_dft<4>(thread float2 *v) {
  fft_dft4(v[0], v[1], v[2], v[3]);
}
template <> NX_FFT_ALWAYS void fft_dft<8>(thread float2 *v) { fft_dft8(v); }
template <> NX_FFT_ALWAYS void fft_dft<16>(thread float2 *v) {
  fft_dft16(v);
}

/* The step */

/* Pass q's radix. */
NX_FFT_INLINE uint fft_radix(uint q) {
  return q == 0   ? fc_r0
         : q == 1 ? fc_r1
         : q == 2 ? fc_r2
         : q == 3 ? fc_r3
         : q == 4 ? fc_r4
         : q == 5 ? fc_r5
                  : fc_r6;
}

/* Whether a contiguous line's plane is padded one point in 16. */
NX_FFT_INLINE bool fft_padded() {
  return (fc_l & (fc_l - 1)) == 0 && fc_l <= 2048;
}

NX_FFT_INLINE uint fft_slot(uint x) {
  return fft_padded() ? x + (x >> 4) : x;
}

NX_FFT_INLINE uint fft_plane_points() {
  return fft_padded() ? fc_l + fc_l / 16 : fc_l;
}

/* w_l^e, e < l, from the inner table: an odd l's whole, an even l's first
   half, w_l^(e + l/2) = -w_l^e. */
NX_FFT_INLINE float2 fft_inner(constant float2 *lo, uint e) {
  if (fc_l % 2 != 0) return lo[e];
  const uint upper = fc_l / 2;
  const bool up = e >= upper;
  const float2 w = lo[up ? e - upper : e];
  return up ? -w : w;
}

/* A line's thread [j], [on] where it belongs to one of the group's lines,
   [held] where that line is below the side's, and its line's sides, as
   complex elements or floats by what they hold; its line's complex plane
   [own] where its lines run contiguous, and where they run side by side,
   the group's float plane [floats], its line [i]'s point x at x [fc_lines]
   + i; the inner table [lo], a halved line's roots of 2 n [hi], and a
   convolution's tables; its line's points [n], and Rader's point 0 [x0],
   conjugated in an inverse. */
struct fft_lane {
  uint j, i, n;
  float2 x0;
  bool on, held, conj;
  device const fft_c64 *s;
  device fft_c64 *d;
  device const float *sf;
  device float *df;
  int ips, ops;
  threadgroup float2 *own;
  threadgroup float *floats;
  constant float2 *lo;
  constant float2 *hi;
  constant float2 *kern;
  constant float2 *chirp;
  constant int *logs;
};

/* Point [p] of the in side as its load reads it, before an inverse's
   conjugation: a complex point; a float, or for a split the pair of floats
   2 p and 2 p + 1; or for a c2r's odd line, its bin p <= l / 2, bin 0's
   imaginary part +0, and past it conj X_(n - p). */
NX_FFT_INLINE float2 fft_in(thread const fft_lane &a, uint p) {
  if (fc_input == NX_FFT_IN_REAL) {
    if (fc_halved == NX_FFT_HALVED_SPLIT)
      return float2(a.sf[int(2 * p) * a.ips], a.sf[int(2 * p + 1) * a.ips]);
    return float2(a.sf[int(p) * a.ips], 0.0f);
  }
  if (fc_input == NX_FFT_IN_BINS && p > a.n / 2) {
    const fft_c64 z = a.s[int(a.n - p) * a.ips];
    return float2(z.re, -z.im);
  }
  const fft_c64 z = a.s[int(p) * a.ips];
  return float2(z.re, fc_input == NX_FFT_IN_BINS && p == 0 ? 0.0f : z.im);
}

/* Writes point [p], conjugated already in an inverse, to the out side: a
   complex point, unless an r2c's odd line's past bin n / 2; or a float, its
   real part, or for a pre-pass both parts at floats 2 p and 2 p + 1. */
NX_FFT_INLINE void fft_out(thread const fft_lane &a, uint p, float2 z) {
  if (fc_output == NX_FFT_OUT_REAL) {
    if (fc_halved == NX_FFT_HALVED_PREPASS) {
      a.df[int(2 * p) * a.ops] = z.x;
      a.df[int(2 * p + 1) * a.ops] = z.y;
    } else {
      a.df[int(p) * a.ops] = z.x;
    }
    return;
  }
  if (fc_output == NX_FFT_OUT_HALF && p > a.n / 2) return;
  a.d[int(p) * a.ops] = fft_c64{z.x, z.y};
}

/* The split's and the pre-pass's pair k <= h / 2 of a halved line of
   h = n points, with the forward root w = w_2h^k from [hi]: from a and b,
   E = a + conj b, D = a - conj b, and U = fma(w.re, D.im, w.im D.re),
   V = fma(-w.re, D.re, w.im D.im), the parts of -i w D. */
struct fft_pair {
  float er, ei, u, v;
};

NX_FFT_INLINE fft_pair fft_pair_of(thread const fft_lane &a, uint k, float2 x,
                                   float2 y) {
  const float2 w = a.hi[k];
  const float br = y.x, bi = -y.y;
  const float dr = x.x - br, di = x.y - bi;
  fft_pair f;
  f.er = x.x + br;
  f.ei = x.y + bi;
  f.u = fma(w.x, di, w.y * dr);
  f.v = fma(-w.x, dr, w.y * di);
  return f;
}

/* A split's pair: from a = Z_k and b = Z_(h-k), Z_0 at k = 0, bins
   X_k = ((E.re + U) / 2, (E.im + V) / 2), then
   X_(h-k) = ((E.re - U) / 2, (V - E.im) / 2), stored. */
NX_FFT_INLINE void fft_split_pair(thread const fft_lane &a, uint k, float2 x,
                                  float2 y) {
  const fft_pair f = fft_pair_of(a, k, x, y);
  fft_out(a, k, fft_flip(float2(0.5f * (f.er + f.u), 0.5f * (f.ei + f.v)),
                         a.conj));
  fft_out(a, a.n - k,
          fft_flip(float2(0.5f * (f.er - f.u), 0.5f * (f.v - f.ei)), a.conj));
}

/* A pre-pass's pair: bins a = X_k and b = X_(h-k), conjugated in an
   inverse, bins 0 and h at k = 0 with imaginary parts +0 after it, into
   Z'_k = (E.re + U, E.im + V) and Z'_(h-k) = (E.re - U, V - E.im). */
NX_FFT_INLINE void fft_made_pair(thread const fft_lane &a, uint k,
                                 thread float2 &zk, thread float2 &zh) {
  float2 x = fft_flip(fft_in(a, k), a.conj);
  float2 y = fft_flip(fft_in(a, a.n - k), a.conj);
  if (k == 0) {
    x.y = 0.0f;
    y.y = 0.0f;
  }
  const fft_pair f = fft_pair_of(a, k, x, y);
  zk = float2(f.er + f.u, f.ei + f.v);
  zh = float2(f.er - f.u, f.v - f.ei);
}

/* The pieces of a pass of radix R, its butterflies K = PM / R a thread:
   butterfly b = j + tl k, b < Q = l / R, takes points b + Q r and writes
   value r to point (b / ns) ns R + b mod ns + ns r. */

/* Pass 0's reads, from the in side, conjugated in an inverse: its value p
   point [first] + p, +0 past the line's. */
template <uint R, uint PM>
NX_FFT_INLINE void fft_load(thread float2 *v, thread const fft_lane &a,
                            uint first) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    NX_FFT_UNROLL for (uint r = 0; r < R; r++) {
      const uint p = first + b + q * r;
      float2 z = float2(0.0f);
      if (a.held && b < q && p < a.n) z = fft_flip(fft_in(a, p), a.conj);
      v[k * R + r] = z;
    }
  }
}

/* Bluestein's chirp on pass 0's values: value p times chirp_p below the
   line's n points, +0 past them. */
template <uint R, uint PM>
NX_FFT_INLINE void fft_chirp_in(thread float2 *v, thread const fft_lane &a) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    NX_FFT_UNROLL for (uint r = 0; r < R; r++) {
      const uint p = b + q * r;
      v[k * R + r] =
          p < a.n ? fft_cmul(v[k * R + r], a.chirp[p]) : float2(0.0f);
    }
  }
}

/* The kernel's product on the last pass's values: value p of the FFT A,
   conj(A_p B_p). Rader's [y0] takes A_0 + x_0 first. */
template <uint R, uint PM>
NX_FFT_INLINE void fft_product(thread float2 *v, thread const fft_lane &a,
                              thread float2 &y0) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    if (b >= q) continue;
    NX_FFT_UNROLL for (uint r = 0; r < R; r++) {
      const uint p = b + q * r;
      if (fc_conv == NX_FFT_CONV_RADER && p == 0) y0 = v[k * R + r] + a.x0;
      v[k * R + r] = fft_conj(fft_cmul(v[k * R + r], a.kern[p]));
    }
  }
}

/* Bluestein's chirp on the last pass's values, in place: value p below the
   line's n points, conj(c_p) chirp_p. */
template <uint R, uint PM>
NX_FFT_INLINE void fft_chirp_out(thread float2 *v, thread const fft_lane &a) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    NX_FFT_UNROLL for (uint r = 0; r < R; r++) {
      const uint p = b + q * r;
      if (p < a.n)
        v[k * R + r] = fft_cmul(fft_conj(v[k * R + r]), a.chirp[p]);
    }
  }
}

/* A pass's butterflies, past pass 0 their points twiddled first. */
template <uint R, uint PM>
NX_FFT_INLINE void fft_butterflies(thread float2 *v, thread const fft_lane &a,
                                   uint ns) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  if (!a.on) return;
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    if (b >= q) continue;
    if (ns > 1) {
      const uint e = (b % ns) * (fc_l / (ns * R));
      NX_FFT_UNROLL for (uint r = 1; r < R; r++) {
        v[k * R + r] = fft_cmul(v[k * R + r], fft_inner(a.lo, e * r));
      }
    }
    fft_dft<R>(v + k * R);
  }
}

/* The last pass's writes, to the out side, conjugated in an inverse: its
   ns is Q, so value r of butterfly b is point b + Q r; those of the line's
   n points alone. */
template <uint R, uint PM>
NX_FFT_INLINE void fft_store(thread const float2 *v, thread const fft_lane &a) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  if (!a.held) return;
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    if (b >= q) continue;
    NX_FFT_UNROLL for (uint r = 0; r < R; r++) {
      const uint p = b + q * r;
      if (p < a.n) fft_out(a, p, fft_flip(v[k * R + r], a.conj));
    }
  }
}

/* A pass's writes to the plane: complex points into the line's own, or
   part [c] alone (0 real, 1 imaginary) into the float plane. */
template <uint R, uint PM, bool Side>
NX_FFT_INLINE void fft_put(thread const float2 *v, thread const fft_lane &a,
                           uint ns, uint c) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  if (!a.on) return;
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    if (b >= q) continue;
    const uint at = (b / ns) * ns * R + b % ns;
    NX_FFT_UNROLL for (uint r = 0; r < R; r++) {
      const uint x = at + ns * r;
      if (Side)
        a.floats[x * fc_lines + a.i] = c == 0 ? v[k * R + r].x : v[k * R + r].y;
      else
        a.own[fft_slot(x)] = v[k * R + r];
    }
  }
}

/* A pass's reads from the plane, as [fft_put] writes it. */
template <uint R, uint PM, bool Side>
NX_FFT_INLINE void fft_get(thread float2 *v, thread const fft_lane &a,
                           uint c) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  if (!a.on) return;
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    if (b >= q) continue;
    NX_FFT_UNROLL for (uint r = 0; r < R; r++) {
      const uint x = b + q * r;
      if (Side) {
        const float f = a.floats[x * fc_lines + a.i];
        if (c == 0) v[k * R + r].x = f;
        else v[k * R + r].y = f;
      } else {
        v[k * R + r] = a.own[fft_slot(x)];
      }
    }
  }
}

/* Plane slot x of part [c] of the lane's line, or the whole point. */
template <bool Side>
NX_FFT_INLINE void fft_plane_set(thread const fft_lane &a, uint x, float2 z,
                                 uint c) {
  if (Side)
    a.floats[x * fc_lines + a.i] = c == 0 ? z.x : z.y;
  else
    a.own[fft_slot(x)] = z;
}

template <bool Side>
NX_FFT_INLINE void fft_plane_read(thread const fft_lane &a, uint x,
                                  thread float2 &z, uint c) {
  if (!Side) {
    z = a.own[fft_slot(x)];
  } else if (c == 0) {
    z.x = a.floats[x * fc_lines + a.i];
  } else {
    z.y = a.floats[x * fc_lines + a.i];
  }
}

/* Rader's load's writes to the plane: pass 0's value p, point p + 1 as
   loaded, at slot logs[p + 1], a part [c] or the whole point. */
template <uint R, uint PM, bool Side>
NX_FFT_INLINE void fft_scatter(thread const float2 *v, thread const fft_lane &a,
                               uint c) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  if (!a.on) return;
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    if (b >= q) continue;
    NX_FFT_UNROLL for (uint r = 0; r < R; r++) {
      const uint p = b + q * r;
      fft_plane_set<Side>(a, uint(a.logs[p + 1]), v[k * R + r], c);
    }
  }
}

/* Rader's store's reads from the plane, which holds the second FFT's c in
   natural order: value p, for point p + 1, takes c_q, q = (l - logs[p + 1])
   mod l, a part [c] or the whole point. */
template <uint R, uint PM, bool Side>
NX_FFT_INLINE void fft_gather(thread float2 *v, thread const fft_lane &a,
                              uint c) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  if (!a.on) return;
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    if (b >= q) continue;
    NX_FFT_UNROLL for (uint r = 0; r < R; r++) {
      const uint s = uint(a.logs[b + q * r + 1]);
      fft_plane_read<Side>(a, s == 0 ? 0 : fc_l - s, v[k * R + r], c);
    }
  }
}

/* Rader's writes: point p + 1, x_0 + conj c_q, and from the line's first
   thread point 0, y_0, each conjugated in an inverse. */
template <uint R, uint PM>
NX_FFT_INLINE void fft_rader_out(thread const float2 *v,
                                 thread const fft_lane &a, float2 y0) {
  constexpr uint K = PM / R;
  const uint q = fc_l / R;
  if (!a.held) return;
  if (a.j == 0) fft_out(a, 0, fft_flip(y0, a.conj));
  NX_FFT_UNROLL for (uint k = 0; k < K; k++) {
    const uint b = a.j + fc_tl * k;
    if (b >= q) continue;
    NX_FFT_UNROLL for (uint r = 0; r < R; r++) {
      const float2 y = a.x0 + fft_conj(v[k * R + r]);
      fft_out(a, b + q * r + 1, fft_flip(y, a.conj));
    }
  }
}

/* [Piece<R, PM>::run] for pass radix [r] where R fits the PM points a thread
   holds, else nothing: a radix no key of the function takes. */
#define NX_FFT_RADIX(r, call)                                                  \
  switch (r) {                                                                 \
  case 2: NX_FFT_FIT(2, call); break;                                          \
  case 3: NX_FFT_FIT(3, call); break;                                          \
  case 4: NX_FFT_FIT(4, call); break;                                          \
  case 5: NX_FFT_FIT(5, call); break;                                          \
  case 7: NX_FFT_FIT(7, call); break;                                          \
  case 8: NX_FFT_FIT(8, call); break;                                          \
  case 16: NX_FFT_FIT(16, call); break;                                        \
  default: break;                                                              \
  }

template <uint R, uint PM, bool Fits = (R <= PM)> struct fft_pieces {
  static NX_FFT_ALWAYS void load(thread float2 *v, thread const fft_lane &a,
                                 uint first) {
    fft_load<R, PM>(v, a, first);
  }
  static NX_FFT_ALWAYS void chirp_in(thread float2 *v,
                                     thread const fft_lane &a) {
    fft_chirp_in<R, PM>(v, a);
  }
  static NX_FFT_ALWAYS void product(thread float2 *v, thread const fft_lane &a,
                                   thread float2 &y0) {
    fft_product<R, PM>(v, a, y0);
  }
  static NX_FFT_ALWAYS void chirp_out(thread float2 *v,
                                      thread const fft_lane &a) {
    fft_chirp_out<R, PM>(v, a);
  }
  static NX_FFT_ALWAYS void butterflies(thread float2 *v,
                                        thread const fft_lane &a, uint ns) {
    fft_butterflies<R, PM>(v, a, ns);
  }
  static NX_FFT_ALWAYS void store(thread const float2 *v,
                                  thread const fft_lane &a) {
    fft_store<R, PM>(v, a);
  }
  template <bool Side>
  static NX_FFT_ALWAYS void put(thread const float2 *v,
                                thread const fft_lane &a, uint ns, uint c) {
    fft_put<R, PM, Side>(v, a, ns, c);
  }
  template <bool Side>
  static NX_FFT_ALWAYS void get(thread float2 *v, thread const fft_lane &a,
                                uint c) {
    fft_get<R, PM, Side>(v, a, c);
  }
  template <bool Side>
  static NX_FFT_ALWAYS void scatter(thread const float2 *v,
                                    thread const fft_lane &a, uint c) {
    fft_scatter<R, PM, Side>(v, a, c);
  }
  template <bool Side>
  static NX_FFT_ALWAYS void gather(thread float2 *v, thread const fft_lane &a,
                                   uint c) {
    fft_gather<R, PM, Side>(v, a, c);
  }
  static NX_FFT_ALWAYS void rader_out(thread const float2 *v,
                                      thread const fft_lane &a, float2 y0) {
    fft_rader_out<R, PM>(v, a, y0);
  }
};
template <uint R, uint PM> struct fft_pieces<R, PM, false> {
  static NX_FFT_ALWAYS void load(thread float2 *, thread const fft_lane &,
                                 uint) {}
  static NX_FFT_ALWAYS void chirp_in(thread float2 *,
                                     thread const fft_lane &) {}
  static NX_FFT_ALWAYS void product(thread float2 *, thread const fft_lane &,
                                   thread float2 &) {}
  static NX_FFT_ALWAYS void chirp_out(thread float2 *,
                                      thread const fft_lane &) {}
  static NX_FFT_ALWAYS void butterflies(thread float2 *,
                                        thread const fft_lane &, uint) {}
  static NX_FFT_ALWAYS void store(thread const float2 *,
                                  thread const fft_lane &) {}
  template <bool Side>
  static NX_FFT_ALWAYS void put(thread const float2 *,
                                thread const fft_lane &, uint, uint) {}
  template <bool Side>
  static NX_FFT_ALWAYS void get(thread float2 *, thread const fft_lane &,
                                uint) {}
  template <bool Side>
  static NX_FFT_ALWAYS void scatter(thread const float2 *,
                                    thread const fft_lane &, uint) {}
  template <bool Side>
  static NX_FFT_ALWAYS void gather(thread float2 *, thread const fft_lane &,
                                   uint) {}
  static NX_FFT_ALWAYS void rader_out(thread const float2 *,
                                      thread const fft_lane &, float2) {}
};

/* The exchange between a pass of radix [rp] after [nsp] points and the
   next, of radix [r]. Contiguous lines move complex points through their
   own planes: a write, a barrier, a read. Lines side by side move one part
   at a time through the float plane, the real parts, then the imaginary:
   a write, a read, each after a barrier. A read of the real parts writes
   only the points' real parts, so the imaginary parts the next write moves
   are still the earlier pass's. No barrier follows the last exchange's
   reads, which no later write of the plane follows. */
template <uint PM, bool Side>
NX_FFT_INLINE void fft_exchange(thread float2 *v, thread const fft_lane &a,
                                uint rp, uint nsp, uint r, bool last) {
  const uint parts = Side ? 2 : 1;
  NX_FFT_UNROLL for (uint c = 0; c < parts; c++) {
#define NX_FFT_FIT(R, call) fft_pieces<R, PM>::template call
    NX_FFT_RADIX(rp, put<Side>(v, a, nsp, c));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NX_FFT_RADIX(r, get<Side>(v, a, c));
#undef NX_FFT_FIT
    if (!last || c + 1 < parts)
      threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

/* An r2c's halved line's store: the last pass's points Z into the plane in
   natural order, after the last exchange's reads, then each thread's
   pairs k = j + tl m <= h / 2 from it, split into bins. Bin h, from bin
   0's pair, lies outside the plane. */
template <uint PM, bool Side>
NX_FFT_INLINE void fft_split(thread const float2 *v, thread const fft_lane &a) {
  constexpr uint M = PM / 2 + 1;
  const uint parts = Side ? 2 : 1;
  const uint rl = fft_radix(fc_passes - 1);
  float2 x[M], y[M];
  NX_FFT_UNROLL for (uint m = 0; m < M; m++) {
    x[m] = float2(0.0f);
    y[m] = float2(0.0f);
  }
  NX_FFT_UNROLL for (uint c = 0; c < parts; c++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#define NX_FFT_FIT(R, call) fft_pieces<R, PM>::template call
    NX_FFT_RADIX(rl, put<Side>(v, a, fc_l / rl, c));
#undef NX_FFT_FIT
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NX_FFT_UNROLL for (uint m = 0; m < M; m++) {
      const uint k = a.j + fc_tl * m;
      if (!a.on || k > a.n / 2) continue;
      fft_plane_read<Side>(a, k, x[m], c);
      fft_plane_read<Side>(a, k == 0 ? 0 : a.n - k, y[m], c);
    }
  }
  if (!a.held) return;
  NX_FFT_UNROLL for (uint m = 0; m < M; m++) {
    const uint k = a.j + fc_tl * m;
    if (k <= a.n / 2) fft_split_pair(a, k, x[m], y[m]);
  }
}

/* A c2r's halved line's load: each thread's pairs k = j + tl m <= h / 2
   made from the bins into the plane, Z'_(h-k) where 0 < k and 2 k != h,
   then pass 0's points read from it, and a barrier before the plane's
   next writes. Bin h, beside bin 0, lies outside the plane. */
template <uint PM, bool Side>
NX_FFT_INLINE void fft_prepass(thread float2 *v, thread const fft_lane &a) {
  constexpr uint M = PM / 2 + 1;
  const uint parts = Side ? 2 : 1;
  float2 zk[M], zh[M];
  NX_FFT_UNROLL for (uint m = 0; m < M; m++) {
    const uint k = a.j + fc_tl * m;
    zk[m] = float2(0.0f);
    zh[m] = float2(0.0f);
    if (a.held && k <= a.n / 2) fft_made_pair(a, k, zk[m], zh[m]);
  }
  NX_FFT_UNROLL for (uint c = 0; c < parts; c++) {
    NX_FFT_UNROLL for (uint m = 0; m < M; m++) {
      const uint k = a.j + fc_tl * m;
      if (!a.on || k > a.n / 2) continue;
      fft_plane_set<Side>(a, k, zk[m], c);
      if (k != 0 && 2 * k != a.n) fft_plane_set<Side>(a, a.n - k, zh[m], c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#define NX_FFT_FIT(R, call) fft_pieces<R, PM>::template call
    NX_FFT_RADIX(fc_r0, get<Side>(v, a, c));
#undef NX_FFT_FIT
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

/* Rader's load: point 0 in every thread's x0 already, then points p + 1,
   p < l, into the plane at slot logs[p + 1], pass 0's points read from
   it, and a barrier before the plane's next writes. */
template <uint PM, bool Side>
NX_FFT_INLINE void fft_rader_load(thread float2 *v, thread const fft_lane &a) {
  const uint parts = Side ? 2 : 1;
#define NX_FFT_FIT(R, call) fft_pieces<R, PM>::call
  NX_FFT_RADIX(fc_r0, load(v, a, 1));
#undef NX_FFT_FIT
#define NX_FFT_FIT(R, call) fft_pieces<R, PM>::template call
  NX_FFT_UNROLL for (uint c = 0; c < parts; c++) {
    NX_FFT_RADIX(fc_r0, scatter<Side>(v, a, c));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NX_FFT_RADIX(fc_r0, get<Side>(v, a, c));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#undef NX_FFT_FIT
}

/* Rader's store: the second FFT's c into the plane in natural order, after
   the last exchange's reads, each thread's values read back permuted, then
   its points. */
template <uint PM, bool Side>
NX_FFT_INLINE void fft_rader_store(thread float2 *v, thread const fft_lane &a,
                                   float2 y0) {
  const uint parts = Side ? 2 : 1;
  const uint rl = fft_radix(fc_passes - 1);
#define NX_FFT_FIT(R, call) fft_pieces<R, PM>::template call
  NX_FFT_UNROLL for (uint c = 0; c < parts; c++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NX_FFT_RADIX(rl, put<Side>(v, a, fc_l / rl, c));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NX_FFT_RADIX(rl, gather<Side>(v, a, c));
  }
#undef NX_FFT_FIT
#define NX_FFT_FIT(R, call) fft_pieces<R, PM>::call
  NX_FFT_RADIX(rl, rader_out(v, a, y0));
#undef NX_FFT_FIT
}

/* An FFT's passes from pass 0's values to the last pass's; [last] where no
   write of the plane follows, so no barrier ends them. */
template <uint PM, bool Side>
NX_FFT_INLINE void fft_passes(thread float2 *v, thread const fft_lane &a,
                              bool last) {
#define NX_FFT_FIT(R, call) fft_pieces<R, PM>::call
  NX_FFT_RADIX(fc_r0, butterflies(v, a, 1));
  uint ns = fc_r0;
  NX_FFT_UNROLL for (uint q = 1; q < NX_FFT_FC_RADICES; q++) {
    if (q < fc_passes) {
      const uint rp = fft_radix(q - 1), r = fft_radix(q);
      const bool end = last && q + 1 == fc_passes;
      fft_exchange<PM, Side>(v, a, rp, ns / rp, r, end);
      NX_FFT_RADIX(r, butterflies(v, a, ns));
      ns *= r;
    }
  }
#undef NX_FFT_FIT
}

/* A step of one point: each part times the table's 1, which flushes a
   subnormal part and keeps every other value; conjugating around it
   changes nothing. A split takes the product as Z_0, a pre-pass makes the
   point from bins 0 and 1 first, and an r2c's line of no point loads
   +0, conjugated in an inverse as a loaded point. */
NX_FFT_INLINE void fft_one(thread const fft_lane &a) {
  if (!a.held || a.j != 0) return;
  const float one = a.lo[0].x;
  float2 z = float2(0.0f);
  if (fc_halved == NX_FFT_HALVED_PREPASS) {
    float2 zh;
    fft_made_pair(a, 0, z, zh);
  } else {
    z = fft_flip(fc_input == NX_FFT_IN_NONE ? float2(0.0f) : fft_in(a, 0),
                 a.conj);
  }
  z = float2(z.x * one, z.y * one);
  if (fc_halved == NX_FFT_HALVED_SPLIT)
    fft_split_pair(a, 0, z, z);
  else
    fft_out(a, 0, fft_flip(z, a.conj));
}

/* A group's lines: lines run contiguous, [fc_tl] threads a line, thread
   tl i + j of line i; or side by side, thread G j + i of line i, G =
   [fc_lines], so that a point's loads and stores of adjacent lines are
   adjacent threads'. */
template <uint PM, bool Side>
NX_FFT_INLINE void fft_step(constant fft_key_words &w,
                            threadgroup float2 *plane, uint g, uint t) {
  const uint tl = fc_tl, lines = fc_lines;
  fft_lane a;
  a.i = Side ? t % lines : t / tl;
  a.j = Side ? t / lines : t - a.i * tl;
  a.on = Side ? a.j < tl : a.i < lines;
  const ulong line = (w.first + g) * lines + a.i;
  a.held = a.on && line < w.lines;
  a.conj = (w.mode & NX_FFT_CONJ) != 0;
  a.own = plane + (Side || !a.on ? 0 : a.i) * fft_plane_points();
  a.floats = (threadgroup float *)plane;
  a.lo = w.lo;
  a.hi = w.hi;
  a.kern = w.kern;
  a.chirp = w.chirp;
  a.logs = w.logs;
  a.n = fc_conv == NX_FFT_CONV_NONE ? fc_l : uint(w.n);
  a.ips = int(w.in.ps);
  a.ops = int(w.out.ps);
  a.s = (device const fft_c64 *)w.src;
  a.d = (device fft_c64 *)w.dst;
  a.sf = (device const float *)w.src;
  a.df = (device float *)w.dst;
  a.x0 = float2(0.0f);
  if (a.held) {
    constant long *in_axes =
        w.in.rank <= NX_FFT_SIDE_HOLDS ? w.in.axes : w.tail;
    constant long *out_axes = w.out.rank <= NX_FFT_SIDE_HOLDS
                                  ? w.out.axes
                                  : w.tail + NX_FFT_AXIS_WORDS * w.in.rank;
    /* Each point an element, complex or a float, at a 32-bit offset from
       its line's first, as the step's sides are not wide, which fft.ml
       checks (fft_index). */
    const long in_first = fft_line(w.in, in_axes, line);
    const long out_first = fft_line(w.out, out_axes, line);
    a.s += in_first;
    a.sf += in_first;
    a.d += out_first;
    a.df += out_first;
    if (fc_conv == NX_FFT_CONV_RADER) a.x0 = fft_flip(fft_in(a, 0), a.conj);
  }

  if (fc_passes == 0) {
    fft_one(a);
    return;
  }

  /* A convolution runs the FFT twice, the kernel's product between, the
     last pass's values back to pass 0's through the plane. */
  float2 v[PM];
  float2 y0 = float2(0.0f);
  const uint rl = fft_radix(fc_passes - 1);
  const bool conv = fc_conv != NX_FFT_CONV_NONE;
  if (fc_halved == NX_FFT_HALVED_PREPASS) {
    fft_prepass<PM, Side>(v, a);
  } else if (fc_conv == NX_FFT_CONV_RADER) {
    fft_rader_load<PM, Side>(v, a);
  } else {
#define NX_FFT_FIT(R, call) fft_pieces<R, PM>::call
    NX_FFT_RADIX(fc_r0, load(v, a, 0));
#undef NX_FFT_FIT
  }
#define NX_FFT_FIT(R, call) fft_pieces<R, PM>::call
  if (fc_conv == NX_FFT_CONV_BLUESTEIN) NX_FFT_RADIX(fc_r0, chirp_in(v, a));
  fft_passes<PM, Side>(v, a, !conv);
  if (conv) {
    NX_FFT_RADIX(rl, product(v, a, y0));
    fft_exchange<PM, Side>(v, a, rl, fc_l / rl, fc_r0, false);
    fft_passes<PM, Side>(v, a, true);
  }
  if (fc_conv == NX_FFT_CONV_BLUESTEIN) NX_FFT_RADIX(rl, chirp_out(v, a));
  if (fc_conv == NX_FFT_CONV_RADER)
    fft_rader_store<PM, Side>(v, a, y0);
  else if (fc_halved == NX_FFT_HALVED_SPLIT)
    fft_split<PM, Side>(v, a);
  else
    NX_FFT_RADIX(rl, store(v, a));
#undef NX_FFT_FIT
}

NX_KERNEL(NX_FFT_KEY_THREADS_P8)
void fft_step_p8(constant fft_key_words &w [[buffer(0)]],
                 threadgroup float2 *plane [[threadgroup(0)]],
                 uint g [[threadgroup_position_in_grid]],
                 uint t [[thread_index_in_threadgroup]]) {
  fft_step<8, false>(w, plane, g, t);
}

NX_KERNEL(NX_FFT_KEY_THREADS_P16)
void fft_step_p16(constant fft_key_words &w [[buffer(0)]],
                  threadgroup float2 *plane [[threadgroup(0)]],
                  uint g [[threadgroup_position_in_grid]],
                  uint t [[thread_index_in_threadgroup]]) {
  fft_step<16, false>(w, plane, g, t);
}

NX_KERNEL(NX_FFT_SIDE_THREADS)
void fft_side_p8(constant fft_key_words &w [[buffer(0)]],
                 threadgroup float2 *plane [[threadgroup(0)]],
                 uint g [[threadgroup_position_in_grid]],
                 uint t [[thread_index_in_threadgroup]]) {
  fft_step<8, true>(w, plane, g, t);
}

NX_KERNEL(NX_FFT_SIDE_THREADS)
void fft_side_p16(constant fft_key_words &w [[buffer(0)]],
                  threadgroup float2 *plane [[threadgroup(0)]],
                  uint g [[threadgroup_position_in_grid]],
                  uint t [[thread_index_in_threadgroup]]) {
  fft_step<16, true>(w, plane, g, t);
}
