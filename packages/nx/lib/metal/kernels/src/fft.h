/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the Fourier transforms' units share (fft.metal, fft_steps.metal): a
   side's words and its lines' positions, complex arithmetic, and the
   butterflies of 4 and 16 points. */

#ifndef NX_METAL_FFT_H
#define NX_METAL_FFT_H

#include "device.h"

using metal::fma;
using metal::mem_flags;
using metal::threadgroup_barrier;

struct fft_side_words {
  long first;
  long ps;
  long rank;
  long width;
  long axes[NX_FFT_AXIS_WORDS * NX_FFT_SIDE_HOLDS];
};

NX_FIELD(fft_side_words, first, NX_FFT_SIDE_FIRST);
NX_FIELD(fft_side_words, ps, NX_FFT_SIDE_PS);
NX_FIELD(fft_side_words, rank, NX_FFT_SIDE_RANK);
NX_FIELD(fft_side_words, width, NX_FFT_SIDE_WIDTH);
NX_FIELD(fft_side_words, axes, NX_FFT_SIDE_AXES);
NX_SIZE(fft_side_words, NX_FFT_SIDE_WORDS);

/* Complex arithmetic */

static inline float2 fft_cmul(float2 a, float2 w) {
  return float2(fma(a.x, w.x, -(a.y * w.y)), fma(a.x, w.y, a.y * w.x));
}

static inline float2 fft_conj(float2 a) { return float2(a.x, -a.y); }

/* [z] conjugated where [c], an inverse's load or store: a negation, exact
   at every value, where a product by -1 would read a subnormal as zero.
   [c] is uniform, so a forward launch takes neither the flip nor its cost. */
static inline float2 fft_flip(float2 z, bool c) {
  if (c) z.y = -z.y;
  return z;
}

/* Indices */

/* The position of line [l]'s first point of side [s], whose axes lie at
   [axes], in elements: its coordinates from the innermost axis, each times
   its stride. */
static inline long fft_line(constant fft_side_words &s, constant long *axes,
                            ulong l) {
  long at = s.first;
  for (int k = int(s.rank) - 1; k >= 1; k--) {
    constant long *axis = axes + NX_FFT_AXIS_WORDS * k;
    const ulong e = ulong(axis[NX_FFT_AXIS_EXTENT]);
    const ulong q = nx_divide(l, e, ulong(axis[NX_FFT_AXIS_DIVISOR]));
    at += long(l - q * e) * axis[NX_FFT_AXIS_STRIDE];
    l = q;
  }
  return at + long(l) * axes[NX_FFT_AXIS_STRIDE];
}

/* A complex element as memory holds it: two floats, aligned as a float, so
   that a line may start at any float. */
struct fft_c64 {
  float re, im;
};

/* Butterflies: forward DFTs in natural order, in place, fft.hip's forms. A
   thread's points are an array indexed by constants alone once the loops
   unroll, so they stay in registers. */

#define NX_FFT_UNROLL _Pragma("clang loop unroll(full)")

static inline __attribute__((always_inline)) void
fft_dft4(thread float2 &a0, thread float2 &a1, thread float2 &a2,
         thread float2 &a3) {
  const float2 t0 = a0 + a2, t1 = a0 - a2, t2 = a1 + a3, t3 = a1 - a3;
  a0 = t0 + t2;
  a2 = t0 - t2;
  /* -i t3 = (t3.im, -t3.re) */
  a1 = float2(t1.x + t3.y, t1.y - t3.x);
  a3 = float2(t1.x - t3.y, t1.y + t3.x);
}

/* x w for w = (C, -Sn), a root of 16 with both parts nonzero. */
static inline __attribute__((always_inline)) float2 fft_w16(float2 x, float C,
                                                            float Sn) {
  return float2(fma(x.x, C, x.y * Sn), fma(x.y, C, -(x.x * Sn)));
}

/* x w^2 for w = w_16, (c, -c): c (x.re + x.im), c (x.im - x.re). */
static inline __attribute__((always_inline)) float2 fft_w16_2(float2 x,
                                                              float c) {
  return float2(c * (x.x + x.y), c * (x.y - x.x));
}

/* x w^6, (-c, -c): c (x.im - x.re), -(c (x.re + x.im)). */
static inline __attribute__((always_inline)) float2 fft_w16_6(float2 x,
                                                              float c) {
  return float2(c * (x.y - x.x), -(c * (x.x + x.y)));
}

/* 4 x 4: a DFT of 4 over n2 of v[n1 + 4 n2], the twiddles w_16^(n1 k2),
   then a DFT of 4 over n1 into v[k2 + 4 k1]. */
static inline __attribute__((always_inline)) void
fft_dft16(thread float2 *v) {
  const float c = 0.70710678118654752440f;
  const float C = 0.92387953251128675613f, Sn = 0.38268343236508977173f;
  NX_FFT_UNROLL for (int n1 = 0; n1 < 4; n1++) {
    fft_dft4(v[n1], v[n1 + 4], v[n1 + 8], v[n1 + 12]);
  }
  v[5] = fft_w16(v[5], C, Sn);
  v[9] = fft_w16_2(v[9], c);
  v[13] = fft_w16(v[13], Sn, C); /* w^3 = (Sn, -C) */
  v[6] = fft_w16_2(v[6], c);
  v[10] = float2(v[10].y, -v[10].x); /* w^4 = -i */
  v[14] = fft_w16_6(v[14], c);
  v[7] = fft_w16(v[7], Sn, C);
  v[11] = fft_w16_6(v[11], c);
  v[15] = -fft_w16(v[15], C, Sn); /* w^9 = -w^1 */
  float2 u[16];
  NX_FFT_UNROLL for (int k = 0; k < 16; k++) { u[k] = v[k]; }
  NX_FFT_UNROLL for (int k2 = 0; k2 < 4; k2++) {
    float2 a0 = u[4 * k2], a1 = u[4 * k2 + 1], a2 = u[4 * k2 + 2],
           a3 = u[4 * k2 + 3];
    fft_dft4(a0, a1, a2, a3);
    v[k2] = a0;
    v[k2 + 4] = a1;
    v[k2 + 8] = a2;
    v[k2 + 12] = a3;
  }
}

#endif
