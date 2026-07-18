/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_fft.c — the backend's owned FFT.

   It uses a self-contained, self-sorting mixed-radix Cooley-Tukey
   transform — dedicated radix 8/4/2/3/5/7 butterflies plus a generic O(ip²)
   odd-prime pass (11, 13) — for every length whose largest prime factor is small
   (≤ 13). This covers all powers of two and smooth composites like 100000 =
   2⁵·5⁵, 1000000 = 2⁶·5⁶, and 44100 = 2²·3²·5²·7². A length with a larger prime
   factor (a large prime, or a composite like 17·k) is transformed by Bluestein's
   chirp-z algorithm, whose two internal length-m FFTs (m a power of two, hence
   smooth) ride the same native core — so the transform is correct for ALL n.
   Everything computes in double precision; c32 inputs upcast on gather and round
   on store, which beats a native single-precision transform on accuracy.

   The core is SELF-SORTING (Stockham/Temperton): each stage reads one buffer and
   writes the other in natural order, so there is NO separate digit-reversal
   pass. The butterflies are written as explicit real arithmetic on an
   interleaved `cx2 {re, im}` element — never a C99 `double _Complex`, whose
   multiply compiles to a `__muldc3` libcall that neither inlines nor vectorizes.
   Interleaved (not split re[]/im[]) is deliberate: at a power-of-two length the
   radix-8 output stride n/8 folds the 8 lanes onto a handful of cache sets, and a
   split layout DOUBLES the number of colliding streams (re and im each fold),
   overrunning both L1 associativity and the hardware prefetcher — measured 3×
   slower at n=65536. Interleaved keeps re/im in one stream, so the concurrent
   stream count stays within budget. The main transform's two ping-pong buffers
   are deliberately UNPADDED (adjacent in the per-line scratch): a cache-line pad
   between them was measured and REGRESSED pow2 (65536 1.20×→1.29×), so FFT_PAD is
   applied only where it measured neutral-or-better — the Bluestein a/scr buffers
   and the irfft g/f/work buffers.

   Backend transforms are UNNORMALIZED, as pinned by the backend contract: fft
   is the -sign DFT, ifft the +sign DFT with no 1/n, so
   ifft(fft x) = n·x and irfft(rfft x) = n·x; the frontend owns the 1/n.

   Plans (mixed-radix schedule + per-stage twiddle tables, or the Bluestein
   chirp/filter + its two native sub-plans) are built once per (length, sign) and
   kept in a grow-only, mutex-guarded cache for the process lifetime — the same
   "lives until exit" trade the engine pool makes, so a plan pointer handed to
   pool workers is never freed under them (no eviction → no use-after-free). The
   driver builds/looks up plans with the runtime lock HELD (before releasing via
   nx_c_parallel_for); pool workers only READ the plan's immutable tables. Every
   twiddle is computed once with libm, its index reduced mod n so the angle stays
   in [0,2π) (full accuracy, no ~n·eps drift at large n).

   Multi-axis is repeated 1-D passes, one transform axis at a time; each pass
   parallelizes over the independent "lines" (the odometer over all other dims),
   gathering each strided length-n line to contiguous scratch, transforming, and
   scattering back. rfft/irfft ride the complex transform (correctness-first). */

/* DEFERRED PERF (correctness-first; none affects correctness, all verified
   against the DFT oracle):
   1. rfft/irfft use a full-size complex FFT + half extraction rather than a
      half-size packed real-FFT — correct but ~2× the ideal on the real axis.
   2. irfft's strided→contiguous tmp-fill (nx_c_irfft_run) runs serially under the
      runtime lock before nx_c_parallel_for; it only reads raw bigarray memory, so
      it could move after the lock release if it ever shows on a profile.
   3. The generic odd-prime pass (11/13) is an O(ip²) direct DFT, not a tuned
      unrolled butterfly, and the native/Bluestein gate is a flat prime bound
      (≤ 13) rather than a cost model. Fine for the admitted small primes; a
      length whose largest prime factor is a mid-size prime (17..~√n) always
      takes Bluestein even where a tuned generic pass could beat it. */

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h"

#define NX_C_PI 3.14159265358979323846
#define NX_C_SQRTH 0.70710678118654752440 /* 1/√2 = cos(π/4) */

/* One cache line (4 cx2) of separation between two co-live scratch buffers, to
   break power-of-two cache-set aliasing. Applied ONLY where it measured
   neutral-or-better: the Bluestein a/scr buffers (bluestein_exec) and the irfft
   g/f/work buffers (irfft_last_body). The MAIN fft/ifft transform's two ping-pong
   buffers are deliberately left adjacent (no pad) — a pad there REGRESSED pow2
   (65536 1.20×→1.29×, measured), so native's ping-pong scratch is exactly n. */
#define FFT_PAD 4

typedef struct {
  double r, i;
} cx2;

/* ── Self-sorting mixed-radix butterfly stages (interleaved, DIF) ──────────

   Every stage combines `radix` sub-transforms of size `ido·l1` read from cc into
   one of size `radix·ido·l1` written to ch, across `l1` blocks. n = radix·ido·l1.
   Reads use the layout CC(a,b,k) = cc[a + ido*(b + radix*k)] (a in [0,ido),
   b in [0,radix), k in [0,l1)); writes use CH(a,k,b) = ch[a + ido*(k + l1*b)] —
   the b-stride flips from `radix·ido` on input to `ido` on output, which is what
   sorts the data in place over the whole schedule. The twiddle applied to output
   lane b≥1 at inner index a is wa[(a-1) + (b-1)*(ido-1)] (a=0 needs none); its
   value is ω_n^{sign·b·l1·a} already sign-folded, so the butterfly does a plain
   complex multiply. */

/* Radix butterflies as macros declaring the `radix` pre-twiddle outputs as named
   cx2 locals m0..m{radix-1} (kept in registers — an m[] array staged the outputs
   on the stack and slowed the compute-bound power-of-two path). The pass loops
   expand these, then apply the per-lane twiddle (or store bare at a=0). The a=0
   case is peeled OUT of the inner loop so it stays straight-line and vectorizes
   across a. `rot`/`h` are pass-scope locals the radix-4/8 macros reference. */

#define NX_C_BFLY3(x0, x1, x2, tw1r, tw1i)                                      \
  double _t1r = (x1).r + (x2).r, _t1i = (x1).i + (x2).i;                       \
  double _t2r = (x1).r - (x2).r, _t2i = (x1).i - (x2).i;                       \
  double _car = (x0).r + (tw1r) * _t1r, _cai = (x0).i + (tw1r) * _t1i;         \
  double _cbr = -(tw1i) * _t2i, _cbi = (tw1i) * _t2r;                          \
  cx2 m0 = {(x0).r + _t1r, (x0).i + _t1i};                                     \
  cx2 m1 = {_car + _cbr, _cai + _cbi};                                         \
  cx2 m2 = {_car - _cbr, _cai - _cbi}

#define NX_C_BFLY4(x0, x1, x2, x3)                                             \
  double _t2r = (x0).r + (x2).r, _t2i = (x0).i + (x2).i;                       \
  double _t1r = (x0).r - (x2).r, _t1i = (x0).i - (x2).i;                       \
  double _t3r = (x1).r + (x3).r, _t3i = (x1).i + (x3).i;                       \
  double _t4r = (x1).r - (x3).r, _t4i = (x1).i - (x3).i;                       \
  double _r4r = rot * _t4i, _r4i = -rot * _t4r;                                \
  cx2 m0 = {_t2r + _t3r, _t2i + _t3i};                                         \
  cx2 m1 = {_t1r + _r4r, _t1i + _r4i};                                         \
  cx2 m2 = {_t2r - _t3r, _t2i - _t3i};                                         \
  cx2 m3 = {_t1r - _r4r, _t1i - _r4i}

#define NX_C_BFLY5(x0, x1, x2, x3, x4, tw1r, tw1i, tw2r, tw2i)                  \
  double _t1r = (x1).r + (x4).r, _t1i = (x1).i + (x4).i;                       \
  double _t4r = (x1).r - (x4).r, _t4i = (x1).i - (x4).i;                       \
  double _t2r = (x2).r + (x3).r, _t2i = (x2).i + (x3).i;                       \
  double _t3r = (x2).r - (x3).r, _t3i = (x2).i - (x3).i;                       \
  double _ca1r = (x0).r + (tw1r) * _t1r + (tw2r) * _t2r;                       \
  double _ca1i = (x0).i + (tw1r) * _t1i + (tw2r) * _t2i;                       \
  double _cb1i = (tw1i) * _t4r + (tw2i) * _t3r;                                \
  double _cb1r = -((tw1i) * _t4i + (tw2i) * _t3i);                             \
  double _ca2r = (x0).r + (tw2r) * _t1r + (tw1r) * _t2r;                       \
  double _ca2i = (x0).i + (tw2r) * _t1i + (tw1r) * _t2i;                       \
  double _cb2i = (tw2i) * _t4r - (tw1i) * _t3r;                                \
  double _cb2r = -((tw2i) * _t4i - (tw1i) * _t3i);                             \
  cx2 m0 = {(x0).r + _t1r + _t2r, (x0).i + _t1i + _t2i};                       \
  cx2 m1 = {_ca1r + _cb1r, _ca1i + _cb1i};                                     \
  cx2 m4 = {_ca1r - _cb1r, _ca1i - _cb1i};                                     \
  cx2 m2 = {_ca2r + _cb2r, _ca2i + _cb2i};                                     \
  cx2 m3 = {_ca2r - _cb2r, _ca2i - _cb2i}

#define NX_C_BFLY7(x0, x1, x2, x3, x4, x5, x6)                                 \
  double _t1r = (x0).r, _t1i = (x0).i;                                        \
  double _t2r = (x1).r + (x6).r, _t2i = (x1).i + (x6).i;                       \
  double _t7r = (x1).r - (x6).r, _t7i = (x1).i - (x6).i;                       \
  double _t3r = (x2).r + (x5).r, _t3i = (x2).i + (x5).i;                       \
  double _t6r = (x2).r - (x5).r, _t6i = (x2).i - (x5).i;                       \
  double _t4r = (x3).r + (x4).r, _t4i = (x3).i + (x4).i;                       \
  double _t5r = (x3).r - (x4).r, _t5i = (x3).i - (x4).i;                       \
  double _ca1r = _t1r + tw1r * _t2r + tw2r * _t3r + tw3r * _t4r;               \
  double _ca1i = _t1i + tw1r * _t2i + tw2r * _t3i + tw3r * _t4i;               \
  double _cb1i = tw1i * _t7r + tw2i * _t6r + tw3i * _t5r;                      \
  double _cb1r = -(tw1i * _t7i + tw2i * _t6i + tw3i * _t5i);                   \
  double _ca2r = _t1r + tw2r * _t2r + tw3r * _t3r + tw1r * _t4r;               \
  double _ca2i = _t1i + tw2r * _t2i + tw3r * _t3i + tw1r * _t4i;               \
  double _cb2i = tw2i * _t7r - tw3i * _t6r - tw1i * _t5r;                      \
  double _cb2r = -(tw2i * _t7i - tw3i * _t6i - tw1i * _t5i);                   \
  double _ca3r = _t1r + tw3r * _t2r + tw1r * _t3r + tw2r * _t4r;               \
  double _ca3i = _t1i + tw3r * _t2i + tw1r * _t3i + tw2r * _t4i;               \
  double _cb3i = tw3i * _t7r - tw1i * _t6r + tw2i * _t5r;                      \
  double _cb3r = -(tw3i * _t7i - tw1i * _t6i + tw2i * _t5i);                   \
  cx2 m0 = {_t1r + _t2r + _t3r + _t4r, _t1i + _t2i + _t3i + _t4i};             \
  cx2 m1 = {_ca1r + _cb1r, _ca1i + _cb1i}, m6 = {_ca1r - _cb1r, _ca1i - _cb1i};\
  cx2 m2 = {_ca2r + _cb2r, _ca2i + _cb2i}, m5 = {_ca2r - _cb2r, _ca2i - _cb2i};\
  cx2 m3 = {_ca3r + _cb3r, _ca3i + _cb3i}, m4 = {_ca3r - _cb3r, _ca3i - _cb3i}

#define NX_C_BFLY8(x0, x1, x2, x3, x4, x5, x6, x7)                              \
  double _a1r = (x1).r + (x5).r, _a1i = (x1).i + (x5).i;                       \
  double _a5r = (x1).r - (x5).r, _a5i = (x1).i - (x5).i;                       \
  double _a3r = (x3).r + (x7).r, _a3i = (x3).i + (x7).i;                       \
  double _a7r = (x3).r - (x7).r, _a7i = (x3).i - (x7).i;                       \
  double _tr = rot * _a7i;                                                     \
  _a7i = -rot * _a7r;                                                          \
  _a7r = _tr;                                                                  \
  double _s1r = _a1r + _a3r, _s1i = _a1i + _a3i;                               \
  double _d13r = _a1r - _a3r, _d13i = _a1i - _a3i;                             \
  _a1r = _s1r;                                                                 \
  _a1i = _s1i;                                                                 \
  _a3r = rot * _d13i;                                                          \
  _a3i = -rot * _d13r;                                                         \
  double _s5r = _a5r + _a7r, _s5i = _a5i + _a7i;                               \
  double _d57r = _a5r - _a7r, _d57i = _a5i - _a7i;                             \
  _a5r = h * (_s5r + rot * _s5i);                                              \
  _a5i = h * (_s5i - rot * _s5r);                                              \
  _a7r = h * (rot * _d57i - _d57r);                                            \
  _a7i = h * (-rot * _d57r - _d57i);                                           \
  double _a0r = (x0).r + (x4).r, _a0i = (x0).i + (x4).i;                       \
  double _a4r = (x0).r - (x4).r, _a4i = (x0).i - (x4).i;                       \
  double _a2r = (x2).r + (x6).r, _a2i = (x2).i + (x6).i;                       \
  double _a6r = (x2).r - (x6).r, _a6i = (x2).i - (x6).i;                       \
  double _s02r = _a0r + _a2r, _s02i = _a0i + _a2i;                             \
  double _d02r = _a0r - _a2r, _d02i = _a0i - _a2i;                             \
  double _r6r = rot * _a6i, _r6i = -rot * _a6r;                                \
  double _s46r = _a4r + _r6r, _s46i = _a4i + _r6i;                             \
  double _d46r = _a4r - _r6r, _d46i = _a4i - _r6i;                             \
  cx2 m0 = {_s02r + _a1r, _s02i + _a1i}, m4 = {_s02r - _a1r, _s02i - _a1i};    \
  cx2 m2 = {_d02r + _a3r, _d02i + _a3i}, m6 = {_d02r - _a3r, _d02i - _a3i};    \
  cx2 m1 = {_s46r + _a5r, _s46i + _a5i}, m5 = {_s46r - _a5r, _s46i - _a5i};    \
  cx2 m3 = {_d46r + _a7r, _d46i + _a7i}, m7 = {_d46r - _a7r, _d46i - _a7i}

/* Apply the sign-folded twiddle w to the pre-twiddle output m, store to dst. */
#define NX_C_TWST(dst, m, w)                                                    \
  (dst).r = (m).r * (w).r - (m).i * (w).i;                                     \
  (dst).i = (m).r * (w).i + (m).i * (w).r

static void pass2(int64_t ido, int64_t l1, const cx2 *restrict cc,
                  cx2 *restrict ch, const cx2 *restrict wa) {
  int64_t cdl = ido * l1; /* CH output b-stride */
  if (ido == 1) {
    for (int64_t k = 0; k < l1; k++) {
      cx2 x0 = cc[2 * k], x1 = cc[2 * k + 1];
      ch[k].r = x0.r + x1.r;
      ch[k].i = x0.i + x1.i;
      ch[k + cdl].r = x0.r - x1.r;
      ch[k + cdl].i = x0.i - x1.i;
    }
    return;
  }
  for (int64_t k = 0; k < l1; k++) {
    const cx2 *c0 = cc + ido * 2 * k;
    cx2 *h0 = ch + ido * k;
    { /* a = 0: no twiddle */
      cx2 x0 = c0[0], x1 = c0[ido];
      h0[0].r = x0.r + x1.r;
      h0[0].i = x0.i + x1.i;
      h0[cdl].r = x0.r - x1.r;
      h0[cdl].i = x0.i - x1.i;
    }
    for (int64_t a = 1; a < ido; a++) {
      cx2 x0 = c0[a], x1 = c0[ido + a];
      double dr = x0.r - x1.r, di = x0.i - x1.i;
      cx2 w = wa[a - 1];
      h0[a].r = x0.r + x1.r;
      h0[a].i = x0.i + x1.i;
      h0[cdl + a].r = dr * w.r - di * w.i;
      h0[cdl + a].i = dr * w.i + di * w.r;
    }
  }
}

static void pass3(int64_t ido, int64_t l1, const cx2 *restrict cc,
                  cx2 *restrict ch, const cx2 *restrict wa, int sign) {
  int64_t cdl = ido * l1, ido1 = ido - 1;
  const double tw1r = -0.5;
  const double tw1i = (double)sign * 0.86602540378443864676; /* sin(2π/3) */
  if (ido == 1) {
    for (int64_t k = 0; k < l1; k++) {
      const cx2 *c0 = cc + 3 * k;
      NX_C_BFLY3(c0[0], c0[1], c0[2], tw1r, tw1i);
      ch[k] = m0;
      ch[k + cdl] = m1;
      ch[k + 2 * cdl] = m2;
    }
    return;
  }
  for (int64_t k = 0; k < l1; k++) {
    const cx2 *c0 = cc + ido * 3 * k;
    cx2 *h0 = ch + ido * k;
    {
      NX_C_BFLY3(c0[0], c0[ido], c0[2 * ido], tw1r, tw1i);
      h0[0] = m0;
      h0[cdl] = m1;
      h0[2 * cdl] = m2;
    }
    for (int64_t a = 1; a < ido; a++) {
      NX_C_BFLY3(c0[a], c0[ido + a], c0[2 * ido + a], tw1r, tw1i);
      cx2 w0 = wa[a - 1], w1 = wa[a - 1 + ido1];
      h0[a] = m0;
      NX_C_TWST(h0[cdl + a], m1, w0);
      NX_C_TWST(h0[2 * cdl + a], m2, w1);
    }
  }
}

static void pass4(int64_t ido, int64_t l1, const cx2 *restrict cc,
                  cx2 *restrict ch, const cx2 *restrict wa, int sign) {
  int64_t cdl = ido * l1, ido1 = ido - 1;
  double rot = -(double)sign; /* ·(sign·i): (r,im) -> (rot·im, -rot·r) */
  if (ido == 1) {
    for (int64_t k = 0; k < l1; k++) {
      const cx2 *c0 = cc + 4 * k;
      NX_C_BFLY4(c0[0], c0[1], c0[2], c0[3]);
      ch[k] = m0;
      ch[k + cdl] = m1;
      ch[k + 2 * cdl] = m2;
      ch[k + 3 * cdl] = m3;
    }
    return;
  }
  for (int64_t k = 0; k < l1; k++) {
    const cx2 *c0 = cc + ido * 4 * k;
    cx2 *h0 = ch + ido * k;
    {
      NX_C_BFLY4(c0[0], c0[ido], c0[2 * ido], c0[3 * ido]);
      h0[0] = m0;
      h0[cdl] = m1;
      h0[2 * cdl] = m2;
      h0[3 * cdl] = m3;
    }
    for (int64_t a = 1; a < ido; a++) {
      NX_C_BFLY4(c0[a], c0[ido + a], c0[2 * ido + a], c0[3 * ido + a]);
      cx2 w0 = wa[a - 1], w1 = wa[a - 1 + ido1], w2 = wa[a - 1 + 2 * ido1];
      h0[a] = m0;
      NX_C_TWST(h0[cdl + a], m1, w0);
      NX_C_TWST(h0[2 * cdl + a], m2, w1);
      NX_C_TWST(h0[3 * cdl + a], m3, w2);
    }
  }
}

static void pass5(int64_t ido, int64_t l1, const cx2 *restrict cc,
                  cx2 *restrict ch, const cx2 *restrict wa, int sign) {
  int64_t cdl = ido * l1, ido1 = ido - 1;
  const double tw1r = 0.30901699437494742410;  /* cos(2π/5) */
  const double tw2r = -0.80901699437494742410; /* cos(4π/5) */
  const double tw1i = (double)sign * 0.95105651629515357212; /* sin(2π/5) */
  const double tw2i = (double)sign * 0.58778525229247312917; /* sin(4π/5) */
  if (ido == 1) {
    for (int64_t k = 0; k < l1; k++) {
      const cx2 *c0 = cc + 5 * k;
      NX_C_BFLY5(c0[0], c0[1], c0[2], c0[3], c0[4], tw1r, tw1i, tw2r, tw2i);
      ch[k] = m0;
      ch[k + cdl] = m1;
      ch[k + 2 * cdl] = m2;
      ch[k + 3 * cdl] = m3;
      ch[k + 4 * cdl] = m4;
    }
    return;
  }
  for (int64_t k = 0; k < l1; k++) {
    const cx2 *c0 = cc + ido * 5 * k;
    cx2 *h0 = ch + ido * k;
    {
      NX_C_BFLY5(c0[0], c0[ido], c0[2 * ido], c0[3 * ido], c0[4 * ido], tw1r,
                tw1i, tw2r, tw2i);
      h0[0] = m0;
      h0[cdl] = m1;
      h0[2 * cdl] = m2;
      h0[3 * cdl] = m3;
      h0[4 * cdl] = m4;
    }
    for (int64_t a = 1; a < ido; a++) {
      NX_C_BFLY5(c0[a], c0[ido + a], c0[2 * ido + a], c0[3 * ido + a],
                c0[4 * ido + a], tw1r, tw1i, tw2r, tw2i);
      cx2 w0 = wa[a - 1], w1 = wa[a - 1 + ido1], w2 = wa[a - 1 + 2 * ido1];
      cx2 w3 = wa[a - 1 + 3 * ido1];
      h0[a] = m0;
      NX_C_TWST(h0[cdl + a], m1, w0);
      NX_C_TWST(h0[2 * cdl + a], m2, w1);
      NX_C_TWST(h0[3 * cdl + a], m3, w2);
      NX_C_TWST(h0[4 * cdl + a], m4, w3);
    }
  }
}

static void pass7(int64_t ido, int64_t l1, const cx2 *restrict cc,
                  cx2 *restrict ch, const cx2 *restrict wa, int sign) {
  int64_t cdl = ido * l1, ido1 = ido - 1;
  const double tw1r = 0.62348980185873353053;  /* cos(2π/7) */
  const double tw2r = -0.22252093395631440429; /* cos(4π/7) */
  const double tw3r = -0.90096886790241912624; /* cos(6π/7) */
  const double tw1i = (double)sign * 0.78183148246802980871; /* sin(2π/7) */
  const double tw2i = (double)sign * 0.97492791218182360702; /* sin(4π/7) */
  const double tw3i = (double)sign * 0.43388373911755812048; /* sin(6π/7) */
  if (ido == 1) {
    for (int64_t k = 0; k < l1; k++) {
      const cx2 *c0 = cc + 7 * k;
      NX_C_BFLY7(c0[0], c0[1], c0[2], c0[3], c0[4], c0[5], c0[6]);
      ch[k] = m0;
      ch[k + cdl] = m1;
      ch[k + 2 * cdl] = m2;
      ch[k + 3 * cdl] = m3;
      ch[k + 4 * cdl] = m4;
      ch[k + 5 * cdl] = m5;
      ch[k + 6 * cdl] = m6;
    }
    return;
  }
  for (int64_t k = 0; k < l1; k++) {
    const cx2 *c0 = cc + ido * 7 * k;
    cx2 *h0 = ch + ido * k;
    {
      NX_C_BFLY7(c0[0], c0[ido], c0[2 * ido], c0[3 * ido], c0[4 * ido],
                c0[5 * ido], c0[6 * ido]);
      h0[0] = m0;
      h0[cdl] = m1;
      h0[2 * cdl] = m2;
      h0[3 * cdl] = m3;
      h0[4 * cdl] = m4;
      h0[5 * cdl] = m5;
      h0[6 * cdl] = m6;
    }
    for (int64_t a = 1; a < ido; a++) {
      NX_C_BFLY7(c0[a], c0[ido + a], c0[2 * ido + a], c0[3 * ido + a],
                c0[4 * ido + a], c0[5 * ido + a], c0[6 * ido + a]);
      cx2 w0 = wa[a - 1], w1 = wa[a - 1 + ido1], w2 = wa[a - 1 + 2 * ido1];
      cx2 w3 = wa[a - 1 + 3 * ido1], w4 = wa[a - 1 + 4 * ido1];
      cx2 w5 = wa[a - 1 + 5 * ido1];
      h0[a] = m0;
      NX_C_TWST(h0[cdl + a], m1, w0);
      NX_C_TWST(h0[2 * cdl + a], m2, w1);
      NX_C_TWST(h0[3 * cdl + a], m3, w2);
      NX_C_TWST(h0[4 * cdl + a], m4, w3);
      NX_C_TWST(h0[5 * cdl + a], m5, w4);
      NX_C_TWST(h0[6 * cdl + a], m6, w5);
    }
  }
}

static void pass8(int64_t ido, int64_t l1, const cx2 *restrict cc,
                  cx2 *restrict ch, const cx2 *restrict wa, int sign) {
  int64_t cdl = ido * l1, ido1 = ido - 1;
  double rot = -(double)sign; /* ROT90:  (r,im) -> (rot·im, -rot·r) */
  double h = NX_C_SQRTH;
  if (ido == 1) {
    /* Final stage: no twiddle and no inner loop. A dedicated loop over k lets
       clang vectorize across k (unit-stride output ch[k+b·l1]); folded into the
       ido>1 path the empty a-loop keeps it scalar. */
    for (int64_t k = 0; k < l1; k++) {
      const cx2 *c0 = cc + 8 * k;
      NX_C_BFLY8(c0[0], c0[1], c0[2], c0[3], c0[4], c0[5], c0[6], c0[7]);
      ch[k] = m0;
      ch[k + cdl] = m1;
      ch[k + 2 * cdl] = m2;
      ch[k + 3 * cdl] = m3;
      ch[k + 4 * cdl] = m4;
      ch[k + 5 * cdl] = m5;
      ch[k + 6 * cdl] = m6;
      ch[k + 7 * cdl] = m7;
    }
    return;
  }
  for (int64_t k = 0; k < l1; k++) {
    const cx2 *c0 = cc + ido * 8 * k;
    cx2 *h0 = ch + ido * k;
    {
      NX_C_BFLY8(c0[0], c0[ido], c0[2 * ido], c0[3 * ido], c0[4 * ido],
                c0[5 * ido], c0[6 * ido], c0[7 * ido]);
      h0[0] = m0;
      h0[cdl] = m1;
      h0[2 * cdl] = m2;
      h0[3 * cdl] = m3;
      h0[4 * cdl] = m4;
      h0[5 * cdl] = m5;
      h0[6 * cdl] = m6;
      h0[7 * cdl] = m7;
    }
    for (int64_t a = 1; a < ido; a++) {
      NX_C_BFLY8(c0[a], c0[ido + a], c0[2 * ido + a], c0[3 * ido + a],
                c0[4 * ido + a], c0[5 * ido + a], c0[6 * ido + a],
                c0[7 * ido + a]);
      cx2 w0 = wa[a - 1], w1 = wa[a - 1 + ido1], w2 = wa[a - 1 + 2 * ido1];
      cx2 w3 = wa[a - 1 + 3 * ido1], w4 = wa[a - 1 + 4 * ido1];
      cx2 w5 = wa[a - 1 + 5 * ido1], w6 = wa[a - 1 + 6 * ido1];
      h0[a] = m0;
      NX_C_TWST(h0[cdl + a], m1, w0);
      NX_C_TWST(h0[2 * cdl + a], m2, w1);
      NX_C_TWST(h0[3 * cdl + a], m3, w2);
      NX_C_TWST(h0[4 * cdl + a], m4, w3);
      NX_C_TWST(h0[5 * cdl + a], m5, w4);
      NX_C_TWST(h0[6 * cdl + a], m6, w5);
      NX_C_TWST(h0[7 * cdl + a], m7, w6);
    }
  }
}

/* Generic odd-prime radix (11, 13): a direct ip-point DFT per butterfly using
   the stage's ip roots `root[t] = ω_ip^{sign·t}`, then the inter-block twiddle on
   lanes b≥1. O(ip²) per butterfly — used only for the small odd primes the plan
   admits natively (largest_prime_factor ≤ 13); larger primes ride Bluestein. Same
   out-of-place ping-pong contract as the dedicated passes. */
static void passg(int64_t ido, int64_t ip, int64_t l1, const cx2 *restrict cc,
                  cx2 *restrict ch, const cx2 *restrict wa,
                  const cx2 *restrict root) {
  int64_t cdl = ido * l1, ido1 = ido - 1;
  for (int64_t k = 0; k < l1; k++) {
    const cx2 *c0 = cc + ido * ip * k;
    cx2 *h0 = ch + ido * k;
    for (int64_t a = 0; a < ido; a++) {
      for (int64_t b = 0; b < ip; b++) {
        double accr = 0.0, acci = 0.0;
        for (int64_t j = 0; j < ip; j++) {
          cx2 w = root[(j * b) % ip];
          cx2 xj = c0[a + ido * j];
          accr += xj.r * w.r - xj.i * w.i;
          acci += xj.r * w.i + xj.i * w.r;
        }
        cx2 m = {accr, acci};
        if (a == 0 || b == 0)
          h0[cdl * b + a] = m;
        else {
          cx2 tw = wa[a - 1 + (b - 1) * ido1];
          NX_C_TWST(h0[cdl * b + a], m, tw);
        }
      }
    }
  }
}

/* ── Plan: native mixed-radix, or Bluestein for a non-smooth length ────────*/
typedef enum { FFT_NATIVE, FFT_BLUESTEIN } fft_kind;

typedef struct {
  int radix;   /* 8, 4, 2, 3, 5, 7, or a generic odd prime (11, 13) */
  int64_t l1;  /* number of blocks entering this stage */
  int64_t ido; /* inner sub-transform size = n / (l1·radix) */
  cx2 *tw;     /* sign-folded twiddles, (radix-1)·(ido-1); NULL when ido==1 */
  cx2 *root;   /* ip sign-folded roots for the generic pass; NULL otherwise */
} fft_stage;

typedef struct fft_plan {
  int64_t n;
  int sign;
  fft_kind kind;
  int64_t work_cx; /* scratch cx2 exec needs beyond the size-n line buffer */
  /* NATIVE */
  int nstages;
  fft_stage *stages;
  /* BLUESTEIN */
  int64_t m;                  /* padded power-of-two length (>= 2n-1) */
  cx2 *chirp;                 /* chirp[j] = exp(sign·iπ j²/n), j in [0,n) */
  cx2 *bfilter;               /* native FFT of the b-sequence, length m */
  struct fft_plan *mplan_fwd; /* native size-m forward plan (sign -1) */
  struct fft_plan *mplan_inv; /* native size-m inverse plan (sign +1) */
  struct fft_plan *next;
} fft_plan;

static fft_plan *g_plans = NULL;
static pthread_mutex_t g_plan_mtx = PTHREAD_MUTEX_INITIALIZER;

/* Only for build-failure and the never-taken free path; cached plans live until
   exit and are never freed. Recurses into the Bluestein sub-plans. */
static void plan_free(fft_plan *p) {
  if (!p) return;
  if (p->stages) {
    for (int i = 0; i < p->nstages; i++) {
      free(p->stages[i].tw);
      free(p->stages[i].root);
    }
    free(p->stages);
  }
  free(p->chirp);
  free(p->bfilter);
  plan_free(p->mplan_fwd);
  plan_free(p->mplan_inv);
  free(p);
}

/* Transform the length-n line `x` in place: ping-pong the self-sorting stages
   between it and the scratch `w` (both length n), copying back if an odd stage
   count leaves the result in the scratch. Never called for n <= 1. */
static void native_exec(const fft_plan *p, cx2 *x, cx2 *w) {
  int64_t n = p->n;
  cx2 *a = x, *b = w;
  for (int i = 0; i < p->nstages; i++) {
    const fft_stage *st = &p->stages[i];
    switch (st->radix) {
      case 8: pass8(st->ido, st->l1, a, b, st->tw, p->sign); break;
      case 4: pass4(st->ido, st->l1, a, b, st->tw, p->sign); break;
      case 2: pass2(st->ido, st->l1, a, b, st->tw); break;
      case 3: pass3(st->ido, st->l1, a, b, st->tw, p->sign); break;
      case 5: pass5(st->ido, st->l1, a, b, st->tw, p->sign); break;
      case 7: pass7(st->ido, st->l1, a, b, st->tw, p->sign); break;
      default: passg(st->ido, st->radix, st->l1, a, b, st->tw, st->root); break;
    }
    cx2 *t = a; a = b; b = t;
  }
  if (a != x) memcpy(x, a, (size_t)n * sizeof(cx2));
}

/* Native mixed-radix plan (dedicated radix 8/4/2/3/5/7 + a generic pass for
   larger odd primes). The caller (plan_build) only routes here for lengths whose
   largest prime factor is small; n <= 1 yields a trivial identity plan. Returns
   NULL only on allocation failure. */
static fft_plan *build_native(int64_t n, int sign) {
  fft_plan *p = calloc(1, sizeof(fft_plan));
  if (!p) return NULL;
  p->n = n;
  p->sign = sign;
  p->kind = FFT_NATIVE;
  if (n <= 1) return p;

  /* Schedule: radix-8s, then radix-4s, then a leftover radix-2 moved to the
     front, then radix-3s, 5s, 7s, then any remaining odd prime factors (the
     generic pass handles 11/13; the caller's gate keeps larger primes on
     Bluestein). Any order transforms correctly (self-sorting adapts); this one
     is tuned. Max stages = log2(n) < 64. */
  int radices[64];
  int s = 0;
  int64_t r = n;
  while (r % 8 == 0) { radices[s++] = 8; r /= 8; }
  while (r % 4 == 0) { radices[s++] = 4; r /= 4; }
  if (r % 2 == 0) {
    r /= 2;
    radices[s++] = 2;
    int t = radices[0]; radices[0] = radices[s - 1]; radices[s - 1] = t;
  }
  while (r % 3 == 0) { radices[s++] = 3; r /= 3; }
  while (r % 5 == 0) { radices[s++] = 5; r /= 5; }
  while (r % 7 == 0) { radices[s++] = 7; r /= 7; }
  for (int64_t d = 11; d * d <= r; d += 2)
    while (r % d == 0) { radices[s++] = (int)d; r /= d; }
  if (r > 1) radices[s++] = (int)r;

  p->nstages = s;
  p->work_cx = n; /* one ping-pong scratch line, adjacent (main junction unpadded) */
  p->stages = calloc((size_t)s, sizeof(fft_stage));
  if (!p->stages) { plan_free(p); return NULL; }

  int64_t l1 = 1;
  for (int i = 0; i < s; i++) {
    int ip = radices[i];
    int64_t ido = n / (l1 * ip);
    p->stages[i].radix = ip;
    p->stages[i].l1 = l1;
    p->stages[i].ido = ido;
    int64_t sz = (int64_t)(ip - 1) * (ido - 1);
    if (sz > 0) {
      p->stages[i].tw = malloc((size_t)sz * sizeof(cx2));
      if (!p->stages[i].tw) { plan_free(p); return NULL; }
      for (int b = 1; b < ip; b++)
        for (int64_t a = 1; a < ido; a++) {
          /* ω_n^{sign·b·l1·a}, index reduced mod n for a small-angle argument */
          int64_t q = (int64_t)(((__int128)b * l1 * a) % n);
          double ang = (double)sign * 2.0 * NX_C_PI * (double)q / (double)n;
          int64_t off = (a - 1) + (int64_t)(b - 1) * (ido - 1);
          p->stages[i].tw[off].r = cos(ang);
          p->stages[i].tw[off].i = sin(ang);
        }
    }
    if (ip != 2 && ip != 3 && ip != 4 && ip != 5 && ip != 7 && ip != 8) {
      /* generic pass: ip sign-folded roots ω_ip^{sign·t} */
      p->stages[i].root = malloc((size_t)ip * sizeof(cx2));
      if (!p->stages[i].root) { plan_free(p); return NULL; }
      for (int t = 0; t < ip; t++) {
        double ang = (double)sign * 2.0 * NX_C_PI * (double)t / (double)ip;
        p->stages[i].root[t].r = cos(ang);
        p->stages[i].root[t].i = sin(ang);
      }
    }
    l1 *= ip;
  }
  return p;
}

/* chirp phase exp(sign·iπ j²/n); j² reduced mod 2n (period of the phase in j²)
   in 128-bit to avoid overflow for large n. */
static cx2 chirp_at(int64_t j, int64_t n, int sign) {
  int64_t r = (int64_t)(((__int128)j * j) % (2 * n));
  double ang = (double)sign * NX_C_PI * (double)r / (double)n;
  cx2 c = {cos(ang), sin(ang)};
  return c;
}

/* Bluestein chirp-z plan for a non-smooth length. The two length-m FFTs (m a
   power of two, hence smooth) ride native sub-plans, one per sign. */
static fft_plan *build_bluestein(int64_t n, int sign) {
  fft_plan *p = calloc(1, sizeof(fft_plan));
  if (!p) return NULL;
  p->n = n;
  p->sign = sign;
  p->kind = FFT_BLUESTEIN;
  int64_t m = 1;
  while (m < 2 * n - 1) m <<= 1;
  p->m = m;
  p->work_cx = 2 * (m + FFT_PAD); /* a-line + de-aliased ping-pong scratch */
  p->chirp = malloc((size_t)n * sizeof(cx2));
  p->bfilter = malloc((size_t)m * sizeof(cx2));
  p->mplan_fwd = build_native(m, -1);
  p->mplan_inv = build_native(m, 1);
  if (!p->chirp || !p->bfilter || !p->mplan_fwd || !p->mplan_inv) {
    plan_free(p);
    return NULL;
  }
  for (int64_t j = 0; j < n; j++) p->chirp[j] = chirp_at(j, n, sign);
  /* b[j] = conj(chirp[j]) with circular symmetry b[m-j]=b[j]; bfilter = FFT(b). */
  cx2 *bseq = calloc((size_t)m, sizeof(cx2));
  cx2 *scr = malloc((size_t)m * sizeof(cx2));
  if (!bseq || !scr) {
    free(bseq); free(scr);
    plan_free(p);
    return NULL;
  }
  bseq[0].r = p->chirp[0].r;
  bseq[0].i = -p->chirp[0].i;
  for (int64_t j = 1; j < n; j++) {
    cx2 bj = {p->chirp[j].r, -p->chirp[j].i};
    bseq[j] = bj;
    bseq[m - j] = bj;
  }
  native_exec(p->mplan_fwd, bseq, scr);
  memcpy(p->bfilter, bseq, (size_t)m * sizeof(cx2));
  free(bseq); free(scr);
  return p;
}

/* Transform the length-n line `x` in place; `work` is >= plan->work_cx cx2. */
static void bluestein_exec(const fft_plan *p, cx2 *x, cx2 *work) {
  int64_t n = p->n, m = p->m;
  cx2 *a = work, *scr = work + (m + FFT_PAD);
  for (int64_t j = 0; j < n; j++) {
    double vr = x[j].r, vi = x[j].i, cr = p->chirp[j].r, ci = p->chirp[j].i;
    a[j].r = vr * cr - vi * ci;
    a[j].i = vr * ci + vi * cr;
  }
  for (int64_t j = n; j < m; j++) { a[j].r = 0.0; a[j].i = 0.0; }
  native_exec(p->mplan_fwd, a, scr);
  for (int64_t k = 0; k < m; k++) {
    double vr = a[k].r, vi = a[k].i, br = p->bfilter[k].r, bi = p->bfilter[k].i;
    a[k].r = vr * br - vi * bi;
    a[k].i = vr * bi + vi * br;
  }
  native_exec(p->mplan_inv, a, scr);
  double inv = 1.0 / (double)m;
  for (int64_t k = 0; k < n; k++) {
    double vr = a[k].r * inv, vi = a[k].i * inv;
    double cr = p->chirp[k].r, ci = p->chirp[k].i;
    x[k].r = vr * cr - vi * ci;
    x[k].i = vr * ci + vi * cr;
  }
}

static int64_t largest_prime_factor(int64_t n) {
  int64_t res = 1;
  while (n % 2 == 0) { res = 2; n /= 2; }
  for (int64_t x = 3; x * x <= n; x += 2)
    while (n % x == 0) { res = x; n /= x; }
  if (n > 1) res = n;
  return res;
}

/* Native mixed-radix when every prime factor is small — the dedicated radices
   2/3/5/7 plus the generic O(ip²) pass for 11/13. A larger prime factor rides
   Bluestein, whose O(m log m) beats the generic butterfly's O(n·ip). n <= 1 is a
   trivial native identity. */
static fft_plan *plan_build(int64_t n, int sign) {
  if (n <= 1 || largest_prime_factor(n) <= 13)
    return build_native(n, sign);
  return build_bluestein(n, sign);
}

/* Lock held by the caller's CAMLprim; the cache mutex serializes cache ops
   across domains. Grow-only: a returned pointer is never freed. NULL on OOM. */
static const fft_plan *plan_get(int64_t n, int sign) {
  pthread_mutex_lock(&g_plan_mtx);
  for (fft_plan *p = g_plans; p; p = p->next)
    if (p->n == n && p->sign == sign) {
      pthread_mutex_unlock(&g_plan_mtx);
      return p;
    }
  fft_plan *p = plan_build(n, sign);
  if (p) {
    p->next = g_plans;
    g_plans = p;
  }
  pthread_mutex_unlock(&g_plan_mtx);
  return p;
}

/* Transform the length-n line `x` in place; `work` is >= plan->work_cx cx2. */
static void plan_exec(const fft_plan *p, cx2 *x, cx2 *work) {
  if (p->n <= 1) return; /* identity: length-1 line already holds its transform */
  if (p->kind == FFT_NATIVE)
    native_exec(p, x, work);
  else
    bluestein_exec(p, x, work);
}

/* ── Gather / scatter one strided line, converting storage <-> cx2 ─────────
   Complex operands: c32 (float _Complex) upcast, c64 (double _Complex) direct.
   Real operands: f32/f64 as the real part on gather (imag 0); real part on
   scatter. Interleaved storage is [re,im] contiguous per element (C99). */
static void gather_cx(nx_c_dtype dt, const char *base, int64_t stride_elems,
                      int64_t esz, int64_t n, cx2 *dst) {
  for (int64_t k = 0; k < n; k++) {
    const char *p = base + k * stride_elems * esz;
    if (dt == NX_C_DTYPE_c64) {
      const double *z = (const double *)p;
      dst[k].r = z[0];
      dst[k].i = z[1];
    } else {
      const float *z = (const float *)p;
      dst[k].r = (double)z[0];
      dst[k].i = (double)z[1];
    }
  }
}
static void scatter_cx(nx_c_dtype dt, char *base, int64_t stride_elems,
                       int64_t esz, int64_t n, const cx2 *src) {
  for (int64_t k = 0; k < n; k++) {
    char *p = base + k * stride_elems * esz;
    if (dt == NX_C_DTYPE_c64) {
      double *z = (double *)p;
      z[0] = src[k].r;
      z[1] = src[k].i;
    } else {
      float *z = (float *)p;
      z[0] = (float)src[k].r;
      z[1] = (float)src[k].i;
    }
  }
}
static void gather_real(nx_c_dtype dt, const char *base, int64_t stride_elems,
                        int64_t esz, int64_t n, cx2 *dst) {
  for (int64_t k = 0; k < n; k++) {
    const char *p = base + k * stride_elems * esz;
    dst[k].r = (dt == NX_C_DTYPE_f64) ? *(const double *)p : (double)*(const float *)p;
    dst[k].i = 0.0;
  }
}
static void scatter_real(nx_c_dtype dt, char *base, int64_t stride_elems,
                         int64_t esz, int64_t n, const cx2 *src) {
  for (int64_t k = 0; k < n; k++) {
    char *p = base + k * stride_elems * esz;
    if (dt == NX_C_DTYPE_f64)
      *(double *)p = src[k].r;
    else
      *(float *)p = (float)src[k].r;
  }
}

/* Element offset of the line at flat index L along `axis` (odometer over the
   other dims); same decode for src and dst so they pair up. */
static int64_t line_base(const nx_c_ndarray *a, int axis, int64_t L) {
  int64_t base = a->offset;
  int64_t rem = L;
  for (int d = a->ndim - 1; d >= 0; d--) {
    if (d == axis) continue;
    int64_t c = rem % a->shape[d];
    rem /= a->shape[d];
    base += c * a->strides[d];
  }
  return base;
}

/* ── One transform axis over all lines, pooled ────────────────────────────
   Reads length-n lines from `src` along `axis`, transforms, writes to `dst`
   along `axis` (src==dst ⇒ in place, decoupled through scratch). Per worker the
   scratch is a line buffer (n_in cx2) followed by plan->work_cx of transform
   scratch. n_in / n_out are the line lengths (equal for fft/ifft; differ for the
   rfft/irfft last-axis packing, handled by the caller). */
typedef struct {
  nx_c_dtype src_dt, dst_dt;
  const nx_c_ndarray *src;
  const nx_c_ndarray *dst;
  int axis;
  int64_t n_in, n_out;
  int64_t src_esz, dst_esz;
  int src_real, dst_real;
  const fft_plan *plan;
  char *scratch;
  int64_t slot; /* per-worker cx2 count */
} axis_ctx;

static void axis_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  const axis_ctx *c = (const axis_ctx *)vctx;
  cx2 *base = (cx2 *)(c->scratch + (int64_t)worker * c->slot * (int64_t)sizeof(cx2));
  cx2 *x = base;
  cx2 *work = base + c->n_in;
  int64_t sstride = c->src->strides[c->axis];
  int64_t dstride = c->dst->strides[c->axis];
  for (int64_t L = lo; L < hi; L++) {
    const char *sb = (const char *)c->src->data + line_base(c->src, c->axis, L) * c->src_esz;
    char *db = (char *)c->dst->data + line_base(c->dst, c->axis, L) * c->dst_esz;
    if (c->src_real)
      gather_real(c->src_dt, sb, sstride, c->src_esz, c->n_in, x);
    else
      gather_cx(c->src_dt, sb, sstride, c->src_esz, c->n_in, x);
    plan_exec(c->plan, x, work);
    /* n_out <= n_in: the caller sizes the output line (full for fft/ifft, the
       half-spectrum for rfft's last axis). */
    if (c->dst_real)
      scatter_real(c->dst_dt, db, dstride, c->dst_esz, c->n_out, x);
    else
      scatter_cx(c->dst_dt, db, dstride, c->dst_esz, c->n_out, x);
  }
}

/* Runs one axis pass: builds/gets the plan (lock held), sizes per-worker scratch
   (a line + plan scratch), releases + pools over the lines via nx_c_parallel_for.
   n_out<=n_in trims the written line. */
static nx_c_status run_axis(nx_c_dtype src_dt, nx_c_dtype dst_dt,
                           const nx_c_ndarray *src, const nx_c_ndarray *dst,
                           int axis, int64_t n_in, int64_t n_out, int sign,
                           int src_real, int dst_real) {
  const fft_plan *plan = plan_get(n_in, sign);
  if (!plan) return NX_C_ERR_ALLOC;

  int64_t lines = 1;
  for (int d = 0; d < dst->ndim; d++)
    if (d != axis) lines *= dst->shape[d];
  if (lines == 0 || n_in == 0) return NX_C_OK;

  int64_t slot = n_in + plan->work_cx; /* cx2 per worker */
  int64_t bytes = lines * n_in * (int64_t)sizeof(cx2);
  int nth = nx_c_threads_for(NX_C_COST_COMPUTE, lines, n_in, bytes);
  if (nth > lines) nth = (int)lines;
  if (nth < 1) nth = 1;

  int64_t slot_bytes = ((slot * (int64_t)sizeof(cx2)) + 63) & ~(int64_t)63;
  char *scratch = aligned_alloc(64, (size_t)slot_bytes * nth);
  if (!scratch) return NX_C_ERR_ALLOC;

  axis_ctx c;
  c.src_dt = src_dt;
  c.dst_dt = dst_dt;
  c.src = src;
  c.dst = dst;
  c.axis = axis;
  c.n_in = n_in;
  c.n_out = n_out;
  c.src_esz = nx_c_elem_size(src_dt);
  c.dst_esz = nx_c_elem_size(dst_dt);
  c.src_real = src_real;
  c.dst_real = dst_real;
  c.plan = plan;
  c.scratch = scratch;
  c.slot = slot_bytes / (int64_t)sizeof(cx2);
  nx_c_parallel_for(nth, lines, bytes, axis_body, &c, scratch);
  return NX_C_OK;
}

/* ── fft / ifft ───────────────────────────────────────────────────────────
   Complex→complex, same shape. First axis reads `in`, later axes transform
   `out` in place. */
static nx_c_status nx_c_fft_run(const nx_c_ndarray *in, const nx_c_ndarray *out,
                              nx_c_dtype dt, const int *axes, int naxes,
                              int sign) {
  if (naxes == 0) return NX_C_OK;
  for (int ai = 0; ai < naxes; ai++) {
    int axis = axes[ai];
    if (axis < 0 || axis >= in->ndim) return NX_C_ERR_AXIS;
    const nx_c_ndarray *src = (ai == 0) ? in : out;
    int64_t n = out->shape[axis];
    nx_c_status s = run_axis(dt, dt, src, out, axis, n, n, sign, 0, 0);
    if (s != NX_C_OK) return s;
  }
  return NX_C_OK;
}

/* ── rfft ─────────────────────────────────────────────────────────────────
   Real→complex. Last transformed axis: real line → full FFT → keep n/2+1 bins.
   Other transformed axes: full complex FFT of `out` in place. */
static nx_c_status nx_c_rfft_run(const nx_c_ndarray *in, const nx_c_ndarray *out,
                               nx_c_dtype in_dt, nx_c_dtype out_dt,
                               const int *axes, int naxes) {
  if (naxes == 0) return NX_C_OK;
  int last = axes[naxes - 1];
  if (last < 0 || last >= in->ndim) return NX_C_ERR_AXIS;
  int64_t n = in->shape[last];
  int64_t half = n / 2 + 1;
  if (out->shape[last] != half) return NX_C_ERR_SHAPE;
  /* last axis: real in (n) → complex out (half) */
  nx_c_status s = run_axis(in_dt, out_dt, in, out, last, n, half, -1, 1, 0);
  if (s != NX_C_OK) return s;
  /* remaining axes: complex fft on out in place */
  for (int ai = 0; ai < naxes - 1; ai++) {
    int axis = axes[ai];
    if (axis < 0 || axis >= out->ndim) return NX_C_ERR_AXIS;
    int64_t m = out->shape[axis];
    s = run_axis(out_dt, out_dt, out, out, axis, m, m, -1, 0, 0);
    if (s != NX_C_OK) return s;
  }
  return NX_C_OK;
}

/* ── irfft ────────────────────────────────────────────────────────────────
   Complex half-spectrum → real. Other axes ifft first (in a complex temp), then
   the last axis: reconstruct the full length-s spectrum via conjugate symmetry,
   inverse FFT, take the real part. Output length s along `last`. */
typedef struct {
  nx_c_dtype out_dt;
  const nx_c_ndarray *out;
  const nx_c_ndarray *tmp; /* complex temp, in's shape */
  int axis;
  int64_t half, s;
  int64_t out_esz, tmp_esz;
  const fft_plan *plan;
  char *scratch;
  int64_t slot;
} irfft_ctx;

static void irfft_last_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  const irfft_ctx *c = (const irfft_ctx *)vctx;
  cx2 *base = (cx2 *)(c->scratch + (int64_t)worker * c->slot * (int64_t)sizeof(cx2));
  cx2 *g = base;                    /* gathered half-spectrum, length half */
  cx2 *f = g + (c->half + FFT_PAD); /* reconstructed spectrum, length s */
  cx2 *work = f + (c->s + FFT_PAD); /* transform scratch */
  int64_t istride = c->tmp->strides[c->axis];
  int64_t ostride = c->out->strides[c->axis];
  for (int64_t L = lo; L < hi; L++) {
    const char *ib = (const char *)c->tmp->data + line_base(c->tmp, c->axis, L) * c->tmp_esz;
    char *ob = (char *)c->out->data + line_base(c->out, c->axis, L) * c->out_esz;
    gather_cx(NX_C_DTYPE_c64, ib, istride, c->tmp_esz, c->half, g);
    /* An explicit output length may require padding or truncating the supplied
       half-spectrum. Missing frequencies are zero; excess ones were excluded
       when c->half was computed. */
    for (int64_t k = 0; k < c->s; k++) {
      f[k].r = 0.0;
      f[k].i = 0.0;
    }
    for (int64_t k = 0; k < c->half; k++) f[k] = g[k];
    for (int64_t k = 1; k < c->half; k++) {
      int64_t mirror = c->s - k;
      if (mirror != k) {
        f[mirror].r = g[k].r;
        f[mirror].i = -g[k].i;
      }
    }
    plan_exec(c->plan, f, work);
    scatter_real(c->out_dt, ob, ostride, c->out_esz, c->s, f);
  }
}

static nx_c_status nx_c_irfft_run(const nx_c_ndarray *in, const nx_c_ndarray *out,
                                nx_c_dtype in_dt, nx_c_dtype out_dt,
                                const int *axes, int naxes, int64_t s_last) {
  if (naxes == 0) return NX_C_OK;
  int last = axes[naxes - 1];
  if (last < 0 || last >= in->ndim) return NX_C_ERR_AXIS;
  int64_t in_half = in->shape[last];
  int64_t s = (s_last > 0) ? s_last : 2 * (in_half - 1);
  if (out->shape[last] != s) return NX_C_ERR_SHAPE;
  /* Match explicit-length irfft semantics: a short half-spectrum is padded with
     zeros and a long one is truncated to the frequencies representable by s. */
  int64_t needed_half = (s / 2) + 1;
  int64_t half = in_half < needed_half ? in_half : needed_half;

  /* complex temp holding in's shape; ifft the non-last transformed axes there.
     ACCEPTED DEVIATION (leak-only, never corruption): tdata spans the whole
     irfft — the non-last-axis run_axis passes AND the last-axis pool region —
     so it outlives multiple nx_c_parallel_for calls and CANNOT be the primitive's
     free_on_exit (that owns a single region's scratch, freed at that region's
     join). It is freed explicitly on every return path here. The only leak is if
     an async pending action (signal/memprof) raises inside a nx_c_parallel_for
     lock re-acquire and longjmps past these frees; that never corrupts memory,
     and every single-region alternative either double-copies the strided input
     or holds the runtime lock across the transform. */
  nx_c_ndarray tmp = *in;
  int64_t nelem = 1;
  for (int d = 0; d < in->ndim; d++) nelem *= in->shape[d];
  int64_t cesz = (int64_t)sizeof(cx2);
  cx2 *tdata = malloc((size_t)(nelem ? nelem : 1) * (size_t)cesz);
  if (!tdata) return NX_C_ERR_ALLOC;
  tmp.data = tdata;
  tmp.offset = 0;
  { /* contiguous strides for tmp */
    int64_t st = 1;
    for (int d = in->ndim - 1; d >= 0; d--) {
      tmp.strides[d] = st;
      st *= in->shape[d];
    }
  }
  /* Fill tmp (contiguous) from in (strided), converting to double complex. */
  {
    int64_t idx[NX_C_MAX_NDIM];
    for (int d = 0; d < in->ndim; d++) idx[d] = 0;
    for (int64_t f = 0; f < nelem; f++) {
      int64_t ioff = in->offset;
      for (int d = 0; d < in->ndim; d++) ioff += idx[d] * in->strides[d];
      const char *ip = (const char *)in->data + ioff * nx_c_elem_size(in_dt);
      if (in_dt == NX_C_DTYPE_c64) {
        const double *w = (const double *)ip;
        tdata[f].r = w[0];
        tdata[f].i = w[1];
      } else {
        const float *w = (const float *)ip;
        tdata[f].r = (double)w[0];
        tdata[f].i = (double)w[1];
      }
      for (int d = in->ndim - 1; d >= 0; d--) {
        if (++idx[d] < in->shape[d]) break;
        idx[d] = 0;
      }
    }
  }
  /* ifft the non-last transformed axes in tmp (complex, unnormalized) */
  for (int ai = 0; ai < naxes - 1; ai++) {
    int axis = axes[ai];
    if (axis < 0 || axis >= in->ndim) {
      free(tdata);
      return NX_C_ERR_AXIS;
    }
    int64_t m = tmp.shape[axis];
    nx_c_status s2 = run_axis(NX_C_DTYPE_c64, NX_C_DTYPE_c64, &tmp, &tmp, axis, m, m,
                             1, 0, 0);
    if (s2 != NX_C_OK) {
      free(tdata);
      return s2;
    }
  }
  /* last axis: half-spectrum → real length s */
  const fft_plan *plan = plan_get(s, 1);
  if (!plan) {
    free(tdata);
    return NX_C_ERR_ALLOC;
  }
  int64_t lines = 1;
  for (int d = 0; d < out->ndim; d++)
    if (d != last) lines *= out->shape[d];
  if (lines > 0 && s > 0) {
    int64_t slot = (half + FFT_PAD) + (s + FFT_PAD) + plan->work_cx; /* cx2 */
    int64_t slot_bytes = ((slot * cesz) + 63) & ~(int64_t)63;
    int nth = nx_c_threads_for(NX_C_COST_COMPUTE, lines, s, lines * s * cesz);
    if (nth > lines) nth = (int)lines;
    if (nth < 1) nth = 1;
    char *scratch = aligned_alloc(64, (size_t)slot_bytes * nth);
    if (!scratch) {
      free(tdata);
      return NX_C_ERR_ALLOC;
    }
    irfft_ctx c;
    c.out_dt = out_dt;
    c.out = out;
    c.tmp = &tmp;
    c.axis = last;
    c.half = half;
    c.s = s;
    c.out_esz = nx_c_elem_size(out_dt);
    c.tmp_esz = cesz;
    c.plan = plan;
    c.scratch = scratch;
    c.slot = slot_bytes / cesz;
    /* scratch handed to the primitive as free_on_exit — do NOT free it here. */
    nx_c_parallel_for(nth, lines, lines * s * cesz, irfft_last_body, &c, scratch);
  }
  free(tdata);
  return NX_C_OK;
}

/* ── FFI stubs ────────────────────────────────────────────────────────────*/
static NX_C_NORETURN void fft_raise(const char *op, nx_c_status s) {
  if (strcmp(s, NX_C_ERR_AXIS) == 0 || strcmp(s, NX_C_ERR_SHAPE) == 0)
    nx_c_raise_invalid(op, s);
  nx_c_raise(op, s);
}

static int read_axes(value vaxes, int *axes) {
  int n = (int)Wosize_val(vaxes);
  if (n > NX_C_MAX_NDIM) n = NX_C_MAX_NDIM;
  for (int i = 0; i < n; i++) axes[i] = (int)Long_val(Field(vaxes, i));
  return n;
}

static void fft_stub(const char *op, value vout, value vin, value vaxes,
                     int sign) {
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) fft_raise(op, s);
  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  if (dt != NX_C_DTYPE_c32 && dt != NX_C_DTYPE_c64) fft_raise(op, NX_C_ERR_BAD_KIND);
  int axes[NX_C_MAX_NDIM];
  int naxes = read_axes(vaxes, axes);
  s = nx_c_fft_run(&in, &out, dt, axes, naxes, sign);
  if (s != NX_C_OK) fft_raise(op, s);
}

CAMLprim value caml_nx_c_fft(value vout, value vin, value vaxes) {
  CAMLparam3(vout, vin, vaxes);
  fft_stub("fft", vout, vin, vaxes, -1);
  CAMLreturn(Val_unit);
}
CAMLprim value caml_nx_c_ifft(value vout, value vin, value vaxes) {
  CAMLparam3(vout, vin, vaxes);
  fft_stub("ifft", vout, vin, vaxes, 1);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_c_rfft(value vout, value vin, value vaxes) {
  CAMLparam3(vout, vin, vaxes);
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) fft_raise("rfft", s);
  nx_c_dtype in_dt = nx_c_dtype_of_value(vin);
  nx_c_dtype out_dt = nx_c_dtype_of_value(vout);
  if ((in_dt != NX_C_DTYPE_f32 && in_dt != NX_C_DTYPE_f64) ||
      (out_dt != NX_C_DTYPE_c32 && out_dt != NX_C_DTYPE_c64))
    fft_raise("rfft", NX_C_ERR_BAD_KIND);
  int axes[NX_C_MAX_NDIM];
  int naxes = read_axes(vaxes, axes);
  s = nx_c_rfft_run(&in, &out, in_dt, out_dt, axes, naxes);
  if (s != NX_C_OK) fft_raise("rfft", s);
  CAMLreturn(Val_unit);
}

/* vs: an int array of output sizes along the transformed axes, or empty to
   infer 2*(half-1) from the last-axis input extent. */
CAMLprim value caml_nx_c_irfft(value vout, value vin, value vaxes, value vs) {
  CAMLparam4(vout, vin, vaxes, vs);
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) fft_raise("irfft", s);
  nx_c_dtype in_dt = nx_c_dtype_of_value(vin);
  nx_c_dtype out_dt = nx_c_dtype_of_value(vout);
  if ((in_dt != NX_C_DTYPE_c32 && in_dt != NX_C_DTYPE_c64) ||
      (out_dt != NX_C_DTYPE_f32 && out_dt != NX_C_DTYPE_f64))
    fft_raise("irfft", NX_C_ERR_BAD_KIND);
  int axes[NX_C_MAX_NDIM];
  int naxes = read_axes(vaxes, axes);
  int64_t s_last = 0;
  int ns = (int)Wosize_val(vs);
  if (ns > 0) s_last = (int64_t)Long_val(Field(vs, ns - 1));
  s = nx_c_irfft_run(&in, &out, in_dt, out_dt, axes, naxes, s_last);
  if (s != NX_C_OK) fft_raise("irfft", s);
  CAMLreturn(Val_unit);
}
