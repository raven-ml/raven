/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The arguments every library computes nx_kinds.h's kinds at, and the digest
   of the results, so that each library's results can be compared with the
   digests test_kinds.ml records.

   The f32 and f64 strata are, for each sign: every binade's first and last
   eight patterns and a fixed draw of significands, the infinities and NaNs
   of several payloads; then the points where the kinds switch formulas or
   saturate, each with its neighbours: every (k + 1/2) ln 2 and k + 1/2,
   the multiples of pi/4 up to the trigonometric switch, every sqrt(2) 2^k,
   and the kinds' constants and clamps; under a million points in each
   precision. Pairs and triples for the kinds of two and three operands
   are drawn from the strata, with every pair of a set of specials. A
   result's digest is FNV-1a over its bits, every NaN taken as one NaN.

   Host C only: C11 with stdint and math. */

#ifndef NX_KINDS_STRATA_H
#define NX_KINDS_STRATA_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* The regions the strata are drawn in, for reports. */
enum nx_region {
  NX_REGION_SUBNORMAL,
  NX_REGION_BINADE,
  NX_REGION_SPECIAL,
  NX_REGION_LN2,
  NX_REGION_HALF,
  NX_REGION_PIO4,
  NX_REGION_SQRT2,
  NX_REGION_SWITCH,
  NX_REGION_COUNT
};

static const char *const nx_region_names[NX_REGION_COUNT] = {
    "subnormals",     "binades",        "infinities and NaNs",
    "ln2 / 2 steps",  "half-integers",  "pi/4 multiples",
    "sqrt(2) powers", "switches and clamps"};

/* SplitMix64: a fixed stream of draws. */
static inline uint64_t nx_strata_mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15u;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9u;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBu;
  return x ^ (x >> 31);
}

/* Appends the pattern p, with its region, when there is room; counts it
   always. */
typedef struct {
  uint64_t *bits;
  uint8_t *region;
  size_t n, cap;
} nx_strata;

static inline void nx_strata_put(nx_strata *s, uint64_t p, int region) {
  if (s->n < s->cap) {
    s->bits[s->n] = p;
    if (s->region) s->region[s->n] = (uint8_t)region;
  }
  s->n++;
}

/* The values a kind switches at, saturates at or clamps to, in either
   precision. */
static const double nx_strata_switches[] = {
    /* exponentials */
    88.72283935546875, 89.0, -104.0, 103.97208, 129.0, 151.0, 128.0, 149.0,
    126.0, 17.328680, 26.0, 709.782712893384, 708.3964185322641,
    745.1332191019411, 710.0, 746.0, 1025.0, 1076.0, 1024.0, 1074.0, 40.0,
    37.42994775023705,
    /* logarithms */
    0.70710678118654757, 1.4142135623730951, 0.29289321881345243,
    0.41421356237309515,
    /* trigonometric switches */
    32768.0, 268435456.0,
    /* inverse trigonometric and erf */
    0.5, 1.0, 0.4375, 0.6875, 1.1875, 2.4375, 0.875, 3.92, 1.75, 2.75, 4.0,
    5.95, 5.921587195794507,
    /* hyperbolic */
    90.0, 89.41598629223294, 712.0, 710.4758600739439, 9.010913339, 19.06154746539849,
    13.0, 22.18070977791825, 44.3614195558365, 88.722839111672999,
    /* tiny arguments */
    0x1p-12, 0x1p-24, 0x1p-25, 0x1p-27, 0x1p-53, 0x1p-54, 0x1p-100, 0x1p24,
    0x1p23, 0x1p52, 0x1p53, 2.0, 3.0};

#define NX_STRATA_COUNT(a) (sizeof(a) / sizeof *(a))

/* Patterns of a float: its bits, as f32 or f64. */
static inline uint64_t nx_strata_bits(double v, int f64) {
  if (f64) {
    uint64_t u;
    memcpy(&u, &v, 8);
    return u;
  }
  float f = (float)v;
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

/* v and its k neighbours on each side, both signs. */
static inline void nx_strata_around(nx_strata *s, double v, int k, int f64,
                                    int region) {
  uint64_t sign = f64 ? UINT64_C(1) << 63 : UINT64_C(1) << 31;
  uint64_t b = nx_strata_bits(fabs(v), f64);
  for (int d = -k; d <= k; d++) {
    uint64_t p = b + (uint64_t)(int64_t)d;
    if (p >= sign) continue;
    nx_strata_put(s, p, region);
    nx_strata_put(s, p | sign, region);
  }
}

/* The strata in f32 (f64 = 0) or f64 (f64 = 1). Run once with cap 0 to
   count them. */
static inline size_t nx_strata_real(nx_strata *s, int f64) {
  int ebits = f64 ? 11 : 8, mbits = f64 ? 52 : 23;
  uint64_t emax = (UINT64_C(1) << ebits) - 1;
  uint64_t mmask = (UINT64_C(1) << mbits) - 1;
  uint64_t sign = UINT64_C(1) << (ebits + mbits);
  int draws = f64 ? 16 : 256;
  for (uint64_t sg = 0; sg < 2; sg++)
    for (uint64_t e = 0; e <= emax; e++) {
      uint64_t top = (sg * sign) | (e << mbits);
      int region = e == 0 ? NX_REGION_SUBNORMAL
                          : e == emax ? NX_REGION_SPECIAL : NX_REGION_BINADE;
      for (uint64_t m = 0; m < 8; m++) {
        nx_strata_put(s, top | m, region);
        nx_strata_put(s, top | (mmask - m), region);
      }
      int n = e == emax ? 8 : draws;
      for (int i = 0; i < n; i++) {
        uint64_t m = nx_strata_mix((sg << 40) ^ (e << 20) ^ (uint64_t)i) & mmask;
        nx_strata_put(s, top | m, region);
      }
      if (e == emax) nx_strata_put(s, top | (mmask >> 1) | 1, region);
    }
  /* both signs come from each k >= 0 */
  double ln2 = 0.69314718055994531;
  for (int k = 0; k <= (f64 ? 1077 : 152); k++)
    nx_strata_around(s, (k + 0.5) * ln2, 8, f64, NX_REGION_LN2);
  for (int k = 0; k <= (f64 ? 1075 : 150); k++)
    nx_strata_around(s, k + 0.5, 8, f64, NX_REGION_HALF);
  /* f32 takes every multiple below its switch at 2^15, f64 the first 2^14
     and a draw up to its switch at 2^28 */
  double pio4 = 0.78539816339744831;
  for (int k = 1; f64 ? k <= 16384 : k * pio4 < 32768.0; k++)
    nx_strata_around(s, k * pio4, 4, f64, NX_REGION_PIO4);
  for (int i = 0; f64 && i < 8192; i++) {
    double k = (double)(nx_strata_mix(UINT64_C(0x5EED) + (uint64_t)i) % 341782637);
    nx_strata_around(s, k * pio4, 4, f64, NX_REGION_PIO4);
  }
  for (int k = f64 ? -1074 : -149; k <= (f64 ? 1023 : 127); k++)
    nx_strata_around(s, ldexp(1.4142135623730951, k), 8, f64, NX_REGION_SQRT2);
  for (size_t i = 0; i < NX_STRATA_COUNT(nx_strata_switches); i++)
    nx_strata_around(s, nx_strata_switches[i], 8, f64, NX_REGION_SWITCH);
  return s->n;
}

/* Values every pair of which the kinds of two operands are computed at. */
static const double nx_strata_specials[] = {
    0.0, -0.0, INFINITY, -INFINITY, NAN, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0,
    3.0, -3.0, 0x1p-149, -0x1p-149, 0x1p-126, 3.4028234663852886e38,
    -3.4028234663852886e38, 0x1p-1074, 1.7976931348623157e308};

/* FNV-1a over 64-bit words. */
#define NX_FNV_START UINT64_C(14695981039346656037)

static inline uint64_t nx_fnv(uint64_t h, uint64_t v) {
  for (int i = 0; i < 8; i++) {
    h ^= (v >> (8 * i)) & 0xFF;
    h *= UINT64_C(1099511628211);
  }
  return h;
}

/* A result's bits for the digest, every NaN one NaN. */
static inline uint64_t nx_digest_f32(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (u & 0x7FFFFFFFu) > 0x7F800000u ? 0x7FC00000u : u;
}

static inline uint64_t nx_digest_f64(double f) {
  uint64_t u;
  memcpy(&u, &f, 8);
  return (u & UINT64_C(0x7FFFFFFFFFFFFFFF)) > UINT64_C(0x7FF0000000000000)
             ? UINT64_C(0x7FF8000000000000)
             : u;
}

#endif /* NX_KINDS_STRATA_H */
