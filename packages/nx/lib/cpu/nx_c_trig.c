/*---------------------------------------------------------------------------
  Copyright (C) 1993 by Sun Microsystems, Inc. (see the notice below).
  Copyright (c) 2005 Bruce D. Evans and Steven G. Kargl, (c) 2011 David
  Schultz. BSD-2-Clause (see LICENSE-msun).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: SunPro AND BSD-2-Clause AND ISC
  ---------------------------------------------------------------------------*/

/* nx_c_trig.c — sin, cos and tan, their float forms, and the complex
   functions built on them, where the C library reduces by x87's 66-bit pi.

   mingw-w64's sin, cos and tan run the x87 unit's fsin, fcos and fptan, which
   reduce their argument by a 66-bit pi: sin(-0x1.fffffffffffffp+31) misses by
   88342 ulps, and cos(0x1.d215304bd75dap+13) by 3. Its complex exp, sin, cos,
   tan, sinh, cosh, tanh and pow go through its sincos. nx_c.h compiles this
   file's functions there (NX_C_OWN_TRIG) and the C library's elsewhere.

   These are fdlibm's, as FreeBSD's msun and musl ship them, each within 1 ulp.
   The argument is reduced by pi/2 in three 33-bit parts up to 2^20 pi/2, and
   past that by 2/pi's bits (Payne and Hanek), exactly enough for every finite
   argument; a polynomial on [-pi/4, pi/4] then gives the sine, cosine or
   tangent. The complex functions are msun's, with C99 Annex G's special
   values, on these and the C library's exp, sinh, cosh and tanh. Nothing
   computes in a type wider than double, and the file compiles with
   -ffp-contract=off (dune). They raise none of C99's flags on purpose, and
   nx reads none.

   The reduction, kernels and tables are fdlibm's (e_rem_pio2.c,
   k_rem_pio2.c, k_sin.c, k_cos.c, k_tan.c and their float forms), which carry
   this notice:

   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved. Developed
   at SunSoft, a Sun Microsystems, Inc. business. Permission to use, copy,
   modify, and distribute this software is freely granted, provided that this
   notice is preserved.

   The complex functions are FreeBSD msun's (s_cexp.c, k_exp.c, s_ccosh.c,
   s_csinh.c, s_ctanh.c and their float forms), under the notices in
   LICENSE-msun. */

#include <math.h>
#include <stdint.h>

#include "nx_c.h"

#ifdef NX_C_OWN_TRIG

/* Bits */

static inline uint64_t asuint64(double f) {
  union {
    double f;
    uint64_t i;
  } u = {f};
  return u.i;
}

static inline double asdouble(uint64_t i) {
  union {
    uint64_t i;
    double f;
  } u = {i};
  return u.f;
}

static inline uint32_t asuint(float f) {
  union {
    float f;
    uint32_t i;
  } u = {f};
  return u.i;
}

static inline float asfloat(uint32_t i) {
  union {
    uint32_t i;
    float f;
  } u = {i};
  return u.f;
}

/* The high word of a double: sign, exponent and the top 20 significand
   bits. */
static inline uint32_t high_word(double x) {
  return (uint32_t)(asuint64(x) >> 32);
}

/* [x] with its low 32 significand bits cleared. */
static inline double clear_low(double x) {
  return asdouble(asuint64(x) & 0xffffffff00000000ULL);
}

/* Adding then subtracting 1.5 * 2^52 rounds a double below 2^51 to an
   integer, in the current rounding mode. */
#define TOINT 0x1.8p52

/* Reduction by 2/pi's bits */

/* 2/pi's bits after the point, 24 to an entry: entry i is bits 24i + 1 to
   24i + 24. A double's exponent of at most 1023 needs the first 66 entries
   (k_rem_pio2.c). */
static const int32_t ipio2[] = {
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C,
    0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649,
    0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44,
    0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C, 0x845F8B,
    0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D,
    0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
    0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330,
    0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3, 0x91615E, 0xE61B08,
    0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
};

/* pi/2 cut into 24-bit parts. */
static const double PIo2[] = {
    1.57079625129699707031e+00, /* 0x3FF921FB, 0x40000000 */
    7.54978941586159635335e-08, /* 0x3E74442D, 0x00000000 */
    5.39030252995776476554e-15, /* 0x3CF84698, 0x80000000 */
    3.28200341580791294123e-22, /* 0x3B78CC51, 0x60000000 */
    1.27065575308067607349e-29, /* 0x39F01B83, 0x80000000 */
    1.22933308981111328932e-36, /* 0x387A2520, 0x40000000 */
    2.73370053816464559624e-44, /* 0x36E38222, 0x80000000 */
    2.16741683877804819444e-51, /* 0x3569F31D, 0x00000000 */
};

/* The precisions [rem_pio2_large] reduces to. */
enum { SINGLE = 0, DOUBLE = 1 };

/* The terms of 2/pi taken at first, less one, by precision: two more than
   the result needs, so that the recomputation test below works. */
static const int init_jk[] = {3, 4};

/* [rem_pio2_large x y e0 nx prec] is the last three bits of n with
   [y = x - n pi/2], [|y| < pi/2], for the positive [x] given as [nx] 24-bit
   integers in doubles, [x[0]]'s exponent [e0]: [x = sum x[i] 2^(e0 - 24i)].
   [y] is one double at SINGLE, and the head and tail of a sum at DOUBLE.

   It multiplies x by 2/pi's bits only from where the product's integer part
   is a multiple of 8, so its cost does not grow with the exponent, and takes
   more bits while the fraction is all zeros, which cancellation can leave. */
static int rem_pio2_large(const double *x, double *y, int e0, int nx,
                          int prec) {
  int32_t jz, jx, jv, jp, jk, carry, n, iq[20], i, j, k, m, q0, ih;
  double z, fw, f[20], fq[20], q[20];

  jk = init_jk[prec];
  jp = jk;

  /* jx, jv and q0, with q0 below 3. */
  jx = nx - 1;
  jv = (e0 - 3) / 24;
  if (jv < 0) jv = 0;
  q0 = e0 - 24 * (jv + 1);

  /* f[0] to f[jx + jk], f[jx + jk] = ipio2[jv + jk]. */
  j = jv - jx;
  m = jx + jk;
  for (i = 0; i <= m; i++, j++) f[i] = j < 0 ? 0.0 : (double)ipio2[j];

  /* q[0] to q[jk]. */
  for (i = 0; i <= jk; i++) {
    for (j = 0, fw = 0.0; j <= jx; j++) fw += x[j] * f[jx + i - j];
    q[i] = fw;
  }

  jz = jk;
recompute:
  /* q[] into 24-bit integers iq[], lowest first. */
  for (i = 0, j = jz, z = q[jz]; j > 0; i++, j--) {
    fw = (double)(int32_t)(0x1p-24 * z);
    iq[i] = (int32_t)(z - 0x1p24 * fw);
    z = q[j - 1] + fw;
  }

  /* n. */
  z = ldexp(z, q0);
  z -= 8.0 * floor(z * 0.125);
  n = (int32_t)z;
  z -= (double)n;
  ih = 0;
  if (q0 > 0) {
    i = iq[jz - 1] >> (24 - q0);
    n += i;
    iq[jz - 1] -= i << (24 - q0);
    ih = iq[jz - 1] >> (23 - q0);
  } else if (q0 == 0)
    ih = iq[jz - 1] >> 23;
  else if (z >= 0.5)
    ih = 2;

  if (ih > 0) { /* the fraction is at least 1/2: take 1 - it */
    n += 1;
    carry = 0;
    for (i = 0; i < jz; i++) {
      j = iq[i];
      if (carry == 0) {
        if (j != 0) {
          carry = 1;
          iq[i] = 0x1000000 - j;
        }
      } else
        iq[i] = 0xffffff - j;
    }
    if (q0 > 0) {
      switch (q0) {
      case 1:
        iq[jz - 1] &= 0x7fffff;
        break;
      case 2:
        iq[jz - 1] &= 0x3fffff;
        break;
      }
    }
    if (ih == 2) {
      z = 1.0 - z;
      if (carry != 0) z -= ldexp(1.0, q0);
    }
  }

  /* All zeros so far: take more of 2/pi's bits. */
  if (z == 0.0) {
    j = 0;
    for (i = jz - 1; i >= jk; i--) j |= iq[i];
    if (j == 0) {
      for (k = 1; iq[jk - k] == 0; k++)
        ;
      for (i = jz + 1; i <= jz + k; i++) {
        f[jx + i] = (double)ipio2[jv + i];
        for (j = 0, fw = 0.0; j <= jx; j++) fw += x[j] * f[jx + i - j];
        q[i] = fw;
      }
      jz += k;
      goto recompute;
    }
  }

  /* Drop the zero terms, or split z into 24-bit parts. */
  if (z == 0.0) {
    jz -= 1;
    q0 -= 24;
    while (iq[jz] == 0) {
      jz--;
      q0 -= 24;
    }
  } else {
    z = ldexp(z, -q0);
    if (z >= 0x1p24) {
      fw = (double)(int32_t)(0x1p-24 * z);
      iq[jz] = (int32_t)(z - 0x1p24 * fw);
      jz += 1;
      q0 += 24;
      iq[jz] = (int32_t)fw;
    } else
      iq[jz] = (int32_t)z;
  }

  /* The integer parts as doubles. */
  fw = ldexp(1.0, q0);
  for (i = jz; i >= 0; i--) {
    q[i] = fw * (double)iq[i];
    fw *= 0x1p-24;
  }

  /* PIo2[0..jp] times q[jz..0]. */
  for (i = jz; i >= 0; i--) {
    for (fw = 0.0, k = 0; k <= jp && k <= jz - i; k++) fw += PIo2[k] * q[i + k];
    fq[jz - i] = fw;
  }

  /* fq[] into y[]. */
  fw = 0.0;
  for (i = jz; i >= 0; i--) fw += fq[i];
  y[0] = ih == 0 ? fw : -fw;
  if (prec == DOUBLE) {
    fw = fq[0] - fw;
    for (i = 1; i <= jz; i++) fw += fq[i];
    y[1] = ih == 0 ? fw : -fw;
  }
  return n & 7;
}

/* Double reduction */

static const double pio4 = 0x1.921fb54442d18p-1,
                    invpio2 = 6.36619772367581382433e-01, /* 53 bits of 2/pi */
    /* pi/2 in three 33-bit parts, and what each leaves. */
    pio2_1 = 1.57079632673412561417e+00,  /* 0x3FF921FB, 0x54400000 */
    pio2_1t = 6.07710050650619224932e-11, /* 0x3DD0B461, 0x1A626331 */
    pio2_2 = 6.07710050630396597660e-11,  /* 0x3DD0B461, 0x1A600000 */
    pio2_2t = 2.02226624879595063154e-21, /* 0x3BA3198A, 0x2E037073 */
    pio2_3 = 2.02226624871116645580e-21,  /* 0x3BA3198A, 0x2E000000 */
    pio2_3t = 8.47842766036889956997e-32; /* 0x397B839A, 0x252049C1 */

/* [x - k pi/2] for a small [k]: one 33-bit part and its rest, good to 85
   bits away from a multiple of pi/2. */
static inline int near(double x, int k, double *y) {
  double z = x - k * pio2_1;
  y[0] = z - k * pio2_1t;
  y[1] = (z - y[0]) - k * pio2_1t;
  return k;
}

/* [rem_pio2 x y] is n with [y[0] + y[1] = x - n pi/2], [|y| <= pi/4]
   about, for [|x|] above about pi/4. */
static int rem_pio2(double x, double *y) {
  uint64_t u = asuint64(x);
  int sign = (int)(u >> 63);
  uint32_t ix = (uint32_t)(u >> 32) & 0x7fffffff;
  double z, w, t, r, fn;
  double tx[3], ty[2];
  int n, ex, ey, i;

  if (ix <= 0x400f6a7a) {           /* |x| ~<= 5pi/4 */
    if ((ix & 0xfffff) == 0x921fb)  /* |x| ~= pi/2 or 2pi/2: cancellation */
      goto medium;
    if (ix <= 0x4002d97c) /* |x| ~<= 3pi/4 */
      return sign ? near(x, -1, y) : near(x, 1, y);
    return sign ? near(x, -2, y) : near(x, 2, y);
  }
  if (ix <= 0x401c463b) {   /* |x| ~<= 9pi/4 */
    if (ix <= 0x4015fdbc) { /* |x| ~<= 7pi/4 */
      if (ix == 0x4012d97c) /* |x| ~= 3pi/2 */
        goto medium;
      return sign ? near(x, -3, y) : near(x, 3, y);
    }
    if (ix == 0x401921fb) /* |x| ~= 4pi/2 */
      goto medium;
    return sign ? near(x, -4, y) : near(x, 4, y);
  }
  if (ix < 0x413921fb) { /* |x| ~< 2^20 pi/2 */
  medium:
    fn = x * invpio2 + TOINT - TOINT;
    n = (int32_t)fn;
    r = x - fn * pio2_1;
    w = fn * pio2_1t; /* good to 85 bits */
    /* Under a directed rounding, fn can be one off. */
    if (r - w < -pio4) {
      n--;
      fn--;
      r = x - fn * pio2_1;
      w = fn * pio2_1t;
    } else if (r - w > pio4) {
      n++;
      fn++;
      r = x - fn * pio2_1;
      w = fn * pio2_1t;
    }
    y[0] = r - w;
    ey = (int)(asuint64(y[0]) >> 52 & 0x7ff);
    ex = (int)(ix >> 20);
    if (ex - ey > 16) { /* second part, good to 118 bits */
      t = r;
      w = fn * pio2_2;
      r = t - w;
      w = fn * pio2_2t - ((t - r) - w);
      y[0] = r - w;
      ey = (int)(asuint64(y[0]) >> 52 & 0x7ff);
      if (ex - ey > 49) { /* third part, good to 151 bits: every case */
        t = r;
        w = fn * pio2_3;
        r = t - w;
        w = fn * pio2_3t - ((t - r) - w);
        y[0] = r - w;
      }
    }
    y[1] = (r - y[0]) - w;
    return n;
  }

  if (ix >= 0x7ff00000) { /* inf or NaN */
    y[0] = y[1] = x - x;
    return 0;
  }

  /* |x| scaled into [2^23, 2^24), cut into three 24-bit parts. */
  z = asdouble((u & (UINT64_MAX >> 12)) | ((uint64_t)(0x3ff + 23) << 52));
  for (i = 0; i < 2; i++) {
    tx[i] = (double)(int32_t)z;
    z = (z - tx[i]) * 0x1p24;
  }
  tx[i] = z;
  while (tx[i] == 0.0) i--;
  n = rem_pio2_large(tx, ty, (int)(ix >> 20) - (0x3ff + 23), i + 1, DOUBLE);
  if (sign) {
    y[0] = -ty[0];
    y[1] = -ty[1];
    return -n;
  }
  y[0] = ty[0];
  y[1] = ty[1];
  return n;
}

/* Double kernels on [-pi/4, pi/4] */

/* sin(x + y), y the tail of x, |y| < ulp(x) / 2; [iy] is 0 when y is 0. */
static double k_sin(double x, double y, int iy) {
  static const double S1 = -1.66666666666666324348e-01,
                      S2 = 8.33333333332248946124e-03,
                      S3 = -1.98412698298579493134e-04,
                      S4 = 2.75573137070700676789e-06,
                      S5 = -2.50507602534068634195e-08,
                      S6 = 1.58969099521155010221e-10;
  double z = x * x, w = z * z;
  double r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
  double v = z * x;
  if (iy == 0) return x + v * (S1 + z * r);
  return x - ((z * (0.5 * y - v * r) - y) - v * S1);
}

/* cos(x + y), y the tail of x. */
static double k_cos(double x, double y) {
  static const double C1 = 4.16666666666666019037e-02,
                      C2 = -1.38888888888741095749e-03,
                      C3 = 2.48015872894767294178e-05,
                      C4 = -2.75573143513906633035e-07,
                      C5 = 2.08757232129817482790e-09,
                      C6 = -1.13596475577881948265e-11;
  double z = x * x, w = z * z;
  double r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
  double hz = 0.5 * z;
  w = 1.0 - hz;
  return w + (((1.0 - w) - hz) + (z * r - x * y));
}

/* tan(x + y), y the tail of x, or -1/tan(x + y) when [odd]. */
static double k_tan(double x, double y, int odd) {
  static const double T[] = {
      3.33333333333334091986e-01,  1.33333333333201242699e-01,
      5.39682539762260521377e-02,  2.18694882948595424599e-02,
      8.86323982359930005737e-03,  3.59207910759131235356e-03,
      1.45620945432529025516e-03,  5.88041240820264096874e-04,
      2.46463134818469906812e-04,  7.81794442939557092300e-05,
      7.14072491382608190305e-05,  -1.85586374855275456654e-05,
      2.59073051863633712884e-05,
  };
  static const double pio4lo = 3.06161699786838301793e-17;
  double z, r, v, w, s, a, w0, a0;
  uint32_t hx = high_word(x);
  int big = (hx & 0x7fffffff) >= 0x3FE59428; /* |x| >= 0.6744 */
  int sign = 0;

  /* Near pi/4, tan(x) from tan(pi/4 - x). */
  if (big) {
    sign = (int)(hx >> 31);
    if (sign) {
      x = -x;
      y = -y;
    }
    x = (pio4 - x) + (pio4lo - y);
    y = 0.0;
  }
  z = x * x;
  w = z * z;
  /* x^5 (T[1] + x^2 T[2] + ...), its odd and even terms apart. */
  r = T[1] + w * (T[3] + w * (T[5] + w * (T[7] + w * (T[9] + w * T[11]))));
  v = z * (T[2] + w * (T[4] + w * (T[6] + w * (T[8] + w * (T[10] + w * T[12])))));
  s = z * x;
  r = y + z * (s * (r + v) + y) + s * T[0];
  w = x + r;
  if (big) {
    s = 1 - 2 * odd;
    v = s - 2.0 * (x + (r - w * w / (w + s)));
    return sign ? -v : v;
  }
  if (!odd) return w;
  /* -1/(x + r), to within an ulp: w0 + v = x + r, w0 and a0 short. */
  w0 = clear_low(w);
  v = r - (w0 - x);
  a0 = a = -1.0 / w;
  a0 = clear_low(a0);
  return a0 + a * (1.0 + a0 * w0 + a0 * v);
}

/* Double functions */

double nx_c_sin(double x) {
  double y[2];
  uint32_t ix = high_word(x) & 0x7fffffff;
  if (ix <= 0x3fe921fb) {       /* |x| ~< pi/4 */
    if (ix < 0x3e500000) return x; /* |x| < 2^-26 */
    return k_sin(x, 0.0, 0);
  }
  if (ix >= 0x7ff00000) return x - x;
  switch (rem_pio2(x, y) & 3) {
  case 0:
    return k_sin(y[0], y[1], 1);
  case 1:
    return k_cos(y[0], y[1]);
  case 2:
    return -k_sin(y[0], y[1], 1);
  default:
    return -k_cos(y[0], y[1]);
  }
}

double nx_c_cos(double x) {
  double y[2];
  uint32_t ix = high_word(x) & 0x7fffffff;
  if (ix <= 0x3fe921fb) {         /* |x| ~< pi/4 */
    if (ix < 0x3e46a09e) return 1.0; /* |x| < 2^-27 sqrt 2 */
    return k_cos(x, 0.0);
  }
  if (ix >= 0x7ff00000) return x - x;
  switch (rem_pio2(x, y) & 3) {
  case 0:
    return k_cos(y[0], y[1]);
  case 1:
    return -k_sin(y[0], y[1], 1);
  case 2:
    return -k_cos(y[0], y[1]);
  default:
    return k_sin(y[0], y[1], 1);
  }
}

double nx_c_tan(double x) {
  double y[2];
  uint32_t ix = high_word(x) & 0x7fffffff;
  if (ix <= 0x3fe921fb) {       /* |x| ~< pi/4 */
    if (ix < 0x3e400000) return x; /* |x| < 2^-27 */
    return k_tan(x, 0.0, 0);
  }
  if (ix >= 0x7ff00000) return x - x;
  int n = rem_pio2(x, y);
  return k_tan(y[0], y[1], n & 1);
}

/* Float reduction */

/* [rem_pio2f x y] is n with [*y = x - n pi/2], [|*y| <= pi/4] about, in
   double: pi/2 in 25 + 53 bits is enough up to 2^28 pi/2. */
static int rem_pio2f(float x, double *y) {
  static const double pio4f = 0x1.921fb6p-1,
                      pio2_1f = 1.57079631090164184570e+00,  /* 25 bits */
                      pio2_1tf = 1.58932547735281966916e-08; /* the rest */
  uint32_t u = asuint(x), ix = u & 0x7fffffff;
  double tx[1], ty[1], fn;
  int n, e0;

  if (ix < 0x4dc90fdb) { /* |x| ~< 2^28 pi/2 */
    fn = (double)x * invpio2 + TOINT - TOINT;
    n = (int32_t)fn;
    *y = x - fn * pio2_1f - fn * pio2_1tf;
    /* Under a directed rounding, fn can be one off. */
    if (*y < -pio4f) {
      n--;
      fn--;
      *y = x - fn * pio2_1f - fn * pio2_1tf;
    } else if (*y > pio4f) {
      n++;
      fn++;
      *y = x - fn * pio2_1f - fn * pio2_1tf;
    }
    return n;
  }
  if (ix >= 0x7f800000) { /* inf or NaN */
    *y = x - x;
    return 0;
  }
  /* |x| scaled into [2^23, 2^24). */
  e0 = (int)(ix >> 23) - (0x7f + 23);
  tx[0] = asfloat(ix - ((uint32_t)e0 << 23));
  n = rem_pio2_large(tx, ty, e0, 1, SINGLE);
  if (u >> 31) {
    *y = -ty[0];
    return -n;
  }
  *y = ty[0];
  return n;
}

/* Float kernels on [-pi/4, pi/4], in double */

/* sin(x), |sin(x)/x - s(x)| < 2^-37.5. */
static float k_sinf(double x) {
  static const double S1 = -0x15555554cbac77.0p-55,
                      S2 = 0x111110896efbb2.0p-59,
                      S3 = -0x1a00f9e2cae774.0p-65,
                      S4 = 0x16cd878c3b46a7.0p-71;
  double z = x * x, w = z * z;
  double r = S3 + z * S4, s = z * x;
  return (float)((x + s * (S1 + z * S2)) + s * w * r);
}

/* cos(x), |cos(x) - c(x)| < 2^-34.1. */
static float k_cosf(double x) {
  static const double C0 = -0x1ffffffd0c5e81.0p-54,
                      C1 = 0x155553e1053a42.0p-57,
                      C2 = -0x16c087e80f1e27.0p-62,
                      C3 = 0x199342e0ee5069.0p-68;
  double z = x * x, w = z * z;
  double r = C2 + z * C3;
  return (float)(((1.0 + z * C0) + w * C1) + (w * z) * r);
}

/* tan(x), or -1/tan(x) when [odd], |tan(x)/x - t(x)| < 2^-25.5. */
static float k_tanf(double x, int odd) {
  static const double T[] = {
      0x15554d3418c99f.0p-54, 0x1112fd38999f72.0p-55, 0x1b54c91d865afe.0p-57,
      0x191df3908c33ce.0p-58, 0x185dadfcecf44e.0p-61, 0x1362b9bf971bcd.0p-59,
  };
  double z = x * x;
  double r = T[4] + z * T[5], t = T[2] + z * T[3], w = z * z, s = z * x;
  double u = T[0] + z * T[1];
  r = (x + s * u) + (s * w) * (t + w * r);
  return (float)(odd ? -1.0 / r : r);
}

/* Float functions */

/* 1 to 4 times pi/2, rounded to double. */
static const double pio2x1 = 1 * 0x1.921fb54442d18p+0,
                    pio2x2 = 2 * 0x1.921fb54442d18p+0,
                    pio2x3 = 3 * 0x1.921fb54442d18p+0,
                    pio2x4 = 4 * 0x1.921fb54442d18p+0;

float nx_c_sinf(float x) {
  double y;
  uint32_t ix = asuint(x);
  int sign = (int)(ix >> 31);
  ix &= 0x7fffffff;
  if (ix <= 0x3f490fda) {       /* |x| ~<= pi/4 */
    if (ix < 0x39800000) return x; /* |x| < 2^-12 */
    return k_sinf(x);
  }
  if (ix <= 0x407b53d1) {   /* |x| ~<= 5pi/4 */
    if (ix <= 0x4016cbe3) /* |x| ~<= 3pi/4 */
      return sign ? -k_cosf(x + pio2x1) : k_cosf(x - pio2x1);
    return k_sinf(sign ? -(x + pio2x2) : -(x - pio2x2));
  }
  if (ix <= 0x40e231d5) {   /* |x| ~<= 9pi/4 */
    if (ix <= 0x40afeddf) /* |x| ~<= 7pi/4 */
      return sign ? k_cosf(x + pio2x3) : -k_cosf(x - pio2x3);
    return k_sinf(sign ? x + pio2x4 : x - pio2x4);
  }
  if (ix >= 0x7f800000) return x - x;
  switch (rem_pio2f(x, &y) & 3) {
  case 0:
    return k_sinf(y);
  case 1:
    return k_cosf(y);
  case 2:
    return k_sinf(-y);
  default:
    return -k_cosf(y);
  }
}

float nx_c_cosf(float x) {
  double y;
  uint32_t ix = asuint(x);
  int sign = (int)(ix >> 31);
  ix &= 0x7fffffff;
  if (ix <= 0x3f490fda) {         /* |x| ~<= pi/4 */
    if (ix < 0x39800000) return 1.0f; /* |x| < 2^-12 */
    return k_cosf(x);
  }
  if (ix <= 0x407b53d1) { /* |x| ~<= 5pi/4 */
    if (ix > 0x4016cbe3)  /* |x| ~> 3pi/4 */
      return -k_cosf(sign ? x + pio2x2 : x - pio2x2);
    return sign ? k_sinf(x + pio2x1) : k_sinf(pio2x1 - x);
  }
  if (ix <= 0x40e231d5) { /* |x| ~<= 9pi/4 */
    if (ix > 0x40afeddf)  /* |x| ~> 7pi/4 */
      return k_cosf(sign ? x + pio2x4 : x - pio2x4);
    return sign ? k_sinf(-x - pio2x3) : k_sinf(x - pio2x3);
  }
  if (ix >= 0x7f800000) return x - x;
  switch (rem_pio2f(x, &y) & 3) {
  case 0:
    return k_cosf(y);
  case 1:
    return k_sinf(-y);
  case 2:
    return -k_cosf(y);
  default:
    return k_sinf(y);
  }
}

float nx_c_tanf(float x) {
  double y;
  uint32_t ix = asuint(x);
  int sign = (int)(ix >> 31);
  ix &= 0x7fffffff;
  if (ix <= 0x3f490fda) {       /* |x| ~<= pi/4 */
    if (ix < 0x39800000) return x; /* |x| < 2^-12 */
    return k_tanf(x, 0);
  }
  if (ix <= 0x407b53d1) {   /* |x| ~<= 5pi/4 */
    if (ix <= 0x4016cbe3) /* |x| ~<= 3pi/4 */
      return k_tanf(sign ? x + pio2x1 : x - pio2x1, 1);
    return k_tanf(sign ? x + pio2x2 : x - pio2x2, 0);
  }
  if (ix <= 0x40e231d5) {   /* |x| ~<= 9pi/4 */
    if (ix <= 0x40afeddf) /* |x| ~<= 7pi/4 */
      return k_tanf(sign ? x + pio2x3 : x - pio2x3, 1);
    return k_tanf(sign ? x + pio2x4 : x - pio2x4, 0);
  }
  if (ix >= 0x7f800000) return x - x;
  int n = rem_pio2f(x, &y);
  return k_tanf(y, n & 1);
}

/* Complex functions */

static inline uint32_t high_word_abs(double x) {
  return high_word(x) & 0x7fffffff;
}

static inline uint32_t low_word(double x) { return (uint32_t)asuint64(x); }

/* [x] with its high word replaced by [hi]. */
static inline double with_high(double x, uint32_t hi) {
  return asdouble((asuint64(x) & 0xffffffffULL) | (uint64_t)hi << 32);
}

/* exp(x) as [m 2^e], [m] in [2^1023, 2^1024), for ln(DBL_MAX) <= x <
   1454.91: exp(x - k ln2) 2^k, k chosen so that exp(k ln2) is near 2^k. */
static double frexp_exp(double x, int *e) {
  static const uint32_t k = 1799;
  static const double kln2 = 1246.97177782734161156;
  double m = exp(x - kln2);
  uint32_t hx = high_word(m);
  *e = (int)(hx >> 20) - (0x3ff + 1023) + (int)k;
  return with_high(m, (hx & 0xfffff) | ((uint32_t)(0x3ff + 1023) << 20));
}

/* exp(z) 2^e, for a real part past ln(DBL_MAX) and [e] 0 or -1: the scale
   goes on in two halves, so that neither overflows. */
static nx_c_complex64 ldexp_cexp(nx_c_complex64 z, int e) {
  double x = creal(z), y = cimag(z);
  int ex;
  double m = frexp_exp(x, &ex);
  e += ex;
  double s1 = asdouble((uint64_t)(0x3ff + e / 2) << 52);
  double s2 = asdouble((uint64_t)(0x3ff + e - e / 2) << 52);
  return CMPLX(nx_c_cos(y) * m * s1 * s2, nx_c_sin(y) * m * s1 * s2);
}

nx_c_complex64 nx_c_cexp(nx_c_complex64 z) {
  static const uint32_t exp_ovfl = 0x40862e42;  /* high word of ~709.78 */
  static const uint32_t cexp_ovfl = 0x4096b8e4; /* ~1454.3 */
  double x = creal(z), y = cimag(z);
  uint32_t hx = high_word(x), hy = high_word_abs(y);
  if ((hy | low_word(y)) == 0) return CMPLX(exp(x), y);
  if (((hx & 0x7fffffff) | low_word(x)) == 0)
    return CMPLX(nx_c_cos(y), nx_c_sin(y));
  if (hy >= 0x7ff00000) {
    if (low_word(x) != 0 || (hx & 0x7fffffff) != 0x7ff00000)
      return CMPLX(y - y, y - y); /* finite or NaN x, inf or NaN y */
    if (hx & 0x80000000) return CMPLX(0.0, 0.0); /* -inf */
    return CMPLX(x, y - y);                       /* +inf */
  }
  if (hx >= exp_ovfl && hx <= cexp_ovfl) return ldexp_cexp(z, 0);
  double ex = exp(x);
  return CMPLX(ex * nx_c_cos(y), ex * nx_c_sin(y));
}

nx_c_complex64 nx_c_ccosh(nx_c_complex64 z) {
  static const double huge = 0x1p1023;
  double x = creal(z), y = cimag(z), h;
  uint32_t hx = high_word(x), ix = hx & 0x7fffffff, lx = low_word(x);
  uint32_t iy = high_word_abs(y), ly = low_word(y);

  if (ix < 0x7ff00000 && iy < 0x7ff00000) {
    if ((iy | ly) == 0) return CMPLX(cosh(x), x * y);
    if (ix < 0x40360000) /* |x| < 22 */
      return CMPLX(cosh(x) * nx_c_cos(y), sinh(x) * nx_c_sin(y));
    if (ix < 0x40862e42) { /* |x| < 710: cosh(x) is exp(|x|) / 2 */
      h = exp(fabs(x)) * 0.5;
      return CMPLX(h * nx_c_cos(y), copysign(h, x) * nx_c_sin(y));
    }
    if (ix < 0x4096bbaa) { /* |x| < 1455: scaled */
      z = ldexp_cexp(CMPLX(fabs(x), y), -1);
      return CMPLX(creal(z), cimag(z) * copysign(1, x));
    }
    h = huge * x; /* overflows */
    return CMPLX(h * h * nx_c_cos(y), h * nx_c_sin(y));
  }
  if ((ix | lx) == 0 && iy >= 0x7ff00000)
    return CMPLX(y - y, copysign(0, x * (y - y)));
  if ((iy | ly) == 0 && ix >= 0x7ff00000) {
    if (((hx & 0xfffff) | lx) == 0) return CMPLX(x * x, copysign(0, x) * y);
    return CMPLX(x * x, copysign(0, (x + x) * y));
  }
  if (ix < 0x7ff00000 && iy >= 0x7ff00000) return CMPLX(y - y, x * (y - y));
  if (ix >= 0x7ff00000 && ((hx & 0xfffff) | lx) == 0) {
    if (iy >= 0x7ff00000) return CMPLX(x * x, x * (y - y));
    return CMPLX((x * x) * nx_c_cos(y), x * nx_c_sin(y));
  }
  return CMPLX((x * x) * (y - y), (x + x) * (y - y));
}

nx_c_complex64 nx_c_csinh(nx_c_complex64 z) {
  static const double huge = 0x1p1023;
  double x = creal(z), y = cimag(z), h;
  uint32_t hx = high_word(x), ix = hx & 0x7fffffff, lx = low_word(x);
  uint32_t iy = high_word_abs(y), ly = low_word(y);

  if (ix < 0x7ff00000 && iy < 0x7ff00000) {
    if ((iy | ly) == 0) return CMPLX(sinh(x), y);
    if (ix < 0x40360000) /* |x| < 22 */
      return CMPLX(sinh(x) * nx_c_cos(y), cosh(x) * nx_c_sin(y));
    if (ix < 0x40862e42) { /* |x| < 710: sinh(x) is ±exp(|x|) / 2 */
      h = exp(fabs(x)) * 0.5;
      return CMPLX(copysign(h, x) * nx_c_cos(y), h * nx_c_sin(y));
    }
    if (ix < 0x4096bbaa) { /* |x| < 1455: scaled */
      z = ldexp_cexp(CMPLX(fabs(x), y), -1);
      return CMPLX(creal(z) * copysign(1, x), cimag(z));
    }
    h = huge * x; /* overflows */
    return CMPLX(h * nx_c_cos(y), h * h * nx_c_sin(y));
  }
  if ((ix | lx) == 0 && iy >= 0x7ff00000)
    return CMPLX(copysign(0, x * (y - y)), y - y);
  if ((iy | ly) == 0 && ix >= 0x7ff00000) {
    if (((hx & 0xfffff) | lx) == 0) return CMPLX(x, y);
    return CMPLX(x, copysign(0, y));
  }
  if (ix < 0x7ff00000 && iy >= 0x7ff00000) return CMPLX(y - y, x * (y - y));
  if (ix >= 0x7ff00000 && ((hx & 0xfffff) | lx) == 0) {
    if (iy >= 0x7ff00000) return CMPLX(x * x, x * (y - y));
    return CMPLX(x * nx_c_cos(y), INFINITY * nx_c_sin(y));
  }
  return CMPLX((x * x) * (y - y), (x + x) * (y - y));
}

nx_c_complex64 nx_c_ctanh(nx_c_complex64 z) {
  double x = creal(z), y = cimag(z);
  uint32_t hx = high_word(x), ix = hx & 0x7fffffff;

  if (ix >= 0x7ff00000) {
    if ((ix & 0xfffff) | low_word(x)) /* NaN */
      return CMPLX(x, y == 0 ? y : x * y);
    x = with_high(x, hx - 0x40000000); /* copysign(1, x) */
    return CMPLX(x, copysign(0, isinf(y) ? y : nx_c_sin(y) * nx_c_cos(y)));
  }
  if (!isfinite(y)) return CMPLX(x != 0 ? y - y : x, y - y);
  if (ix >= 0x40360000) { /* |x| >= 22 */
    double e = exp(-fabs(x));
    return CMPLX(copysign(1, x), 4 * nx_c_sin(y) * nx_c_cos(y) * e * e);
  }
  /* tanh(x + iy) = (beta rho s + i t) / (1 + beta s^2), t = tan(y),
     beta = 1/cos^2(y), s = sinh(x), rho = cosh(x). */
  double t = nx_c_tan(y);
  double beta = 1.0 + t * t;
  double s = sinh(x);
  double rho = sqrt(1 + s * s);
  double denom = 1 + beta * s * s;
  return CMPLX((beta * rho * s) / denom, t / denom);
}

/* sin(z) = -i sinh(iz), cos(z) = cosh(iz), tan(z) = -i tanh(iz). */
nx_c_complex64 nx_c_csin(nx_c_complex64 z) {
  z = nx_c_csinh(CMPLX(-cimag(z), creal(z)));
  return CMPLX(cimag(z), -creal(z));
}

nx_c_complex64 nx_c_ccos(nx_c_complex64 z) {
  return nx_c_ccosh(CMPLX(-cimag(z), creal(z)));
}

nx_c_complex64 nx_c_ctan(nx_c_complex64 z) {
  z = nx_c_ctanh(CMPLX(-cimag(z), creal(z)));
  return CMPLX(cimag(z), -creal(z));
}

nx_c_complex64 nx_c_cpow(nx_c_complex64 z, nx_c_complex64 c) {
  return nx_c_cexp(c * clog(z));
}

/* Float complex functions */

/* exp(x) as [m 2^e], [m] in [2^127, 2^128), for ln(FLT_MAX) <= x < 192.7. */
static float frexp_expf(float x, int *e) {
  static const uint32_t k = 235;
  static const float kln2 = 162.88958740f;
  float m = expf(x - kln2);
  uint32_t hx = asuint(m);
  *e = (int)(hx >> 23) - (0x7f + 127) + (int)k;
  return asfloat((hx & 0x7fffff) | ((uint32_t)(0x7f + 127) << 23));
}

static nx_c_complex32 ldexp_cexpf(nx_c_complex32 z, int e) {
  float x = crealf(z), y = cimagf(z);
  int ex;
  float m = frexp_expf(x, &ex);
  e += ex;
  float s1 = asfloat((uint32_t)(0x7f + e / 2) << 23);
  float s2 = asfloat((uint32_t)(0x7f + e - e / 2) << 23);
  return CMPLXF(nx_c_cosf(y) * m * s1 * s2, nx_c_sinf(y) * m * s1 * s2);
}

nx_c_complex32 nx_c_cexpf(nx_c_complex32 z) {
  static const uint32_t exp_ovfl = 0x42b17218;  /* ~88.72 */
  static const uint32_t cexp_ovfl = 0x43400074; /* ~192.0 */
  float x = crealf(z), y = cimagf(z);
  uint32_t hx = asuint(x), hy = asuint(y) & 0x7fffffff;
  if (hy == 0) return CMPLXF(expf(x), y);
  if ((hx & 0x7fffffff) == 0) return CMPLXF(nx_c_cosf(y), nx_c_sinf(y));
  if (hy >= 0x7f800000) {
    if ((hx & 0x7fffffff) != 0x7f800000) return CMPLXF(y - y, y - y);
    if (hx & 0x80000000) return CMPLXF(0.0f, 0.0f);
    return CMPLXF(x, y - y);
  }
  if (hx >= exp_ovfl && hx <= cexp_ovfl) return ldexp_cexpf(z, 0);
  float ex = expf(x);
  return CMPLXF(ex * nx_c_cosf(y), ex * nx_c_sinf(y));
}

nx_c_complex32 nx_c_ccoshf(nx_c_complex32 z) {
  static const float huge = 0x1p127f;
  float x = crealf(z), y = cimagf(z), h;
  uint32_t hx = asuint(x), ix = hx & 0x7fffffff, iy = asuint(y) & 0x7fffffff;

  if (ix < 0x7f800000 && iy < 0x7f800000) {
    if (iy == 0) return CMPLXF(coshf(x), x * y);
    if (ix < 0x41100000) /* |x| < 9 */
      return CMPLXF(coshf(x) * nx_c_cosf(y), sinhf(x) * nx_c_sinf(y));
    if (ix < 0x42b17218) { /* |x| < 88.7 */
      h = expf(fabsf(x)) * 0.5f;
      return CMPLXF(h * nx_c_cosf(y), copysignf(h, x) * nx_c_sinf(y));
    }
    if (ix < 0x4340b1e7) { /* |x| < 192.7 */
      z = ldexp_cexpf(CMPLXF(fabsf(x), y), -1);
      return CMPLXF(crealf(z), cimagf(z) * copysignf(1, x));
    }
    h = huge * x;
    return CMPLXF(h * h * nx_c_cosf(y), h * nx_c_sinf(y));
  }
  if (ix == 0 && iy >= 0x7f800000)
    return CMPLXF(y - y, copysignf(0, x * (y - y)));
  if (iy == 0 && ix >= 0x7f800000) {
    if ((hx & 0x7fffff) == 0) return CMPLXF(x * x, copysignf(0, x) * y);
    return CMPLXF(x * x, copysignf(0, (x + x) * y));
  }
  if (ix < 0x7f800000 && iy >= 0x7f800000) return CMPLXF(y - y, x * (y - y));
  if (ix >= 0x7f800000 && (hx & 0x7fffff) == 0) {
    if (iy >= 0x7f800000) return CMPLXF(x * x, x * (y - y));
    return CMPLXF((x * x) * nx_c_cosf(y), x * nx_c_sinf(y));
  }
  return CMPLXF((x * x) * (y - y), (x + x) * (y - y));
}

nx_c_complex32 nx_c_csinhf(nx_c_complex32 z) {
  static const float huge = 0x1p127f;
  float x = crealf(z), y = cimagf(z), h;
  uint32_t hx = asuint(x), ix = hx & 0x7fffffff, iy = asuint(y) & 0x7fffffff;

  if (ix < 0x7f800000 && iy < 0x7f800000) {
    if (iy == 0) return CMPLXF(sinhf(x), y);
    if (ix < 0x41100000) /* |x| < 9 */
      return CMPLXF(sinhf(x) * nx_c_cosf(y), coshf(x) * nx_c_sinf(y));
    if (ix < 0x42b17218) { /* |x| < 88.7 */
      h = expf(fabsf(x)) * 0.5f;
      return CMPLXF(copysignf(h, x) * nx_c_cosf(y), h * nx_c_sinf(y));
    }
    if (ix < 0x4340b1e7) { /* |x| < 192.7 */
      z = ldexp_cexpf(CMPLXF(fabsf(x), y), -1);
      return CMPLXF(crealf(z) * copysignf(1, x), cimagf(z));
    }
    h = huge * x;
    return CMPLXF(h * nx_c_cosf(y), h * h * nx_c_sinf(y));
  }
  if (ix == 0 && iy >= 0x7f800000)
    return CMPLXF(copysignf(0, x * (y - y)), y - y);
  if (iy == 0 && ix >= 0x7f800000) {
    if ((hx & 0x7fffff) == 0) return CMPLXF(x, y);
    return CMPLXF(x, copysignf(0, y));
  }
  if (ix < 0x7f800000 && iy >= 0x7f800000) return CMPLXF(y - y, x * (y - y));
  if (ix >= 0x7f800000 && (hx & 0x7fffff) == 0) {
    if (iy >= 0x7f800000) return CMPLXF(x * x, x * (y - y));
    return CMPLXF(x * nx_c_cosf(y), INFINITY * nx_c_sinf(y));
  }
  return CMPLXF((x * x) * (y - y), (x + x) * (y - y));
}

nx_c_complex32 nx_c_ctanhf(nx_c_complex32 z) {
  float x = crealf(z), y = cimagf(z);
  uint32_t hx = asuint(x), ix = hx & 0x7fffffff;

  if (ix >= 0x7f800000) {
    if (ix & 0x7fffff) return CMPLXF(x, y == 0 ? y : x * y); /* NaN */
    x = asfloat(hx - 0x40000000); /* copysign(1, x) */
    return CMPLXF(x,
                  copysignf(0, isinf(y) ? y : nx_c_sinf(y) * nx_c_cosf(y)));
  }
  if (!isfinite(y)) return CMPLXF(ix ? y - y : x, y - y);
  if (ix >= 0x41300000) { /* |x| >= 11 */
    float e = expf(-fabsf(x));
    return CMPLXF(copysignf(1, x), 4 * nx_c_sinf(y) * nx_c_cosf(y) * e * e);
  }
  float t = nx_c_tanf(y);
  float beta = 1.0f + t * t;
  float s = sinhf(x);
  float rho = sqrtf(1 + s * s);
  float denom = 1 + beta * s * s;
  return CMPLXF((beta * rho * s) / denom, t / denom);
}

nx_c_complex32 nx_c_csinf(nx_c_complex32 z) {
  z = nx_c_csinhf(CMPLXF(-cimagf(z), crealf(z)));
  return CMPLXF(cimagf(z), -crealf(z));
}

nx_c_complex32 nx_c_ccosf(nx_c_complex32 z) {
  return nx_c_ccoshf(CMPLXF(-cimagf(z), crealf(z)));
}

nx_c_complex32 nx_c_ctanf(nx_c_complex32 z) {
  z = nx_c_ctanhf(CMPLXF(-cimagf(z), crealf(z)));
  return CMPLXF(cimagf(z), -crealf(z));
}

nx_c_complex32 nx_c_cpowf(nx_c_complex32 z, nx_c_complex32 c) {
  return nx_c_cexpf(c * clogf(z));
}

#endif /* NX_C_OWN_TRIG */
