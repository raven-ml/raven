/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Encodings of the float formats narrower than binary32: bfloat16, float16
   and the float8 formats e4m3 and e5m2. */

#ifndef NX_DTYPE_H
#define NX_DTYPE_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/* BFloat16 conversions */
static inline uint16_t float_to_bfloat16(float f) {
  union {
    float f;
    uint32_t i;
  } u = {.f = f};
  /* NaN first: the rounding bias below could carry a small NaN significand
     into the exponent and turn it into inf. */
  if ((u.i & 0x7FFFFFFFu) > 0x7F800000u) {
    return (uint16_t)((u.i >> 16) | 0x0040u); /* quiet, keep the sign */
  }
  /* Round to nearest even */
  uint32_t rounding_bias = ((u.i >> 16) & 1) + 0x7FFF;
  return (u.i + rounding_bias) >> 16;
}

static inline float bfloat16_to_float(uint16_t bf16) {
  union {
    float f;
    uint32_t i;
  } u;
  u.i = ((uint32_t)bf16) << 16;
  return u.f;
}

/* Float16 (IEEE 754 half-precision) conversions.
   Round to nearest, ties to even, with subnormal support. */
static inline uint16_t float_to_half(float f) {
  union {
    float f;
    uint32_t i;
  } u = {.f = f};
  uint32_t f_bits = u.i;
  uint16_t h_sgn = (uint16_t)((f_bits & 0x80000000u) >> 16);
  uint32_t f_exp = f_bits & 0x7F800000u;
  uint32_t f_sig = f_bits & 0x007FFFFFu;

  /* Exponent overflow/NaN converts to signed inf/NaN. */
  if (f_exp >= 0x47800000u) {
    if (f_exp == 0x7F800000u && f_sig != 0) {
      /* NaN: propagate the significand bits, keeping it a NaN. */
      uint16_t ret = (uint16_t)(0x7C00u + (f_sig >> 13));
      ret += (ret == 0x7C00u);
      return h_sgn + ret;
    }
    return h_sgn + 0x7C00u; /* inf, or finite overflow to inf */
  }

  /* Exponent underflow converts to a subnormal half or signed zero. */
  if (f_exp <= 0x38000000u) {
    if (f_exp < 0x33000000u) return h_sgn; /* below 2^-25: signed zero */
    /* Make the subnormal significand. */
    f_exp >>= 23;
    f_sig += 0x00800000u; /* implicit bit */
    f_sig >>= (113 - f_exp);
    /* Round to nearest, ties to even. The shift above can lose up to 11
       bits, so the low bits of the original word break the apparent tie. */
    if (((f_sig & 0x00003FFFu) != 0x00001000u) || (f_bits & 0x000007FFu)) {
      f_sig += 0x00001000u;
    }
    /* A rounding carry into the exponent field yields the smallest normal:
       the correct result. */
    return h_sgn + (uint16_t)(f_sig >> 13);
  }

  /* Regular case. */
  uint16_t h_exp = (uint16_t)((f_exp - 0x38000000u) >> 13);
  /* Round to nearest, ties to even: add half an ulp except on a tie with
     the even bit already clear. */
  if ((f_sig & 0x00003FFFu) != 0x00001000u) {
    f_sig += 0x00001000u;
  }
  uint16_t h_sig = (uint16_t)(f_sig >> 13);
  /* A rounding carry increments the exponent, possibly to 31: overflow to
     signed inf, the correct result. */
  return h_sgn + h_exp + h_sig;
}

static inline float half_to_float(uint16_t h) {
  uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
  uint32_t exp = (h & 0x7C00u) >> 10;
  uint32_t mant = h & 0x3FFu;

  if (exp == 0x1F) { /* Inf/NaN */
    exp = 0xFFu << 23;
    mant = (mant != 0) ? (mant << 13) | 0x400000u : 0;
  } else if (exp == 0) { /* Denorm or zero */
    if (mant == 0) {
      exp = 0;
    } else { /* Denorm */
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FFu;
      exp = (exp + 112) << 23;
      mant <<= 13;
    }
  } else { /* Normal */
    exp = (exp + 112) << 23;
    mant <<= 13;
  }

  union {
    float f;
    uint32_t i;
  } u;
  u.i = sign | exp | mant;
  return u.f;
}

/* FP8 E4M3 conversions (OCP "fn" variant: no infinities, S.1111.111 is NaN,
   exponent 15 is otherwise normal up to the max finite 448, subnormals scale
   by 2^-6). Finite overflow and infinities convert to NaN, matching the
   ml_dtypes and PyTorch e4m3fn casts; saturate before casting if clamping is
   wanted. */
static inline uint8_t float_to_fp8_e4m3(float f) {
  if (isnan(f) || isinf(f)) return signbit(f) ? 0xFF : 0x7F;

  union {
    float f;
    uint32_t i;
  } u = {.f = f};
  uint32_t sign = (u.i >> 31) << 7;
  int exp = ((u.i >> 23) & 0xFF) - 127;

  if (exp >= -6) { /* Normal range */
    uint32_t sig = u.i & 0x7FFFFF;
    /* Round the 23-bit significand to 3 bits, nearest, ties to even. */
    uint32_t q = sig >> 20;
    uint32_t rem = sig & 0xFFFFF;
    if (rem > 0x80000 || (rem == 0x80000 && (q & 1))) q++;
    /* A rounding carry propagates into the exponent field. */
    uint32_t bits = ((uint32_t)(exp + 7) << 3) + q;
    if (bits >= 0x7F) return sign | 0x7F; /* Overflow past 448 to NaN */
    return sign | bits;
  }

  /* Subnormal or zero: denormalize to the 2^-6 scale, keeping all shifted-out
     bits for the rounding decision. */
  uint32_t sig = (u.i & 0x7FFFFF) | 0x800000; /* Implicit one */
  int shift = 20 + (-6 - exp);
  if (shift > 24) return sign; /* Below half the min subnormal 2^-9 */
  uint32_t q = sig >> shift;
  uint32_t rem = sig & ((1u << shift) - 1);
  uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (q & 1))) q++;
  /* q == 8 after rounding is the min normal; the bit pattern lines up. */
  return sign | q;
}

static inline float fp8_e4m3_to_float(uint8_t fp8) {
  bool negative = (fp8 & 0x80) != 0;
  uint32_t exp = (fp8 >> 3) & 0xF;
  uint32_t mant = fp8 & 0x7;

  /* No infinities: exponent 15 is normal except S.1111.111. */
  if (exp == 0xF && mant == 0x7) return NAN;

  float v;
  if (exp == 0) {
    v = ldexpf((float)mant, -9); /* Subnormal: mant/8 * 2^-6 */
  } else {
    v = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
  }
  return negative ? -v : v;
}

/* FP8 E5M2 conversions (IEEE-like: has infinities and subnormals). Finite
   overflow rounds to infinity. */
static inline uint8_t float_to_fp8_e5m2(float f) {
  if (isnan(f)) return signbit(f) ? 0xFF : 0x7F;
  if (isinf(f)) return signbit(f) ? 0xFC : 0x7C;

  union {
    float f;
    uint32_t i;
  } u = {.f = f};
  uint32_t sign = (u.i >> 31) << 7;
  int exp = ((u.i >> 23) & 0xFF) - 127;

  if (exp >= -14) { /* Normal range */
    uint32_t sig = u.i & 0x7FFFFF;
    /* Round the 23-bit significand to 2 bits, nearest, ties to even. */
    uint32_t q = sig >> 21;
    uint32_t rem = sig & 0x1FFFFF;
    if (rem > 0x100000 || (rem == 0x100000 && (q & 1))) q++;
    /* A rounding carry propagates into the exponent field. */
    uint32_t bits = ((uint32_t)(exp + 15) << 2) + q;
    if (bits >= 0x7C) return sign | 0x7C; /* Overflow to Inf */
    return sign | bits;
  }

  /* Subnormal or zero: denormalize to the 2^-14 scale, keeping all
     shifted-out bits for the rounding decision. */
  uint32_t sig = (u.i & 0x7FFFFF) | 0x800000; /* Implicit one */
  int shift = 21 + (-14 - exp);
  if (shift > 24) return sign; /* Below half the min subnormal 2^-16 */
  uint32_t q = sig >> shift;
  uint32_t rem = sig & ((1u << shift) - 1);
  uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (q & 1))) q++;
  /* q == 4 after rounding is the min normal; the bit pattern lines up. */
  return sign | q;
}

static inline float fp8_e5m2_to_float(uint8_t fp8) {
  bool negative = (fp8 & 0x80) != 0;
  uint32_t exp = (fp8 >> 2) & 0x1F;
  uint32_t mant = fp8 & 0x3;

  if (exp == 0x1F) { /* Inf/NaN */
    if (mant == 0) return negative ? -INFINITY : INFINITY;
    return NAN;
  }

  float value;
  if (exp == 0) {
    if (mant == 0) return negative ? -0.0f : 0.0f;
    /* Subnormal: mantissa has no implicit leading 1 */
    float frac = (float)mant / 4.0f;
    value = ldexpf(frac, 1 - 15); /* 2^(1-bias) */
  } else {
    float frac = 1.0f + (float)mant / 4.0f;
    value = ldexpf(frac, (int)exp - 15);
  }

  return negative ? -value : value;
}

/* Encoders from binary64. A double narrows to binary32 by rounding to odd
   (truncating, then setting the last bit if a discarded bit was set), and the
   binary32 encoder rounds that to nearest even. Binary32's grid is at least
   four times finer than bfloat16's or float8's at every magnitude, so the odd
   last bit stands for the discarded bits without creating or breaking a tie:
   the result is the double rounded once. Casting to float first would round
   twice and move a value next to a tie onto it. */
static inline float double_to_float_odd(double x) {
  union {
    float f;
    uint32_t i;
  } u = {.f = (float)x};
  /* Stepping back from a result that rounded away from zero truncates, which
     also brings an overflow to inf back to FLT_MAX. NaN is neither. */
  uint32_t away = fabs((double)u.f) > fabs(x);
  uint32_t inexact = ((double)u.f != x) & (x == x);
  u.i = (u.i - away) | inexact;
  return u.f;
}

static inline uint16_t double_to_bfloat16(double x) {
  return float_to_bfloat16(double_to_float_odd(x));
}

static inline uint8_t double_to_fp8_e4m3(double x) {
  return float_to_fp8_e4m3(double_to_float_odd(x));
}

static inline uint8_t double_to_fp8_e5m2(double x) {
  return float_to_fp8_e5m2(double_to_float_odd(x));
}

#endif /* NX_DTYPE_H */
