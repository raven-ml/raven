/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Encodings of the float formats narrower than binary32: bfloat16, float16
   and the float8 formats e4m3, e5m2 and their fnuz variants. Every encoder
   rounds once to nearest, ties to even, from its argument's width. A finite
   value past the largest finite one encodes as the format's infinity, or as
   NaN where the format has none. */

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

/* Float8. Each format has [m] fraction bits and exponent bias [bias], and its
   least normal exponent is 1 - bias. A code's low seven bits are its
   magnitude, which orders the finite values, and its top bit is the sign. */

/* The magnitude code of the finite binary32 [f], rounded to nearest, ties to
   even; a code past the format's largest finite one means [f] overflows. */
static inline uint32_t nx_fp8_round(float f, int m, int bias) {
  union {
    float f;
    uint32_t i;
  } u = {.f = f};
  int exp = (int)((u.i >> 23) & 0xFF) - 127;
  uint32_t sig = u.i & 0x7FFFFF;
  uint32_t base = 0;
  int shift = 23 - m;
  if (exp >= 1 - bias) {
    base = (uint32_t)(exp + bias) << m;
  } else {
    /* Subnormal or zero: denormalize to the 2^(1-bias) scale, keeping all
       shifted-out bits for the rounding decision. */
    sig |= 0x800000; /* Implicit one */
    shift += 1 - bias - exp;
    if (shift > 24) return 0; /* Below half the least subnormal */
  }
  uint32_t q = sig >> shift;
  uint32_t rem = sig & ((1u << shift) - 1);
  uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (q & 1))) q++;
  /* A rounding carry propagates into the exponent field, and a subnormal
     that rounds up to 2^m is the least normal: the bit patterns line up. */
  return base + q;
}

/* The value of the finite magnitude code [q]. */
static inline float nx_fp8_value(uint32_t q, int m, int bias) {
  uint32_t exp = q >> m;
  uint32_t frac = q & ((1u << m) - 1);
  if (exp == 0) return ldexpf((float)frac, 1 - bias - m);
  return ldexpf((float)(frac | (1u << m)), (int)exp - bias - m);
}

/* E4M3 (the OCP "fn" variant): no infinities, S.1111.111 is NaN and exponent
   15 is otherwise normal, up to the largest finite value 448. Finite overflow
   and infinities convert to NaN, matching the ml_dtypes and PyTorch e4m3fn
   casts; saturate before casting if clamping is wanted. */
static inline uint8_t float_to_fp8_e4m3(float f) {
  uint8_t sign = signbit(f) ? 0x80 : 0;
  if (!isfinite(f)) return sign | 0x7F;
  uint32_t q = nx_fp8_round(f, 3, 7);
  return sign | (q >= 0x7F ? 0x7F : q);
}

static inline float fp8_e4m3_to_float(uint8_t c) {
  if ((c & 0x7F) == 0x7F) return NAN;
  float v = nx_fp8_value(c & 0x7F, 3, 7);
  return (c & 0x80) ? -v : v;
}

/* E5M2: IEEE-like, with infinities. Finite overflow rounds to infinity. */
static inline uint8_t float_to_fp8_e5m2(float f) {
  uint8_t sign = signbit(f) ? 0x80 : 0;
  if (isnan(f)) return sign | 0x7F;
  if (isinf(f)) return sign | 0x7C;
  uint32_t q = nx_fp8_round(f, 2, 15);
  return sign | (q >= 0x7C ? 0x7C : q);
}

static inline float fp8_e5m2_to_float(uint8_t c) {
  uint32_t q = c & 0x7F;
  if (q > 0x7C) return NAN;
  float v = q == 0x7C ? INFINITY : nx_fp8_value(q, 2, 15);
  return (c & 0x80) ? -v : v;
}

/* The fnuz formats: no infinities and no negative zero, and 0x80 is their
   one NaN. Finite overflow and infinities convert to NaN. */
static inline uint8_t nx_fp8_fnuz(float f, int m, int bias) {
  if (!isfinite(f)) return 0x80;
  uint32_t q = nx_fp8_round(f, m, bias);
  if (q > 0x7F) return 0x80;
  if (q == 0) return 0;
  return (signbit(f) ? 0x80 : 0) | q;
}

static inline float nx_fp8_fnuz_value(uint8_t c, int m, int bias) {
  if (c == 0x80) return NAN;
  float v = nx_fp8_value(c & 0x7F, m, bias);
  return (c & 0x80) ? -v : v;
}

/* E4M3 fnuz: exponent bias 8, largest finite value 240. */
static inline uint8_t float_to_fp8_e4m3fnuz(float f) {
  return nx_fp8_fnuz(f, 3, 8);
}

static inline float fp8_e4m3fnuz_to_float(uint8_t c) {
  return nx_fp8_fnuz_value(c, 3, 8);
}

/* E5M2 fnuz: exponent bias 16, largest finite value 57344. */
static inline uint8_t float_to_fp8_e5m2fnuz(float f) {
  return nx_fp8_fnuz(f, 2, 16);
}

static inline float fp8_e5m2fnuz_to_float(uint8_t c) {
  return nx_fp8_fnuz_value(c, 2, 16);
}

/* Encoders from binary64. A double narrows to binary32 by rounding to odd
   (truncating, then setting the last bit if a discarded bit was set), and the
   binary32 encoder rounds that to nearest even. Binary32's grid is at least
   four times finer than float16's, bfloat16's or float8's at every
   magnitude, so the odd last bit stands for the discarded bits without
   creating or breaking a tie: the result is the double rounded once. Casting
   to float first would round twice and move a value next to a tie onto it. */
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

static inline uint16_t double_to_half(double x) {
  return float_to_half(double_to_float_odd(x));
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

static inline uint8_t double_to_fp8_e4m3fnuz(double x) {
  return float_to_fp8_e4m3fnuz(double_to_float_odd(x));
}

static inline uint8_t double_to_fp8_e5m2fnuz(double x) {
  return float_to_fp8_e5m2fnuz(double_to_float_odd(x));
}

#endif /* NX_DTYPE_H */
