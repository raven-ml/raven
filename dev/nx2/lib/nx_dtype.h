/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Dtypes: their codes, their facts, and every conversion into them.

   A dtype's code is its NX_<NAME> constant, the index of its row in
   NX_DTYPES. The OCaml library's Dtype.code gives the same codes and its
   rows the same facts.

   Every store of a double into a dtype follows one rule:

     x                        f64 f32 f16 bf16   e5m2     e4m3   e2m1
     finite, in range         nearest, ties to even, rounded once
     finite, past the largest  ±inf               ±57344   ±448   ±6
     below the least          subnormal or ±0
     ±inf                     ±inf               ±inf     NaN    ±6
     NaN                      NaN                NaN      NaN    +0

   Integers truncate toward zero, saturate to their range and take NaN to 0.
   Booleans are x != 0.

   This header needs no OCaml header, so GPU kernel sources compile it. */

#ifndef NX_DTYPE_H
#define NX_DTYPE_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Codes and facts */

enum nx_kind {
  NX_KIND_FLOAT,
  NX_KIND_COMPLEX,
  NX_KIND_SIGNED,
  NX_KIND_UNSIGNED,
  NX_KIND_BOOLEAN
};

/* X(NAME, name, bits, kind) for every dtype, in code order. */
#define NX_DTYPES(X)                                   \
  X(FLOAT64, "float64", 64, NX_KIND_FLOAT)             \
  X(FLOAT32, "float32", 32, NX_KIND_FLOAT)             \
  X(FLOAT16, "float16", 16, NX_KIND_FLOAT)             \
  X(BFLOAT16, "bfloat16", 16, NX_KIND_FLOAT)           \
  X(FLOAT8_E4M3, "float8_e4m3", 8, NX_KIND_FLOAT)      \
  X(FLOAT8_E5M2, "float8_e5m2", 8, NX_KIND_FLOAT)      \
  X(FLOAT4_E2M1, "float4_e2m1", 4, NX_KIND_FLOAT)      \
  X(INT64, "int64", 64, NX_KIND_SIGNED)                \
  X(UINT64, "uint64", 64, NX_KIND_UNSIGNED)            \
  X(INT32, "int32", 32, NX_KIND_SIGNED)                \
  X(UINT32, "uint32", 32, NX_KIND_UNSIGNED)            \
  X(INT16, "int16", 16, NX_KIND_SIGNED)                \
  X(UINT16, "uint16", 16, NX_KIND_UNSIGNED)            \
  X(INT8, "int8", 8, NX_KIND_SIGNED)                   \
  X(UINT8, "uint8", 8, NX_KIND_UNSIGNED)               \
  X(INT4, "int4", 4, NX_KIND_SIGNED)                   \
  X(UINT4, "uint4", 4, NX_KIND_UNSIGNED)               \
  X(COMPLEX128, "complex128", 128, NX_KIND_COMPLEX)    \
  X(COMPLEX64, "complex64", 64, NX_KIND_COMPLEX)       \
  X(BOOL, "bool", 8, NX_KIND_BOOLEAN)                  \
  X(BIT, "bit", 1, NX_KIND_BOOLEAN)

#define NX_CODE(NAME, name, bits, kind) NX_##NAME,
enum nx_dtype { NX_DTYPES(NX_CODE) NX_DTYPE_COUNT };
#undef NX_CODE

typedef struct {
  const char *name;
  int bits;
  enum nx_kind kind;
} nx_dtype_row;

/* The row of the dtype [dt], a code. */
static inline nx_dtype_row nx_dtype_row_of(int dt) {
#define NX_ROW(NAME, name, bits, kind) {name, bits, kind},
  static const nx_dtype_row rows[NX_DTYPE_COUNT] = {NX_DTYPES(NX_ROW)};
#undef NX_ROW
  return rows[dt];
}

/* Narrowing a double

   A double narrows to binary32 by rounding to odd: truncating, then setting
   the last bit if a discarded bit was set. Binary32's grid is at least four
   times finer than float16's, bfloat16's or float8's at every magnitude, so
   the odd last bit stands for the discarded bits without creating or
   breaking a tie, and the binary32 encoders below then round the double
   once. Rounding to nearest twice would move a value next to a tie onto
   it. */

static inline float nx_float_odd(double x) {
  union {
    float f;
    uint32_t i;
  } u = {.f = (float)x};
  /* Stepping back from a result that rounded away from zero truncates,
     which also brings an overflow to inf back to FLT_MAX. NaN is neither. */
  uint32_t away = fabs((double)u.f) > fabs(x);
  uint32_t inexact = ((double)u.f != x) & (x == x);
  u.i = (u.i - away) | inexact;
  return u.f;
}

/* bfloat16: binary32's top half */

static inline uint16_t nx_float_to_bf16(float f) {
  union {
    float f;
    uint32_t i;
  } u = {.f = f};
  /* NaN first: the rounding bias could carry a small NaN significand into
     the exponent and turn it into inf. */
  if ((u.i & 0x7FFFFFFFu) > 0x7F800000u)
    return (uint16_t)((u.i >> 16) | 0x0040u);
  return (uint16_t)((u.i + 0x7FFFu + ((u.i >> 16) & 1)) >> 16);
}

static inline float nx_bf16_to_float(uint16_t c) {
  union {
    float f;
    uint32_t i;
  } u = {.i = (uint32_t)c << 16};
  return u.f;
}

/* float16: IEEE 754 binary16 */

static inline uint16_t nx_float_to_f16(float f) {
  union {
    float f;
    uint32_t i;
  } u = {.f = f};
  uint32_t bits = u.i;
  uint16_t sign = (uint16_t)((bits & 0x80000000u) >> 16);
  uint32_t exp = bits & 0x7F800000u;
  uint32_t sig = bits & 0x007FFFFFu;

  /* Past the largest exponent: inf, or a quiet NaN keeping its top payload
     bits, as the hardware's conversion gives. */
  if (exp >= 0x47800000u) {
    if (exp == 0x7F800000u && sig != 0)
      return sign | 0x7E00u | (uint16_t)(sig >> 13);
    return sign + 0x7C00u;
  }

  /* Below the least normal: a subnormal or zero. */
  if (exp <= 0x38000000u) {
    if (exp < 0x33000000u) return sign; /* below 2^-25 */
    exp >>= 23;
    sig += 0x00800000u;
    sig >>= (113 - exp);
    /* The shift drops up to 11 bits: the word's low bits break a tie. */
    if (((sig & 0x00003FFFu) != 0x00001000u) || (bits & 0x000007FFu))
      sig += 0x00001000u;
    /* A carry into the exponent gives the least normal, as it should. */
    return sign + (uint16_t)(sig >> 13);
  }

  uint16_t hexp = (uint16_t)((exp - 0x38000000u) >> 13);
  if ((sig & 0x00003FFFu) != 0x00001000u) sig += 0x00001000u;
  /* A carry may raise the exponent to 31: inf, as it should. */
  return sign + hexp + (uint16_t)(sig >> 13);
}

static inline float nx_f16_to_float(uint16_t c) {
  uint32_t sign = ((uint32_t)(c & 0x8000u)) << 16;
  uint32_t exp = (c & 0x7C00u) >> 10;
  uint32_t mant = c & 0x3FFu;
  if (exp == 0x1F) {
    exp = 0xFFu << 23;
    mant = mant != 0 ? (mant << 13) | 0x400000u : 0;
  } else if (exp == 0) {
    if (mant != 0) {
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FFu;
      exp = (exp + 112) << 23;
      mant <<= 13;
    }
  } else {
    exp = (exp + 112) << 23;
    mant <<= 13;
  }
  union {
    float f;
    uint32_t i;
  } u = {.i = sign | exp | mant};
  return u.f;
}

/* Minifloats: float8 e4m3, e5m2 and float4 e2m1

   A format of [m] fraction bits and exponent bias [bias] has its least normal
   exponent at 1 - bias. A code's low bits below the sign are its magnitude,
   which orders the finite values. */

/* The magnitude code of the finite binary32 [f], rounded to nearest, ties to
   even; a code past the format's largest finite one means [f] overflows. */
static inline uint32_t nx_mini_round(float f, int m, int bias) {
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
    /* Subnormal or zero: denormalise to the least normal's scale, keeping
       every shifted-out bit for the rounding. */
    sig |= 0x800000;
    shift += 1 - bias - exp;
    if (shift > 24) return 0; /* below half the least subnormal */
  }
  uint32_t q = sig >> shift;
  uint32_t rem = sig & ((1u << shift) - 1);
  uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (q & 1))) q++;
  /* A carry runs into the exponent, and a subnormal that rounds up to 2^m is
     the least normal: the codes line up. */
  return base + q;
}

static inline float nx_mini_value(uint32_t q, int m, int bias) {
  uint32_t exp = q >> m;
  uint32_t frac = q & ((1u << m) - 1);
  if (exp == 0) return ldexpf((float)frac, 1 - bias - m);
  return ldexpf((float)(frac | (1u << m)), (int)exp - bias - m);
}

static inline uint32_t nx_mini_saturate(float f, int m, int bias,
                                        uint32_t max) {
  uint32_t q = nx_mini_round(f, m, bias);
  return q > max ? max : q;
}

/* e4m3 (OCP "fn"): no infinity; S.1111.111 is NaN; largest finite 448. */

static inline uint8_t nx_float_to_e4m3(float f) {
  uint8_t sign = signbit(f) ? 0x80 : 0;
  if (!isfinite(f)) return sign | 0x7F;
  return sign | (uint8_t)nx_mini_saturate(f, 3, 7, 0x7E);
}

static inline float nx_e4m3_to_float(uint8_t c) {
  float s = (c & 0x80) ? -1.f : 1.f;
  if ((c & 0x7F) == 0x7F) return copysignf(NAN, s);
  return s * nx_mini_value(c & 0x7F, 3, 7);
}

/* e5m2: IEEE-like, with infinities; largest finite 57344. */

static inline uint8_t nx_float_to_e5m2(float f) {
  uint8_t sign = signbit(f) ? 0x80 : 0;
  if (isnan(f)) return sign | 0x7F;
  if (isinf(f)) return sign | 0x7C;
  return sign | (uint8_t)nx_mini_saturate(f, 2, 15, 0x7B);
}

static inline float nx_e5m2_to_float(uint8_t c) {
  float s = (c & 0x80) ? -1.f : 1.f;
  uint32_t q = c & 0x7F;
  if (q > 0x7C) return copysignf(NAN, s);
  return s * (q == 0x7C ? INFINITY : nx_mini_value(q, 2, 15));
}

/* e2m1 (OCP FP4): no infinity and no NaN, values ±{0, 0.5, 1, 1.5, 2, 3, 4,
   6}. NaN stores as +0, as it does in integers, and an infinity
   saturates. A code is the low four bits of its byte. */

static inline uint8_t nx_float_to_e2m1(float f) {
  if (isnan(f)) return 0;
  uint8_t sign = signbit(f) ? 0x8 : 0;
  if (isinf(f)) return sign | 0x7;
  return sign | (uint8_t)nx_mini_saturate(f, 1, 1, 0x7);
}

static inline float nx_e2m1_to_float(uint8_t c) {
  float v = nx_mini_value(c & 0x7, 1, 1);
  return (c & 0x8) ? -v : v;
}

/* From a double, rounded once */

static inline uint16_t nx_double_to_f16(double x) {
  return nx_float_to_f16(nx_float_odd(x));
}

static inline uint16_t nx_double_to_bf16(double x) {
  return nx_float_to_bf16(nx_float_odd(x));
}

static inline uint8_t nx_double_to_e4m3(double x) {
  return nx_float_to_e4m3(nx_float_odd(x));
}

static inline uint8_t nx_double_to_e5m2(double x) {
  return nx_float_to_e5m2(nx_float_odd(x));
}

static inline uint8_t nx_double_to_e2m1(double x) {
  return nx_float_to_e2m1(nx_float_odd(x));
}

/* Integers

   A double truncates toward zero and saturates to the range; NaN is 0.
   Comparing against the range's bounds as doubles is exact: each bound
   below is a power of two, or one less than a power of two where the bound
   is below 2^53. */

static inline int64_t nx_double_to_i64(double x) {
  if (x != x) return 0;
  if (x <= -9223372036854775808.0) return INT64_MIN;
  if (x >= 9223372036854775808.0) return INT64_MAX;
  return (int64_t)x;
}

static inline uint64_t nx_double_to_u64(double x) {
  if (!(x > 0.0)) return 0; /* NaN too */
  if (x >= 18446744073709551616.0) return UINT64_MAX;
  return (uint64_t)x;
}

/* Saturates [x] to [[lo, hi]], a range of at most 32 bits. */
static inline int64_t nx_double_to_int(double x, int64_t lo, int64_t hi) {
  if (x != x) return 0;
  if (x <= (double)lo) return lo;
  if (x >= (double)hi) return hi;
  return (int64_t)x;
}

/* The bits a store of [x] writes into an element of the integer, boolean or
   narrow float dtype [dt], in the low bits of the result: sign-extended for
   signed integers. [dt] is none of float64, float32 and the complex
   dtypes. */
static inline int64_t nx_double_to_bits(int dt, double x) {
  switch (dt) {
    case NX_FLOAT16: return nx_double_to_f16(x);
    case NX_BFLOAT16: return nx_double_to_bf16(x);
    case NX_FLOAT8_E4M3: return nx_double_to_e4m3(x);
    case NX_FLOAT8_E5M2: return nx_double_to_e5m2(x);
    case NX_FLOAT4_E2M1: return nx_double_to_e2m1(x);
    case NX_INT64: return nx_double_to_i64(x);
    case NX_UINT64: return (int64_t)nx_double_to_u64(x);
    case NX_INT32: return nx_double_to_int(x, INT32_MIN, INT32_MAX);
    case NX_UINT32: return nx_double_to_int(x, 0, UINT32_MAX);
    case NX_INT16: return nx_double_to_int(x, INT16_MIN, INT16_MAX);
    case NX_UINT16: return nx_double_to_int(x, 0, UINT16_MAX);
    case NX_INT8: return nx_double_to_int(x, INT8_MIN, INT8_MAX);
    case NX_UINT8: return nx_double_to_int(x, 0, UINT8_MAX);
    case NX_INT4: return nx_double_to_int(x, -8, 7);
    case NX_UINT4: return nx_double_to_int(x, 0, 15);
    default: return x != 0.0; /* bool, bit */
  }
}

/* The value of the bits [c] of an element of the narrow float dtype [dt]. */
static inline float nx_bits_to_float(int dt, uint32_t c) {
  switch (dt) {
    case NX_FLOAT16: return nx_f16_to_float((uint16_t)c);
    case NX_BFLOAT16: return nx_bf16_to_float((uint16_t)c);
    case NX_FLOAT8_E4M3: return nx_e4m3_to_float((uint8_t)c);
    case NX_FLOAT8_E5M2: return nx_e5m2_to_float((uint8_t)c);
    default: return nx_e2m1_to_float((uint8_t)c);
  }
}

/* Runs

   The same conversions over [n] contiguous elements, where the hardware has
   them: arm64 converts float16 in FCVTL and FCVTN, which round as the
   scalar forms do. A sub-byte format has no run form: its elements share
   bytes. */

static inline void nx_f16_to_double_run(const uint16_t *src, double *dst,
                                        size_t n) {
#if defined(__aarch64__)
  const __fp16 *h = (const __fp16 *)src;
  for (size_t i = 0; i < n; i++) dst[i] = (double)h[i];
#else
  for (size_t i = 0; i < n; i++) dst[i] = nx_f16_to_float(src[i]);
#endif
}

static inline void nx_double_to_f16_run(const double *src, uint16_t *dst,
                                        size_t n) {
#if defined(__aarch64__)
  /* A double converts to __fp16 in one rounding, FCVT from d to h. */
  __fp16 *h = (__fp16 *)dst;
  for (size_t i = 0; i < n; i++) h[i] = (__fp16)src[i];
#else
  for (size_t i = 0; i < n; i++) dst[i] = nx_double_to_f16(src[i]);
#endif
}

static inline void nx_bf16_to_double_run(const uint16_t *src, double *dst,
                                         size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = nx_bf16_to_float(src[i]);
}

static inline void nx_double_to_bf16_run(const double *src, uint16_t *dst,
                                         size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = nx_double_to_bf16(src[i]);
}

#endif /* NX_DTYPE_H */
