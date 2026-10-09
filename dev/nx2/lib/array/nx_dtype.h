/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Dtypes: their codes, their facts, and every conversion into them.

   A dtype's code is its NX_<NAME> constant, the index of its row in
   NX_DTYPES. The OCaml library's Dtype.code gives the same codes and its
   rows the same facts.

   The encoders below implement the store rule that Dtype.of_float states
   (dtype.mli), in the floating-point environment's default: rounding to
   nearest, subnormals kept. A kernel stores into their formats through
   them, or through a conversion that gives the same bits. The float16 and
   minifloat codecs give the same bits under flush-to-zero; a conversion
   through C's double-to-float cast, as bfloat16's from a double, does
   not.

   C, CUDA and HIP sources compile this header with no OCaml header, their
   device code included. Metal sources compile it too, without the row
   table and the functions of doubles, which Metal lacks. */

#ifndef NX_DTYPE_H
#define NX_DTYPE_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <math.h>
#include <stdint.h>
#include <string.h>
#endif

/* Every function here is inline, and under CUDA and HIP a device function
   too. C++ needs no static: an inline function may be defined in every
   unit that uses it, and nvcc warns of an unused static one. HIP spells
   the attributes itself: nx.amd compiles with no HIP header, which is
   what defines __host__ and __device__. It stays defined for the headers beside this one (nx_kinds.h) to qualify
   theirs. */
#if defined(__CUDACC__)
#define NX_INLINE inline __host__ __device__
#elif defined(__HIP__)
#define NX_INLINE inline __attribute__((host, device))
#else
#define NX_INLINE static inline
#endif

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
  X(FLOAT8_E4M3FN, "float8_e4m3fn", 8, NX_KIND_FLOAT)      \
  X(FLOAT8_E5M2, "float8_e5m2", 8, NX_KIND_FLOAT)      \
  X(FLOAT4_E2M1FN, "float4_e2m1fn", 4, NX_KIND_FLOAT)      \
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

#ifndef __METAL_VERSION__
typedef struct {
  const char *name;
  int bits;
  enum nx_kind kind;
} nx_dtype_row;

/* The row of the dtype [dt], a code. */
NX_INLINE nx_dtype_row nx_dtype_row_of(int dt) {
#define NX_ROW(NAME, name, bits, kind) {name, bits, kind},
  static const nx_dtype_row rows[NX_DTYPE_COUNT] = {NX_DTYPES(NX_ROW)};
#undef NX_ROW
  return rows[dt];
}
#endif

/* A binary32's bits, and back. */

#ifdef __METAL_VERSION__
NX_INLINE uint32_t nx_float_bits(float f) { return as_type<uint32_t>(f); }
NX_INLINE float nx_bits_float(uint32_t i) { return as_type<float>(i); }
#else
NX_INLINE uint32_t nx_float_bits(float f) {
  uint32_t i;
  memcpy(&i, &f, 4);
  return i;
}

NX_INLINE float nx_bits_float(uint32_t i) {
  float f;
  memcpy(&f, &i, 4);
  return f;
}
#endif

/* The sign and NaN-ness of [f], from its bits: a compiler folds a float
   predicate such as isnan away under -ffinite-math-only, which a kernel
   library's flags may set. */
NX_INLINE uint32_t nx_float_sign(float f) { return nx_float_bits(f) >> 31; }

NX_INLINE int nx_float_nan(float f) {
  return (nx_float_bits(f) & 0x7FFFFFFFu) > 0x7F800000u;
}

/* bfloat16: binary32's top half */

NX_INLINE uint16_t nx_float_to_bf16(float f) {
  uint32_t i = nx_float_bits(f);
  /* NaN first: the rounding bias could carry a small NaN significand into
     the exponent and turn it into inf. */
  if ((i & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((i >> 16) | 0x0040u);
  return (uint16_t)((i + 0x7FFFu + ((i >> 16) & 1)) >> 16);
}

/* A NaN is quieted, as every decoder here does: a signalling bfloat16 NaN
   keeps its payload and gains binary32's quiet bit, bit 6 of the code. A
   magnitude [m] past the infinity's makes [0x7F80 - m] wrap in 16 bits,
   setting its top bit, which the shift moves to bit 6: 16-bit lanes, so a
   loop of decodes vectorises nearly as the plain shift does. */
NX_INLINE float nx_bf16_to_float(uint16_t c) {
  uint16_t past = (uint16_t)(0x7F80u - (c & 0x7FFFu));
  uint16_t q = (uint16_t)(c | ((past >> 9) & 0x40u));
  return nx_bits_float((uint32_t)q << 16);
}

/* float16: IEEE 754 binary16 */

NX_INLINE uint16_t nx_float_to_f16(float f) {
#if defined(__aarch64__) && !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__)
  /* FCVT, from s to h, rounds and keeps NaN payloads as the code below
     does. */
  __fp16 h = (__fp16)f;
  uint16_t c;
  memcpy(&c, &h, 2);
  return c;
#else
  uint32_t bits = nx_float_bits(f);
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
#endif
}

NX_INLINE float nx_f16_to_float(uint16_t c) {
#if defined(__aarch64__) && !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__)
  /* FCVT, from h to s, is exact and keeps NaN payloads as the code below
     does. */
  __fp16 h;
  memcpy(&h, &c, 2);
  return (float)h;
#else
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
  return nx_bits_float(sign | exp | mant);
#endif
}

/* Minifloats: float8 e4m3fn, e5m2 and float4 e2m1fn

   A format of [m] fraction bits and exponent bias [bias] has its least normal
   exponent at 1 - bias. A code's low bits below the sign are its magnitude,
   which orders the finite values. */

/* The magnitude code of the binary32 [f], not NaN, rounded to nearest, ties
   to even; a code past the format's largest finite one means [f] overflows,
   as an infinity does. Both cases below are computed with shifts by
   constants and one is selected, so a loop over values vectorises.

   Below the least normal, 2^(1-bias), adding 2^(24-bias-m) makes the
   hardware round the value to a multiple of the least subnormal,
   2^(1-bias-m), to nearest even: the sum's bits less the addend's count
   those multiples. A value that rounds up to the least normal counts 2^m,
   which is its code. This needs the round-to-nearest mode stores assume.

   Above it, the exponent is rebiased and the fraction rounded to m bits in
   the integer; a carry runs into the exponent. */
NX_INLINE uint32_t nx_mini_round(float f, int m, int bias) {
  uint32_t u = nx_float_bits(f) & 0x7FFFFFFFu;
  float magic = nx_bits_float((uint32_t)(127 + 24 - bias - m) << 23);
  uint32_t sub = nx_float_bits(nx_bits_float(u) + magic) - nx_float_bits(magic);
  uint32_t odd = (u >> (23 - m)) & 1;
  uint32_t normal = (u - ((uint32_t)(127 - bias) << 23) +
                     (1u << (22 - m)) - 1 + odd) >> (23 - m);
  return u < (uint32_t)(128 - bias) << 23 ? sub : normal;
}

/* The value of the magnitude code [q]. */
#if defined(__aarch64__) && !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__)
/* Through binary16, which FCVT widens: the code's bits moved to binary16's
   exponent and fraction fields read as its value times 2^(bias-15),
   exactly, subnormals included, so one exact scale finishes it. */
NX_INLINE float nx_mini_value(uint32_t q, int m, int bias) {
  return nx_f16_to_float((uint16_t)(q << (10 - m))) *
         nx_bits_float((uint32_t)(127 + 15 - bias) << 23);
}
#else
/* A subnormal's value is frac·2^(1-bias-m), a normal's binary32 bits are
   its fields moved into place. Both are exact and both are computed, so a
   loop over codes does not branch. */
NX_INLINE float nx_mini_value(uint32_t q, int m, int bias) {
  uint32_t exp = q >> m;
  uint32_t frac = q & ((1u << m) - 1);
  float sub = (float)frac * nx_bits_float((uint32_t)(128 - bias - m) << 23);
  float normal =
      nx_bits_float(((exp + 127 - (uint32_t)bias) << 23) | (frac << (23 - m)));
  return exp == 0 ? sub : normal;
}
#endif

/* [v], positive, with the sign bit of [sign] (0 or 1). */
NX_INLINE float nx_with_sign(float v, uint32_t sign) {
  return nx_bits_float(nx_float_bits(v) | (sign << 31));
}

NX_INLINE uint32_t nx_mini_saturate(float f, int m, int bias,
                                    uint32_t max) {
  uint32_t q = nx_mini_round(f, m, bias);
  return q > max ? max : q;
}

/* e4m3fn: no infinity; S.1111.111 is NaN; largest finite 448. */

NX_INLINE uint8_t nx_float_to_e4m3fn(float f) {
  uint32_t q = nx_mini_saturate(f, 3, 7, 0x7E);
  return (uint8_t)((nx_float_sign(f) << 7) | (nx_float_nan(f) ? 0x7F : q));
}

NX_INLINE float nx_e4m3fn_to_float(uint8_t c) {
  uint32_t q = c & 0x7F;
  return nx_with_sign(q == 0x7F ? NAN : nx_mini_value(q, 3, 7), c >> 7);
}

/* e5m2: IEEE-like, with infinities; largest finite 57344. */

NX_INLINE uint8_t nx_float_to_e5m2(float f) {
  uint32_t q = nx_mini_saturate(f, 2, 15, 0x7B);
  return (uint8_t)((nx_float_sign(f) << 7) | (nx_float_nan(f) ? 0x7F : q));
}

NX_INLINE float nx_e5m2_to_float(uint8_t c) {
  uint32_t q = c & 0x7F;
  float v = q > 0x7C ? NAN : q == 0x7C ? INFINITY : nx_mini_value(q, 2, 15);
  return nx_with_sign(v, c >> 7);
}

/* e2m1fn: no infinity and no NaN, values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. A
   code is the low four bits of its byte. */

NX_INLINE uint8_t nx_float_to_e2m1fn(float f) {
  uint32_t q = nx_mini_saturate(f, 1, 1, 0x7);
  return (uint8_t)(nx_float_nan(f) ? 0 : (nx_float_sign(f) << 3) | q);
}

NX_INLINE float nx_e2m1fn_to_float(uint8_t c) {
  return nx_with_sign(nx_mini_value(c & 0x7, 1, 1), (c >> 3) & 1);
}

/* The value of the bits [c] of an element of the narrow float dtype [dt]. */
NX_INLINE float nx_bits_to_float(int dt, uint32_t c) {
  switch (dt) {
    case NX_FLOAT16: return nx_f16_to_float((uint16_t)c);
    case NX_BFLOAT16: return nx_bf16_to_float((uint16_t)c);
    case NX_FLOAT8_E4M3FN: return nx_e4m3fn_to_float((uint8_t)c);
    case NX_FLOAT8_E5M2: return nx_e5m2_to_float((uint8_t)c);
    default: return nx_e2m1fn_to_float((uint8_t)c);
  }
}

/* Doubles

   A double narrows to binary32 by rounding to odd: truncating, then setting
   the last bit if a discarded bit was set. Binary32's grid is at least four
   times finer than float16's, bfloat16's or float8's at every magnitude, so
   the odd last bit stands for the discarded bits without creating or
   breaking a tie, and the binary32 encoders above then round the double
   once. Rounding to nearest twice would move a value next to a tie onto
   it. */

#ifndef __METAL_VERSION__

NX_INLINE float nx_float_odd(double x) {
  float f = (float)x;
  uint32_t i = nx_float_bits(f);
  /* Stepping back from a result that rounded away from zero truncates,
     which also brings an overflow to inf back to FLT_MAX. NaN is neither. */
  uint32_t away = fabs((double)f) > fabs(x);
  uint32_t inexact = ((double)f != x) & (x == x);
  return nx_bits_float((i - away) | inexact);
}

/* From a double, rounded once */

NX_INLINE uint16_t nx_double_to_f16(double x) {
  return nx_float_to_f16(nx_float_odd(x));
}

NX_INLINE uint16_t nx_double_to_bf16(double x) {
  return nx_float_to_bf16(nx_float_odd(x));
}

NX_INLINE uint8_t nx_double_to_e4m3fn(double x) {
  return nx_float_to_e4m3fn(nx_float_odd(x));
}

NX_INLINE uint8_t nx_double_to_e5m2(double x) {
  return nx_float_to_e5m2(nx_float_odd(x));
}

NX_INLINE uint8_t nx_double_to_e2m1fn(double x) {
  return nx_float_to_e2m1fn(nx_float_odd(x));
}

/* From 64-bit integers, rounded once

   Past 2^53 an integer narrows to binary64 by rounding to odd from its exact
   value, as a double narrows to binary32 above: rounding to odd twice is
   rounding to odd once at the coarser precision, so the encoders from a
   double then round the integer once, as in nx_double_to_bf16(nx_i64_odd(v)).
   Converting to double rounds to nearest, which can land on a tie of a narrow
   format. A value that rounds to 2^63 or 2^64 cannot be converted back, and
   rounded away from zero. */

NX_INLINE uint64_t nx_double_bits(double d) {
  uint64_t i;
  memcpy(&i, &d, 8);
  return i;
}

NX_INLINE double nx_bits_double(uint64_t i) {
  double d;
  memcpy(&d, &i, 8);
  return d;
}

NX_INLINE double nx_u64_odd(uint64_t a) {
  double d = (double)a;
  uint64_t top = d >= 18446744073709551616.0;
  uint64_t back = top ? 0 : (uint64_t)d;
  uint64_t away = top | (back > a);
  uint64_t inexact = top | (back != a);
  return nx_bits_double((nx_double_bits(d) - away) | inexact);
}

NX_INLINE double nx_i64_odd(int64_t v) {
  double d = (double)v;
  uint64_t top = d >= 9223372036854775808.0;
  int64_t back = top ? 0 : (int64_t)d;
  uint64_t away = top | (v > 0 ? back > v : back < v);
  uint64_t inexact = top | (back != v);
  return nx_bits_double((nx_double_bits(d) - away) | inexact);
}

/* Integers

   Comparing a double against a range's bounds is exact: each bound below
   is a power of two, or one less than a power of two where the bound is
   below 2^53. */

NX_INLINE int64_t nx_double_to_i64(double x) {
  if (x != x) return 0;
  if (x <= -9223372036854775808.0) return INT64_MIN;
  if (x >= 9223372036854775808.0) return INT64_MAX;
  return (int64_t)x;
}

NX_INLINE uint64_t nx_double_to_u64(double x) {
  if (!(x > 0.0)) return 0; /* NaN too */
  if (x >= 18446744073709551616.0) return UINT64_MAX;
  return (uint64_t)x;
}

/* Saturates [x] to [[lo, hi]], a range of at most 32 bits; NaN gives 0. The
   bounds are selected before the truncation, with no branch, so a loop over
   values vectorises. */
NX_INLINE int64_t nx_double_to_int(double x, int64_t lo, int64_t hi) {
  double y = x == x ? x : 0.0;
  y = y < (double)lo ? (double)lo : y;
  y = y > (double)hi ? (double)hi : y;
  return (int64_t)y;
}

/* The bits a store of [x] writes into an element of the integer, boolean or
   narrow float dtype [dt], in the low bits of the result: sign-extended for
   signed integers. [dt] is none of float64, float32 and the complex
   dtypes. */
NX_INLINE int64_t nx_double_to_bits(int dt, double x) {
  switch (dt) {
    case NX_FLOAT16: return nx_double_to_f16(x);
    case NX_BFLOAT16: return nx_double_to_bf16(x);
    case NX_FLOAT8_E4M3FN: return nx_double_to_e4m3fn(x);
    case NX_FLOAT8_E5M2: return nx_double_to_e5m2(x);
    case NX_FLOAT4_E2M1FN: return nx_double_to_e2m1fn(x);
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

/* Runs

   The same conversions over [n] contiguous elements, where the hardware has
   them. arm64 widens float16 with FCVT and FCVTL, exactly. A double narrows
   to __fp16 in one rounding: clang narrows a vector through FCVTXN (round to
   odd) then FCVTN, gcc converts each element with FCVT from d to h. A
   sub-byte format has no run form: its elements share bytes. */

#if defined(__aarch64__) && !defined(__CUDA_ARCH__)
/* The run forms read and write uint16_t memory as __fp16. */
typedef __fp16 __attribute__((may_alias)) nx_fp16;
#endif

NX_INLINE void nx_f16_to_double_run(const uint16_t *src, double *dst,
                                    size_t n) {
#if defined(__aarch64__) && !defined(__CUDA_ARCH__)
  const nx_fp16 *h = (const nx_fp16 *)src;
  for (size_t i = 0; i < n; i++) dst[i] = (double)h[i];
#else
  for (size_t i = 0; i < n; i++) dst[i] = nx_f16_to_float(src[i]);
#endif
}

NX_INLINE void nx_double_to_f16_run(const double *src, uint16_t *dst,
                                    size_t n) {
#if defined(__aarch64__) && !defined(__CUDA_ARCH__)
  nx_fp16 *h = (nx_fp16 *)dst;
  for (size_t i = 0; i < n; i++) h[i] = (__fp16)src[i];
#else
  for (size_t i = 0; i < n; i++) dst[i] = nx_double_to_f16(src[i]);
#endif
}

NX_INLINE void nx_bf16_to_double_run(const uint16_t *src, double *dst,
                                     size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = nx_bf16_to_float(src[i]);
}

NX_INLINE void nx_double_to_bf16_run(const double *src, uint16_t *dst,
                                     size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = nx_double_to_bf16(src[i]);
}

#endif /* __METAL_VERSION__ */

#endif /* NX_DTYPE_H */
