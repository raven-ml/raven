#ifndef NX_BIGARRAY_EXT_H
#define NX_BIGARRAY_EXT_H

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/* Additional types not in standard bigarray, following stdlib naming convention */
typedef uint16_t caml_ba_bfloat16;      /* BFloat16 */
typedef uint8_t caml_ba_fp8_e4m3;       /* 8-bit float: 1 sign, 4 exponent, 3 mantissa */
typedef uint8_t caml_ba_fp8_e5m2;       /* 8-bit float: 1 sign, 5 exponent, 2 mantissa */
typedef struct { uint16_t re, im; } caml_ba_complex16; /* Complex with half-precision components */
typedef uint8_t caml_ba_bool;           /* Bool as byte (0/1) */
/* Note: int4/uint4 pack 2 values per byte - no single-element typedef makes sense */
typedef int8_t caml_ba_qint8;           /* Quantized signed 8-bit */
typedef uint8_t caml_ba_quint8;         /* Quantized unsigned 8-bit */

/* Extended kind enumeration that continues from OCaml's bigarray kinds */
enum nx_ba_extended_kind {
  NX_BA_BFLOAT16 = CAML_BA_FIRST_UNIMPLEMENTED_KIND,
  NX_BA_BOOL,
  NX_BA_INT4,
  NX_BA_UINT4,
  NX_BA_FP8_E4M3,
  NX_BA_FP8_E5M2,
  NX_BA_COMPLEX16,
  NX_BA_QINT8,
  NX_BA_QUINT8,
  NX_BA_LAST_KIND
};

/* Extended element sizes, aligning with stdlib's caml_ba_element_size[] */
extern int caml_ba_extended_element_size[];

/* Conversion functions for extended types */

/* BFloat16 conversions */
static inline uint16_t float_to_bfloat16(float f) {
  union {
    float f;
    uint32_t i;
  } u = {.f = f};
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

/* Float16 (IEEE 754 half-precision) conversions */
static inline uint16_t float_to_half(float f) {
  union {
    float f;
    uint32_t i;
  } u = {.f = f};
  uint32_t i = u.i;
  uint16_t h_sgn = (uint16_t)((i & 0x80000000u) >> 16);
  uint32_t f_m = i & 0x00FFFFFFu;
  uint32_t f_e = (i & 0x7F800000u) >> 23;

  if (f_e == 0xFF) {  // Inf or NaN
    h_sgn |= 0x7C00u;
    h_sgn |= (f_m != 0);  // NaN if mant !=0
    return h_sgn;
  }
  if (f_e == 0) {  // Denormal or zero
    return h_sgn;  // Flush to zero for simplicity
  }
  int exp = (int)f_e - 127 + 15;
  if (exp >= 31) return h_sgn | 0x7C00u;  // Inf
  if (exp <= 0) return h_sgn;             // Underflow

  uint32_t mant = f_m >> 13;
  uint32_t round = (f_m >> 12) & 1;
  if (round) {
    mant += 1;
    if (mant >= (1u << 10)) {
      mant = 0;
      exp += 1;
    }
  }
  return h_sgn | (exp << 10) | (mant & 0x3FFu);
}

static inline float half_to_float(uint16_t h) {
  uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
  uint32_t exp = (h & 0x7C00u) >> 10;
  uint32_t mant = h & 0x3FFu;

  if (exp == 0x1F) {  // Inf/NaN
    exp = 0xFFu << 23;
    mant = (mant != 0) ? (mant << 13) | 0x400000u : 0;  // NaN or Inf
  } else if (exp == 0) {                                // Denorm or zero
    if (mant == 0)
      exp = 0;
    else {  // Denorm
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FFu;
      exp = (exp + 112) << 23;
      mant <<= 13;
    }
  } else {  // Normal
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

/* FP8 E4M3 conversions with proper rounding and handling */
static inline uint8_t float_to_fp8_e4m3(float f) {
  if (isnan(f)) return 0x7F;  // NaN
  if (isinf(f))
    return signbit(f) ? 0xFF
                      : 0x7F;  // Clamp Inf to NaN (per some specs) or max

  union {
    float f;
    uint32_t i;
  } u = {.f = f};
  uint32_t sign = (u.i >> 31) << 7;
  int exp = ((u.i >> 23) & 0xFF) - 127;
  uint32_t mant = u.i & 0x7FFFFF;

  if (exp > 7) return sign | 0x7E;  // Clamp to max finite (448.0 or -448.0)
  if (exp < -8) return sign;        // Underflow to +/-0

  // Normalize mantissa for rounding (add implicit 1)
  mant |= (1 << 23);
  exp -= 1;  // Adjust for implicit bit

  // Shift to 3-bit mantissa position (23 - 3 = 20 bits shift)
  uint32_t mant_shifted = mant >> 20;
  uint32_t round_bit = (mant >> 19) & 1;
  uint32_t sticky_bits = mant & ((1 << 19) - 1);

  // Round to nearest, ties to even
  if (round_bit && (sticky_bits || (mant_shifted & 1))) {
    mant_shifted += 1;
    if (mant_shifted >= (1 << 4)) {  // Overflow from rounding
      mant_shifted >>= 1;
      exp += 1;
      if (exp > 7) return sign | 0x7E;
    }
  }

  uint8_t exp_bits = (exp + 7) & 0xF;  // Bias 7 for E4M3
  uint8_t mant_bits = mant_shifted & 0x7;

  return sign | (exp_bits << 3) | mant_bits;
}

static inline float fp8_e4m3_to_float(uint8_t fp8) {
  uint32_t sign = (fp8 >> 7) ? 0x80000000 : 0;
  uint32_t exp = (fp8 >> 3) & 0xF;
  uint32_t mant = fp8 & 0x7;

  if (exp ==
      0xF) {  // NaN if mant != 0, else clamp or Inf (but E4M3 has no Inf)
    if (mant != 0) return NAN;
    // Max finite instead of Inf
    exp = 0x86;  // 7 + 127
    mant = 0x700000;
  } else if (exp == 0) {
    if (mant == 0) return sign ? -0.0f : 0.0f;
    // No subnormals in E4M3; treat as normal
    exp = 0x7F - 7;  // Min normal
  } else {
    exp = exp - 7 + 127;
    exp <<= 23;
    mant <<= 20;
  }

  union {
    float f;
    uint32_t i;
  } u;
  u.i = sign | exp | mant;
  return u.f;
}

/* FP8 E5M2 conversions with proper rounding and handling */
static inline uint8_t float_to_fp8_e5m2(float f) {
  if (isnan(f)) return 0x7F;                      // NaN
  if (isinf(f)) return signbit(f) ? 0xFF : 0x7F;  // Inf to special

  union {
    float f;
    uint32_t i;
  } u = {.f = f};
  uint32_t sign = (u.i >> 31) << 7;
  int exp = ((u.i >> 23) & 0xFF) - 127;
  uint32_t mant = u.i & 0x7FFFFF;

  if (exp > 15) return sign | 0x7C;  // Clamp to Inf
  if (exp < -25) return sign;  // Underflow to 0 (including subnormals flush)

  bool subnormal = (exp < -14);
  if (subnormal) {
    // Denormalize
    mant |= (1 << 23);  // Implicit 1
    int shift = -14 - exp;
    mant >>= shift;
    exp = 0;
  } else {
    mant |= (1 << 23);
  }

  // Round to 2-bit mantissa (shift 21 bits)
  uint32_t mant_shifted = mant >> 21;
  uint32_t round_bit = (mant >> 20) & 1;
  uint32_t sticky_bits = mant & ((1 << 20) - 1);

  // Round nearest, ties even
  if (round_bit && (sticky_bits || (mant_shifted & 1))) {
    mant_shifted += 1;
    if (mant_shifted >= (1 << 3)) {  // Overflow
      mant_shifted = 0;
      exp += 1;
      if (exp >= 0x1F) return sign | 0x7C;  // To Inf
    }
  }

  uint8_t exp_bits = subnormal ? 0 : ((exp + 15) & 0x1F);  // Bias 15
  uint8_t mant_bits = mant_shifted & 0x3;

  return sign | (exp_bits << 2) | mant_bits;
}

static inline float fp8_e5m2_to_float(uint8_t fp8) {
  uint32_t sign = (fp8 >> 7) ? 0x80000000 : 0;
  uint32_t exp = (fp8 >> 2) & 0x1F;
  uint32_t mant = fp8 & 0x3;

  if (exp == 0x1F) {  // Inf/NaN
    if (mant == 0) return sign ? -INFINITY : INFINITY;
    return NAN;
  }

  bool subnormal = (exp == 0);
  int bias = 15;
  if (subnormal) {
    if (mant == 0) return sign ? -0.0f : 0.0f;
    exp = 1 - bias;  // Subnormal exp
  } else {
    mant |= 0x4;  // Implicit 1 for 2-bit mant (1.m1 m0)
    exp -= bias;
  }

  exp += 127;
  exp <<= 23;
  mant <<= 21;  // Align to FP32 mantissa

  union {
    float f;
    uint32_t i;
  } u;
  u.i = sign | exp | mant;
  return u.f;
}

#endif /* NX_BIGARRAY_EXT_H */