#ifndef NX_BIGARRAY_EXT_H
#define NX_BIGARRAY_EXT_H

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

/* Additional types not in standard bigarray */
typedef uint16_t bfloat16;
typedef uint8_t fp8_e4m3; /* 8-bit float: 1 sign, 4 exponent, 3 mantissa */
typedef uint8_t fp8_e5m2; /* 8-bit float: 1 sign, 5 exponent, 2 mantissa */

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

/* Conversion functions for extended types */

/* BFloat16 conversions */
static inline uint16_t float_to_bfloat16(float f) {
  union { float f; uint32_t i; } u = { .f = f };
  /* Round to nearest even */
  uint32_t rounding_bias = ((u.i >> 16) & 1) + 0x7FFF;
  return (u.i + rounding_bias) >> 16;
}

static inline float bfloat16_to_float(uint16_t bf16) {
  union { float f; uint32_t i; } u;
  u.i = ((uint32_t)bf16) << 16;
  return u.f;
}

/* FP8 E4M3 conversions with proper rounding */
static inline uint8_t float_to_fp8_e4m3(float f) {
  if (isnan(f)) return 0x7F;
  if (isinf(f)) return f > 0 ? 0x7E : 0xFE;
  
  union { float f; uint32_t i; } u = { .f = f };
  uint32_t sign = (u.i >> 31) << 7;
  int exp = ((u.i >> 23) & 0xFF) - 127;
  
  /* Extract mantissa with extra bits for rounding */
  uint32_t mant = (u.i & 0x7FFFFF);
  
  /* Clamp exponent to 4 bits (-7 to 8) */
  if (exp < -7) return sign;  /* Underflow to zero */
  if (exp > 8) return sign | 0x7E;  /* Overflow to max */
  
  /* Round mantissa to 3 bits */
  /* We need bits 20-22 from the 23-bit mantissa, with bit 19 for rounding */
  uint32_t mant_3bit = (mant >> 20) & 0x7;
  uint32_t round_bit = (mant >> 19) & 0x1;
  
  /* Round to nearest, ties away from zero (matching typical FP8 behavior) */
  if (round_bit) {
    mant_3bit++;
    if (mant_3bit > 7) {
      mant_3bit = 0;
      exp++;
      if (exp > 8) return sign | 0x7E;  /* Overflow after rounding */
    }
  }
  
  uint8_t exp_bits = (exp + 7) & 0xF;
  return sign | (exp_bits << 3) | mant_3bit;
}

static inline float fp8_e4m3_to_float(uint8_t fp8) {
  uint32_t sign = (fp8 & 0x80) << 24;
  uint32_t exp = (fp8 >> 3) & 0xF;
  uint32_t mant = fp8 & 0x7;
  
  if (exp == 0xF) {
    /* Special values */
    if (mant == 0x7) return NAN;
    if (mant == 0x6) return sign ? -INFINITY : INFINITY;
  }
  
  if (exp == 0) {
    /* Subnormal numbers or zero */
    if (mant == 0) {
      /* Zero */
      union { float f; uint32_t i; } u;
      u.i = sign;
      return u.f;
    }
    /* Subnormal - not fully implemented */
    exp = 1;
  }
  
  /* Normal numbers */
  uint32_t float_exp = ((int)exp - 7 + 127) << 23;
  uint32_t float_mant = mant << 20;
  
  union { float f; uint32_t i; } u;
  u.i = sign | float_exp | float_mant;
  return u.f;
}

/* FP8 E5M2 conversions (simplified) */
static inline uint8_t float_to_fp8_e5m2(float f) {
  /* Simplified conversion */
  if (isnan(f)) return 0xFF;
  if (isinf(f)) return f > 0 ? 0x7C : 0xFC;
  
  union { float f; uint32_t i; } u = { .f = f };
  uint32_t sign = (u.i >> 31) << 7;
  int exp = ((u.i >> 23) & 0xFF) - 127;
  uint32_t mant = (u.i >> 21) & 0x3;
  
  /* Clamp exponent to 5 bits (-15 to 16) */
  if (exp < -15) return sign;  /* Underflow to zero */
  if (exp > 16) return sign | 0x7C;  /* Overflow to max */
  
  uint8_t exp_bits = (exp + 15) & 0x1F;
  return sign | (exp_bits << 2) | mant;
}

static inline float fp8_e5m2_to_float(uint8_t fp8) {
  /* Simplified conversion */
  uint32_t sign = (fp8 & 0x80) << 24;
  uint32_t exp = (fp8 >> 2) & 0x1F;
  uint32_t mant = fp8 & 0x3;
  
  if (exp == 0x1F) {
    /* Special values */
    if (mant != 0) return NAN;
    return sign ? -INFINITY : INFINITY;
  }
  
  exp = ((exp - 15) + 127) << 23;
  mant = mant << 21;
  
  union { float f; uint32_t i; } u;
  u.i = sign | exp | mant;
  return u.f;
}

#endif /* NX_BIGARRAY_EXT_H */