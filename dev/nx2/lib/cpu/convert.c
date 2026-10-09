/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The conversion runs, compiled once per target, and the fill of its table.

   dune compiles this file as itself for the base target, the instructions
   every host of the architecture has: SSE2 on x86-64, NEON on arm64. It
   compiles it again as convert_v3.c with NX_CPU_V3 and, on x86-64, AVX2,
   FMA, F16C and BMI2, the headers' inline codecs and nx_array.h's runs
   included. Everything here but the fill is static, so no code compiled
   for v3 reaches the base table. Elsewhere the v3 unit is empty.

   One loop per pair of dtypes, each element converted by an expression
   built from nx_dtype.h's codecs: the store rule of Dtype.of_float from a
   float, the ring's map from an integer. The codecs select rather than
   branch, so every loop vectorises. */

#include "cpu.h"

#if !defined(NX_CPU_V3) || defined(__x86_64__)

typedef struct {
  float re, im;
} c64;

typedef struct {
  double re, im;
} c128;

/* An integer as a double that a narrower format rounds once: rounded to
   odd from its exact value past 2^53, and exact below. */
#define ODD(x)                           \
  _Generic((x),                          \
      int64_t: nx_i64_odd((int64_t)(x)),   \
      uint64_t: nx_u64_odd((uint64_t)(x)), \
      default: (double)(x))

/* An integer as a float32, rounded once. A 64-bit one goes through ODD:
   compilers vectorise its conversion to float32 through float64 rounded to
   nearest, which rounds twice. */
#define F32(x)                                    \
  _Generic((x),                                   \
      int64_t: (float)nx_i64_odd((int64_t)(x)),   \
      uint64_t: (float)nx_u64_odd((uint64_t)(x)), \
      default: (float)(x))

/* Destinations: X(D, storage, kind, from a float x, from a double x, from
   an integer x). The kind says how a complex number converts: by its real
   part (R), by being non-zero (B), or part by part (C). An integer converts
   to another modulo the width, as narrowing to a signed type does in gcc
   and clang; a float saturates. */
#define NARROW(X, D, T, f)                                              \
  X(D, T, R, nx_float_to_##f(x), nx_double_to_##f(x),                   \
    nx_double_to_##f(ODD(x)))
#define SATURATE(X, D, T, lo, hi)                                       \
  X(D, T, R, (T)nx_float_to_int(x, lo, hi), (T)nx_double_to_int(x, lo, hi), \
    (T)(x))

#define DSTS(X)                                                         \
  X(FLOAT64, double, R, (double)x, x, (double)x)                        \
  X(FLOAT32, float, R, x, (float)x, F32(x))                             \
  NARROW(X, FLOAT16, uint16_t, f16)                                     \
  NARROW(X, BFLOAT16, uint16_t, bf16)                                   \
  NARROW(X, FLOAT8_E4M3FN, uint8_t, e4m3fn)                             \
  NARROW(X, FLOAT8_E5M2, uint8_t, e5m2)                                 \
  NARROW(X, FLOAT4_E2M1FN, uint8_t, e2m1fn)                             \
  X(INT64, int64_t, R, nx_double_to_i64(x), nx_double_to_i64(x), (int64_t)x) \
  X(UINT64, uint64_t, R, nx_double_to_u64(x), nx_double_to_u64(x),      \
    (uint64_t)x)                                                        \
  SATURATE(X, INT32, int32_t, INT32_MIN, INT32_MAX)                     \
  SATURATE(X, UINT32, uint32_t, 0, UINT32_MAX)                          \
  SATURATE(X, INT16, int16_t, INT16_MIN, INT16_MAX)                     \
  SATURATE(X, UINT16, uint16_t, 0, UINT16_MAX)                          \
  SATURATE(X, INT8, int8_t, INT8_MIN, INT8_MAX)                         \
  SATURATE(X, UINT8, uint8_t, 0, UINT8_MAX)                             \
  SATURATE(X, INT4, uint8_t, -8, 7)                                     \
  SATURATE(X, UINT4, uint8_t, 0, 15)                                    \
  X(COMPLEX128, c128, C, ((c128){x, 0}), ((c128){x, 0}),                \
    ((c128){(double)x, 0}))                                             \
  X(COMPLEX64, c64, C, ((c64){x, 0}), ((c64){(float)x, 0}),             \
    ((c64){F32(x), 0}))                                                 \
  X(BOOL, uint8_t, B, x != 0, x != 0, x != 0)                           \
  X(BIT, uint8_t, B, x != 0, x != 0, x != 0)

/* Sources, the carriers: X(S, storage, class). A float reads as a float, a
   double as a double, an integer and a bool's 0 or 1 as an integer, and a
   complex number by its real part where the destination's kind is R. */
#define SRCS(X, D, DT, K, A, B, C)                                      \
  X(FLOAT64, double, F64, D, DT, K, A, B, C)                            \
  X(FLOAT32, float, F32, D, DT, K, A, B, C)                             \
  X(INT64, int64_t, I, D, DT, K, A, B, C)                               \
  X(UINT64, uint64_t, I, D, DT, K, A, B, C)                             \
  X(INT32, int32_t, I, D, DT, K, A, B, C)                               \
  X(UINT32, uint32_t, I, D, DT, K, A, B, C)                             \
  X(INT16, int16_t, I, D, DT, K, A, B, C)                               \
  X(UINT16, uint16_t, I, D, DT, K, A, B, C)                             \
  X(INT8, int8_t, I, D, DT, K, A, B, C)                                 \
  X(UINT8, uint8_t, I, D, DT, K, A, B, C)                               \
  X(COMPLEX128, c128, Z64, D, DT, K, A, B, C)                           \
  X(COMPLEX64, c64, Z32, D, DT, K, A, B, C)                             \
  X(BOOL, uint8_t, Q, D, DT, K, A, B, C)

/* d[i] from s[i], by the source's class and the destination's kind. */
#define F64_R(ST, DT, A, B, C) { double x = s[i]; d[i] = B; }
#define F32_R(ST, DT, A, B, C) { float x = s[i]; d[i] = A; }
#define I_R(ST, DT, A, B, C) { ST x = s[i]; d[i] = C; }
/* A bool reads as 1 where its byte is not zero: the top bit of the byte
   or-ed with its negation. gcc vectorises no other form of x != 0 that a
   float conversion follows. */
#define Q_R(ST, DT, A, B, C)                       \
  {                                                \
    uint8_t x = (uint8_t)(s[i] | -s[i]) >> 7;      \
    d[i] = C;                                      \
  }
#define Z64_R(ST, DT, A, B, C) { double x = s[i].re; d[i] = B; }
#define Z32_R(ST, DT, A, B, C) { float x = s[i].re; d[i] = A; }
#define Z64_B(ST, DT, A, B, C) d[i] = s[i].re != 0 || s[i].im != 0;
#define Z64_C(ST, DT, A, B, C) d[i] = (DT){s[i].re, s[i].im};
#define Z32_B Z64_B
#define Z32_C Z64_C
#define F64_B F64_R
#define F64_C F64_R
#define F32_B F32_R
#define F32_C F32_R
#define I_B I_R
#define I_C I_R
#define Q_B Q_R
#define Q_C Q_R

#define PAIR(S, ST, CL, D, DT, K, A, B, C)                              \
  static void S##_##D(const void *src, void *dst, int64_t n) {          \
    const ST *s = src;                                                  \
    DT *d = dst;                                                        \
    for (int64_t i = 0; i < n; i++) CL##_##K(ST, DT, A, B, C)               \
  }
#define PAIRS(D, DT, K, A, B, C) SRCS(PAIR, D, DT, K, A, B, C)
DSTS(PAIRS)

/* Narrow floats to float32: a sub-byte one's codes, one per byte. float16
   and e4m3fn convert in nx_array.h's runs, below. */
#define DECODE(S, ST, f)                                                \
  static void S##_FLOAT32(const void *src, void *dst, int64_t n) {      \
    const ST *s = src;                                                  \
    float *d = dst;                                                     \
    for (int64_t i = 0; i < n; i++) d[i] = nx_##f##_to_float(s[i]);     \
  }
DECODE(BFLOAT16, uint16_t, bf16)
DECODE(FLOAT8_E5M2, uint8_t, e5m2)
DECODE(FLOAT4_E2M1FN, uint8_t, e2m1fn)

/* nx_array.h's runs, where they are faster than a loop of a codec. */
#define RUN(S, D, f)                                                   \
  static void run_##S##_##D(const void *src, void *dst, int64_t n) {   \
    f(src, dst, (size_t)n);                                             \
  }
RUN(FLOAT16, FLOAT32, nx_f16_to_float_run)
RUN(FLOAT32, FLOAT16, nx_float_to_f16_run)
RUN(FLOAT32, INT32, nx_float_to_i32_run)
RUN(FLOAT32, INT8, nx_float_to_i8_run)
RUN(FLOAT32, UINT8, nx_float_to_u8_run)
RUN(FLOAT64, FLOAT16, nx_double_to_f16_run)
RUN(FLOAT32, BFLOAT16, nx_float_to_bf16_run)
RUN(FLOAT32, FLOAT8_E4M3FN, nx_float_to_e4m3fn_run)
RUN(FLOAT8_E4M3FN, FLOAT32, nx_e4m3fn_to_float_run)
RUN(INT64, FLOAT32, nx_i64_to_float_run)
RUN(UINT64, FLOAT32, nx_u64_to_float_run)

/* The table: a loop for every pair, then the runs that differ from one. */
#define ENTRY(S, ST, CL, D, DT, K, A, B, C) t->convert[NX_##S][NX_##D] = S##_##D;
#define ENTRIES(D, DT, K, A, B, C) SRCS(ENTRY, D, DT, K, A, B, C)

static void fill(nx_cpu_target *t) {
  DSTS(ENTRIES)
  t->convert[NX_BFLOAT16][NX_FLOAT32] = BFLOAT16_FLOAT32;
  t->convert[NX_FLOAT8_E5M2][NX_FLOAT32] = FLOAT8_E5M2_FLOAT32;
  t->convert[NX_FLOAT4_E2M1FN][NX_FLOAT32] = FLOAT4_E2M1FN_FLOAT32;
  t->convert[NX_FLOAT16][NX_FLOAT32] = run_FLOAT16_FLOAT32;
  t->convert[NX_FLOAT32][NX_FLOAT16] = run_FLOAT32_FLOAT16;
  t->convert[NX_FLOAT32][NX_INT32] = run_FLOAT32_INT32;
  t->convert[NX_FLOAT32][NX_INT8] = run_FLOAT32_INT8;
  t->convert[NX_FLOAT32][NX_UINT8] = run_FLOAT32_UINT8;
  t->convert[NX_FLOAT64][NX_FLOAT16] = run_FLOAT64_FLOAT16;
  t->convert[NX_FLOAT32][NX_BFLOAT16] = run_FLOAT32_BFLOAT16;
  t->convert[NX_FLOAT32][NX_FLOAT8_E4M3FN] = run_FLOAT32_FLOAT8_E4M3FN;
  t->convert[NX_FLOAT8_E4M3FN][NX_FLOAT32] = run_FLOAT8_E4M3FN_FLOAT32;
  t->convert[NX_INT64][NX_FLOAT32] = run_INT64_FLOAT32;
  t->convert[NX_UINT64][NX_FLOAT32] = run_UINT64_FLOAT32;
}

#if defined(NX_CPU_V3)
void nx_cpu_fill_v3(nx_cpu_target *t) { fill(t); }
#else
void nx_cpu_fill_base(nx_cpu_target *t) { fill(t); }
#endif

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_v3;

#endif
