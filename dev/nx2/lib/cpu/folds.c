/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The folds of reductions and scans, compiled once per target: as itself
   for base and as its copy folds_v3.c for v3.

   A fold is a monoid at a dtype: its identity, a lane fold that adds terms
   into NX_CPU_LANES lanes by their index, and a combine that adds a row of
   terms into a row of accumulators, element by element. Integers compute in
   their own width, unsigned, so they wrap as a wider sum stored back would;
   booleans fold as any and all of their bytes' being non-zero. A float sum
   or product uses the instruction: its NaN may differ from nx_kinds.h's, and
   fold.c replaces every NaN result. Maxima and minima are nx_kinds.h's,
   whose ties order -0 below +0. Contiguous terms take a loop of all 16
   lanes, which the compiler vectorises. */

#include "cpu.h"

#if !defined(NX_CPU_V3) || defined(__x86_64__)

#include <string.h>

#include "nx_kinds.h"

/* A lane fold and a combine of [F] over elements of [T]. */
#define FOLD(NAME, T, F)                                                     \
  static void lanes_##NAME(const uint8_t *x_, int64_t s, int64_t n,          \
                           uint8_t *l_, int first) {                        \
    const T *x = (const T *)x_;                                              \
    T *l = (T *)l_;                                                          \
    int64_t i = 0;                                                           \
    for (; i < n && ((first + i) & 15) != 0; i++)                            \
      l[(first + i) & 15] = F(l[(first + i) & 15], x[i * s]);                \
    if (s == 1 && n - i >= 16) {                                             \
      T v[16];                                                               \
      memcpy(v, l, sizeof v);                                                \
      for (; i + 16 <= n; i += 16)                                           \
        for (int k = 0; k < 16; k++) v[k] = F(v[k], x[i + k]);               \
      memcpy(l, v, sizeof v);                                                \
    }                                                                        \
    for (; i < n; i++) l[(first + i) & 15] = F(l[(first + i) & 15], x[i * s]); \
  }                                                                          \
  static void combine_##NAME(uint8_t *a_, const uint8_t *x_, int64_t s,      \
                             int64_t n) {                                    \
    T *a = (T *)a_;                                                          \
    const T *x = (const T *)x_;                                              \
    if (s == 1)                                                              \
      for (int64_t i = 0; i < n; i++) a[i] = F(a[i], x[i]);                  \
    else                                                                     \
      for (int64_t i = 0; i < n; i++) a[i] = F(a[i], x[i * s]);              \
  }                                                                          \
  static void scan_##NAME(uint8_t *a_, const uint8_t *x_, int64_t s,         \
                          uint8_t *y_, int64_t sy, int64_t n) {             \
    const T *x = (const T *)x_;                                              \
    T *y = (T *)y_, a;                                                       \
    memcpy(&a, a_, sizeof a);                                                \
    for (int64_t i = 0; i < n; i++) y[i * sy] = a = F(a, x[i * s]);          \
    memcpy(a_, &a, sizeof a);                                                \
  }

#define ADD(a, b) ((a) + (b))
#define MUL(a, b) ((a) * (b))
#define ANY(a, b) ((uint8_t)((a) | ((b) != 0)))
#define ALL(a, b) ((uint8_t)((a) & ((b) != 0)))

/* The four monoids of an integer dtype of storage [T], unsigned [U], at the
   compute suffix [S] for its extremes, [C] its compute type. */
#define INTS(D, T, U, C, S)                                                  \
  static inline T add_##D(T a, T b) { return (T)(U)((U)a + (U)b); }          \
  static inline T mul_##D(T a, T b) { return (T)(U)((U)a * (U)b); }          \
  static inline T max_##D(T a, T b) { return (T)nx_maximum_##S((C)a, (C)b); } \
  static inline T min_##D(T a, T b) { return (T)nx_minimum_##S((C)a, (C)b); } \
  FOLD(sum_##D, T, add_##D)                                                  \
  FOLD(prod_##D, T, mul_##D)                                                 \
  FOLD(max_##D, T, max_##D)                                                  \
  FOLD(min_##D, T, min_##D)

INTS(i8, int8_t, uint8_t, int32_t, i32)
INTS(u8, uint8_t, uint8_t, uint32_t, u32)
INTS(i16, int16_t, uint16_t, int32_t, i32)
INTS(u16, uint16_t, uint16_t, uint32_t, u32)
INTS(i32, int32_t, uint32_t, int32_t, i32)
INTS(u32, uint32_t, uint32_t, uint32_t, u32)
INTS(i64, int64_t, uint64_t, int64_t, i64)
INTS(u64, uint64_t, uint64_t, uint64_t, u64)

FOLD(sum_f32, float, ADD)
FOLD(prod_f32, float, MUL)
FOLD(max_f32, float, nx_maximum_f32)
FOLD(min_f32, float, nx_minimum_f32)
FOLD(sum_f64, double, ADD)
FOLD(prod_f64, double, MUL)
FOLD(max_f64, double, nx_maximum_f64)
FOLD(min_f64, double, nx_minimum_f64)
FOLD(max_b, uint8_t, ANY)
FOLD(min_b, uint8_t, ALL)

/* The table's folds, by monoid then dtype: NULL where declined. */
#define SET(M, DT, F)                                                        \
  t->fold[M][DT] = (nx_cpu_fold){lanes_##F, combine_##F, scan_##F}

#define ALL4(DT, D)                                                          \
  SET(NX_SUM, DT, sum_##D);                                                  \
  SET(NX_PROD, DT, prod_##D);                                                \
  SET(NX_MAX, DT, max_##D);                                                  \
  SET(NX_MIN, DT, min_##D)

static void set(nx_cpu_target *t) {
  ALL4(NX_FLOAT32, f32);
  ALL4(NX_FLOAT64, f64);
  ALL4(NX_INT8, i8);
  ALL4(NX_UINT8, u8);
  ALL4(NX_INT16, i16);
  ALL4(NX_UINT16, u16);
  ALL4(NX_INT32, i32);
  ALL4(NX_UINT32, u32);
  ALL4(NX_INT64, i64);
  ALL4(NX_UINT64, u64);
  SET(NX_MAX, NX_BOOL, max_b);
  SET(NX_MIN, NX_BOOL, min_b);
}

#if defined(NX_CPU_V3)
void nx_cpu_set_folds_v3(nx_cpu_target *t) { set(t); }
#else
void nx_cpu_set_folds_base(nx_cpu_target *t) { set(t); }
#endif

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_v3;

#endif
