/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Every kind as a loop over contiguous operands, for one target, compiled
   as nx.cpu's convert.c is: as itself for base, and as its copy
   nx_kinds_support_loops_v3.c with NX_CPU_V3 and, on x86-64, v3's
   instructions, the headers included. Elsewhere the v3 table is NULL. The
   trigonometric loops compute the kind below the switch on every lane and
   redo the lanes past it, as nx_kinds.h says a vector loop does, testing a
   block for such a lane in a loop that vectorises first. */

#include <stddef.h>

#include "nx_kinds.h"
#include "nx_kinds_support.h"

#if !defined(NX_CPU_V3) || defined(__x86_64__)

#define NX_F_UNARY(X)                                                         \
  X(exp) X(exp2) X(expm1) X(log) X(log2) X(log1p) X(asin) X(acos) X(atan)     \
  X(sinh) X(cosh) X(tanh) X(erf) X(neg) X(recip) X(abs) X(sign) X(sqrt)        \
  X(floor) X(ceil) X(round) X(trunc)
#define NX_F_TRIG(X) X(sin) X(cos) X(tan)
#define NX_F_BINARY(X)                                                        \
  X(add) X(sub) X(mul) X(fdiv) X(mod) X(pow) X(atan2) X(maximum) X(minimum)
#define NX_F_COMPARE(X) X(equal) X(not_equal) X(less) X(less_equal)
#define NX_I_UNARY(X) X(neg) X(abs) X(sign) X(recip)
#define NX_I_BINARY(X)                                                        \
  X(add) X(sub) X(mul) X(idiv) X(mod) X(pow) X(maximum) X(minimum) X(and)     \
  X(or) X(xor) X(equal) X(not_equal) X(less) X(less_equal)

#define NX_BLOCK 1024

#define NX_UNARY_LOOPS(k)                                                     \
  static void k##_f32(const float *x, const float *z, const float *w,         \
                      float *y, long n) {                                     \
    (void)z;                                                                  \
    (void)w;                                                                  \
    for (long i = 0; i < n; i++) y[i] = nx_##k##_f32(x[i]);                   \
  }                                                                           \
  static void k##_f64(const double *x, const double *z, const double *w,      \
                      double *y, long n) {                                    \
    (void)z;                                                                  \
    (void)w;                                                                  \
    for (long i = 0; i < n; i++) y[i] = nx_##k##_f64(x[i]);                   \
  }
NX_F_UNARY(NX_UNARY_LOOPS)

/* The scalar call that redoes a lane past the switch, kept out of the
   loop: inlined there, it made the loop's own call too large to inline. */
#if defined(__GNUC__)
#define NX_COLD __attribute__((noinline))
#else
#define NX_COLD
#endif

#define NX_TRIG_LOOPS(k)                                                      \
  NX_COLD static float k##_big_f32(float x) { return nx_##k##_f32(x); }       \
  NX_COLD static double k##_big_f64(double x) { return nx_##k##_f64(x); }     \
  static void k##_f32(const float *x, const float *z, const float *w,         \
                      float *y, long n) {                                     \
    (void)z;                                                                  \
    (void)w;                                                                  \
    for (long b = 0; b < n; b += NX_BLOCK) {                                  \
      long e = n - b < NX_BLOCK ? n : b + NX_BLOCK;                           \
      int big = 0;                                                            \
      for (long i = b; i < e; i++) y[i] = nx_##k##_near_f32(x[i]);            \
      for (long i = b; i < e; i++) big |= nx_trig_big_f32(x[i]);              \
      for (long i = b; big && i < e; i++)                                     \
        if (nx_trig_big_f32(x[i])) y[i] = k##_big_f32(x[i]);                  \
    }                                                                         \
  }                                                                           \
  static void k##_f64(const double *x, const double *z, const double *w,      \
                      double *y, long n) {                                    \
    (void)z;                                                                  \
    (void)w;                                                                  \
    for (long b = 0; b < n; b += NX_BLOCK) {                                  \
      long e = n - b < NX_BLOCK ? n : b + NX_BLOCK;                           \
      int big = 0;                                                            \
      for (long i = b; i < e; i++) y[i] = nx_##k##_near_f64(x[i]);            \
      for (long i = b; i < e; i++) big |= nx_trig_big_f64(x[i]);              \
      for (long i = b; big && i < e; i++)                                     \
        if (nx_trig_big_f64(x[i])) y[i] = k##_big_f64(x[i]);                  \
    }                                                                         \
  }
NX_F_TRIG(NX_TRIG_LOOPS)

#define NX_BINARY_LOOPS(k)                                                    \
  static void k##_f32(const float *x, const float *z, const float *w,         \
                      float *y, long n) {                                     \
    (void)w;                                                                  \
    for (long i = 0; i < n; i++) y[i] = nx_##k##_f32(x[i], z[i]);             \
  }                                                                           \
  static void k##_f64(const double *x, const double *z, const double *w,      \
                      double *y, long n) {                                    \
    (void)w;                                                                  \
    for (long i = 0; i < n; i++) y[i] = nx_##k##_f64(x[i], z[i]);             \
  }
NX_F_BINARY(NX_BINARY_LOOPS)

/* A comparison gives 1 or 0 in the operands' type. */
#define NX_COMPARE_LOOPS(k)                                                   \
  static void k##_f32(const float *x, const float *z, const float *w,         \
                      float *y, long n) {                                     \
    (void)w;                                                                  \
    for (long i = 0; i < n; i++) y[i] = (float)nx_##k##_f32(x[i], z[i]);      \
  }                                                                           \
  static void k##_f64(const double *x, const double *z, const double *w,      \
                      double *y, long n) {                                    \
    (void)w;                                                                  \
    for (long i = 0; i < n; i++) y[i] = (double)nx_##k##_f64(x[i], z[i]);     \
  }
NX_F_COMPARE(NX_COMPARE_LOOPS)

/* where selects on a non-zero first operand. */
static void where_f32(const float *x, const float *z, const float *w, float *y,
                      long n) {
  for (long i = 0; i < n; i++) y[i] = nx_where_f32(x[i] != 0.0f, z[i], w[i]);
}

static void where_f64(const double *x, const double *z, const double *w,
                      double *y, long n) {
  for (long i = 0; i < n; i++) y[i] = nx_where_f64(x[i] != 0.0, z[i], w[i]);
}

static void fma_f32(const float *x, const float *z, const float *w, float *y,
                    long n) {
  for (long i = 0; i < n; i++) y[i] = nx_fma_f32(x[i], z[i], w[i]);
}

static void fma_f64(const double *x, const double *z, const double *w,
                    double *y, long n) {
  for (long i = 0; i < n; i++) y[i] = nx_fma_f64(x[i], z[i], w[i]);
}

/* An integer kind at i32, u32, i64 and u64, results as 64-bit patterns:
   y[4i] to y[4i + 3]. */
#define NX_INT_UNARY_LOOP(k)                                                  \
  static void int_##k(const uint64_t *x, const uint64_t *z,                   \
                      const uint64_t *w, uint64_t *y, long n) {               \
    (void)z;                                                                  \
    (void)w;                                                                  \
    for (long i = 0; i < n; i++) {                                            \
      y[4 * i] = (uint64_t)(uint32_t)nx_##k##_i32((int32_t)x[i]);             \
      y[4 * i + 1] = nx_##k##_u32((uint32_t)x[i]);                            \
      y[4 * i + 2] = (uint64_t)nx_##k##_i64((int64_t)x[i]);                   \
      y[4 * i + 3] = nx_##k##_u64(x[i]);                                      \
    }                                                                         \
  }
NX_I_UNARY(NX_INT_UNARY_LOOP)

#define NX_INT_BINARY_LOOP(k)                                                 \
  static void int_##k(const uint64_t *x, const uint64_t *z,                   \
                      const uint64_t *w, uint64_t *y, long n) {               \
    (void)w;                                                                  \
    for (long i = 0; i < n; i++) {                                            \
      y[4 * i] =                                                              \
          (uint64_t)(uint32_t)nx_##k##_i32((int32_t)x[i], (int32_t)z[i]);     \
      y[4 * i + 1] = nx_##k##_u32((uint32_t)x[i], (uint32_t)z[i]);            \
      y[4 * i + 2] = (uint64_t)nx_##k##_i64((int64_t)x[i], (int64_t)z[i]);    \
      y[4 * i + 3] = nx_##k##_u64(x[i], z[i]);                                \
    }                                                                         \
  }
NX_I_BINARY(NX_INT_BINARY_LOOP)

static void int_fma(const uint64_t *x, const uint64_t *z, const uint64_t *w,
                    uint64_t *y, long n) {
  for (long i = 0; i < n; i++) {
    y[4 * i] = (uint64_t)(uint32_t)nx_fma_i32((int32_t)x[i], (int32_t)z[i],
                                              (int32_t)w[i]);
    y[4 * i + 1] = nx_fma_u32((uint32_t)x[i], (uint32_t)z[i], (uint32_t)w[i]);
    y[4 * i + 2] =
        (uint64_t)nx_fma_i64((int64_t)x[i], (int64_t)z[i], (int64_t)w[i]);
    y[4 * i + 3] = nx_fma_u64(x[i], z[i], w[i]);
  }
}

static void int_where(const uint64_t *x, const uint64_t *z, const uint64_t *w,
                      uint64_t *y, long n) {
  for (long i = 0; i < n; i++) {
    y[4 * i] = (uint64_t)(uint32_t)nx_where_i32(x[i] != 0, (int32_t)z[i],
                                                (int32_t)w[i]);
    y[4 * i + 1] = nx_where_u32(x[i] != 0, (uint32_t)z[i], (uint32_t)w[i]);
    y[4 * i + 2] = (uint64_t)nx_where_i64(x[i] != 0, (int64_t)z[i], (int64_t)w[i]);
    y[4 * i + 3] = nx_where_u64(x[i] != 0, z[i], w[i]);
  }
}

static void int_threefry(const uint64_t *x, const uint64_t *z,
                         const uint64_t *w, uint64_t *y, long n) {
  (void)w;
  for (long i = 0; i < n; i++) {
    y[4 * i] = nx_threefry_u64(x[i], z[i]);
    y[4 * i + 1] = y[4 * i + 2] = y[4 * i + 3] = 0;
  }
}

#define NX_REAL_ROW(k, arity) {#k, arity, k##_f32, k##_f64},
#define NX_ROW1(k) NX_REAL_ROW(k, 1)
#define NX_ROW2(k) NX_REAL_ROW(k, 2)
#define NX_INT_ROW1(k) {#k, 1, int_##k},
#define NX_INT_ROW2(k) {#k, 2, int_##k},

static const nx_kinds_loops loops = {
    {NX_F_UNARY(NX_ROW1) NX_F_TRIG(NX_ROW1) NX_F_BINARY(NX_ROW2)
         NX_F_COMPARE(NX_ROW2) NX_REAL_ROW(fma, 3) NX_REAL_ROW(where, 3)},
    {NX_I_UNARY(NX_INT_ROW1) NX_I_BINARY(NX_INT_ROW2) {"fma", 3, int_fma},
     {"where", 3, int_where},
     {"threefry", 2, int_threefry}}};

#if defined(NX_CPU_V3)
const nx_kinds_loops *const nx_kinds_loops_v3 = &loops;
#else
const nx_kinds_loops *const nx_kinds_loops_base = &loops;
#endif

#else

const nx_kinds_loops *const nx_kinds_loops_v3 = NULL;

#endif
