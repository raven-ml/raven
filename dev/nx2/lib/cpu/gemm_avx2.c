/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* x86-64's contraction kernels, in the v3 table: dune compiles this file
   with AVX2 and FMA.

   A step of k adds 6 rows × two vectors of outputs, 6 × 16 of float32 or
   6 × 8 of float64: 12 accumulators, two vectors of b and one broadcast of
   a, 15 of the 16 vector registers. Each output adds a[i] · b[j] by VFMADD,
   one rounding, in increasing k. Two FMA ports with a latency of four want
   eight independent accumulators; twelve leave room for the loads. The
   accumulators are named variables: gcc keeps an array of them in memory
   and stores each at every step. */

#include <math.h>

#include "cpu.h"

#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)

/* The kernels' speed (cpu.h): 155 GFLOP/s on a performance core of
   kimchi, against the 146 GB/s its memcpy moves. */
#define SPEED 1

#define MR 6

/* Row i's two accumulators. */
#define DECL(V, i) V c##i##0, c##i##1
#define LOADC(LOAD, ZERO, W, i)                                    \
  c##i##0 = from == NX_CPU_FROM_ZERO ? ZERO : LOAD(y + i * ldc);    \
  c##i##1 = from == NX_CPU_FROM_ZERO ? ZERO : LOAD(y + i * ldc + W)
#define STEP(BCAST, FMA, i)          \
  ai = BCAST(a + i);                 \
  c##i##0 = FMA(ai, b0, c##i##0);    \
  c##i##1 = FMA(ai, b1, c##i##1)
#define STOREC(STORE, W, i)          \
  STORE(y + i * ldc, c##i##0);       \
  STORE(y + i * ldc + W, c##i##1)
#define ROWS(M, ...)                                                     \
  M(__VA_ARGS__, 0); M(__VA_ARGS__, 1); M(__VA_ARGS__, 2);               \
  M(__VA_ARGS__, 3); M(__VA_ARGS__, 4); M(__VA_ARGS__, 5)

#define KERNEL(name, T, V, W, LOAD, STORE, BCAST, FMA, ZERO)              \
  static void name(int64_t k, const void *va, int64_t lda,               \
                   const void *vb, void *vc, int64_t ldc,                \
                   nx_cpu_from from) {                                   \
    const T *a = va, *b = vb;                                            \
    T *y = vc;                                                           \
    ROWS(DECL, V);                                                       \
    ROWS(LOADC, LOAD, ZERO, W);                                          \
    for (int64_t p = 0; p < k; p++, a += lda, b += 2 * W) {              \
      V b0 = LOAD(b), b1 = LOAD(b + W), ai;                              \
      ROWS(STEP, BCAST, FMA);                                            \
    }                                                                    \
    ROWS(STOREC, STORE, W);                                              \
  }

KERNEL(kernel_f32, float, __m256, 8, _mm256_loadu_ps, _mm256_storeu_ps,
       _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_setzero_ps())
KERNEL(kernel_f64, double, __m256d, 4, _mm256_loadu_pd, _mm256_storeu_pd,
       _mm256_broadcast_sd, _mm256_fmadd_pd, _mm256_setzero_pd())

/* Lane order's dots and axpys: 16 lanes are 2 vectors of float32, 4 of
   float64; 4 rows of float32 or 2 of float64 hold 8 accumulators and b's
   vectors. */
#define VT __m256
#define LOAD _mm256_loadu_ps
#define STORE _mm256_storeu_ps
#define BCAST _mm256_set1_ps
#define FMA(c, a, b) _mm256_fmadd_ps(a, b, c)
#include "gemm_lanes.h"
DOT(dot1_f32, float, 8, 2, 1, fmaf)
DOT(dot2_f32, float, 8, 2, 2, fmaf)
DOT(dot4_f32, float, 8, 2, 4, fmaf)
DOTS(dot_f32, float, 4, dot4_f32, dot2_f32, dot1_f32)
AXPY(axpy_f32, float, 8, fmaf)
#undef VT
#undef LOAD
#undef STORE
#undef BCAST
#undef FMA
#define VT __m256d
#define LOAD _mm256_loadu_pd
#define STORE _mm256_storeu_pd
#define BCAST _mm256_set1_pd
#define FMA(c, a, b) _mm256_fmadd_pd(a, b, c)
DOT(dot1_f64, double, 4, 4, 1, fma)
DOT(dot2_f64, double, 4, 4, 2, fma)
DOTS(dot_f64, double, 2, dot2_f64, dot2_f64, dot1_f64)
AXPY(axpy_f64, double, 4, fma)

void nx_cpu_set_avx2(nx_cpu_target *t) {
  t->gemm[NX_FLOAT32] = (nx_cpu_gemm){
      .kernel = {kernel_f32, MR, 16, SPEED, 1},
      .mc = 96,
      .kc = 384,
      .nc = 3072};
  t->gemm[NX_FLOAT64] = (nx_cpu_gemm){
      .kernel = {kernel_f64, MR, 8, SPEED, 1},
      .mc = 96,
      .kc = 256,
      .nc = 3072};
  t->dot[NX_FLOAT32] = dot_f32;
  t->dot[NX_FLOAT64] = dot_f64;
  t->axpy[NX_FLOAT32] = axpy_f32;
  t->axpy[NX_FLOAT64] = axpy_f64;
}

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_avx2;

#endif
