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

#include "cpu.h"

#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)

#define MR 6

/* Row i's two accumulators. */
#define DECL(V, i) V c##i##0, c##i##1
#define LOADC(LOAD, W, i)            \
  c##i##0 = LOAD(y + i * ldc);       \
  c##i##1 = LOAD(y + i * ldc + W)
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

#define KERNEL(name, T, V, W, LOAD, STORE, BCAST, FMA)                    \
  static void name(int64_t k, const void *va, int64_t lda,               \
                   const void *vb, void *vc, int64_t ldc) {              \
    const T *a = va, *b = vb;                                            \
    T *y = vc;                                                           \
    ROWS(DECL, V);                                                       \
    ROWS(LOADC, LOAD, W);                                                \
    for (int64_t p = 0; p < k; p++, a += lda, b += 2 * W) {              \
      V b0 = LOAD(b), b1 = LOAD(b + W), ai;                              \
      ROWS(STEP, BCAST, FMA);                                            \
    }                                                                    \
    ROWS(STOREC, STORE, W);                                              \
  }

KERNEL(kernel_f32, float, __m256, 8, _mm256_loadu_ps, _mm256_storeu_ps,
       _mm256_broadcast_ss, _mm256_fmadd_ps)
KERNEL(kernel_f64, double, __m256d, 4, _mm256_loadu_pd, _mm256_storeu_pd,
       _mm256_broadcast_sd, _mm256_fmadd_pd)

/* Thin tiles: 12 accumulators, the rows' broadcasts of a, and b read by
   the fused adds from memory: 16 registers at 4 rows. */
#define FMA(c, a, b) _mm256_fmadd_ps(a, b, c)
#define LOAD _mm256_loadu_ps
#define STORE _mm256_storeu_ps
#define BCAST _mm256_broadcast_ss
#include "gemm_thin.h"
THIN(thin1_f32, float, __m256, 8, 1, 12)
THIN(thin2_f32, float, __m256, 8, 2, 6)
THIN(thin4_f32, float, __m256, 8, 4, 3)
#undef FMA
#undef LOAD
#undef STORE
#undef BCAST
#define FMA(c, a, b) _mm256_fmadd_pd(a, b, c)
#define LOAD _mm256_loadu_pd
#define STORE _mm256_storeu_pd
#define BCAST _mm256_broadcast_sd
THIN(thin1_f64, double, __m256d, 4, 1, 12)
THIN(thin2_f64, double, __m256d, 4, 2, 6)
THIN(thin4_f64, double, __m256d, 4, 4, 3)

void nx_cpu_set_avx2(nx_cpu_target *t) {
  t->gemm[NX_FLOAT32] = (nx_cpu_gemm){
      .kernel = kernel_f32,
      .dot = t->gemm[NX_FLOAT32].dot,
      .mr = MR,
      .nr = 16,
      .mc = 96,
      .kc = 384,
      .nc = 3072,
      .thin = {{thin1_f32, 96}, {thin2_f32, 48}, {thin4_f32, 24}}};
  t->gemm[NX_FLOAT64] = (nx_cpu_gemm){
      .kernel = kernel_f64,
      .dot = t->gemm[NX_FLOAT64].dot,
      .mr = MR,
      .nr = 8,
      .mc = 96,
      .kc = 256,
      .nc = 3072,
      .thin = {{thin1_f64, 48}, {thin2_f64, 24}, {thin4_f64, 12}}};
}

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_avx2;

#endif
