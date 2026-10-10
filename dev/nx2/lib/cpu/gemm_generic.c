/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The portable contraction kernels, compiled once per target as convert.c
   is: as itself for base, and as gemm_generic_v3.c for v3.

   A microkernel of 4 × 4 outputs and lane order's dot and axpy, each
   product fused into its addition by C's fma, which rounds once: the same
   bits as the vector kernels' fused instructions. A target whose
   instructions have no fused multiply-add (base on x86-64) calls the C
   library's, which is exact and slow; every x86-64 host measured runs v3.
   A target's own kernels replace these where it has them. */

#include <math.h>

#include "cpu.h"

#if !defined(NX_CPU_V3) || defined(__x86_64__)

#define MR 4
#define NR 4

/* The kernels' speed (cpu.h), as the vector kernels': a target without
   their own runs these on every product. */
#define SPEED 1

#define KERNEL(name, T, FMA)                                                \
  static void name(int64_t k, const void *va, int64_t lda, const void *vb,  \
                   void *vc, int64_t ldc, nx_cpu_from from) {               \
    const T *a = va, *b = vb;                                               \
    T *c = vc, t[MR][NR];                                                   \
    for (int i = 0; i < MR; i++)                                            \
      for (int j = 0; j < NR; j++)                                          \
        t[i][j] = from == NX_CPU_FROM_ZERO ? 0 : c[i * ldc + j];            \
    for (int64_t p = 0; p < k; p++, a += lda, b += NR)                      \
      for (int i = 0; i < MR; i++)                                          \
        for (int j = 0; j < NR; j++) t[i][j] = FMA(a[i], b[j], t[i][j]);    \
    for (int i = 0; i < MR; i++)                                            \
      for (int j = 0; j < NR; j++) c[i * ldc + j] = t[i][j];                \
  }

/* Adds, for each of [r] rows of [a], [lda] apart, the [n] products with
   [b] into the row's lanes at [l]: term t into lane t modulo
   NX_CPU_LANES. */
#define DOT(name, T, FMA)                                                   \
  static void name(const void *va, int64_t lda, int r, const void *vb,      \
                   int64_t n, void *vl) {                                   \
    const T *a = va, *b = vb;                                               \
    T *l = vl;                                                              \
    for (int i = 0; i < r; i++)                                             \
      for (int64_t t = 0; t < n; t++) {                                     \
        T *x = l + i * NX_CPU_LANES + t % NX_CPU_LANES;                     \
        *x = FMA(a[i * lda + t], b[t], *x);                                 \
      }                                                                     \
  }

/* Adds each of [r] rows' element of [a], [lda] apart, times each of the
   [n] elements of [b] into the row's accumulators at [y], [ldy] apart,
   having prefetched [next]. */
#define AXPY(name, T, FMA)                                                  \
  static void name(const void *va, int64_t lda, int r, const void *vb,      \
                   int64_t n, void *vy, int64_t ldy, const void *next) {    \
    const T *a = va, *b = vb;                                               \
    for (int64_t q = 0; q < n * (int64_t)sizeof(T); q += 64)                \
      __builtin_prefetch((const char *)next + q);                           \
    for (int i = 0; i < r; i++) {                                           \
      T x = a[i * lda], *y = (T *)vy + i * ldy;                             \
      for (int64_t j = 0; j < n; j++) y[j] = FMA(x, b[j], y[j]);            \
    }                                                                       \
  }

KERNEL(kernel_f32, float, fmaf)
DOT(dot_f32, float, fmaf)
AXPY(axpy_f32, float, fmaf)
KERNEL(kernel_f64, double, fma)
DOT(dot_f64, double, fma)
AXPY(axpy_f64, double, fma)

static void set(nx_cpu_target *t) {
  t->gemm[NX_FLOAT32] = (nx_cpu_gemm){
      .kernel = {kernel_f32, MR, NR, SPEED, 1},
      .mc = 64,
      .kc = 256,
      .nc = 1024};
  t->gemm[NX_FLOAT64] = (nx_cpu_gemm){
      .kernel = {kernel_f64, MR, NR, SPEED, 1},
      .mc = 64,
      .kc = 256,
      .nc = 1024};
  t->dot[NX_FLOAT32] = dot_f32;
  t->dot[NX_FLOAT64] = dot_f64;
  t->axpy[NX_FLOAT32] = axpy_f32;
  t->axpy[NX_FLOAT64] = axpy_f64;
}

#if defined(NX_CPU_V3)
void nx_cpu_set_gemm_v3(nx_cpu_target *t) { set(t); }
#else
void nx_cpu_set_gemm_base(nx_cpu_target *t) { set(t); }
#endif

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_v3;

#endif
