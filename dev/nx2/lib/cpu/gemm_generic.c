/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The portable contraction kernels, compiled once per target as convert.c
   is: as itself for base, and as gemm_generic_v3.c for v3.

   A microkernel of 4 × 4 outputs and the lanes of a dot, each product
   fused into its addition by C's fma, which rounds once: the same bits as
   the vector kernels' fused instructions. A target whose instructions
   have no fused multiply-add (base on x86-64) calls the C library's, which
   is exact and slow; every x86-64 host measured runs v3. A target's own
   microkernels replace these where it has them; every target runs this
   dot. */

#include <math.h>

#include "cpu.h"

#if !defined(NX_CPU_V3) || defined(__x86_64__)

#define MR 4
#define NR 4

#define KERNEL(name, T, FMA)                                                \
  static void name(int64_t k, const void *va, int64_t lda, const void *vb,  \
                   void *vc, int64_t ldc) {                                 \
    const T *a = va, *b = vb;                                               \
    T *c = vc, t[MR][NR];                                                   \
    for (int i = 0; i < MR; i++)                                            \
      for (int j = 0; j < NR; j++) t[i][j] = c[i * ldc + j];                \
    for (int64_t p = 0; p < k; p++, a += lda, b += NR)                      \
      for (int i = 0; i < MR; i++)                                          \
        for (int j = 0; j < NR; j++) t[i][j] = FMA(a[i], b[j], t[i][j]);    \
    for (int i = 0; i < MR; i++)                                            \
      for (int j = 0; j < NR; j++) c[i * ldc + j] = t[i][j];                \
  }

/* Adds the [n] products of [a] and [b] into lanes [l]: term t into lane t
   modulo NX_CPU_LANES. */
#define DOT(name, T, FMA)                                                   \
  static void name(const void *va, const void *vb, int64_t n, void *vl) {   \
    const T *a = va, *b = vb;                                               \
    T *l = vl;                                                              \
    int64_t t = 0;                                                          \
    for (; t + NX_CPU_LANES <= n; t += NX_CPU_LANES)                        \
      for (int i = 0; i < NX_CPU_LANES; i++)                                \
        l[i] = FMA(a[t + i], b[t + i], l[i]);                               \
    for (int i = 0; t + i < n; i++) l[i] = FMA(a[t + i], b[t + i], l[i]);   \
  }

KERNEL(kernel_f32, float, fmaf)
DOT(dot_f32, float, fmaf)
KERNEL(kernel_f64, double, fma)
DOT(dot_f64, double, fma)

static void set(nx_cpu_target *t) {
  t->gemm[NX_FLOAT32] = (nx_cpu_gemm){
      .kernel = {kernel_f32, MR, NR}, .mc = 64, .kc = 256, .nc = 1024};
  t->gemm[NX_FLOAT64] = (nx_cpu_gemm){
      .kernel = {kernel_f64, MR, NR}, .mc = 64, .kc = 256, .nc = 1024};
  t->dot[NX_FLOAT32] = dot_f32;
  t->dot[NX_FLOAT64] = dot_f64;
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
