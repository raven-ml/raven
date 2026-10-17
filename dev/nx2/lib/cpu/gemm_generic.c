/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The portable contraction kernels, compiled once per target as convert.c
   is: as itself for base, and as gemm_generic_v3.c for v3.

   A microkernel of 4 × 4 outputs and lane order's dot and axpy, each
   product fused into its addition by C's fma, which rounds once: the same
   bits as the vector kernels' fused instructions. The integers' microkernels
   multiply and add modulo their width, as unsigned numbers, which gives a
   signed accumulator's bits too. A target whose
   instructions have no fused multiply-add (base on x86-64) calls the C
   library's, which is exact and slow; every x86-64 host measured runs v3.
   A target's own kernels replace these where it has them. */

#include <math.h>

#include "cpu.h"

#if !defined(NX_CPU_V3) || defined(__x86_64__)

#define MR 4
#define NR 4

/* The kernels' speed (cpu.h): a target without vector kernels runs these
   on every product. It is measured for x86-64's base with glibc, where
   each fused multiply-add is a call of the C library's fma: on a
   performance core of fudge they run 1.4 GFLOP/s of float32 and 1.3 of
   float64, against the 151 GB/s its memcpy moves (floor-copy-256K-1t).
   Another target or C library runs them at its own speed. */
#define SPEED 0.01

#define KERNEL(name, T, FMA)                                                \
  static void name(int64_t m, int64_t n, int64_t k, const void *va,         \
                   int64_t a_row, int64_t a_k, const void *vb, int64_t b_k, \
                   void *vc, int64_t ldc, nx_cpu_from from) {               \
    for (int64_t i0 = 0; i0 < m; i0 += MR) {                                \
      int64_t rows = m - i0 < MR ? m - i0 : MR;                             \
      const T *a = (const T *)va + i0 * a_row, *b = vb;                     \
      T *c = (T *)vc + i0 * ldc, t[MR][NR];                                 \
      for (int64_t i = 0; i < rows; i++)                                    \
        for (int64_t j = 0; j < n; j++)                                     \
          t[i][j] = from == NX_CPU_FROM_ZERO ? 0 : c[i * ldc + j];          \
      for (int64_t p = 0; p < k; p++, a += a_k, b += b_k)                   \
        for (int64_t i = 0; i < rows; i++)                                  \
          for (int64_t j = 0; j < n; j++)                                   \
            t[i][j] = FMA(a[i * a_row], b[j], t[i][j]);                     \
      for (int64_t i = 0; i < rows; i++)                                    \
        for (int64_t j = 0; j < n; j++) c[i * ldc + j] = t[i][j];           \
    }                                                                       \
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

/* Sets y[j], for each of the [c] columns of b, to lane order's sum of
   the block of [k] products of a's row and the column (cpu.h's lane
   tile). */
#define TILE(name, T, FMA)                                                  \
  static void name(const void *va, int64_t a_k, const void *vb,             \
                   int64_t b_k, int64_t k, int64_t c, void *vy) {           \
    const T *a = va, *b = vb;                                               \
    T *y = vy;                                                              \
    for (int64_t j = 0; j < c; j++) {                                       \
      T s[NX_CPU_LANES] = {0};                                              \
      for (int64_t t = 0; t < k; t++)                                       \
        s[t % NX_CPU_LANES] =                                               \
            FMA(a[t * a_k], b[t * b_k + j], s[t % NX_CPU_LANES]);           \
      for (int w = NX_CPU_LANES / 2; w > 0; w /= 2)                         \
        for (int q = 0; q < w; q++) s[q] = s[q] + s[q + w];                 \
      y[j] = s[0];                                                          \
    }                                                                       \
  }

/* Adds into each of the [m] elements of [y] the [k] products of a's
   column p, [lda] apart, and [x]'s pth, in increasing p. */
#define GEMV(name, T, FMA)                                                  \
  static void name(int64_t m, int64_t k, const void *va, int64_t lda,       \
                   const void *vx, void *vy) {                              \
    const T *a = va, *x = vx;                                               \
    T *y = vy;                                                              \
    for (int64_t i = 0; i < m; i++) {                                       \
      T s = y[i];                                                           \
      for (int64_t p = 0; p < k; p++) s = FMA(a[i + p * lda], x[p], s);     \
      y[i] = s;                                                             \
    }                                                                       \
  }

KERNEL(kernel_f32, float, fmaf)
DOT(dot_f32, float, fmaf)
AXPY(axpy_f32, float, fmaf)
TILE(tile_f32, float, fmaf)
GEMV(gemv_f32, float, fmaf)
KERNEL(kernel_f64, double, fma)
DOT(dot_f64, double, fma)
AXPY(axpy_f64, double, fma)
TILE(tile_f64, double, fma)
GEMV(gemv_f64, double, fma)

/* z + x·y modulo the width of the [T] it is stored in: computed in an
   unsigned type no narrower than int, which C does not promote to int. */
#define MAD32(x, y, z) ((z) + (uint32_t)(x) * (uint32_t)(y))
#define MAD64(x, y, z) ((z) + (uint64_t)(x) * (uint64_t)(y))
KERNEL(kernel_u8, uint8_t, MAD32)
KERNEL(kernel_u16, uint16_t, MAD32)
KERNEL(kernel_u32, uint32_t, MAD32)
KERNEL(kernel_u64, uint64_t, MAD64)

/* The rows of a block of a, and those up to which a product streams b
   past all of a (cpu.h's mc and few). */
#define MC 64

static nx_cpu_gemm gemm_of(nx_cpu_kernel f) {
  return (nx_cpu_gemm){.kernel = {.f = f,
                                  .mr = MR,
                                  .nr = NR,
                                  .a = NX_CPU_A_ELEMENTS,
                                  .b = NX_CPU_B_COLUMNS,
                                  .flops_per_byte = SPEED,
                                  .shared = 1},
                       .mc = MC,
                       .kc = 256,
                       .nc = 1024,
                       .few = MC};
}

static void set(nx_cpu_target *t) {
  t->gemm[NX_FLOAT32] = gemm_of(kernel_f32);
  t->gemm[NX_FLOAT64] = gemm_of(kernel_f64);
  t->gemm[NX_INT8] = t->gemm[NX_UINT8] = gemm_of(kernel_u8);
  t->gemm[NX_INT16] = t->gemm[NX_UINT16] = gemm_of(kernel_u16);
  t->gemm[NX_INT32] = t->gemm[NX_UINT32] = gemm_of(kernel_u32);
  t->gemm[NX_INT64] = t->gemm[NX_UINT64] = gemm_of(kernel_u64);
  t->dot[NX_FLOAT32] = dot_f32;
  t->dot[NX_FLOAT64] = dot_f64;
  t->axpy[NX_FLOAT32] = axpy_f32;
  t->axpy[NX_FLOAT64] = axpy_f64;
  t->tile[NX_FLOAT32] = tile_f32;
  t->tile[NX_FLOAT64] = tile_f64;
  t->gemv[NX_FLOAT32] = gemv_f32;
  t->gemv[NX_FLOAT64] = gemv_f64;
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
