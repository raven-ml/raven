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
   and stores each at every step.

   a's elements are broadcast one at a time, so that its rows are read in
   place wherever they lie. A tile of fewer rows points its missing rows at
   its last and stores only its own; one of fewer columns loads b and its
   outputs and stores them through masks, so that it reads no column past
   its own (NX_CPU_B_COLUMNS). Before a tile, the next tile's first lines
   of a and of the outputs are prefetched: its rows start cold, where the
   hardware's prefetchers follow each row once it streams. */

#include <math.h>

#include "cpu.h"

#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)

/* The kernels' speed (cpu.h): 155 GFLOP/s on a performance core of
   kimchi, against the 146 GB/s its memcpy moves. */
#define SPEED 1

#define MR 6

/* The masks of a vector's first q lanes, q from 0 to its lanes: a run of
   ones then zeros, read from q lanes before the zeros. */
static const int32_t ones32[16] = {-1, -1, -1, -1, -1, -1, -1, -1};
static const int64_t ones64[8] = {-1, -1, -1, -1};
#define MASK32(q) _mm256_loadu_si256((const __m256i *)(ones32 + 8 - (q)))
#define MASK64(q) _mm256_loadu_si256((const __m256i *)(ones64 + 4 - (q)))

/* Row i's two accumulators and its row of a. */
#define DECL(V, T, i)                                       \
  V c##i##0, c##i##1;                                       \
  const T *a##i = a + (i < rows ? i : rows - 1) * a_row
#define LOADC(LOAD, MLOAD, ZERO, W, i)                             \
  if (from == NX_CPU_FROM_ZERO || i >= rows) {                     \
    c##i##0 = ZERO;                                                \
    c##i##1 = ZERO;                                                \
  } else if (n == 2 * W) {                                         \
    c##i##0 = LOAD(y + i * ldc);                                   \
    c##i##1 = LOAD(y + i * ldc + W);                               \
  } else {                                                         \
    c##i##0 = MLOAD(y + i * ldc, m0);                              \
    c##i##1 = MLOAD(y + i * ldc + W, m1);                          \
  }
#define STEP(BCAST, FMA, i)          \
  ai = BCAST(a##i + q);              \
  c##i##0 = FMA(ai, b0, c##i##0);    \
  c##i##1 = FMA(ai, b1, c##i##1)
#define STOREC(STORE, MSTORE, W, i)            \
  if (i < rows && n == 2 * W) {                \
    STORE(y + i * ldc, c##i##0);               \
    STORE(y + i * ldc + W, c##i##1);           \
  } else if (i < rows) {                       \
    MSTORE(y + i * ldc, m0, c##i##0);          \
    MSTORE(y + i * ldc + W, m1, c##i##1);      \
  }
#define ROWS(M, ...)                                                     \
  M(__VA_ARGS__, 0); M(__VA_ARGS__, 1); M(__VA_ARGS__, 2);               \
  M(__VA_ARGS__, 3); M(__VA_ARGS__, 4); M(__VA_ARGS__, 5)

#define KERNEL(name, T, V, W, MASK, LOAD, MLOAD, BLOAD, STORE, MSTORE,      \
               BCAST, FMA, ZERO)                                            \
  static void name(int64_t m, int64_t n, int64_t k, const void *va,         \
                   int64_t a_row, int64_t a_k, const void *vb, int64_t b_k, \
                   void *vc, int64_t ldc, nx_cpu_from from) {               \
    __m256i m0 = MASK(n < W ? n : W), m1 = MASK(n > W ? n - W : 0);         \
    for (int64_t i0 = 0; i0 < m; i0 += MR) {                                \
      int64_t rows = m - i0 < MR ? m - i0 : MR;                             \
      const T *a = (const T *)va + i0 * a_row, *b = vb;                     \
      T *y = (T *)vc + i0 * ldc;                                            \
      for (int64_t i = MR; i < 2 * MR && i0 + i < m; i++) {                 \
        _mm_prefetch((const char *)(a + i * a_row), _MM_HINT_T0);           \
        _mm_prefetch((const char *)(y + i * ldc), _MM_HINT_T0);             \
      }                                                                     \
      ROWS(DECL, V, T);                                                     \
      ROWS(LOADC, LOAD, MLOAD, ZERO, W);                                    \
      for (int64_t p = 0, q = 0; p < k; p++, q += a_k, b += b_k) {          \
        V b0 = BLOAD(b, m0), b1 = BLOAD(b + W, m1), ai;                     \
        ROWS(STEP, BCAST, FMA);                                             \
      }                                                                     \
      ROWS(STOREC, STORE, MSTORE, W);                                       \
    }                                                                       \
  }

/* A tile of every column loads b whole, one of fewer through masks: two
   instances, so that the masks cost the whole tile's loop nothing. */
#define WHOLE32(p, m) _mm256_loadu_ps(p)
#define WHOLE64(p, m) _mm256_loadu_pd(p)
#define TILES(name, T, NR)                                                   \
  static void name(int64_t m, int64_t n, int64_t k, const void *a,          \
                   int64_t a_row, int64_t a_k, const void *b, int64_t b_k,  \
                   void *c, int64_t ldc, nx_cpu_from from) {                \
    (n == NR ? whole_##T : edge_##T)(m, n, k, a, a_row, a_k, b, b_k, c, ldc, \
                                     from);                                 \
  }

KERNEL(whole_f32, float, __m256, 8, MASK32, _mm256_loadu_ps,
       _mm256_maskload_ps, WHOLE32, _mm256_storeu_ps, _mm256_maskstore_ps,
       _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_setzero_ps())
KERNEL(edge_f32, float, __m256, 8, MASK32, _mm256_loadu_ps,
       _mm256_maskload_ps, _mm256_maskload_ps, _mm256_storeu_ps,
       _mm256_maskstore_ps, _mm256_broadcast_ss, _mm256_fmadd_ps,
       _mm256_setzero_ps())
TILES(kernel_f32, f32, 16)
KERNEL(whole_f64, double, __m256d, 4, MASK64, _mm256_loadu_pd,
       _mm256_maskload_pd, WHOLE64, _mm256_storeu_pd, _mm256_maskstore_pd,
       _mm256_broadcast_sd, _mm256_fmadd_pd, _mm256_setzero_pd())
KERNEL(edge_f64, double, __m256d, 4, MASK64, _mm256_loadu_pd,
       _mm256_maskload_pd, _mm256_maskload_pd, _mm256_storeu_pd,
       _mm256_maskstore_pd, _mm256_broadcast_sd, _mm256_fmadd_pd,
       _mm256_setzero_pd())
TILES(kernel_f64, f64, 8)

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

/* The rows up to which a product streams b past all of a (gemm.c): its
   one job, which packs b's slivers as it computes them, takes less time
   than chain order's jobs, which pack a panel of b on few threads before
   computing it. On the i9-9900K's 16 threads a 256³ float32 product takes
   36.5 µs so against 51, a 4096 × 4096 weight against 128 rows 5.10 ms
   against 8.35. */
#define FEW 256

void nx_cpu_set_avx2(nx_cpu_target *t) {
  t->gemm[NX_FLOAT32] = (nx_cpu_gemm){
      .kernel = {kernel_f32, MR, 16, NX_CPU_A_ELEMENTS, NX_CPU_B_COLUMNS,
                 SPEED, 1},
      .mc = 96,
      .kc = 384,
      .nc = 3072,
      .few = FEW};
  t->gemm[NX_FLOAT64] = (nx_cpu_gemm){
      .kernel = {kernel_f64, MR, 8, NX_CPU_A_ELEMENTS, NX_CPU_B_COLUMNS,
                 SPEED, 1},
      .mc = 96,
      .kc = 256,
      .nc = 3072,
      .few = FEW};
  t->dot[NX_FLOAT32] = dot_f32;
  t->dot[NX_FLOAT64] = dot_f64;
  t->axpy[NX_FLOAT32] = axpy_f32;
  t->axpy[NX_FLOAT64] = axpy_f64;
}

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_avx2;

#endif
