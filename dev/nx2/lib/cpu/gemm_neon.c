/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* arm64's contraction kernels, in the base table: NEON is in every arm64
   host.

   float32 adds 8 × 12 outputs per step of k: 24 accumulators of four lanes,
   three vectors of b and a broadcast of a, 28 of the 32 vector registers;
   float64 8 × 6, 24 of two lanes. Each output adds b[j] · a[i] by FMLA, one
   rounding, in increasing k. The M1's P-core issues four FMLA a cycle with
   a latency of four, so 24 independent accumulators keep its pipes full;
   its three load ports take a step's eight broadcasts (LD1R) and three
   vectors of b in four of the six cycles its 24 FMLA take.

   a's elements are broadcast one at a time, so that its rows are read in
   place wherever they lie. A tile of fewer rows points its missing rows at
   its last and stores only its own; one of fewer columns moves its outputs
   through a row on the stack. */

#include <math.h>
#include <string.h>

#include "cpu.h"

#if defined(__aarch64__)

/* The kernels' speed (cpu.h): 99.8 GFLOP/s of float32 and 50.2 of float64
   on a performance core of the M1 Max, against the 100-160 GB/s its
   memcpy moves. */
#define SPEED32 1.0
#define SPEED64 0.5

#define MR 8
#define NV 3

/* A kernel of MR rows by NV vectors of [L] lanes of [T]. */
#define KERNEL(name, T, V, L, LOAD, STORE, DUP, FMA, ZERO)                   \
  static void name(int64_t m, int64_t n, int64_t k, const void *va,         \
                   int64_t a_row, int64_t a_k, const void *vb, int64_t b_k, \
                   void *vc, int64_t ldc, nx_cpu_from from) {               \
    for (int64_t i0 = 0; i0 < m; i0 += MR) {                                \
      int64_t rows = m - i0 < MR ? m - i0 : MR;                             \
      const T *a = (const T *)va + i0 * a_row, *b = vb, *ar[MR];            \
      T *y = (T *)vc + i0 * ldc, t[NV * L] = {0};                           \
      V c[MR][NV];                                                          \
      for (int i = 0; i < MR; i++) {                                        \
        ar[i] = a + (i < rows ? i : rows - 1) * a_row;                      \
        if (from == NX_CPU_FROM_ZERO || i >= rows) {                        \
          for (int v = 0; v < NV; v++) c[i][v] = ZERO;                      \
          continue;                                                         \
        }                                                                   \
        const T *r = y + i * ldc;                                           \
        if (n < NV * L) r = memcpy(t, r, (size_t)n * sizeof(T));            \
        for (int v = 0; v < NV; v++) c[i][v] = LOAD(r + L * v);             \
      }                                                                     \
      for (int64_t p = 0, q = 0; p < k; p++, q += a_k, b += b_k) {          \
        V b0 = LOAD(b), b1 = LOAD(b + L), b2 = LOAD(b + 2 * L);             \
        for (int i = 0; i < MR; i++) {                                      \
          V x = DUP(ar[i] + q);                                             \
          c[i][0] = FMA(c[i][0], b0, x);                                    \
          c[i][1] = FMA(c[i][1], b1, x);                                    \
          c[i][2] = FMA(c[i][2], b2, x);                                    \
        }                                                                   \
      }                                                                     \
      for (int i = 0; i < rows; i++) {                                      \
        T *r = n < NV * L ? t : y + i * ldc;                                \
        for (int v = 0; v < NV; v++) STORE(r + L * v, c[i][v]);             \
        if (n < NV * L) memcpy(y + i * ldc, t, (size_t)n * sizeof(T));      \
      }                                                                     \
    }                                                                       \
  }

KERNEL(kernel_f32, float, float32x4_t, 4, vld1q_f32, vst1q_f32,
       vld1q_dup_f32, vfmaq_f32, vdupq_n_f32(0))
KERNEL(kernel_f64, double, float64x2_t, 2, vld1q_f64, vst1q_f64,
       vld1q_dup_f64, vfmaq_f64, vdupq_n_f64(0))

/* The integers of 8 to 32 bits by MLA, modulo their width, as unsigned
   numbers, which gives a signed accumulator's bits too; 64-bit ones have
   no vector multiply and keep the portable kernel. */
KERNEL(kernel_u8, uint8_t, uint8x16_t, 16, vld1q_u8, vst1q_u8, vld1q_dup_u8,
       vmlaq_u8, vdupq_n_u8(0))
KERNEL(kernel_u16, uint16_t, uint16x8_t, 8, vld1q_u16, vst1q_u16,
       vld1q_dup_u16, vmlaq_u16, vdupq_n_u16(0))
KERNEL(kernel_u32, uint32_t, uint32x4_t, 4, vld1q_u32, vst1q_u32,
       vld1q_dup_u32, vmlaq_u32, vdupq_n_u32(0))

/* Quads of int8 or uint8 by SDOT or UDOT, where the host has them: each
   lane adds the four products of a word of b and the broadcast word of a,
   16 products an instruction where MLA makes 4. */
#if defined(__ARM_FEATURE_DOTPROD)
#define SDOT(c, b, x)                                                     \
  vreinterpretq_u32_s32(vdotq_s32(vreinterpretq_s32_u32(c),              \
                                  vreinterpretq_s8_u32(b),               \
                                  vreinterpretq_s8_u32(x)))
#define UDOT(c, b, x) \
  vdotq_u32(c, vreinterpretq_u8_u32(b), vreinterpretq_u8_u32(x))
KERNEL(kernel_sdot, uint32_t, uint32x4_t, 4, vld1q_u32, vst1q_u32,
       vld1q_dup_u32, SDOT, vdupq_n_u32(0))
KERNEL(kernel_udot, uint32_t, uint32x4_t, 4, vld1q_u32, vst1q_u32,
       vld1q_dup_u32, UDOT, vdupq_n_u32(0))
#endif

/* Lane order's dots and axpys: 16 lanes are 4 vectors of float32, 8 of
   float64; 4 rows of float32 or 2 of float64 hold 16 accumulators and b's
   vectors. */
#define VT float32x4_t
#define LOAD vld1q_f32
#define STORE vst1q_f32
#define BCAST vdupq_n_f32
#define FMA vfmaq_f32
#include "gemm_lanes.h"
DOT(dot1_f32, float, 4, 4, 1, fmaf)
DOT(dot2_f32, float, 4, 4, 2, fmaf)
DOT(dot4_f32, float, 4, 4, 4, fmaf)
DOTS(dot_f32, float, 4, dot4_f32, dot2_f32, dot1_f32)
AXPY(axpy_f32, float, 4, fmaf)
#undef VT
#undef LOAD
#undef STORE
#undef BCAST
#undef FMA
#define VT float64x2_t
#define LOAD vld1q_f64
#define STORE vst1q_f64
#define BCAST vdupq_n_f64
#define FMA vfmaq_f64
DOT(dot1_f64, double, 2, 8, 1, fma)
DOT(dot2_f64, double, 2, 8, 2, fma)
DOTS(dot_f64, double, 2, dot2_f64, dot2_f64, dot1_f64)
AXPY(axpy_f64, double, 2, fma)

/* The rows of a block of a, and those up to which a product streams b
   past all of a (cpu.h's mc and few). */
#define MC 128

/* The integers' kernels of [lanes] lanes: a sliver of b is as many bytes
   along k as float32's. */
static nx_cpu_gemm integers(nx_cpu_kernel f, int lanes) {
  return (nx_cpu_gemm){.kernel = {.f = f,
                                  .mr = MR,
                                  .nr = NV * lanes,
                                  .a = NX_CPU_A_ELEMENTS,
                                  .b = NX_CPU_B_SLIVER,
                                  .flops_per_byte = SPEED32 * lanes / 4,
                                  .shared = 1},
                       .mc = MC,
                       .kc = 512,
                       .nc = 3072,
                       .few = MC};
}

void nx_cpu_set_neon(nx_cpu_target *t) {
  t->gemm[NX_FLOAT32] = (nx_cpu_gemm){
      .kernel = {.f = kernel_f32,
                 .mr = MR,
                 .nr = 12,
                 .a = NX_CPU_A_ELEMENTS,
                 .b = NX_CPU_B_SLIVER,
                 .flops_per_byte = SPEED32,
                 .shared = 1},
      .mc = MC,
      .kc = 512,
      .nc = 3072,
      .few = MC};
  t->gemm[NX_FLOAT64] = (nx_cpu_gemm){
      .kernel = {.f = kernel_f64,
                 .mr = MR,
                 .nr = 6,
                 .a = NX_CPU_A_ELEMENTS,
                 .b = NX_CPU_B_SLIVER,
                 .flops_per_byte = SPEED64,
                 .shared = 1},
      .mc = MC,
      .kc = 256,
      .nc = 3072,
      .few = MC};
  t->dot[NX_FLOAT32] = dot_f32;
  t->dot[NX_FLOAT64] = dot_f64;
  t->axpy[NX_FLOAT32] = axpy_f32;
  t->axpy[NX_FLOAT64] = axpy_f64;
  t->gemm[NX_INT8] = t->gemm[NX_UINT8] = integers(kernel_u8, 16);
  t->gemm[NX_INT16] = t->gemm[NX_UINT16] = integers(kernel_u16, 8);
  t->gemm[NX_INT32] = t->gemm[NX_UINT32] = integers(kernel_u32, 4);
#if defined(__ARM_FEATURE_DOTPROD)
  t->quads[NX_INT8] = integers(kernel_sdot, 4);
  t->quads[NX_UINT8] = integers(kernel_udot, 4);
#endif
}

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_neon;

#endif
