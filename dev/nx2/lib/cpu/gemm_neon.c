/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* arm64's contraction kernels, in the base table: NEON is in every arm64
   host.

   float32 adds 8 × 12 outputs per step of k: 24 accumulators of four lanes,
   two vectors of a and three of b, 29 of the 32 vector registers. Each
   output adds b[j] · a[i] by FMLA by element, one rounding, in increasing
   k. The M1's P-core issues four FMLA a cycle with a latency of four, so
   24 independent accumulators keep its pipes full; loads take five of its
   three load ports' cycles per 24 FMLA. */

#include "cpu.h"

#if defined(__aarch64__)

#define F32_MR 8
#define F32_NR 12

/* Row i of the tile, a[i] being lane [lane] of [av]. */
#define ROW(i, av, lane)                                  \
  c[i][0] = vfmaq_laneq_f32(c[i][0], b0, av, lane);       \
  c[i][1] = vfmaq_laneq_f32(c[i][1], b1, av, lane);       \
  c[i][2] = vfmaq_laneq_f32(c[i][2], b2, av, lane);

static void kernel_f32(int64_t k, const void *va, const void *vb, void *vc,
                       int64_t ldc) {
  const float *a = va, *b = vb;
  float *y = vc;
  float32x4_t c[F32_MR][3];
  for (int i = 0; i < F32_MR; i++)
    for (int v = 0; v < 3; v++) c[i][v] = vld1q_f32(y + i * ldc + 4 * v);
  for (int64_t p = 0; p < k; p++, a += F32_MR, b += F32_NR) {
    float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
    float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4),
                b2 = vld1q_f32(b + 8);
    ROW(0, a0, 0)
    ROW(1, a0, 1)
    ROW(2, a0, 2)
    ROW(3, a0, 3)
    ROW(4, a1, 0)
    ROW(5, a1, 1)
    ROW(6, a1, 2)
    ROW(7, a1, 3)
  }
  for (int i = 0; i < F32_MR; i++)
    for (int v = 0; v < 3; v++) vst1q_f32(y + i * ldc + 4 * v, c[i][v]);
}

#define F64_MR 8
#define F64_NR 6

/* Row i of the float64 tile, a[i] being lane [lane] of [av]. */
#define ROW64(i, av, lane)                                \
  c[i][0] = vfmaq_laneq_f64(c[i][0], b0, av, lane);       \
  c[i][1] = vfmaq_laneq_f64(c[i][1], b1, av, lane);       \
  c[i][2] = vfmaq_laneq_f64(c[i][2], b2, av, lane);

/* float64 adds 8 × 6 outputs per step: 24 accumulators of two lanes, four
   vectors of a and three of b, 31 registers. */
static void kernel_f64(int64_t k, const void *va, const void *vb, void *vc,
                       int64_t ldc) {
  const double *a = va, *b = vb;
  double *y = vc;
  float64x2_t c[F64_MR][3];
  for (int i = 0; i < F64_MR; i++)
    for (int v = 0; v < 3; v++) c[i][v] = vld1q_f64(y + i * ldc + 2 * v);
  for (int64_t p = 0; p < k; p++, a += F64_MR, b += F64_NR) {
    float64x2_t a0 = vld1q_f64(a), a1 = vld1q_f64(a + 2),
                a2 = vld1q_f64(a + 4), a3 = vld1q_f64(a + 6);
    float64x2_t b0 = vld1q_f64(b), b1 = vld1q_f64(b + 2),
                b2 = vld1q_f64(b + 4);
    ROW64(0, a0, 0)
    ROW64(1, a0, 1)
    ROW64(2, a1, 0)
    ROW64(3, a1, 1)
    ROW64(4, a2, 0)
    ROW64(5, a2, 1)
    ROW64(6, a3, 0)
    ROW64(7, a3, 1)
  }
  for (int i = 0; i < F64_MR; i++)
    for (int v = 0; v < 3; v++) vst1q_f64(y + i * ldc + 2 * v, c[i][v]);
}

/* Thin tiles: 16 accumulators, the rows' broadcasts of a and a vector of
   b. */
#define FMA vfmaq_f32
#define LOAD vld1q_f32
#define STORE vst1q_f32
#define BCAST vld1q_dup_f32
#include "gemm_thin.h"
THIN(thin1_f32, float, float32x4_t, 4, 1, 16)
THIN(thin2_f32, float, float32x4_t, 4, 2, 8)
THIN(thin4_f32, float, float32x4_t, 4, 4, 4)
#undef FMA
#undef LOAD
#undef STORE
#undef BCAST
#define FMA vfmaq_f64
#define LOAD vld1q_f64
#define STORE vst1q_f64
#define BCAST vld1q_dup_f64
THIN(thin1_f64, double, float64x2_t, 2, 1, 16)
THIN(thin2_f64, double, float64x2_t, 2, 2, 8)
THIN(thin4_f64, double, float64x2_t, 2, 4, 4)

void nx_cpu_fill_neon(nx_cpu_target *t) {
  t->gemm[NX_FLOAT32] = (nx_cpu_gemm){
      .kernel = kernel_f32,
      .dot = t->gemm[NX_FLOAT32].dot,
      .mr = F32_MR,
      .nr = F32_NR,
      .mc = 128,
      .kc = 512,
      .nc = 3072,
      .thin = {{thin1_f32, 64}, {thin2_f32, 32}, {thin4_f32, 16}}};
  t->gemm[NX_FLOAT64] = (nx_cpu_gemm){
      .kernel = kernel_f64,
      .dot = t->gemm[NX_FLOAT64].dot,
      .mr = F64_MR,
      .nr = F64_NR,
      .mc = 128,
      .kc = 256,
      .nc = 3072,
      .thin = {{thin1_f64, 32}, {thin2_f64, 16}, {thin4_f64, 8}}};
}

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_neon;

#endif
