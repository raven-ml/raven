/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Floors of contractions: the host's fused multiply-adds at their peak, and
   a read of the bytes a product of few rows streams. The peak runs
   independent chains of vector fused multiply-adds, as many as the pipes
   hold in flight: 16 vectors on arm64 (four pipes, four cycles), 12 on
   x86-64, whose 16 registers spill at 16: 95.5 against 155 GFLOP/s on
   kimchi. The read sums the bytes into eight vector accumulators.
   Both slice the work on the pool's threads as the other floors do; on
   x86-64 they compile for v3's instructions. */

#if defined(__x86_64__)
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,f16c,bmi2"))), \
                             apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c,bmi2")
#endif
#endif

#include <stdint.h>

#include <caml/bigarray.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include "nx_array.h"
#include "rig_pool.h"

/* Slices of a job per thread, as nx_cpu_bench_stubs.c's. */
#define SLICES 8

typedef struct {
  int f64;
  int64_t steps, total; /* steps of every chain, over [total] slices */
  const uint8_t *s;
  int64_t n;
} peak_job;

/* Where each slice leaves its result, so that no compiler drops the work. */
static volatile double sink;

/* Chain i starts from i: chains from one value compute one value, which gcc
   computes once. */
#if defined(__aarch64__)
#define CHAINS 16
#define LANES32 4
#define LANES64 2
#define F32(i) float32x4_t c##i = vdupq_n_f32((float)i)
#define STEP32(i) c##i = vfmaq_f32(c##i, x, y)
#define F64(i) float64x2_t c##i = vdupq_n_f64((double)i)
#define STEP64(i) c##i = vfmaq_f64(c##i, x, y)
#define SUM32(i) vgetq_lane_f32(c##i, 0)
#define SUM64(i) vgetq_lane_f64(c##i, 0)
#define X32 float32x4_t x = vdupq_n_f32(1.0001f), y = vdupq_n_f32(1e-7f)
#define X64 float64x2_t x = vdupq_n_f64(1.0001), y = vdupq_n_f64(1e-7)
#else
#define CHAINS 12
#define LANES32 8
#define LANES64 4
#define F32(i) __m256 c##i = _mm256_set1_ps((float)i)
#define STEP32(i) c##i = _mm256_fmadd_ps(c##i, x, y)
#define F64(i) __m256d c##i = _mm256_set1_pd((double)i)
#define STEP64(i) c##i = _mm256_fmadd_pd(c##i, x, y)
#define SUM32(i) _mm256_cvtss_f32(c##i)
#define SUM64(i) _mm256_cvtsd_f64(c##i)
#define X32 __m256 x = _mm256_set1_ps(1.0001f), y = _mm256_set1_ps(1e-7f)
#define X64 __m256d x = _mm256_set1_pd(1.0001), y = _mm256_set1_pd(1e-7)
#endif

#define EACH12(M) M(0); M(1); M(2); M(3); M(4); M(5); M(6); M(7); M(8); M(9); M(10); M(11)
#if CHAINS == 16
#define EACH(M) EACH12(M); M(12); M(13); M(14); M(15)
#define TOTAL(S) (S(0) + S(1) + S(2) + S(3) + S(4) + S(5) + S(6) + S(7) + \
                  S(8) + S(9) + S(10) + S(11) + S(12) + S(13) + S(14) + S(15))
#else
#define EACH(M) EACH12(M)
#define TOTAL(S) (S(0) + S(1) + S(2) + S(3) + S(4) + S(5) + S(6) + S(7) + \
                  S(8) + S(9) + S(10) + S(11))
#endif

static double peak32(int64_t steps) {
  X32;
  EACH(F32);
  for (int64_t p = 0; p < steps; p++) {
    EACH(STEP32);
  }
  return TOTAL(SUM32);
}

static double peak64(int64_t steps) {
  X64;
  EACH(F64);
  for (int64_t p = 0; p < steps; p++) {
    EACH(STEP64);
  }
  return TOTAL(SUM64);
}

static void peak_slice(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  peak_job *f = ctx;
  int64_t steps = f->steps * hi / f->total - f->steps * lo / f->total;
  sink = f->f64 ? peak64(steps) : peak32(steps);
}

static void read_slice(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  peak_job *f = ctx;
  int64_t first = f->n * lo / f->total / 64 * 64;
  int64_t last = f->n * hi / f->total / 64 * 64;
  uint64_t acc[8] = {0};
  const uint64_t *s = (const uint64_t *)(f->s + first);
  for (int64_t i = 0; i < (last - first) / 8; i += 8)
    for (int j = 0; j < 8; j++) acc[j] += s[i + j];
  uint64_t t = 0;
  for (int j = 0; j < 8; j++) t += acc[j];
  sink = (double)t;
}

static void run(int t, peak_job *f, rig_pool_body body) {
  f->total = t * SLICES;
  if (t == 1) {
    f->total = 1;
    body(0, 1, 0, f);
    return;
  }
  caml_enter_blocking_section_no_pending();
  rig_pool_run(t, f->total, f->total, body, f);
  caml_leave_blocking_section();
}

/* [floor_fma threads f64 flops] runs [flops] floating-point operations of
   fused multiply-adds, float64 if [f64], on [threads] threads. */
value nx_cpu_bench_floor_fma(value threads, value f64, value flops) {
  int lanes = Bool_val(f64) ? LANES64 : LANES32;
  peak_job f = {.f64 = Bool_val(f64)};
  f.steps = Long_val(flops) / (2 * lanes * CHAINS);
  run(Int_val(threads), &f, peak_slice);
  return Val_unit;
}

/* [floor_read threads src] reads every byte of [src] on [threads]
   threads. */
value nx_cpu_bench_floor_read(value threads, value src) {
  peak_job f = {.s = Caml_ba_data_val(src),
                .n = (int64_t)Caml_ba_array_val(src)->dim[0]};
  run(Int_val(threads), &f, read_slice);
  return Val_unit;
}

#if defined(__x86_64__)
#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
#endif
