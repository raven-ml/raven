/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Floors of the conversions whose own operations can cost more than their
   bytes' move: the fastest loop of the codec, nx_array.h's run or a loop of
   nx_dtype.h's scalar codec, over the cast's elements, sliced on the pool's
   threads as the other floors are. Neither CPU converts to or from bfloat16
   or e4m3fn, neither converts a vector of 64-bit integers or narrows a
   double to float16 at once on x86-64, and the M1's conversions of float32
   to float16 and int8 cost more than an integer narrowing. On x86-64 the
   loops compile for AVX2, F16C and FMA, as nx.cpu's v3 runs do, and run
   where the host has AVX2. */

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

/* The codecs, in the order of the bench's [codec] constructors. */
enum { F32_BF16, F32_E4M3, E4M3_F32, I64_F32, F64_F16, F32_F16, F32_I8 };

/* Slices of a job per thread, as nx_cpu_bench_stubs.c's. */
#define SLICES 8

typedef struct {
  int codec;
  uint8_t *d;
  const uint8_t *s;
  int64_t n, total;
} codec_job;

static void slice(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const codec_job *c = ctx;
  int64_t first = c->n * lo / c->total;
  size_t n = (size_t)(c->n * hi / c->total - first);
  switch (c->codec) {
    case F32_BF16:
      nx_float_to_bf16_run((const float *)c->s + first,
                           (uint16_t *)c->d + first, n);
      return;
    case F32_E4M3:
      nx_float_to_e4m3fn_run((const float *)c->s + first, c->d + first, n);
      return;
    case E4M3_F32:
      nx_e4m3fn_to_float_run(c->s + first, (float *)c->d + first, n);
      return;
    case I64_F32:
      nx_i64_to_float_run((const int64_t *)c->s + first, (float *)c->d + first,
                          n);
      return;
    case F64_F16:
      nx_double_to_f16_run((const double *)c->s + first,
                           (uint16_t *)c->d + first, n);
      return;
    case F32_F16:
      nx_float_to_f16_run((const float *)c->s + first, (uint16_t *)c->d + first,
                          n);
      return;
    default: /* F32_I8 */
      nx_float_to_i8_run((const float *)c->s + first, (int8_t *)c->d + first,
                         n);
      return;
  }
}

/* [floor_codec threads codec dst src n] converts [n] elements of [src] into
   [dst] with [codec] on [threads] of the pool's threads. */
value nx_cpu_bench_floor_codec(value threads, value codec, value dst,
                               value src, value n) {
  int t = Int_val(threads);
  codec_job c = {Int_val(codec), Caml_ba_data_val(dst), Caml_ba_data_val(src),
                 Long_val(n), t * SLICES};
  if (t == 1) {
    c.total = 1;
    slice(0, 1, 0, &c);
    return Val_unit;
  }
  caml_enter_blocking_section_no_pending();
  rig_pool_run(t, c.total, c.total, slice, &c);
  caml_leave_blocking_section();
  return Val_unit;
}

/* Whether the host runs the loops. */
value nx_cpu_bench_codecs_run(value unit) {
  (void)unit;
#if defined(__x86_64__)
  return Val_bool(__builtin_cpu_supports("avx2"));
#else
  return Val_true;
#endif
}

#if defined(__x86_64__)
#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
#endif
