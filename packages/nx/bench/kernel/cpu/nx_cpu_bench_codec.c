/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Floors of the conversions whose own operations can cost more than their
   bytes' move: a loop of nx_dtype.h's scalar codec over the cast's
   elements, compiled as nx.cpu's units are and sliced on the pool's
   threads as the other floors are. nx_dtype.h's codecs select rather than
   branch, so the loop vectorises at the target's width; a kernel that
   converts a vector in fewer instructions than the compiler finds can run
   under this floor. Neither CPU converts to or from bfloat16 or e4m3fn,
   neither converts a vector of 64-bit integers or narrows a double to
   float16 at once on x86-64, and the M1's conversions of float32 to
   float16 and int8 cost more than an integer narrowing. */

#include <stddef.h>
#include <stdint.h>

#include <caml/bigarray.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include "nx_dtype.h"
#include "rig_pool.h"

/* The codecs, in the order of the bench's [codec] constructors. */
enum {
  F32_BF16,
  F32_E4M3,
  E4M3_F32,
  I64_F32,
  U64_F32,
  F64_F16,
  F32_F16,
  F32_I8
};

/* Slices of a job per thread, as nx_cpu_bench_stubs.c's. */
#define SLICES 8

typedef struct {
  int codec;
  uint8_t *d;
  const uint8_t *s;
  int64_t n, total;
} codec_job;

/* d[i] from s[i] through [codec], over the slice's [n] elements. */
#define LOOP(TS, TD, codec)                                \
  {                                                        \
    const TS *s = (const TS *)c->s + first;                \
    TD *d = (TD *)c->d + first;                            \
    for (size_t i = 0; i < n; i++) d[i] = codec(s[i]);     \
    return;                                                \
  }

/* The codecs that nx_dtype.h composes: a 64-bit integer becomes a double
   rounded to odd, which rounds once to float32, and a float32 saturates
   to int8. */
#define I64_F32_CODEC(x) ((float)nx_i64_odd(x))
#define U64_F32_CODEC(x) ((float)nx_u64_odd(x))
#define F32_I8_CODEC(x) ((int8_t)nx_float_to_int(x, INT8_MIN, INT8_MAX))

static void slice(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const codec_job *c = ctx;
  int64_t first = c->n * lo / c->total;
  size_t n = (size_t)(c->n * hi / c->total - first);
  switch (c->codec) {
    case F32_BF16: LOOP(float, uint16_t, nx_float_to_bf16);
    case F32_E4M3: LOOP(float, uint8_t, nx_float_to_e4m3fn);
    case E4M3_F32: LOOP(uint8_t, float, nx_e4m3fn_to_float);
    case I64_F32: LOOP(int64_t, float, I64_F32_CODEC);
    case U64_F32: LOOP(uint64_t, float, U64_F32_CODEC);
    case F64_F16: LOOP(double, uint16_t, nx_double_to_f16);
    case F32_F16: LOOP(float, uint16_t, nx_float_to_f16);
    default: /* F32_I8 */ LOOP(float, int8_t, F32_I8_CODEC);
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
