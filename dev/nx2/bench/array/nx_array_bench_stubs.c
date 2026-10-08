/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdatomic.h>
#include <stdlib.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>

#include "nx_array.h"
#include "rig.h"
#include "rig_edge.h"

/* The door alone: three operands read and released, with no kernel. */

value nx_array_bench_read_3(value z, value x, value y) {
  int dt = nx_array_dtype(z);
  nx_operand in[3] = {{z, dt, 1}, {x, dt, 0}, {y, dt, 0}};
  nx_array a[3];
  int e = nx_read(3, in, a);
  if (!e) nx_done(3, a);
  return Val_int(e);
}

/* The door and the coalescer: three operands read, one loop formed, the
   operands released. */

value nx_array_bench_loop_3(value z, value x, value y) {
  int dt = nx_array_dtype(z);
  nx_operand in[3] = {{z, dt, 1}, {x, dt, 0}, {y, dt, 0}};
  nx_array a[3];
  nx_loop l;
  int e = nx_read(3, in, a);
  if (e) return Val_int(e);
  e = nx_coalesce(3, a, &l);
  nx_done(3, a);
  return Val_int(e ? e : l.rank);
}

/* The door's floor: the claims nx_read takes on three buffers, for writing
   the first, and their release. */

value nx_array_bench_claim_3(value z, value x, value y) {
  enum rig_claim cz = rig_buffer_claim(z, RIG_READ_WRITE);
  enum rig_claim cx = rig_buffer_claim(x, RIG_READ);
  enum rig_claim cy = rig_buffer_claim(y, RIG_READ);
  if (cz == RIG_CLAIMED) rig_buffer_release(z);
  if (cx == RIG_CLAIMED) rig_buffer_release(x);
  if (cy == RIG_CLAIMED) rig_buffer_release(y);
  return Val_unit;
}

/* Conversions over runs

   Each stub converts the elements of the bigarray [src] into [dst], of the
   same length, as a kernel's inner loop does: one call of nx_dtype.h's
   function per element, or one call of a run form over all of them. A
   float4 code takes a byte of its own here. */

#define BENCH_EACH(name, S, D, f)                         \
  value nx_array_bench_##name(value src, value dst) {     \
    const S *s = Caml_ba_data_val(src);                   \
    D *d = Caml_ba_data_val(dst);                         \
    intnat n = Caml_ba_array_val(src)->dim[0];            \
    for (intnat i = 0; i < n; i++) d[i] = f(s[i]);        \
    return Val_unit;                                      \
  }

#define BENCH_RUN(name, S, D, f)                             \
  value nx_array_bench_##name(value src, value dst) {        \
    f((const S *)Caml_ba_data_val(src), (D *)Caml_ba_data_val(dst), \
      (size_t)Caml_ba_array_val(src)->dim[0]);               \
    return Val_unit;                                         \
  }

static inline int32_t to_i32(double x) {
  return (int32_t)nx_double_to_int(x, INT32_MIN, INT32_MAX);
}

BENCH_EACH(f32_to_f16, float, uint16_t, nx_float_to_f16)
BENCH_EACH(f16_to_f32, uint16_t, float, nx_f16_to_float)
BENCH_EACH(f32_to_bf16, float, uint16_t, nx_float_to_bf16)
BENCH_EACH(bf16_to_f32, uint16_t, float, nx_bf16_to_float)
BENCH_EACH(f32_to_e4m3fn, float, uint8_t, nx_float_to_e4m3fn)
BENCH_EACH(e4m3fn_to_f32, uint8_t, float, nx_e4m3fn_to_float)
BENCH_EACH(f32_to_e5m2, float, uint8_t, nx_float_to_e5m2)
BENCH_EACH(e5m2_to_f32, uint8_t, float, nx_e5m2_to_float)
BENCH_EACH(f32_to_e2m1fn, float, uint8_t, nx_float_to_e2m1fn)
BENCH_EACH(e2m1fn_to_f32, uint8_t, float, nx_e2m1fn_to_float)
BENCH_EACH(f64_to_f16, double, uint16_t, nx_double_to_f16)
BENCH_EACH(f64_to_i32, double, int32_t, to_i32)
BENCH_EACH(f64_to_e4m3fn, double, uint8_t, nx_double_to_e4m3fn)
BENCH_RUN(f64_to_f16_run, double, uint16_t, nx_double_to_f16_run)
BENCH_RUN(f16_to_f64_run, uint16_t, double, nx_f16_to_double_run)
BENCH_RUN(f64_to_bf16_run, double, uint16_t, nx_double_to_bf16_run)
BENCH_RUN(bf16_to_f64_run, uint16_t, double, nx_bf16_to_double_run)

/* Floors: loops of the same bytes in and out whose work per element is one
   instruction. */

static inline uint16_t top16(float f) { return (uint16_t)(nx_float_bits(f) >> 16); }
static inline uint8_t top8(float f) { return (uint8_t)(nx_float_bits(f) >> 24); }
static inline float of_u16(uint16_t c) { return (float)c; }
static inline float of_u8(uint8_t c) { return (float)c; }
static inline uint16_t top16_f64(double x) { return top16((float)x); }
static inline int32_t cast_i32(double x) { return (int32_t)x; }

BENCH_EACH(floor_f32_to_u16, float, uint16_t, top16)
BENCH_EACH(floor_f32_to_u8, float, uint8_t, top8)
BENCH_EACH(floor_u16_to_f32, uint16_t, float, of_u16)
BENCH_EACH(floor_u8_to_f32, uint8_t, float, of_u8)
BENCH_EACH(floor_f64_to_u16, double, uint16_t, top16_f64)
BENCH_EACH(floor_f64_to_i32, double, int32_t, cast_i32)

/* Late: a device over host memory whose work completes only when a wait
   sleeps on it. A submit records its value; the next sleep, or the stop,
   makes it the word's. It stands for a device still running the work a
   kernel's operand waits for. */

struct late {
  _Atomic uint64_t word;
  _Atomic uint64_t last;
};

value nx_array_bench_late_new(value unit) {
  (void)unit;
  struct late *l = calloc(1, sizeof *l);
  if (l == NULL) caml_raise_out_of_memory();
  return caml_copy_nativeint((intnat)l);
}

value nx_array_bench_late_publish(value self) {
  struct late *l = (struct late *)Nativeint_val(self);
  uint64_t v = atomic_load_explicit(&l->last, memory_order_acquire);
  atomic_store_explicit(&l->word, v, memory_order_release);
  return Val_unit;
}

value nx_array_bench_late_signaled(value self) {
  struct late *l = (struct late *)Nativeint_val(self);
  return Val_long((intnat)atomic_load_explicit(&l->word, memory_order_acquire));
}

/* Late runs no part: a submission of none fits. */
static int late_room(void *self, const struct rig_part *parts, int n) {
  (void)self;
  (void)parts;
  return n == 0 ? RIG_FITS : RIG_NEVER;
}

static int late_submit(void *self, uint64_t v, const struct rig_wait *waits,
                       int nwaits, const struct rig_part *parts, int nparts,
                       const uint64_t *handles, int nhandles,
                       const char **failure) {
  (void)waits;
  (void)nwaits;
  (void)parts;
  (void)nparts;
  (void)handles;
  (void)nhandles;
  (void)failure;
  atomic_store_explicit(&((struct late *)self)->last, v, memory_order_release);
  return RIG_OK;
}

value nx_array_bench_late_room(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&late_room);
}

value nx_array_bench_late_submit(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&late_submit);
}

/* [v_n] bytes and 64 more, so the region can start on a multiple of 64; 0
   if malloc fails. */
value nx_array_bench_malloc(value v_n) {
  return Val_long((intnat)malloc((size_t)Long_val(v_n) + 64));
}

value nx_array_bench_free(value v_p) {
  free((void *)Long_val(v_p));
  return Val_unit;
}
