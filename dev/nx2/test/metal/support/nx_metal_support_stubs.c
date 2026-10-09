/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The harness's metallib (harness.metallib), included by the assembler,
   and its kernels' names; launch records; host views of device memory;
   the timed fill over a run; and the probes' host side. Objective-C on
   macOS, where the views run; elsewhere no Metal device opens and nothing
   reaches them. Every stub holds the runtime: none blocks. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "harness.h"
#include "nx_dtype.h"
#include "nx_metal.h"

/* The harness */

#define STR_(x) #x
#define STR(x) STR_(x)

#if defined(__APPLE__)
#define SECTION ".const"
#define SYMBOL(s) "_" s
#elif defined(_WIN32)
#define SECTION ".section .rdata,\"dr\""
#define SYMBOL(s) s
#else
#define SECTION ".section .rodata"
#define SYMBOL(s) s
#endif

__asm__(SECTION "\n"
        ".balign 16\n"
        ".globl " SYMBOL("nx_metal_harness") "\n"
        SYMBOL("nx_metal_harness") ":\n"
        ".incbin \"" STR(NX_HARNESS_METALLIB) "\"\n"
        ".globl " SYMBOL("nx_metal_harness_end") "\n"
        SYMBOL("nx_metal_harness_end") ":\n"
        ".text\n");

extern const char nx_metal_harness[], nx_metal_harness_end[];

value nx_metal_test_metallib(value unit) {
  (void)unit;
  return caml_alloc_initialized_string(
      (mlsize_t)(nx_metal_harness_end - nx_metal_harness), nx_metal_harness);
}

static const char *names[] = {
#define NX_HARNESS_NAME(name) #name,
    NX_HARNESS_KERNELS(NX_HARNESS_NAME)
#undef NX_HARNESS_NAME
        NULL};

value nx_metal_test_threads(value unit) {
  (void)unit;
  return Val_int(NX_HARNESS_THREADS);
}

/* nx.metal's metallib and its kernels' names, by their nx_metal_kernel. */
value nx_metal_test_library(value unit) {
  CAMLparam1(unit);
  CAMLlocal3(v, b, names);
  size_t n;
  const char *m = nx_metal_metallib(&n);
  b = caml_alloc_initialized_string(n, m);
  names = caml_alloc_tuple(NX_METAL_KERNEL_COUNT);
  for (int i = 0; i < NX_METAL_KERNEL_COUNT; i++)
    Store_field(names, i, caml_copy_string(nx_metal_kernel_names[i]));
  v = caml_alloc_tuple(2);
  Store_field(v, 0, b);
  Store_field(v, 1, names);
  CAMLreturn(v);
}

/* The harness's kernels' names, by their nx_harness_kernel. */
value nx_metal_test_kernels(value unit) {
  (void)unit;
  return caml_copy_string_array(names);
}

/* Runs */

/* The record of a launch of kernel [v_k] over [v_groups] threadgroups of
   [v_threads] threads, with the parameters [v_params], whose first
   [v_addrs] words are addresses. */
value nx_metal_test_launch(value v_k, value v_groups, value v_threads,
                           value v_params, value v_addrs) {
  CAMLparam5(v_k, v_groups, v_threads, v_params, v_addrs);
  CAMLlocal1(s);
  uint32_t g[3], t[3];
  for (int i = 0; i < 3; i++) {
    g[i] = (uint32_t)Long_val(Field(v_groups, i));
    t[i] = (uint32_t)Long_val(Field(v_threads, i));
  }
  nx_metal_records r = {NULL, 0, 0};
  if (nx_metal_add(&r, (uint32_t)Long_val(v_k), g, t, String_val(v_params),
                   (uint32_t)caml_string_length(v_params),
                   (uint32_t)Long_val(v_addrs), 0))
    caml_raise_out_of_memory();
  s = caml_alloc_initialized_string(r.len, (const char *)r.bytes);
  free(r.bytes);
  CAMLreturn(s);
}

/* A fill's argument: the times of the command buffer that ran the run,
   which [split] writes, then the run, its records after it. */
struct timed {
  int (*split)(void *queue, uint64_t *start, uint64_t *end);
  uint64_t start, end;
  nx_metal_run run;
};

/* Fills the run, then splits, so the command buffer that ends holds
   exactly the run's dispatches. */
static int timed(void *queue, void *arg, uint64_t v) {
  struct timed *t = arg;
  int rc = nx_metal_fill(queue, &t->run, v);
  return rc != 0 ? rc : t->split(queue, &t->start, &t->end);
}

value nx_metal_test_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)timed);
}

/* The argument of [timed] over the records [v_records], with [v_split]
   and the pipelines [v_pipelines], an int64 bigarray the caller keeps
   alive: a uint8 bigarray. */
value nx_metal_test_arg(value v_split, value v_pipelines, value v_records) {
  CAMLparam3(v_split, v_pipelines, v_records);
  CAMLlocal1(v);
  size_t n = caml_string_length(v_records);
  v = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                         (intnat)(sizeof(struct timed) + n));
  struct timed *t = Caml_ba_data_val(v);
  t->split = (void *)Nativeint_val(v_split);
  t->start = t->end = 0;
  t->run.pipelines = Caml_ba_data_val(v_pipelines);
  t->run.count = (uint64_t)Caml_ba_array_val(v_pipelines)->dim[0];
  t->run.bytes = n;
  memcpy(t + 1, String_val(v_records), n);
  CAMLreturn(v);
}

/* The GPU time, in nanoseconds, of the command buffer that last ran the
   argument [v_arg]. */
value nx_metal_test_span(value v_arg) {
  struct timed *t = Caml_ba_data_val(v_arg);
  return Val_long((intnat)(t->end - t->start));
}

/* Host memory */

/* A bigarray of kind [v_kind] over [v_n] elements at the host address
   [v_p], which the caller keeps alive. */
value nx_metal_test_view(value v_kind, value v_p, value v_n) {
  return caml_ba_alloc_dims(Int_val(v_kind) | CAML_BA_C_LAYOUT, 1,
                            (void *)Long_val(v_p), Long_val(v_n));
}

#ifdef __APPLE__

#import <Metal/Metal.h>

/* The host address of the MTLBuffer [v_handle]'s first byte. */
value nx_metal_test_contents(value v_handle) {
  id<MTLBuffer> b = (id)(intptr_t)Nativeint_val(v_handle);
  return Val_long((intnat)[b contents]);
}

#else

value nx_metal_test_contents(value v_handle) {
  (void)v_handle;
  abort();
}

#endif

/* Probes: the GPU's results against the host's. Each counts three kinds
   of difference and prints its first [v_show] wrong cases. */

#define Floats(v) ((const float *)Caml_ba_data_val(v))
#define Words(v) ((const uint32_t *)Caml_ba_data_val(v))
#define Length(v) (Caml_ba_array_val(v)->dim[0])

/* Equal as results: the same bits, or both NaN. */
static int same(float x, float y) {
  return nx_float_bits(x) == nx_float_bits(y) || (isnan(x) && isnan(y));
}

/* x with a float32 subnormal flushed to a zero of its sign, as the GPU
   reads and writes them. */
static float flushed(float x) {
  return fpclassify(x) == FP_SUBNORMAL ? copysignf(0.f, x) : x;
}

static void example(int64_t *shown, int64_t show, const char *what,
                    const float *in, int arity, float got, float want) {
  if ((*shown)++ >= show) return;
  printf("  %s(", what);
  for (int j = 0; j < arity; j++) printf("%s%a", j ? ", " : "", in[j]);
  printf(") = %a, expected %a\n", got, want);
  fflush(stdout);
}

static value counts(int64_t a, int64_t b, int64_t c) {
  value v = caml_alloc_tuple(3);
  Store_field(v, 0, Val_long(a));
  Store_field(v, 1, Val_long(b));
  Store_field(v, 2, Val_long(c));
  return v;
}

/* Over the triples [v_in], the GPU's a·b + c [v_one] and fma [v_fma]:
   how many a·b + c differ from the product and sum rounded apart, how
   many from the fused result, and how many fma differ from it, all with
   float32 subnormals flushed. */
value nx_metal_test_contract(value v_show, value v_in, value v_one,
                             value v_fma) {
  const float *t = Floats(v_in), *one = Floats(v_one), *f = Floats(v_fma);
  int64_t show = Long_val(v_show), apart = 0, fused = 0, fma_wrong = 0,
          shown = 0;
  for (intnat i = 0; i < Length(v_one); i++) {
    float a = flushed(t[3 * i]), b = flushed(t[3 * i + 1]),
          c = flushed(t[3 * i + 2]);
    float twice = flushed(flushed(a * b) + c), once = flushed(fmaf(a, b, c));
    if (!same(one[i], twice))
      apart++, example(&shown, show, "a*b+c", t + 3 * i, 3, one[i], twice);
    fused += !same(one[i], once);
    if (!same(f[i], once))
      fma_wrong++, example(&shown, show, "fma", t + 3 * i, 3, f[i], once);
  }
  return counts(apart, fused, fma_wrong);
}

/* Over the pairs [v_in], the GPU's x / y [v_div] and sqrt x [v_sqrt]:
   how many differ from the correctly rounded results with float32
   subnormals flushed, and how many correct quotients are subnormal. */
value nx_metal_test_div_sqrt(value v_show, value v_in, value v_div,
                             value v_sqrt) {
  const float *t = Floats(v_in), *d = Floats(v_div), *s = Floats(v_sqrt);
  int64_t show = Long_val(v_show), div_wrong = 0, sqrt_wrong = 0,
          subnormal = 0, shown = 0;
  for (intnat i = 0; i < Length(v_div); i++) {
    float x = flushed(t[2 * i]), q = x / flushed(t[2 * i + 1]);
    float r = flushed(sqrtf(x));
    subnormal += fpclassify(q) == FP_SUBNORMAL;
    q = flushed(q);
    if (!same(d[i], q))
      div_wrong++, example(&shown, show, "div", t + 2 * i, 2, d[i], q);
    if (!same(s[i], r))
      sqrt_wrong++, example(&shown, show, "sqrt", t + 2 * i, 1, s[i], r);
  }
  return counts(div_wrong, sqrt_wrong, subnormal);
}

/* Equal as half codes: the same bits, or both NaN. */
static int same_half(uint32_t x, uint32_t y) {
  return (x & 0xffff) == (y & 0xffff) ||
         ((x & 0x7fff) > 0x7c00 && (y & 0x7fff) > 0x7c00);
}

/* Over the words [v_in]: how many of the GPU's half(x) [v_to] differ from
   nx_float_to_f16's, how many floats of the low half's code [v_from] from
   nx_f16_to_float's, and how many sums h + h of that code [v_sum] from
   the exact double rounded to half. */
value nx_metal_test_half(value v_in, value v_to, value v_from, value v_sum) {
  const uint32_t *x = Words(v_in), *to = Words(v_to), *from = Words(v_from),
                 *sum = Words(v_sum);
  int64_t to_wrong = 0, from_wrong = 0, sum_wrong = 0;
  for (intnat i = 0; i < Length(v_to); i++) {
    uint16_t h = (uint16_t)x[i];
    float f = nx_f16_to_float(h);
    to_wrong += !same_half(to[i], nx_float_to_f16(nx_bits_float(x[i])));
    from_wrong += !same(nx_bits_float(from[i]), f);
    sum_wrong += !same_half(sum[i], nx_float_to_f16(f + f));
  }
  return counts(to_wrong, from_wrong, sum_wrong);
}

/* For the narrow format [v_dtype]: how many of the GPU's decodings
   [v_decoded] of the codes [v_codes] differ from the host's, and how many
   of its encodings [v_encoded] of the float bits [v_floats]. */
value nx_metal_test_codec(value v_dtype, value v_codes, value v_decoded,
                          value v_floats, value v_encoded) {
  const uint32_t *c = Words(v_codes), *d = Words(v_decoded),
                 *f = Words(v_floats), *e = Words(v_encoded);
  int dt = Int_val(v_dtype);
  int64_t decode_wrong = 0, encode_wrong = 0;
  for (intnat i = 0; i < Length(v_codes); i++)
    decode_wrong += !same(nx_bits_float(d[i]), nx_bits_to_float(dt, c[i]));
  for (intnat i = 0; i < Length(v_floats); i++) {
    float x = nx_bits_float(f[i]);
    uint32_t want = dt == NX_FLOAT16    ? nx_float_to_f16(x)
                    : dt == NX_BFLOAT16 ? nx_float_to_bf16(x)
                    : dt == NX_FLOAT8_E4M3FN ? nx_float_to_e4m3fn(x)
                    : dt == NX_FLOAT8_E5M2   ? nx_float_to_e5m2(x)
                                             : nx_float_to_e2m1fn(x);
    encode_wrong += e[i] != want;
  }
  return counts(decode_wrong, encode_wrong, 0);
}

/* Contract */

/* The operand of the OCaml triple (address, dtype, (s0, s1, s2)). */
static nx_metal_operand operand(value v) {
  value s = Field(v, 2);
  nx_metal_operand o = {(uint64_t)Long_val(Field(v, 0)), Int_val(Field(v, 1)),
                        {Long_val(Field(s, 0)), Long_val(Field(s, 1)),
                         Long_val(Field(s, 2))}};
  return o;
}

/* The records of the contraction of the dims (batch, m, n, k, acc) over the
   operands a, b, out and init (an option), with the scratch bytes they
   address: Some (records, bytes), or None if the planner declines. */
value nx_metal_test_plan_contract(value v_dims, value v_a, value v_b,
                                  value v_out, value v_init) {
  CAMLparam5(v_dims, v_a, v_b, v_out, v_init);
  CAMLlocal2(s, v);
  nx_metal_contract_in c = {(uint32_t)Long_val(Field(v_dims, 0)),
                            (uint32_t)Long_val(Field(v_dims, 1)),
                            (uint32_t)Long_val(Field(v_dims, 2)),
                            (uint32_t)Long_val(Field(v_dims, 3)),
                            Int_val(Field(v_dims, 4))};
  nx_metal_operand a = operand(v_a), b = operand(v_b), out = operand(v_out),
                   init = Is_some(v_init) ? operand(Some_val(v_init)) : a;
  nx_metal_records r = {NULL, 0, 0};
  size_t scratch;
  int e = nx_metal_plan_contract(&c, &a, &b, &out,
                                 Is_some(v_init) ? &init : NULL, &r, &scratch);
  if (e == -2) caml_raise_out_of_memory();
  if (e == NX_NOT_COMPUTED) CAMLreturn(Val_none);
  s = caml_alloc_initialized_string(r.len, (const char *)r.bytes);
  free(r.bytes);
  v = caml_alloc_tuple(2);
  Store_field(v, 0, s);
  Store_field(v, 1, Val_long(scratch));
  CAMLreturn(caml_alloc_some(v));
}

/* The value of element [i] of the host memory [p] of the float dtype [dt],
   one [floats] accepts. */
static double element(const void *p, int dt, int64_t i) {
  switch (dt) {
  case NX_FLOAT16: return nx_f16_to_float(((const uint16_t *)p)[i]);
  case NX_BFLOAT16: return nx_bf16_to_float(((const uint16_t *)p)[i]);
  default: return ((const float *)p)[i];
  }
}

/* Raises Invalid_argument unless the reference reads every dtype [ds] of
   [n]: float32, float16 and bfloat16 if [floats], the integers of 8 to 64
   bits if not. */
static void readable(int floats, const int *ds, int n) {
  for (int i = 0; i < n; i++) {
    int d = ds[i];
    int ok = floats ? d == NX_FLOAT32 || d == NX_FLOAT16 || d == NX_BFLOAT16
                    : d == NX_INT8 || d == NX_UINT8 || d == NX_INT16 ||
                          d == NX_UINT16 || d == NX_INT32 || d == NX_UINT32 ||
                          d == NX_INT64 || d == NX_UINT64;
    if (!ok)
      caml_invalid_argument("the contraction reference reads no such dtype");
  }
}

/* The magnitude from which rounding to [dt] gives infinity. */
static double overflow(int dt) {
  return dt == NX_FLOAT16    ? 65520.
         : dt == NX_BFLOAT16 ? 0x1.ffp127
                             : 0x1.ffffffp127;
}

/* The largest error of rounding a float32 x to [dt]: |x| times the unit
   roundoff, or half the spacing of the subnormals. */
static double rounding(int dt, double x) {
  if (dt == NX_FLOAT16) return fmax(0x1p-11 * x, 0x1p-25);
  if (dt == NX_BFLOAT16) return fmax(0x1p-8 * x, 0x1p-134);
  return 0;
}

/* x as the GPU's float32 arithmetic reads an operand of dtype [dt]: a
   float32 or bfloat16 subnormal is a zero of its sign; half keeps its
   subnormals. */
static double device_read(int dt, double x) {
  if (dt != NX_FLOAT16 && fabs(x) < 0x1p-126) return copysign(0, x);
  return x;
}

/* Over every output of the contraction (batch, m, n, k), the largest
   |out - s| / allowed, s the sum in double: allowed is the contraction's
   bound γ(k + 1, 2u)·(|init| + Σ|a||b|) at float32's u, plus
   2^-126·(1 + Σ(1 + |a| + |b|)) for the flushed subnormals, operands,
   products and sums, widened by the rounding to out's dtype. An output
   whose terms, as the GPU reads them, sum to a NaN or an infinity in
   IEEE arithmetic must be a NaN, or that infinity; its ratio is 0 if it
   is and infinite if not. Operands are host triples (address, dtype,
   strides); out is C-contiguous. With the worst element's index. */
value nx_metal_test_contract_error(value v_dims, value v_a, value v_b,
                                   value v_out, value v_init) {
  CAMLparam5(v_dims, v_a, v_b, v_out, v_init);
  CAMLlocal1(v);
  int64_t batch = Long_val(Field(v_dims, 0)), m = Long_val(Field(v_dims, 1)),
          n = Long_val(Field(v_dims, 2)), k = Long_val(Field(v_dims, 3));
  nx_metal_operand a = operand(v_a), b = operand(v_b), out = operand(v_out);
  int has_init = Is_some(v_init);
  nx_metal_operand init = has_init ? operand(Some_val(v_init)) : a;
  int ds[] = {a.dtype, b.dtype, out.dtype, init.dtype};
  readable(1, ds, 4);
  double u = 0x1p-24, g = (k + 1) * 2 * u / (1 - (k + 1) * 2 * u);
  double worst = 0;
  int64_t at = 0;
  for (int64_t p = 0; p < batch; p++)
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < n; j++) {
        double s = 0, mag = 0, flush = 1, read = 0;
        for (int64_t l = 0; l < k; l++) {
          double x = element((void *)a.address, a.dtype,
                             p * a.strides[0] + i * a.strides[1] +
                                 l * a.strides[2]);
          double y = element((void *)b.address, b.dtype,
                             p * b.strides[0] + l * b.strides[1] +
                                 j * b.strides[2]);
          s += x * y;
          mag += fabs(x * y);
          flush += 1 + fabs(x) + fabs(y);
          read += device_read(a.dtype, x) * device_read(b.dtype, y);
        }
        if (has_init) {
          double z = element((void *)init.address, init.dtype,
                             p * init.strides[0] + i * init.strides[1] +
                                 j * init.strides[2]);
          s += z;
          mag += fabs(z);
          read += z;
        }
        flush *= 0x1p-126;
        int64_t o = (p * m + i) * n + j;
        double got = element((void *)out.address, out.dtype, o);
        if (!isfinite(read)) {
          int same = isnan(read) ? isnan(got) : got == read;
          if (!same && worst < INFINITY) {
            worst = INFINITY;
            at = o;
          }
          continue;
        }
        double allowed = g * mag + flush +
                         1.01 * rounding(out.dtype, fabs(s) + g * mag + flush);
        double e = fabs(got - s);
        double ratio = allowed > 0 ? e / allowed : (e > 0 ? INFINITY : 0);
        /* A float32 result within the bound past out's range rounds to the
           infinity of its sign. */
        double far = copysign(1, got) * s + g * mag + flush;
        if (isinf(got) && far >= overflow(out.dtype)) ratio = 0;
        if (!(ratio <= worst)) {
          worst = ratio;
          at = o;
        }
      }
  v = caml_alloc_tuple(2);
  Store_field(v, 0, caml_copy_double(worst));
  Store_field(v, 1, Val_long(at));
  CAMLreturn(v);
}

/* The integer at [i] of host memory [p] of dtype [dt], widened by its
   sign. */
static uint64_t widened(const void *p, int dt, int64_t i) {
  switch (dt) {
  case NX_INT8: return (uint64_t)(int64_t)((const int8_t *)p)[i];
  case NX_UINT8: return ((const uint8_t *)p)[i];
  case NX_INT16: return (uint64_t)(int64_t)((const int16_t *)p)[i];
  case NX_UINT16: return ((const uint16_t *)p)[i];
  case NX_INT32: return (uint64_t)(int64_t)((const int32_t *)p)[i];
  case NX_UINT32: return ((const uint32_t *)p)[i];
  default: return ((const uint64_t *)p)[i];
  }
}

/* x's low [bits] bits, widened by their sign. */
static uint64_t widen_sign(uint64_t x, int bits) {
  uint64_t top = 1ull << (bits - 1), low = x & ((top << 1) - 1);
  return (low ^ top) - top;
}

/* Over every output of the integer contraction (batch, m, n, k, acc), how
   many differ from the sum wrapped to acc's width, widened by acc's sign,
   then wrapped to out's, as a cast from acc does, with the first such
   output's index, or -1. */
value nx_metal_test_contract_wrong(value v_dims, value v_a, value v_b,
                                   value v_out, value v_init) {
  CAMLparam5(v_dims, v_a, v_b, v_out, v_init);
  CAMLlocal1(v);
  int64_t batch = Long_val(Field(v_dims, 0)), m = Long_val(Field(v_dims, 1)),
          n = Long_val(Field(v_dims, 2)), k = Long_val(Field(v_dims, 3));
  int acc = Int_val(Field(v_dims, 4));
  int acc_bits = nx_dtype_row_of(acc).bits;
  int acc_signed = acc == NX_INT8 || acc == NX_INT16 || acc == NX_INT32 ||
                   acc == NX_INT64;
  uint64_t acc_mask = acc_bits == 64 ? ~0ull : (1ull << acc_bits) - 1;
  nx_metal_operand a = operand(v_a), b = operand(v_b), out = operand(v_out);
  int has_init = Is_some(v_init);
  nx_metal_operand init = has_init ? operand(Some_val(v_init)) : a;
  int ds[] = {a.dtype, b.dtype, out.dtype, init.dtype, acc};
  readable(0, ds, 5);
  int bits = nx_dtype_row_of(out.dtype).bits;
  uint64_t out_mask = bits == 64 ? ~0ull : (1ull << bits) - 1;
  int64_t wrong = 0, first = -1;
  for (int64_t p = 0; p < batch; p++)
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < n; j++) {
        uint64_t s = 0;
        for (int64_t l = 0; l < k; l++)
          s += widened((void *)a.address, a.dtype,
                       p * a.strides[0] + i * a.strides[1] + l * a.strides[2]) *
               widened((void *)b.address, b.dtype,
                       p * b.strides[0] + l * b.strides[1] + j * b.strides[2]);
        if (has_init)
          s += widened((void *)init.address, init.dtype,
                       p * init.strides[0] + i * init.strides[1] +
                           j * init.strides[2]);
        int64_t o = (p * m + i) * n + j;
        uint64_t got = widened((void *)out.address, out.dtype, o);
        uint64_t want = acc_signed ? widen_sign(s, acc_bits) : s & acc_mask;
        if ((want & out_mask) != (got & out_mask)) {
          if (first < 0) first = o;
          wrong++;
        }
      }
  v = caml_alloc_tuple(2);
  Store_field(v, 0, Val_long(wrong));
  Store_field(v, 1, Val_long(first));
  CAMLreturn(v);
}

/* The records [v_r] with the scratch at GPU address [v_base] placed. */
value nx_metal_test_rebase(value v_r, value v_base) {
  CAMLparam2(v_r, v_base);
  CAMLlocal1(s);
  size_t n = caml_string_length(v_r);
  s = caml_alloc_initialized_string(n, String_val(v_r));
  nx_metal_rebase((unsigned char *)Bytes_val(s), n, (uint64_t)Long_val(v_base));
  CAMLreturn(s);
}

/* The kernel of each launch of the records [v_r], in order. */
value nx_metal_test_entries(value v_r) {
  CAMLparam1(v_r);
  CAMLlocal1(v);
  const unsigned char *r = (const unsigned char *)String_val(v_r);
  size_t n = caml_string_length(v_r), count = 0;
  for (size_t at = 0; at < n; count++)
    at += sizeof(nx_metal_launch) + ((const nx_metal_launch *)(r + at))->bytes;
  v = caml_alloc_tuple(count);
  r = (const unsigned char *)String_val(v_r);
  for (size_t at = 0, i = 0; at < n; i++) {
    const nx_metal_launch *l = (const nx_metal_launch *)(r + at);
    Store_field(v, i, Val_long(l->entry));
    at += sizeof *l + l->bytes;
  }
  CAMLreturn(v);
}
