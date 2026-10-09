/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The harness's metallib (harness.metallib), included by the assembler,
   and its kernels' names; launch records; host views of device memory;
   the timed fill over a run; the probes' host side; and the machine's GPU
   lock. Objective-C on macOS, where the views run; elsewhere no Metal
   device opens and nothing reaches them. Every stub but the lock's holds
   the runtime: none blocks. */

#define _GNU_SOURCE

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#endif

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

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

/* The machine's GPU lock */

/* 0 once this process holds the lock [v_path], writing [v_holder] and its
   pid into it; -1 after a 100 ms nap if another process holds it; an
   errno otherwise. */
value nx_metal_test_lock(value v_path, value v_holder) {
#if defined(_WIN32)
  (void)v_path;
  (void)v_holder;
  return Val_int(ENOSYS);
#else
  /* The descriptor that holds the lock once taken. The suites take it from
     one domain. */
  static int held = -1;
  if (held >= 0) return Val_int(0);
  const char *path = String_val(v_path);
  int fd = open(path, O_RDWR | O_CLOEXEC);
  /* O_EXCL: Linux refuses O_CREAT on another user's file in /tmp
     (fs.protected_regular). */
  if (fd < 0 && errno == ENOENT) {
    fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (fd < 0 && errno == EEXIST) fd = open(path, O_RDWR | O_CLOEXEC);
    else if (fd >= 0 && fchmod(fd, 0666) != 0) {
      int e = errno;
      close(fd);
      return Val_int(e);
    }
  }
  if (fd < 0) return Val_int(errno);
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    int e = errno;
    close(fd);
    if (e != EWOULDBLOCK) return Val_int(e);
    struct timespec nap = {0, 100 * 1000 * 1000};
    caml_release_runtime_system();
    nanosleep(&nap, NULL);
    caml_acquire_runtime_system();
    return Val_int(-1);
  }
  char note[1024] = "";
  snprintf(note, sizeof note, "%s, pid %ld\n", String_val(v_holder),
           (long)getpid());
  size_t len = strlen(note);
  if (ftruncate(fd, 0) != 0 || pwrite(fd, note, len, 0) != (ssize_t)len) {
    int e = errno;
    close(fd);
    return Val_int(e);
  }
  held = fd;
  return Val_int(0);
#endif
}
