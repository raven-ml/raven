/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Loops of nx_kinds.h's transcendental kinds over contiguous bigarrays, as
   nx.cpu compiles them: on x86-64 for its v3 target (AVX2 and FMA), which
   every machine the rows record has, elsewhere for the base one. The
   trigonometric loops reduce every lane by Cody and Waite and redo the
   lanes past the switch, as the header says a vector loop does. */

#if defined(__x86_64__)
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,f16c,bmi2"))), \
                             apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c,bmi2")
#endif
#endif

#include <string.h>

#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>

#include "nx_kinds.h"

#define BLOCK 1024

#define UNARY(T, S, kind)                                                    \
  static void kind##_##S(const T *x, const T *z, T *y, long n) {            \
    (void)z;                                                                 \
    for (long i = 0; i < n; i++) y[i] = nx_##kind##_##S(x[i]);              \
  }

#define BINARY(T, S, kind)                                                   \
  static void kind##_##S(const T *x, const T *z, T *y, long n) {            \
    for (long i = 0; i < n; i++) y[i] = nx_##kind##_##S(x[i], z[i]);        \
  }

#define TRIG(T, S, kind, BIG)                                                \
  static void kind##_##S(const T *x, const T *z, T *y, long n) {            \
    (void)z;                                                                 \
    for (long b = 0; b < n; b += BLOCK) {                                    \
      long e = n - b < BLOCK ? n : b + BLOCK;                                \
      for (long i = b; i < e; i++)                                           \
        y[i] = nx_##kind##_of_##S(nx_rem_pio2_cw_##S(x[i]));                 \
      for (long i = b; i < e; i++)                                           \
        if (nx_abs_bits_##S(x[i]) >= BIG) y[i] = nx_##kind##_##S(x[i]);      \
    }                                                                        \
  }

#define KINDS(T, S, BIG)                                                     \
  UNARY(T, S, exp)                                                           \
  UNARY(T, S, exp2)                                                          \
  UNARY(T, S, expm1)                                                         \
  UNARY(T, S, log)                                                           \
  UNARY(T, S, log2)                                                          \
  UNARY(T, S, log1p)                                                         \
  TRIG(T, S, sin, BIG)                                                       \
  TRIG(T, S, cos, BIG)                                                       \
  TRIG(T, S, tan, BIG)                                                       \
  UNARY(T, S, asin)                                                          \
  UNARY(T, S, acos)                                                          \
  UNARY(T, S, atan)                                                          \
  UNARY(T, S, sinh)                                                          \
  UNARY(T, S, cosh)                                                          \
  UNARY(T, S, tanh)                                                          \
  UNARY(T, S, erf)                                                           \
  BINARY(T, S, pow)                                                          \
  BINARY(T, S, atan2)

KINDS(float, f32, NX_PIO2_BIG_F32)
KINDS(double, f64, NX_PIO2_BIG_F64)

#define ROW(kind) {#kind, kind##_f32, kind##_f64}

static const struct {
  const char *name;
  void (*f32)(const float *, const float *, float *, long);
  void (*f64)(const double *, const double *, double *, long);
} rows[] = {ROW(exp),  ROW(exp2), ROW(expm1), ROW(log),  ROW(log2),
            ROW(log1p), ROW(sin), ROW(cos),   ROW(tan),  ROW(asin),
            ROW(acos), ROW(atan), ROW(sinh),  ROW(cosh), ROW(tanh),
            ROW(erf),  ROW(pow),  ROW(atan2)};

/* y.{i} <- kind x.{i} (z.{i}): x, z and y float32 or float64 bigarrays of
   one length. It keeps the runtime: a call takes milliseconds. */
value nx_kinds_bench_run(value kind, value x, value z, value y) {
  const char *k = String_val(kind);
  long n = (long)Caml_ba_array_val(x)->dim[0];
  int f64 = (Caml_ba_array_val(x)->flags & CAML_BA_KIND_MASK) == CAML_BA_FLOAT64;
  for (size_t i = 0; i < sizeof rows / sizeof *rows; i++) {
    if (strcmp(rows[i].name, k)) continue;
    if (f64)
      rows[i].f64(Caml_ba_data_val(x), Caml_ba_data_val(z), Caml_ba_data_val(y), n);
    else
      rows[i].f32(Caml_ba_data_val(x), Caml_ba_data_val(z), Caml_ba_data_val(y), n);
    return Val_unit;
  }
  caml_invalid_argument(k);
}

#if defined(__x86_64__)
#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
#endif
