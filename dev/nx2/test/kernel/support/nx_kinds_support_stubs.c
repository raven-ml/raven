/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_kinds.h's kinds, called by name through the loops: every call runs
   the loop nx.cpu would run on this CPU. f32 operands and results travel
   as their bits, f64 ones as OCaml floats, integers as int64 values. Each
   stub takes microseconds and keeps the runtime. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_kinds.h"
#include "nx_kinds_support.h"

/* The loops nx.cpu would run on this CPU. */
static const nx_kinds_loops *const loops = &nx_kinds_support_loops;

static const nx_real_loop *nx_kinds_real_loop(const nx_kinds_loops *l,
                                              const char *name) {
  for (int i = 0; i < NX_REAL_LOOPS; i++)
    if (strcmp(l->real[i].name, name) == 0) return &l->real[i];
  return NULL;
}

static const nx_int_loop *nx_kinds_int_loop(const nx_kinds_loops *l,
                                            const char *name) {
  for (int i = 0; i < NX_INT_LOOPS; i++)
    if (strcmp(l->ints[i].name, name) == 0) return &l->ints[i];
  return NULL;
}

/* Raises Invalid_argument naming the string [v]. The message is copied out
   of [v] first: raising allocates, which may move [v]. */
static _Noreturn void refuse(value v) {
  char m[128];
  snprintf(m, sizeof m, "%s", String_val(v));
  caml_invalid_argument(m);
}

static const nx_real_loop *real_loop(const nx_kinds_loops *l, value name) {
  const nx_real_loop *k = nx_kinds_real_loop(l, String_val(name));
  if (k == NULL) refuse(name);
  return k;
}

/* Scalar calls */

value nx_kinds_support_f32(value name, value args) {
  const nx_real_loop *k = real_loop(loops, name);
  if ((int)Wosize_val(args) != k->arity) refuse(name);
  float x[3] = {0, 0, 0}, y;
  for (int i = 0; i < k->arity; i++)
    x[i] = nx_bits_float((uint32_t)Long_val(Field(args, i)));
  k->f32(&x[0], &x[1], &x[2], &y, 1);
  return Val_long(nx_float_bits(y));
}

value nx_kinds_support_f64(value name, value args) {
  CAMLparam2(name, args);
  const nx_real_loop *k = real_loop(loops, name);
  if ((int)(Wosize_val(args) / Double_wosize) != k->arity)
    refuse(name);
  double x[3] = {0, 0, 0}, y;
  for (int i = 0; i < k->arity; i++) x[i] = Double_flat_field(args, i);
  k->f64(&x[0], &x[1], &x[2], &y, 1);
  CAMLreturn(caml_copy_double(y));
}

/* The integer kind [name] at [ty]: the result's int64, sign-extended from
   i32 and zero-extended from u32. */
value nx_kinds_support_int(value ty, value name, value args) {
  CAMLparam3(ty, name, args);
  const nx_int_loop *k = nx_kinds_int_loop(loops, String_val(name));
  if (k == NULL || (int)Wosize_val(args) != k->arity)
    refuse(name);
  uint64_t x[3] = {0, 0, 0}, y[4];
  for (int i = 0; i < k->arity; i++) x[i] = (uint64_t)Int64_val(Field(args, i));
  k->run(&x[0], &x[1], &x[2], y, 1);
  const char *t = String_val(ty);
  int64_t r;
  if (!strcmp(t, "i32")) r = (int32_t)(uint32_t)y[0];
  else if (!strcmp(t, "u32")) r = (int64_t)(uint32_t)y[1];
  else if (!strcmp(t, "i64")) r = (int64_t)y[2];
  else if (!strcmp(t, "u64")) r = (int64_t)y[3];
  else refuse(ty);
  CAMLreturn(caml_copy_int64(r));
}

/* Writes the complex kind [k] of the [n] operands [x] into [y] and answers
   1, or answers 0 for an unknown kind or arity; a comparison's 1 or 0 is
   the real part. */
#define COMPLEX_CALL(C, T)                                                    \
  static int call_##C(const char *k, int n, const nx_##C *x, nx_##C *y) {     \
    if (n == 1) {                                                             \
      if (!strcmp(k, "neg")) *y = nx_neg_##C(x[0]);                           \
      else if (!strcmp(k, "recip")) *y = nx_recip_##C(x[0]);                  \
      else if (!strcmp(k, "sign")) *y = nx_sign_##C(x[0]);                    \
      else if (!strcmp(k, "sqrt")) *y = nx_sqrt_##C(x[0]);                    \
      else if (!strcmp(k, "exp")) *y = nx_exp_##C(x[0]);                      \
      else if (!strcmp(k, "exp2")) *y = nx_exp2_##C(x[0]);                    \
      else if (!strcmp(k, "log")) *y = nx_log_##C(x[0]);                      \
      else if (!strcmp(k, "log2")) *y = nx_log2_##C(x[0]);                    \
      else if (!strcmp(k, "sin")) *y = nx_sin_##C(x[0]);                      \
      else if (!strcmp(k, "cos")) *y = nx_cos_##C(x[0]);                      \
      else if (!strcmp(k, "tan")) *y = nx_tan_##C(x[0]);                      \
      else if (!strcmp(k, "asin")) *y = nx_asin_##C(x[0]);                    \
      else if (!strcmp(k, "acos")) *y = nx_acos_##C(x[0]);                    \
      else if (!strcmp(k, "atan")) *y = nx_atan_##C(x[0]);                    \
      else if (!strcmp(k, "sinh")) *y = nx_sinh_##C(x[0]);                    \
      else if (!strcmp(k, "cosh")) *y = nx_cosh_##C(x[0]);                    \
      else if (!strcmp(k, "tanh")) *y = nx_tanh_##C(x[0]);                    \
      else if (!strcmp(k, "asinh")) *y = nx_asinh_##C(x[0]);                  \
      else if (!strcmp(k, "acosh")) *y = nx_acosh_##C(x[0]);                  \
      else if (!strcmp(k, "atanh")) *y = nx_atanh_##C(x[0]);                  \
      else return 0;                                                          \
    } else if (n != 2) return 0;                                              \
    else if (!strcmp(k, "pow")) *y = nx_pow_##C(x[0], x[1]);                  \
    else if (!strcmp(k, "add")) *y = nx_add_##C(x[0], x[1]);                  \
    else if (!strcmp(k, "sub")) *y = nx_sub_##C(x[0], x[1]);                  \
    else if (!strcmp(k, "mul")) *y = nx_mul_##C(x[0], x[1]);                  \
    else if (!strcmp(k, "fdiv")) *y = nx_fdiv_##C(x[0], x[1]);                \
    else if (!strcmp(k, "maximum")) *y = nx_maximum_##C(x[0], x[1]);          \
    else if (!strcmp(k, "minimum")) *y = nx_minimum_##C(x[0], x[1]);          \
    else if (!strcmp(k, "equal"))                                             \
      *y = nx_##C##_of((T)nx_equal_##C(x[0], x[1]), 0);                       \
    else if (!strcmp(k, "not_equal"))                                         \
      *y = nx_##C##_of((T)nx_not_equal_##C(x[0], x[1]), 0);                   \
    else if (!strcmp(k, "less"))                                              \
      *y = nx_##C##_of((T)nx_less_##C(x[0], x[1]), 0);                        \
    else if (!strcmp(k, "less_equal"))                                        \
      *y = nx_##C##_of((T)nx_less_equal_##C(x[0], x[1]), 0);                  \
    else return 0;                                                            \
    return 1;                                                                 \
  }
COMPLEX_CALL(c64, float)
COMPLEX_CALL(c128, double)

value nx_kinds_support_c64(value name, value args) {
  CAMLparam2(name, args);
  CAMLlocal1(r);
  int n = (int)Wosize_val(args) / 2;
  nx_c64 x[2] = {{0, 0}, {0, 0}}, y;
  if (n < 1 || n > 2 || (int)Wosize_val(args) != 2 * n) refuse(name);
  for (int i = 0; i < n; i++)
    x[i] = nx_c64_of(nx_bits_float((uint32_t)Long_val(Field(args, 2 * i))),
                     nx_bits_float((uint32_t)Long_val(Field(args, 2 * i + 1))));
  if (!call_c64(String_val(name), n, x, &y)) refuse(name);
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_long(nx_float_bits(y.re)));
  Store_field(r, 1, Val_long(nx_float_bits(y.im)));
  CAMLreturn(r);
}

value nx_kinds_support_c128(value name, value args) {
  CAMLparam2(name, args);
  CAMLlocal1(r);
  int n = (int)(Wosize_val(args) / Double_wosize) / 2;
  nx_c128 x[2] = {{0, 0}, {0, 0}}, y;
  if (n < 1 || n > 2 || (int)(Wosize_val(args) / Double_wosize) != 2 * n)
    refuse(name);
  for (int i = 0; i < n; i++)
    x[i] = nx_c128_of(Double_flat_field(args, 2 * i),
                      Double_flat_field(args, 2 * i + 1));
  if (!call_c128(String_val(name), n, x, &y)) refuse(name);
  r = caml_alloc_float_array(2);
  Store_double_flat_field(r, 0, y.re);
  Store_double_flat_field(r, 1, y.im);
  CAMLreturn(r);
}

value nx_kinds_support_threefry(value counter, value key) {
  CAMLparam2(counter, key);
  uint64_t r = nx_threefry_u64((uint64_t)Int64_val(counter), (uint64_t)Int64_val(key));
  CAMLreturn(caml_copy_int64((int64_t)r));
}

/* [run kind x z y]: y.{i} <- kind x.{i} (z.{i}), x, z and y float32 or
   float64 bigarrays of one length. */
value nx_kinds_support_run(value name, value x, value z, value y) {
  const nx_real_loop *k = real_loop(loops, name);
  long n = (long)Caml_ba_array_val(x)->dim[0];
  if ((Caml_ba_array_val(x)->flags & CAML_BA_KIND_MASK) == CAML_BA_FLOAT64)
    k->f64(Caml_ba_data_val(x), Caml_ba_data_val(z), NULL, Caml_ba_data_val(y), n);
  else
    k->f32(Caml_ba_data_val(x), Caml_ba_data_val(z), NULL, Caml_ba_data_val(y), n);
  return Val_unit;
}
