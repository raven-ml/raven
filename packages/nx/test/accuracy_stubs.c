/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The float64 oracle of test_accuracy: the C library's long double functions,
   rounded once to double. It is an oracle only where long double is wider than
   double, such as x86_64 and aarch64 Linux; on macOS they are the same. */

#include <float.h>
#include <math.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

value nx_test_ldbl_wide(value unit) {
  (void)unit;
  return Val_bool(LDBL_MANT_DIG > DBL_MANT_DIG);
}

/* The functions by the index test_accuracy gives them. */
static long double ldbl_unary(int k, long double x) {
  switch (k) {
  case 0: return expl(x);
  case 1: return logl(x);
  case 2: return log1pl(x);
  case 3: return expm1l(x);
  case 4: return sinl(x);
  case 5: return cosl(x);
  case 6: return tanl(x);
  case 7: return asinl(x);
  case 8: return acosl(x);
  case 9: return atanl(x);
  case 10: return sinhl(x);
  case 11: return coshl(x);
  case 12: return tanhl(x);
  default: return erfl(x);
  }
}

/* [nx_test_ldbl_unary k xs ys] writes function [k] of each element of the
   float array [xs] into [ys], of the same length. */
value nx_test_ldbl_unary(value k, value xs, value ys) {
  CAMLparam3(k, xs, ys);
  mlsize_t n = Wosize_val(xs) / Double_wosize;
  for (mlsize_t i = 0; i < n; i++)
    Store_double_flat_field(
        ys, i,
        (double)ldbl_unary(Int_val(k), (long double)Double_flat_field(xs, i)));
  CAMLreturn(Val_unit);
}

/* [nx_test_ldbl_binary k as bs ys]: [pow] for [k = 0], [atan2] otherwise. */
value nx_test_ldbl_binary(value k, value as, value bs, value ys) {
  CAMLparam4(k, as, bs, ys);
  mlsize_t n = Wosize_val(as) / Double_wosize;
  for (mlsize_t i = 0; i < n; i++) {
    long double a = Double_flat_field(as, i), b = Double_flat_field(bs, i);
    Store_double_flat_field(ys, i,
                            (double)(Int_val(k) == 0 ? powl(a, b)
                                                     : atan2l(a, b)));
  }
  CAMLreturn(Val_unit);
}
