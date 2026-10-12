/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The float64 oracle of Nx_test.Accuracy: the C library's long double functions,
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

/* Trigonometry reduces its argument here, by pi/2 in four parts, 160 bits in
   all, and calls the C library on [-pi/4, pi/4] alone: mingw-w64's sinl, cosl
   and tanl reduce by x87's 66-bit pi, and its tanl missed tan(7389.98) by 3
   ulps. The first three parts have 32 significant bits, so that [k] times
   each is exact for [|k| < 2^32]; past that the C library reduces. */
static const long double pio2_1 = 0x1.921fb544p+0L;
static const long double pio2_2 = 0x1.0b4611a6p-34L;
static const long double pio2_3 = 0x1.3198a2ep-69L;
static const long double pio2_4 = 0x1.b839a252049c1114p-104L;
static const long double two_over_pi = 0x1.45f306dc9c882a52p-1L;

/* [x - k pi/2], [k] the quadrant of [x] mod 4, for [|x| / (pi/2) < 2^32]. */
static int reduced(long double x, long double *r) {
  long double k = nearbyintl(x * two_over_pi);
  *r = (((x - k * pio2_1) - k * pio2_2) - k * pio2_3) - k * pio2_4;
  return (int)((long long)k & 3);
}

static int reducible(long double x) { return fabsl(x) < 0x1p32L; }

static long double sin_q(long double x) {
  long double r;
  if (!reducible(x)) return sinl(x);
  switch (reduced(x, &r)) {
  case 0: return sinl(r);
  case 1: return cosl(r);
  case 2: return -sinl(r);
  default: return -cosl(r);
  }
}

static long double cos_q(long double x) {
  long double r;
  if (!reducible(x)) return cosl(x);
  switch (reduced(x, &r)) {
  case 0: return cosl(r);
  case 1: return -sinl(r);
  case 2: return -cosl(r);
  default: return sinl(r);
  }
}

static long double tan_q(long double x) {
  long double r;
  if (!reducible(x)) return tanl(x);
  return reduced(x, &r) & 1 ? -1.0L / tanl(r) : tanl(r);
}

/* The functions by the index Nx_test.Accuracy gives them. */
static long double ldbl_unary(int k, long double x) {
  switch (k) {
  case 0: return expl(x);
  case 1: return logl(x);
  case 2: return log1pl(x);
  case 3: return expm1l(x);
  case 4: return sin_q(x);
  case 5: return cos_q(x);
  case 6: return tan_q(x);
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
