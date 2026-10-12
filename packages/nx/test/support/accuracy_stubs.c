/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The float64 oracle of Nx_test.Accuracy: the C library's long double functions,
   rounded once to double. It is an oracle only where long double is wider than
   double, such as x86_64 and aarch64 Linux; on macOS they are the same. */

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

value nx_test_ldbl_wide(value unit) {
  (void)unit;
  return Val_bool(LDBL_MANT_DIG > DBL_MANT_DIG);
}

/* Trigonometry reduces its argument here, exactly enough for any double, and
   calls the C library on [-pi/4, pi/4] alone: mingw-w64's sinl, cosl and tanl
   reduce by x87's 66-bit pi, and gave tan(-0x1.fffffffffffffp+1023) as
   0x1.534b915c58e89p-2 against 0x1.4530cfe729484p-8. The reduction multiplies
   the argument's integer significand by 192 of 2/pi's bits, from the first
   whose product is not a multiple of 4: the integer part of the product, mod
   4, is the quadrant, and its fraction is within 2^-137 of the exact one,
   while x 2/pi is never within 2^-62 of an integer for a double x. */

/* 2/pi's bits after the point, 64 to a word, after a word of zeros, so that
   bit i, from 1, is at position i + 63 from the first word's top. A double's
   exponent needs bits up to 1162. Computed with mpmath:
   floor(2/pi * 2^1216). */
static const uint64_t two_over_pi[20] = {
    0,
    0xa2f9836e4e441529ULL, 0xfc2757d1f534ddc0ULL, 0xdb6295993c439041ULL,
    0xfe5163abdebbc561ULL, 0xb7246e3a424dd2e0ULL, 0x06492eea09d1921cULL,
    0xfe1deb1cb129a73eULL, 0xe88235f52ebb4484ULL, 0xe99c7026b45f7e41ULL,
    0x3991d639835339f4ULL, 0x9c845f8bbdf9283bULL, 0x1ff897ffde05980fULL,
    0xef2f118b5a0a6d1fULL, 0x6d367ecf27cb09b7ULL, 0x4f463f669e5fea2dULL,
    0x7527bac7ebe5f17bULL, 0x3d0739f78a5292eaULL, 0x6bfb5fb11f8d5d08ULL,
    0x56033046fc7b6babULL,
};

static const long double pio2 = 0x1.921fb54442d18469898cc51701b8p+0L;

/* The 64 bits of two_over_pi from position [p]. */
static uint64_t bits_at(int p) {
  int w = p / 64, sh = p % 64;
  if (sh == 0) return two_over_pi[w];
  return two_over_pi[w] << sh | two_over_pi[w + 1] >> (64 - sh);
}

/* [|x| - k pi/2] in [*r], [|r| <= pi/4], and [k] mod 4, for the finite [x]. */
static int reduced(double x, long double *r) {
  typedef unsigned __int128 u128;
  int e;
  double m = frexp(fabs(x), &e);
  uint64_t n = (uint64_t)ldexp(m, 53); /* |x| = n 2^(e - 53) */
  e -= 53;
  /* |x| 2/pi = n sum b_i 2^(e - i), and the bits with i <= e - 2 add
     multiples of 4. From b_(e - 1), at position e + 62: the product is
     |x| 2/pi 2^190, mod 2^192. */
  uint64_t w2 = bits_at(e + 62), w1 = bits_at(e + 126), w0 = bits_at(e + 190);
  u128 t = (u128)n * w0;
  uint64_t p0 = (uint64_t)t;
  t = (u128)n * w1 + (t >> 64);
  uint64_t p1 = (uint64_t)t;
  t = (u128)n * w2 + (t >> 64);
  uint64_t p2 = (uint64_t)t;
  int k = (int)(p2 >> 62);
  /* The fraction, 190 bits; from 1/2 on, the next quadrant less its
     complement. */
  p2 &= (1ULL << 62) - 1;
  int negative = (int)(p2 >> 61);
  if (negative) {
    k++;
    p0 = ~p0 + 1;
    p1 = ~p1 + (p0 == 0);
    p2 = (~p2 + (p0 == 0 && p1 == 0)) & ((1ULL << 62) - 1);
  }
  /* Each word converts exactly, and the two sums round by 2^-64 at most. */
  long double f = ((long double)p2 * 0x1p-62L + (long double)p1 * 0x1p-126L) +
                  (long double)p0 * 0x1p-190L;
  *r = (negative ? -f : f) * pio2;
  return k & 3;
}

static int small(double x) { return fabs(x) <= 0x1.921fb54442d18p-1; }

static long double sin_q(double x) {
  long double r;
  if (small(x) || !isfinite(x)) return sinl(x);
  long double s;
  switch (reduced(x, &r)) {
  case 0: s = sinl(r); break;
  case 1: s = cosl(r); break;
  case 2: s = -sinl(r); break;
  default: s = -cosl(r); break;
  }
  return x < 0 ? -s : s;
}

static long double cos_q(double x) {
  long double r;
  if (small(x) || !isfinite(x)) return cosl(x);
  switch (reduced(x, &r)) {
  case 0: return cosl(r);
  case 1: return -sinl(r);
  case 2: return -cosl(r);
  default: return sinl(r);
  }
}

static long double tan_q(double x) {
  long double r;
  if (small(x) || !isfinite(x)) return tanl(x);
  long double t = reduced(x, &r) & 1 ? -1.0L / tanl(r) : tanl(r);
  return x < 0 ? -t : t;
}

/* The functions by the index Nx_test.Accuracy gives them. */
static long double ldbl_unary(int k, long double x) {
  switch (k) {
  case 0: return expl(x);
  case 1: return logl(x);
  case 2: return log1pl(x);
  case 3: return expm1l(x);
  case 4: return sin_q((double)x);
  case 5: return cos_q((double)x);
  case 6: return tan_q((double)x);
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
