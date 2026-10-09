/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The reference computes in double, with compensated sums where a bound
   needs more. Elements read through nx_dtype.h, so no format is restated
   here. */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "nx_dtype.h"
#include "ref.h"

typedef nx_ref_view view;

static int64_t at(const view *v, int64_t z, int64_t r, int64_t c) {
  return z * v->s[0] + r * v->s[1] + c * v->s[2];
}

static double float_of(const view *v, int64_t z, int64_t r, int64_t c) {
  const uint8_t *p = (const uint8_t *)v->base;
  int64_t i = at(v, z, r, c);
  switch (v->dtype) {
  case NX_FLOAT64: { double x; memcpy(&x, p + 8 * i, 8); return x; }
  case NX_FLOAT32: { float x; memcpy(&x, p + 4 * i, 4); return x; }
  case NX_FLOAT16: { uint16_t x; memcpy(&x, p + 2 * i, 2); return nx_f16_to_float(x); }
  case NX_BFLOAT16: { uint16_t x; memcpy(&x, p + 2 * i, 2); return nx_bf16_to_float(x); }
  case NX_FLOAT8_E4M3FN: return nx_e4m3fn_to_float(p[i]);
  case NX_FLOAT8_E5M2: return nx_e5m2_to_float(p[i]);
  }
  return NAN;
}

/* An integer element, sign-extended if signed, as 64 bits. */
static uint64_t int_of(const view *v, int64_t z, int64_t r, int64_t c) {
  const uint8_t *p = (const uint8_t *)v->base;
  int64_t i = at(v, z, r, c);
  switch (v->dtype) {
  case NX_INT64: case NX_UINT64: { uint64_t x; memcpy(&x, p + 8 * i, 8); return x; }
  case NX_INT32: { int32_t x; memcpy(&x, p + 4 * i, 4); return (uint64_t)(int64_t)x; }
  case NX_UINT32: { uint32_t x; memcpy(&x, p + 4 * i, 4); return x; }
  case NX_INT16: { int16_t x; memcpy(&x, p + 2 * i, 2); return (uint64_t)(int64_t)x; }
  case NX_UINT16: { uint16_t x; memcpy(&x, p + 2 * i, 2); return x; }
  case NX_INT8: return (uint64_t)(int64_t)(int8_t)p[i];
  case NX_UINT8: case NX_BOOL: return p[i];
  }
  return 0;
}

/* [x] as an integer of [dt]'s width, sign-extended if signed. */
static uint64_t wrap(uint64_t x, int dt) {
  switch (dt) {
  case NX_INT32: return (uint64_t)(int64_t)(int32_t)x;
  case NX_UINT32: return (uint32_t)x;
  case NX_INT16: return (uint64_t)(int64_t)(int16_t)x;
  case NX_UINT16: return (uint16_t)x;
  case NX_INT8: return (uint64_t)(int64_t)(int8_t)x;
  case NX_UINT8: return (uint8_t)x;
  }
  return x;
}

/* [x] rounded once to the float dtype [dt], as a double. */
static double rounded(int dt, double x) {
  switch (dt) {
  case NX_FLOAT64: return x;
  case NX_FLOAT32: return (float)x;
  }
  return nx_bits_to_float(dt, (uint32_t)nx_double_to_bits(dt, x));
}

/* The bits of element (z, r, c), in their low bits. */
static uint64_t bits_of(const view *v, int64_t z, int64_t r, int64_t c) {
  const uint8_t *p = (const uint8_t *)v->base;
  int64_t i = at(v, z, r, c);
  uint64_t x = 0;
  int w = nx_dtype_row_of(v->dtype).bits / 8;
  memcpy(&x, p + w * i, w);
  return x;
}

/* An integer [x], as a signed or unsigned accumulator holds it, cast to the
   dtype [dt]: wrapped to an integer, rounded once to a float, nonzero for
   bool; its bits. */
static uint64_t cast_int(uint64_t x, int sign, int dt) {
  double d, odd;
  float f;
  switch (dt) {
  case NX_FLOAT64:
    d = sign ? (double)(int64_t)x : (double)x;
    return nx_double_bits(d);
  case NX_FLOAT32:
    f = sign ? (float)(int64_t)x : (float)x;
    return nx_float_bits(f);
  case NX_BOOL: return x != 0;
  }
  if (nx_dtype_row_of(dt).kind != NX_KIND_FLOAT) return x;
  odd = sign ? nx_i64_odd((int64_t)x) : nx_u64_odd(x);
  return (uint64_t)nx_double_to_bits(dt, odd);
}

static int is_float(int dt) {
  return nx_dtype_row_of(dt).kind == NX_KIND_FLOAT;
}

/* The unit roundoff of a float dtype, and the gap between [x] and the
   next value of its magnitude in that dtype. */
static void format(int dt, int *mant, int *emin) {
  switch (dt) {
  case NX_FLOAT64: *mant = 52, *emin = -1022; break;
  case NX_FLOAT32: *mant = 23, *emin = -126; break;
  case NX_FLOAT16: *mant = 10, *emin = -14; break;
  case NX_BFLOAT16: *mant = 7, *emin = -126; break;
  case NX_FLOAT8_E4M3FN: *mant = 3, *emin = -6; break;
  default: *mant = 2, *emin = -14; break;
  }
}

static double ulp(int dt, double x) {
  int mant, emin, e;
  format(dt, &mant, &emin);
  frexp(fabs(x), &e);
  return ldexp(1.0, (e - 1 < emin ? emin : e - 1) - mant);
}

/* An error-free product and sum: x·y = p + e, a + b = s + t. */
static void two_prod(double x, double y, double *p, double *e) {
  *p = x * y;
  *e = fma(x, y, -*p);
}

static void two_sum(double a, double b, double *s, double *t) {
  *s = a + b;
  double v = *s - a;
  *t = (a - (*s - v)) + (b - v);
}

/* Output (z, i, j) of y = init + a·b over [k], against the error bound:
   |y - s| / (γ(k + 1, 2u)·(|init| + Σ|a||b|) + ulp_y(y) / 2), s the exact
   sum (Dot2: about twice double's precision), u the accumulator's unit
   roundoff; with [flush], 2^-126·(1 + Σ(1 + |a| + |b|)) more. The exact
   sum rounded once to y passes, past y's range too. A NaN or an infinity
   among the operands or init asks for IEEE's answer, which the plain sum
   gives in any order: NaN from a NaN, ∞·0 or ∞ - ∞, else the infinity. An
   integer or bool y lies between the casts of the ends of the sums the
   bound allows. -1 if y misses an answer the cast fixes, or is NaN where
   the sum is finite. */
static double float_ratio(const view *a, const view *b, const view *init,
                          const view *y, int64_t k, int acc, int flush,
                          int64_t z, int64_t i, int64_t j) {
  double s = init ? float_of(init, z, i, j) : 0, c = 0, mag = fabs(s);
  double plain = s, operands = 0;
  int special = !isfinite(s);
  for (int64_t q = 0; q < k; q++) {
    double x = float_of(a, z, i, q), w = float_of(b, z, j, q), p, e, t;
    plain += x * w;
    operands += 1 + fabs(x) + fabs(w);
    special |= !isfinite(x) || !isfinite(w);
    two_prod(x, w, &p, &e);
    two_sum(s, p, &s, &t);
    c += t + e;
    mag += fabs(x * w);
  }
  s += c;
  int mant, emin;
  format(acc, &mant, &emin);
  double u2 = ldexp(1.0, -mant), g = (k + 1) * u2 / (1 - (k + 1) * u2);
  double slack = g * mag + (flush ? ldexp(1.0, -126) * (1 + operands) : 0);
  if (!is_float(y->dtype)) {
    /* A cast to an integer or bool is monotone: y lies between the casts
       of the ends of the sums the bound allows, or is the cast of IEEE's
       answer. */
    const int dt = y->dtype;
    uint64_t got = wrap(bits_of(y, z, i, j), dt);
    if (special || !isfinite(s))
      return got == wrap(nx_double_to_bits(dt, plain), dt) ? 0 : -1;
    int64_t lo = nx_double_to_bits(dt, s - slack);
    int64_t hi = nx_double_to_bits(dt, s + slack);
    if (dt == NX_BOOL)
      return got == 1 || (s - slack <= 0 && 0 <= s + slack && got == 0) ? 0
                                                                          : -1;
    if (nx_dtype_row_of(dt).kind == NX_KIND_UNSIGNED)
      return (uint64_t)lo <= got && got <= (uint64_t)hi ? 0 : -1;
    return lo <= (int64_t)got && (int64_t)got <= hi ? 0 : -1;
  }
  double got = float_of(y, z, i, j);
  if (special) return (isnan(plain) && isnan(got)) || plain == got ? 0 : -1;
  if (!isfinite(s) || !isfinite(mag))
    return (isnan(s) && isnan(got)) || s == got ? 0 : INFINITY;
  /* The exact sum rounded once is right, past y's range too. */
  double r = rounded(y->dtype, s);
  if (got == r || (isnan(got) && isnan(r))) return 0;
  /* Its error would be NaN, which no bound orders. */
  if (isnan(got)) return -1;
  double allowed = slack + ulp(y->dtype, fabs(s) + g * mag) / 2;
  double err = fabs(got - s);
  return err == 0 ? 0 : err / allowed;
}

/* Whether output (z, i, j) of an integer contraction holds init + a·b
   wrapped to the accumulator, widened by its sign, then cast to y. */
static int int_exact(const view *a, const view *b, const view *init,
                     const view *y, int64_t k, int acc, int64_t z, int64_t i,
                     int64_t j) {
  uint64_t s = init ? int_of(init, z, i, j) : 0;
  for (int64_t q = 0; q < k; q++) s += int_of(a, z, i, q) * int_of(b, z, j, q);
  int sign = nx_dtype_row_of(acc).kind == NX_KIND_SIGNED;
  uint64_t want = cast_int(wrap(s, acc), sign, y->dtype);
  return wrap(want, y->dtype) == wrap(bits_of(y, z, i, j), y->dtype);
}

nx_ref_result nx_ref_contract(const view *a, const view *b,
                              const view *init, const view *y, int64_t batch,
                              int64_t m, int64_t n, int64_t k, int acc,
                              int flush, int64_t samples) {
  nx_ref_result r = {0, 0, -1};
  int64_t total = batch * m * n;
  for (int64_t t = 0; t < (samples < total ? samples : total); t++) {
    int64_t o = samples < total
                    ? (int64_t)((uint64_t)t * 0x9E3779B97F4A7C15ull % total)
                    : t;
    int64_t z = o / (m * n), i = o / n % m, j = o % n;
    double x = is_float(acc)
                   ? float_ratio(a, b, init, y, k, acc, flush, z, i, j)
                   : int_exact(a, b, init, y, k, acc, z, i, j) ? 0 : -1;
    if (x < 0) {
      if (r.wrong++ == 0) r.at = o;
    } else if (x != 0 && !(x <= r.worst)) {
      r.worst = x;
      if (r.wrong == 0) r.at = o;
    }
  }
  return r;
}
