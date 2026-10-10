/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The float kinds of one compute type, written once: nx_kinds.h includes
   this file for f32 and, but on Metal, for f64, after defining

   - NX_T, the type, and NX_U, the unsigned integer of its width;
   - NX_(f), which names f at the type: NX_(nx_exp) is nx_exp_f32;
   - NX_C(c), the type's constant c: NX_C(SHIFT) is NX_SHIFT_F32;
   - NX_FMA, NX_SQRT, NX_BITS, NX_OF_BITS and NX_ISNAN;
   - the polynomials (NX_(nx_exp_q), NX_(nx_exp2_poly), NX_(nx_log_t),
     NX_(nx_log_tp), NX_(nx_sincos_p), NX_(nx_asin_b)), the pair and
     reduction types, and the prototypes of the trigonometric reductions.

   It defines every kind but those whose algorithm differs between the
   types: the trigonometric reductions, atan, atan2, erf and mod. */

#define NX_ONE ((NX_T)1)
#define NX_SIGN_BIT ((NX_U)1 << (8 * sizeof(NX_U) - 1))

/* Types */

/* An integer as a float and an int. */
typedef struct {
  NX_T f;
  int32_t i;
} NX_(nx_int);

/* x = k ln2 + r, and rc the rounding error of r. */
typedef struct {
  NX_T r, rc, kf;
  int32_t k;
} NX_(nx_exp_red);

/* e^x - 1 = 2^k (v.hi + v.lo). */
typedef struct {
  NX_(nx_pair) v;
  int32_t k;
} NX_(nx_expm1_red);

/* u = 2^k (1 + f). */
typedef struct {
  NX_T f, k;
} NX_(nx_log_red);

/* asin's argument s and the tail of its asin, s = sqrt((1 - |x|)/2) when
   big. */
typedef struct {
  NX_T s, tail;
  int big;
} NX_(nx_asin_red);

/* expm1 r = em and w = 1 + em as pairs, and 2^-2k. */
typedef struct {
  NX_(nx_pair) em, w;
  NX_T m;
  int32_t k;
} NX_(nx_hyp_red);

/* Helpers */

NX_INLINE NX_T NX_(nx_abs_bits)(NX_T x) {
  return NX_OF_BITS(NX_BITS(x) & ~NX_SIGN_BIT);
}

NX_INLINE NX_T NX_(nx_copysign)(NX_T mag, NX_T sgn) {
  return NX_OF_BITS((NX_BITS(mag) & ~NX_SIGN_BIT) | (NX_BITS(sgn) & NX_SIGN_BIT));
}

/* 2^k for a normal power. */
NX_INLINE NX_T NX_(nx_pow2)(int32_t k) {
  return NX_OF_BITS((NX_U)(k + NX_C(BIAS)) << NX_C(MBITS));
}

/* y 2^k rounded once: the first product is exact while y 2^(k/2) is
   normal, which holds for y in [1/2, 2] and every k the kinds pass. */
NX_INLINE NX_T NX_(nx_scale)(NX_T y, int32_t k) {
  int32_t k1 = k / 2;
  return y * NX_(nx_pow2)(k1) * NX_(nx_pow2)(k - k1);
}

/* The sum a + b as hi + lo exactly (Knuth's TwoSum). */
NX_INLINE NX_(nx_pair) NX_(nx_two_sum)(NX_T a, NX_T b) {
  NX_(nx_pair) r;
  r.hi = a + b;
  NX_T bb = r.hi - a;
  r.lo = (a - (r.hi - bb)) + (b - bb);
  return r;
}

/* hi + lo renormalised, |hi| >= |lo| (Fast2Sum). */
NX_INLINE NX_(nx_pair) NX_(nx_fast_two_sum)(NX_T hi, NX_T lo) {
  NX_(nx_pair) r;
  r.hi = hi + lo;
  r.lo = lo - (r.hi - hi);
  return r;
}

/* (n.hi + n.lo) / (d.hi + d.lo), renormalised so that the quotient's
   error and the tails are first-order corrections: one division, the
   quotient's head the numerator times the reciprocal, its tail the exact
   residual times it. */
NX_INLINE NX_(nx_pair) NX_(nx_div_pair)(NX_(nx_pair) n, NX_(nx_pair) d) {
  n = NX_(nx_fast_two_sum)(n.hi, n.lo);
  d = NX_(nx_fast_two_sum)(d.hi, d.lo);
  NX_T rd = NX_ONE / d.hi;
  NX_(nx_pair) q;
  q.hi = n.hi * rd;
  q.lo = (NX_FMA(-q.hi, d.hi, n.hi) + NX_FMA(-q.hi, d.lo, n.lo)) * rd;
  return q;
}

/* From t = v + NX_C(SHIFT), |v| below half the shifter, the integer nearest
   v, as a float and as an int. */
NX_INLINE NX_(nx_int) NX_(nx_nearest)(NX_T t) {
  NX_(nx_int) k;
  k.f = t - NX_C(SHIFT);
  k.i = (int32_t)(int64_t)(NX_BITS(t) - NX_BITS(NX_C(SHIFT)));
  return k;
}

/* NaN results. nx_nan1 and nx_nan2 give r, computed from one or two
   operands, as a kind gives it: the first NaN operand itself, else the
   quiet NaN of a clear sign bit where r is a NaN. On x86 an operation of
   two NaN operands gives one of them by their order, which a compiler sets
   per target and differently in vector and scalar code, and a negation
   flips a NaN's sign; these selects make the bits the same everywhere. */

NX_INLINE NX_T NX_(nx_nan1)(NX_T a, NX_T r) {
  r = r != r ? NX_C(NAN) : r;
  return a != a ? a : r;
}

NX_INLINE NX_T NX_(nx_nan2)(NX_T a, NX_T b, NX_T r) {
  r = NX_(nx_nan1)(b, r);
  return NX_ISNAN(a) ? a : r;
}

/* Exact kinds */

NX_INLINE NX_T NX_(nx_add)(NX_T a, NX_T b) { return NX_(nx_nan2)(a, b, a + b); }
NX_INLINE NX_T NX_(nx_sub)(NX_T a, NX_T b) { return NX_(nx_nan2)(a, b, a - b); }
NX_INLINE NX_T NX_(nx_mul)(NX_T a, NX_T b) { return NX_(nx_nan2)(a, b, a * b); }
NX_INLINE NX_T NX_(nx_fdiv)(NX_T a, NX_T b) { return NX_(nx_nan2)(a, b, a / b); }

NX_INLINE NX_T NX_(nx_fma)(NX_T a, NX_T b, NX_T c) {
  NX_T r = NX_(nx_nan1)(c, NX_FMA(a, b, c));
  r = NX_ISNAN(b) ? b : r;
  return NX_ISNAN(a) ? a : r;
}

/* The sign bit flipped: CUDA's negation quiets a signalling NaN. */
NX_INLINE NX_T NX_(nx_neg)(NX_T a) { return NX_OF_BITS(NX_BITS(a) ^ NX_SIGN_BIT); }
NX_INLINE NX_T NX_(nx_abs)(NX_T a) { return NX_(nx_abs_bits)(a); }
NX_INLINE NX_T NX_(nx_sign)(NX_T a) {
  return NX_ISNAN(a) ? a : (NX_T)((a > 0) - (a < 0));
}
NX_INLINE NX_T NX_(nx_recip)(NX_T a) { return NX_(nx_nan1)(a, NX_ONE / a); }
NX_INLINE NX_T NX_(nx_sqrt)(NX_T a) { return NX_(nx_nan1)(a, NX_SQRT(a)); }

NX_INLINE NX_T NX_(nx_trunc)(NX_T a) {
  NX_T m = NX_(nx_abs_bits)(a);
  NX_T t = (m + NX_C(INTEGRAL)) - NX_C(INTEGRAL);
  t = t > m ? t - 1 : t;
  NX_T r = NX_OF_BITS(NX_BITS(t) | (NX_BITS(a) & NX_SIGN_BIT));
  return m < NX_C(INTEGRAL) ? r : a;
}

NX_INLINE NX_T NX_(nx_floor)(NX_T a) {
  NX_T t = NX_(nx_trunc)(a);
  return t > a ? t - 1 : t;
}

NX_INLINE NX_T NX_(nx_ceil)(NX_T a) {
  NX_T t = NX_(nx_trunc)(a);
  return t < a ? t + 1 : t;
}

NX_INLINE NX_T NX_(nx_round)(NX_T a) {
  NX_T t = NX_(nx_trunc)(a);
  NX_T d = NX_(nx_abs_bits)(a - t);
  return d >= (NX_T)0.5 ? (a < 0 ? t - 1 : t + 1) : t;
}

NX_INLINE NX_T NX_(nx_maximum)(NX_T a, NX_T b) {
  NX_T g = (a > b || NX_ISNAN(a)) ? a : b;
  NX_U tie = (NX_U)0 - (NX_U)(a == b);
  return NX_OF_BITS(NX_BITS(g) & (NX_BITS(a) | ~tie));
}

NX_INLINE NX_T NX_(nx_minimum)(NX_T a, NX_T b) {
  NX_T g = (a < b || NX_ISNAN(a)) ? a : b;
  NX_U tie = (NX_U)0 - (NX_U)(a == b);
  return NX_OF_BITS(NX_BITS(g) | (NX_BITS(a) & tie));
}

NX_INLINE int NX_(nx_equal)(NX_T a, NX_T b) { return a == b; }
NX_INLINE int NX_(nx_not_equal)(NX_T a, NX_T b) { return a != b; }
NX_INLINE int NX_(nx_less)(NX_T a, NX_T b) { return a < b; }
NX_INLINE int NX_(nx_less_equal)(NX_T a, NX_T b) { return a <= b; }
NX_INLINE NX_T NX_(nx_where)(int c, NX_T a, NX_T b) { return c ? a : b; }

/* Exponentials

   x = k ln2 + r with k the integer nearest x / ln2, so |r| <= ln2 / 2.
   ln2's head has few enough bits that k ln2_hi is exact over the kinds'
   range of k, so the first fma is exact; rc is r's rounding error. */

NX_INLINE NX_(nx_exp_red) NX_(nx_exp_reduce)(NX_T x) {
  NX_(nx_exp_red) e;
  NX_(nx_int) k = NX_(nx_nearest)(NX_FMA(x, NX_C(LOG2E), NX_C(SHIFT)));
  e.kf = k.f;
  e.k = k.i;
  NX_T rh = NX_FMA(-e.kf, NX_C(LN2_HI), x);
  e.r = NX_FMA(-e.kf, NX_C(LN2_LO), rh);
  e.rc = NX_FMA(-e.kf, NX_C(LN2_LO), rh - e.r);
  return e;
}

/* exp: e^x = 2^k (1 + expm1 r), expm1 r = r + r^2 Q(r); its NaN bits
   unpinned, for the kinds that call it. */
NX_INLINE NX_T NX_(nx_exp_of)(NX_T x) {
  x = x > NX_C(EXP_HI) ? NX_C(EXP_HI) : x;
  x = x < NX_C(EXP_LO) ? NX_C(EXP_LO) : x;
  NX_(nx_exp_red) e = NX_(nx_exp_reduce)(x);
  NX_T em = NX_FMA(e.r * e.r, NX_(nx_exp_q)(e.r), e.r);
  return NX_(nx_scale)(NX_ONE + em, e.k);
}

NX_INLINE NX_T NX_(nx_exp)(NX_T x) {
  return NX_(nx_nan1)(x, NX_(nx_exp_of)(x));
}

/* exp2: x = k + r, |r| <= 1/2, 2^r = 1 + r P(r). Exact at integers whose
   power is in the type. */
NX_INLINE NX_T NX_(nx_exp2)(NX_T x) {
  x = x > NX_C(EXP2_HI) ? NX_C(EXP2_HI) : x;
  x = x < NX_C(EXP2_LO) ? NX_C(EXP2_LO) : x;
  NX_(nx_int) k = NX_(nx_nearest)(x + NX_C(SHIFT));
  return NX_(nx_nan1)(x, NX_(nx_scale)(NX_(nx_exp2_poly)(x - k.f), k.i));
}

/* expm1: e^x - 1 = 2^k ((1 - 2^-k) + r + r^2 Q(r)), the bracket summed
   exactly but for the last rounding: a pair, which tanh divides. r's
   rounding error rc is added scaled by e^r. Past NX_C(EXPM1_KCAP) the
   2^-k term is below 2^-64 of the result. */
NX_INLINE NX_(nx_expm1_red) NX_(nx_expm1_pair)(NX_T x) {
  x = x > NX_C(EXP_HI) ? NX_C(EXP_HI) : x;
  x = x < NX_C(EXPM1_LO) ? NX_C(EXPM1_LO) : x;
  NX_(nx_exp_red) e = NX_(nx_exp_reduce)(x);
  NX_T h = NX_FMA(e.r * e.r, NX_(nx_exp_q)(e.r), NX_FMA(e.rc, e.r, e.rc));
  NX_T m = NX_(nx_pow2)(-(e.k > NX_C(EXPM1_KCAP) ? NX_C(EXPM1_KCAP) : e.k));
  NX_(nx_pair) u = NX_(nx_two_sum)(NX_ONE, -m);
  NX_(nx_pair) sum = NX_(nx_two_sum)(u.hi, e.r);
  NX_(nx_expm1_red) r;
  r.v.hi = sum.hi;
  r.v.lo = sum.lo + (u.lo + h);
  r.k = e.k;
  return r;
}

/* Below NX_C(EXPM1_TINY), expm1 x rounds to x, whose zero sign the sums
   lose. */
NX_INLINE NX_T NX_(nx_expm1)(NX_T x) {
  NX_(nx_expm1_red) e = NX_(nx_expm1_pair)(x);
  NX_T y = NX_(nx_scale)(e.v.hi + e.v.lo, e.k);
  return NX_(nx_nan1)(x, NX_(nx_abs_bits)(x) < NX_C(EXPM1_TINY) ? x : y);
}

/* Logarithms

   u = 2^k m with m in [sqrt(1/2), sqrt(2)), and f = m - 1 exactly. With
   s = f / (2 + f), log(1 + f) = 2 atanh s = f - f^2/2 + s (f^2/2 + R),
   R = s^2 T(s^2). */

/* The reduction of a positive u; a subnormal is scaled up first. */
NX_INLINE NX_(nx_log_red) NX_(nx_log_reduce)(NX_T u) {
  NX_(nx_log_red) l;
  int sub = NX_BITS(u) < NX_C(MIN_NORMAL_BITS);
  u = sub ? u * NX_C(SUB_SCALE) : u;
  NX_U ix = NX_BITS(u) + (NX_C(ONE_BITS) - NX_C(SQRT_HALF_BITS));
  l.k = (NX_T)((int32_t)(ix >> NX_C(MBITS)) - NX_C(BIAS) - (sub ? NX_C(MBITS) : 0));
  l.f = NX_OF_BITS((ix & NX_C(MMASK)) + NX_C(SQRT_HALF_BITS)) - NX_ONE;
  return l;
}

/* k ln2 + log(1 + f) + c, for a correction c small against the result. */
NX_INLINE NX_T NX_(nx_log_core)(NX_(nx_log_red) l, NX_T c) {
  NX_T f = l.f;
  NX_T s = f / ((NX_T)2 + f);
  NX_T z = s * s;
  NX_T hfsq = (NX_T)0.5 * f * f;
  NX_T sr = NX_FMA(s, NX_FMA(z, NX_(nx_log_t)(z), hfsq),
                   NX_FMA(l.k, NX_C(LN2_LO), c));
  return NX_FMA(l.k, NX_C(LN2_HI), -((hfsq - sr) - f));
}

/* The value at u = 1 + x outside (0, inf), and at a NaN x. */
NX_INLINE NX_T NX_(nx_log_special)(NX_T u, NX_T x, NX_T y) {
  y = u == NX_C(INF) ? u : y;
  y = u == 0 ? -NX_C(INF) : y;
  y = u < 0 ? NX_C(NAN) : y;
  return NX_(nx_nan1)(x, y);
}

NX_INLINE NX_T NX_(nx_log)(NX_T x) {
  NX_T y = NX_(nx_log_core)(NX_(nx_log_reduce)(x), 0);
  return NX_(nx_log_special)(x, x, y);
}

/* log2: log(1 + f) as hi + lo, hi truncated so that its product by
   1/ln2's head is exact. Exact at powers of two. */
NX_INLINE NX_T NX_(nx_log2)(NX_T x) {
  NX_(nx_log_red) l = NX_(nx_log_reduce)(x);
  NX_T f = l.f;
  NX_T s = f / ((NX_T)2 + f);
  NX_T z = s * s;
  NX_T hfsq = (NX_T)0.5 * f * f;
  NX_T hi = NX_OF_BITS(NX_BITS(f - hfsq) & NX_C(LOG2_HI_MASK));
  NX_T lo = NX_FMA(s, NX_FMA(z, NX_(nx_log_t)(z), hfsq), (f - hi) - hfsq);
  NX_T v = NX_FMA(lo, NX_C(IVLN2_HI), (lo + hi) * NX_C(IVLN2_LO));
  NX_T y = NX_FMA(hi, NX_C(IVLN2_HI), v) + l.k;
  return NX_(nx_log_special)(x, x, y);
}

/* log1p: u = 1 + x rounded, c its exact error; log(u + c) = log u + c/u
   to the bound. */
NX_INLINE NX_T NX_(nx_log1p)(NX_T x) {
  NX_(nx_pair) u = NX_(nx_two_sum)(NX_ONE, x);
  NX_T y = NX_(nx_log_core)(NX_(nx_log_reduce)(u.hi), u.lo / u.hi);
  y = x == 0 ? x : y;
  return NX_(nx_log_special)(u.hi, x, y);
}

/* Trigonometric functions

   x = q pi/2 + r, r = hi + lo with |r| <= pi/4: sin r = r + r^3 S(r^2) and
   cos r = 1 - r^2/2 + r^4 C(r^2), each a head and a tail an eighth of it
   or less, which tan divides as pairs; q mod 4 picks the signs and the
   swap. Below NX_C(TRIG_TINY), sin x and tan x round to x. */

NX_INLINE NX_(nx_pair) NX_(nx_sin_kernel)(NX_T hi, NX_T lo) {
  NX_T z = hi * hi;
  NX_(nx_pair) p;
  p.hi = hi;
  p.lo = NX_FMA(hi * z, NX_(nx_sincos_p)(z, 0), lo);
  return p;
}

NX_INLINE NX_(nx_pair) NX_(nx_cos_kernel)(NX_T hi, NX_T lo) {
  NX_T z = hi * hi;
  NX_T zl = NX_FMA(hi, hi, -z);
  NX_T hz = (NX_T)0.5 * z;
  NX_(nx_pair) p;
  p.hi = NX_ONE - hz;
  p.lo = ((NX_ONE - p.hi) - hz) +
         NX_FMA(z * z, NX_(nx_sincos_p)(z, 1), NX_FMA(-hi, lo, (NX_T)-0.5 * zl));
  return p;
}

NX_INLINE NX_T NX_(nx_sin_at)(NX_T x, NX_(nx_rem) r) {
  NX_(nx_pair) s = NX_(nx_sin_kernel)(r.hi, r.lo);
  NX_(nx_pair) c = NX_(nx_cos_kernel)(r.hi, r.lo);
  NX_T y = (r.q & 1) ? c.hi + c.lo : s.hi + s.lo;
  y = (r.q & 2) ? -y : y;
  return NX_(nx_nan1)(x, NX_(nx_abs_bits)(x) < NX_C(TRIG_TINY) ? x : y);
}

NX_INLINE NX_T NX_(nx_cos_at)(NX_T x, NX_(nx_rem) r) {
  NX_(nx_pair) s = NX_(nx_sin_kernel)(r.hi, r.lo);
  NX_(nx_pair) c = NX_(nx_cos_kernel)(r.hi, r.lo);
  NX_T y = (r.q & 1) ? s.hi + s.lo : c.hi + c.lo;
  return NX_(nx_nan1)(x, ((r.q + 1) & 2) ? -y : y);
}

NX_INLINE NX_T NX_(nx_tan_at)(NX_T x, NX_(nx_rem) r) {
  NX_(nx_pair) s = NX_(nx_sin_kernel)(r.hi, r.lo);
  NX_(nx_pair) c = NX_(nx_cos_kernel)(r.hi, r.lo);
  /* -cos / sin in odd quadrants, sin / cos in even */
  int odd = (int)(r.q & 1);
  NX_(nx_pair) n = {odd ? -c.hi : s.hi, odd ? -c.lo : s.lo};
  NX_(nx_pair) d = {odd ? s.hi : c.hi, odd ? s.lo : c.lo};
  NX_(nx_pair) q = NX_(nx_div_pair)(n, d);
  return NX_(nx_nan1)(x, NX_(nx_abs_bits)(x) < NX_C(TRIG_TINY) ? x : q.hi + q.lo);
}

/* Whether x takes the integer reduction. */
NX_INLINE int NX_(nx_trig_big)(NX_T x) {
  NX_T a = NX_(nx_abs_bits)(x);
  return a >= NX_C(PIO2_BIG) && a < NX_C(INF);
}

/* sin, cos and tan below the switch */
NX_INLINE NX_T NX_(nx_sin_near)(NX_T x) {
  return NX_(nx_sin_at)(x, NX_(nx_rem_pio2_cw)(x));
}

NX_INLINE NX_T NX_(nx_cos_near)(NX_T x) {
  return NX_(nx_cos_at)(x, NX_(nx_rem_pio2_cw)(x));
}

NX_INLINE NX_T NX_(nx_tan_near)(NX_T x) {
  return NX_(nx_tan_at)(x, NX_(nx_rem_pio2_cw)(x));
}

NX_INLINE NX_T NX_(nx_sin)(NX_T x) {
  if (NX_(nx_trig_big)(x)) return NX_(nx_sin_at)(x, NX_(nx_rem_pio2_big)(x));
  return NX_(nx_sin_near)(x);
}

NX_INLINE NX_T NX_(nx_cos)(NX_T x) {
  if (NX_(nx_trig_big)(x)) return NX_(nx_cos_at)(x, NX_(nx_rem_pio2_big)(x));
  return NX_(nx_cos_near)(x);
}

NX_INLINE NX_T NX_(nx_tan)(NX_T x) {
  if (NX_(nx_trig_big)(x)) return NX_(nx_tan_at)(x, NX_(nx_rem_pio2_big)(x));
  return NX_(nx_tan_near)(x);
}

/* Inverse sine and cosine

   asin s = s + s^3 B(s^2) on |s| <= 1/2. Past 1/2, asin a = pi/2 - 2 asin s
   with s = sqrt((1 - a)/2), whose rounding c is kept. */

/* asin |x| as s + tail: |x| itself at or below 1/2, else sqrt((1 - |x|)/2),
   whose asin the caller doubles. */
NX_INLINE NX_(nx_asin_red) NX_(nx_asin_reduce)(NX_T a) {
  NX_(nx_asin_red) r;
  r.big = a > (NX_T)0.5;
  NX_T z = r.big ? (NX_ONE - a) * (NX_T)0.5 : a * a;
  NX_T s = r.big ? NX_SQRT(z) : a;
  NX_T c = r.big && z > 0 ? NX_FMA(-s, s, z) / (s + s) : 0;
  r.s = s;
  r.tail = NX_FMA(s * z, NX_(nx_asin_b)(z), c);
  return r;
}

NX_INLINE NX_T NX_(nx_asin)(NX_T x) {
  NX_T a = NX_(nx_abs_bits)(x);
  NX_(nx_asin_red) r = NX_(nx_asin_reduce)(a);
  NX_T d = r.s + r.s;
  NX_T hi = NX_C(PIO2_HI) - d;
  NX_T lo = ((NX_C(PIO2_HI) - hi) - d) + NX_FMA((NX_T)-2, r.tail, NX_C(PIO2_LO));
  NX_T y = r.big ? hi + lo : r.s + r.tail;
  y = a > NX_ONE ? NX_C(NAN) : y;
  return NX_(nx_nan1)(x, NX_(nx_copysign)(y, x));
}

/* acos: pi/2 - asin x within 1/2, 2 asin s above, pi - 2 asin s below; the
   asin term adds where |x| > 1/2 and x > 0 are both true or both false. */
NX_INLINE NX_T NX_(nx_acos)(NX_T x) {
  NX_T a = NX_(nx_abs_bits)(x);
  NX_(nx_asin_red) r = NX_(nx_asin_reduce)(a);
  int neg = (int)(NX_BITS(x) >> (8 * sizeof(NX_U) - 1));
  NX_T term = r.big ? r.s + r.s : r.s;
  NX_T tail = r.big ? r.tail + r.tail : r.tail;
  NX_T base_hi = r.big ? (neg ? NX_C(PI_HI) : 0) : NX_C(PIO2_HI);
  NX_T base_lo = r.big ? (neg ? NX_C(PI_LO) : 0) : NX_C(PIO2_LO);
  int plus = r.big != neg;
  term = plus ? term : -term;
  tail = plus ? tail : -tail;
  NX_(nx_pair) h = NX_(nx_two_sum)(base_hi, term);
  NX_T y = h.hi + (h.lo + (base_lo + tail));
  y = a > NX_ONE ? NX_C(NAN) : y;
  return NX_(nx_nan1)(x, y);
}

/* Hyperbolic functions

   For a = |x| = k ln2 + r, with em = expm1 r and w = 1 + em:
   sinh a = 2^(k-1) (em + (em + 1 - 2^-2k) / w) and
   cosh a = 2^(k-1) (w + 2^-2k / w), which have no cancellation at k = 0.
   em and w are pairs; past NX_C(HYP_KCAP) the 2^-2k term is below 2^-64 of
   the result. */

NX_INLINE NX_(nx_hyp_red) NX_(nx_hyp_reduce)(NX_T a) {
  NX_(nx_hyp_red) h;
  a = a > NX_C(HYP_HI) ? NX_C(HYP_HI) : a;
  NX_(nx_exp_red) e = NX_(nx_exp_reduce)(a);
  h.em.hi = e.r;
  h.em.lo = NX_FMA(e.r * e.r, NX_(nx_exp_q)(e.r), NX_FMA(e.rc, e.r, e.rc));
  h.w = NX_(nx_two_sum)(NX_ONE, e.r);
  h.w.lo += h.em.lo;
  h.k = e.k;
  h.m = NX_(nx_pow2)(-2 * (e.k > NX_C(HYP_KCAP) ? NX_C(HYP_KCAP) : e.k));
  return h;
}

NX_INLINE NX_T NX_(nx_sinh)(NX_T x) {
  NX_(nx_hyp_red) h = NX_(nx_hyp_reduce)(NX_(nx_abs_bits)(x));
  NX_(nx_pair) n = NX_(nx_two_sum)(NX_ONE - h.m, h.em.hi);
  n.lo += h.em.lo;
  NX_(nx_pair) q = NX_(nx_div_pair)(n, h.w);
  NX_(nx_pair) v = NX_(nx_two_sum)(h.em.hi, q.hi);
  NX_T y = v.hi + (v.lo + (h.em.lo + q.lo));
  return NX_(nx_nan1)(x, NX_(nx_copysign)(NX_(nx_scale)(y, h.k - 1), x));
}

NX_INLINE NX_T NX_(nx_cosh)(NX_T x) {
  NX_(nx_hyp_red) h = NX_(nx_hyp_reduce)(NX_(nx_abs_bits)(x));
  NX_(nx_pair) m = {h.m, 0};
  NX_(nx_pair) q = NX_(nx_div_pair)(m, h.w);
  NX_(nx_pair) v = NX_(nx_two_sum)(h.w.hi, q.hi);
  NX_T y = v.hi + (v.lo + (h.w.lo + q.lo));
  return NX_(nx_nan1)(x, NX_(nx_scale)(y, h.k - 1));
}

/* tanh |x| = -em / (em + 2) with em = expm1(-2|x|), divided as pairs. */
NX_INLINE NX_T NX_(nx_tanh)(NX_T x) {
  NX_(nx_expm1_red) e = NX_(nx_expm1_pair)((NX_T)-2 * NX_(nx_abs_bits)(x));
  NX_T sh = NX_(nx_pow2)(e.k);
  NX_(nx_pair) n = {-e.v.hi * sh, -e.v.lo * sh};
  NX_(nx_pair) d = NX_(nx_two_sum)((NX_T)2, -n.hi);
  d.lo -= n.lo;
  NX_(nx_pair) q = NX_(nx_div_pair)(n, d);
  NX_T y = q.hi + q.lo;
  NX_T a = NX_(nx_abs_bits)(x);
  return NX_(nx_nan1)(x, NX_(nx_copysign)(a < NX_C(EXPM1_TINY) ? a : y, x));
}

/* pow

   2^(y log2|x|), log2|x| as a pair good to 2^-(p+9), p the type's
   precision, so that y log2|x| errs by about 2^-(p+2) of the result; the
   exponential is exp2's. f = m - 1 and s = f / (2 + f) are carried as
   pairs, and log(1 + f) = 2s + (2/3) s^3 + s^5 T'(s^2). */
NX_INLINE NX_T NX_(nx_pow)(NX_T x, NX_T y) {
  NX_T ax = NX_(nx_abs_bits)(x), ay = NX_(nx_abs_bits)(y);
  NX_(nx_log_red) l = NX_(nx_log_reduce)(ax);
  NX_T f = l.f;
  NX_T d = (NX_T)2 + f;
  NX_T dl = f - (d - (NX_T)2);
  NX_T rd = NX_ONE / d;
  NX_T sh = f * rd;
  NX_T sl = NX_FMA(-sh, dl, NX_FMA(-sh, d, f)) * rd;
  /* s^3 as c + cl, sl's share 3 s^2 sl included */
  NX_T z = sh * sh;
  NX_T zl = NX_FMA(sh, sh, -z);
  NX_T c = sh * z;
  NX_T cl = NX_FMA((NX_T)3 * z, sl, NX_FMA(sh, zl, NX_FMA(sh, z, -c)));
  /* (2/3) s^3 as a + al */
  NX_T a = c * NX_C(TWO_THIRDS_HI);
  NX_T al = NX_FMA(c, NX_C(TWO_THIRDS_HI), -a) +
            NX_FMA(c, NX_C(TWO_THIRDS_LO), cl * NX_C(TWO_THIRDS_HI));
  NX_T tp = NX_(nx_log_tp)(z);
  NX_T lh = sh + sh;
  NX_T head = lh + a;
  NX_T ll = ((lh - head) + a) + (sl + sl + NX_FMA(c * z, tp, al));
  lh = head;
  /* log2|x| = k + (lh + ll) / ln2 */
  NX_T ph = lh * NX_C(LOG2E);
  NX_T pl = NX_FMA(lh, NX_C(LOG2E), -ph);
  pl = NX_FMA(lh, NX_C(LOG2E_LO), NX_FMA(ll, NX_C(LOG2E), pl));
  NX_(nx_pair) g = NX_(nx_two_sum)(l.k, ph);
  NX_T gh = g.hi + (g.lo + pl);
  NX_T gl = (g.lo + pl) - (gh - g.hi);
  gh = ax == NX_C(INF) ? ax : (ax == 0 ? -NX_C(INF) : gh);
  gl = ax == NX_C(INF) || ax == 0 ? 0 : gl;
  /* y log2|x|; past the range, the result is 0 or inf and the tail may be
     NaN */
  NX_T yh = y * gh;
  NX_T yl = NX_FMA(y, gl, NX_FMA(y, gh, -yh));
  yl = yh <= NX_C(POW_HI) && yh >= NX_C(POW_LO) ? yl : 0;
  yh = yh > NX_C(POW_HI) ? NX_C(POW_HI) : yh;
  yh = yh < NX_C(POW_LO) ? NX_C(POW_LO) : yh;
  NX_(nx_int) k = NX_(nx_nearest)(yh + NX_C(SHIFT));
  NX_T v = NX_(nx_scale)(NX_(nx_exp2_poly)((yh - k.f) + yl), k.i);
  /* y's parity: ay + 2^(p-1) rounds ay to an integer, its last bit worth
     1; from 2^(p-1) on y is an integer, its last bit worth 1 below 2^p and
     more above, so even */
  NX_T ty = ay + NX_C(INTEGRAL);
  int integer = ay >= NX_C(INTEGRAL) || ty - NX_C(INTEGRAL) == ay;
  NX_U last = ay >= NX_C(INTEGRAL) ? NX_BITS(ay) : NX_BITS(ty);
  int odd = ay < (NX_T)2 * NX_C(INTEGRAL) && integer && (last & 1);
  int neg = (int)(NX_BITS(x) >> (8 * sizeof(NX_U) - 1));
  v = neg && odd ? -v : v;
  v = neg && !integer && ax != 0 && ax < NX_C(INF) ? NX_C(NAN) : v;
  v = ax == NX_ONE && ay == NX_C(INF) ? NX_ONE : v;
  v = NX_(nx_nan2)(x, y, v);
  return x == NX_ONE || y == 0 ? NX_ONE : v;
}

#undef NX_ONE
#undef NX_SIGN_BIT
