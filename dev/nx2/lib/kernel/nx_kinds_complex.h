/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The complex kinds of one part type, written once: nx_kinds.h includes
   this file at the end of f32's block, for c64, and, but on Metal, of
   f64's, for c128, once the part type's kinds are defined, with
   nx_kinds_real.h's parameters and

   - NX_CT, the complex type;
   - NX_Z(f), which names f at the complex type: NX_Z(nx_exp) is
     nx_exp_c64;
   - NX_MK, the function that makes a complex number of two parts.

   prog.mli's Results section states what these kinds compute. Each
   transcendental kind makes its operands' NaN parts quiet first and is
   computed from the part type's kinds in one order, so that its bits, NaNs
   aside, are the same on every target. A kind whose formula is odd or even
   in a part computes that part's sign apart, so that its symmetries hold
   bit for bit. */

#define NX_ONE ((NX_T)1)
#define NX_HALF ((NX_T)0.5)
#define NX_SIGN_BIT ((NX_U)1 << (8 * sizeof(NX_U) - 1))

/* Arithmetic and order */

typedef struct {
  NX_T re, im;
} NX_CT;

NX_INLINE NX_CT NX_MK(NX_T re, NX_T im) {
  NX_CT z;
  z.re = re;
  z.im = im;
  return z;
}

NX_INLINE int NX_Z(nx_nan)(NX_CT a) { return NX_ISNAN(a.re) || NX_ISNAN(a.im); }

NX_INLINE NX_CT NX_Z(nx_neg)(NX_CT a) {
  return NX_MK(NX_(nx_neg)(a.re), NX_(nx_neg)(a.im));
}
NX_INLINE NX_CT NX_Z(nx_add)(NX_CT a, NX_CT b) {
  return NX_MK(NX_(nx_add)(a.re, b.re), NX_(nx_add)(a.im, b.im));
}
NX_INLINE NX_CT NX_Z(nx_sub)(NX_CT a, NX_CT b) {
  return NX_MK(NX_(nx_sub)(a.re, b.re), NX_(nx_sub)(a.im, b.im));
}
NX_INLINE NX_CT NX_Z(nx_mul)(NX_CT a, NX_CT b) {
  NX_T ii = a.im * b.im, ir = a.im * b.re;
  return NX_MK(NX_FMA(a.re, b.re, -ii), NX_FMA(a.re, b.im, ir));
}

/* Smith's algorithm with selects for its branch, which vectorise: p is the
   divisor's larger part, q the other, x and y the dividend's parts in the
   same roles. Where the real part is the larger, the quotient is
   ((x + r y) + (y - r x) i) / d, else ((x + r y) + (r x - y) i) / d. A
   dividend whose parts are below the least normal over u is scaled up first,
   and the quotient back: unscaled, a fused multiply-add of its parts lands
   subnormal and drops bits, which a small divisor magnifies into a normal
   quotient. The scaling back is exact where the quotient is normal. */
NX_INLINE NX_CT NX_Z(nx_fdiv)(NX_CT a, NX_CT b) {
  int tiny = NX_(nx_abs_bits)(a.re) < NX_C(TINY_DIVIDEND) &&
             NX_(nx_abs_bits)(a.im) < NX_C(TINY_DIVIDEND);
  NX_T up = tiny ? NX_C(DIVIDEND_UP) : NX_ONE;
  NX_T down = tiny ? NX_C(DIVIDEND_DOWN) : NX_ONE;
  a = NX_MK(a.re * up, a.im * up);
  int big = NX_(nx_abs_bits)(b.re) >= NX_(nx_abs_bits)(b.im);
  NX_T p = big ? b.re : b.im, q = big ? b.im : b.re;
  NX_T x = big ? a.re : a.im, y = big ? a.im : a.re;
  NX_T r = q / p, d = NX_FMA(r, q, p);
  NX_T im = big ? NX_FMA(-r, x, y) : NX_FMA(r, x, -y);
  return NX_MK(NX_FMA(r, y, x) / d * down, im / d * down);
}
NX_INLINE NX_CT NX_Z(nx_recip)(NX_CT a) {
  return NX_Z(nx_fdiv)(NX_MK(NX_ONE, (NX_T)0), a);
}

NX_INLINE int NX_Z(nx_equal)(NX_CT a, NX_CT b) {
  return a.re == b.re && a.im == b.im;
}
NX_INLINE int NX_Z(nx_not_equal)(NX_CT a, NX_CT b) {
  return !NX_Z(nx_equal)(a, b);
}
NX_INLINE int NX_Z(nx_less)(NX_CT a, NX_CT b) {
  int n = NX_Z(nx_nan)(a) || NX_Z(nx_nan)(b);
  return !n && (a.re < b.re || (a.re == b.re && a.im < b.im));
}
NX_INLINE int NX_Z(nx_less_equal)(NX_CT a, NX_CT b) {
  int n = NX_Z(nx_nan)(a) || NX_Z(nx_nan)(b);
  return !n && (a.re < b.re || (a.re == b.re && a.im <= b.im));
}

/* 1 where x follows y in the order with -0 below +0, -1 where it precedes,
   0 where they are one value; neither is NaN. */
NX_INLINE int NX_(nx_order)(NX_T x, NX_T y) {
  int c = (x > y) - (x < y);
  int sx = (NX_BITS(x) & NX_SIGN_BIT) != 0, sy = (NX_BITS(y) & NX_SIGN_BIT) != 0;
  return c != 0 ? c : sy - sx;
}
NX_INLINE int NX_Z(nx_order)(NX_CT a, NX_CT b) {
  int c = NX_(nx_order)(a.re, b.re);
  return c != 0 ? c : NX_(nx_order)(a.im, b.im);
}
NX_INLINE NX_CT NX_Z(nx_maximum)(NX_CT a, NX_CT b) {
  if (NX_Z(nx_nan)(a)) return a;
  if (NX_Z(nx_nan)(b)) return b;
  return NX_Z(nx_order)(a, b) >= 0 ? a : b;
}
NX_INLINE NX_CT NX_Z(nx_minimum)(NX_CT a, NX_CT b) {
  if (NX_Z(nx_nan)(a)) return a;
  if (NX_Z(nx_nan)(b)) return b;
  return NX_Z(nx_order)(a, b) <= 0 ? a : b;
}
NX_INLINE NX_CT NX_Z(nx_where)(int c, NX_CT a, NX_CT b) { return c ? a : b; }

/* Parts */

/* v with its sign flipped where s's sign bit is set. */
NX_INLINE NX_T NX_(nx_flip)(NX_T v, NX_T s) {
  return NX_OF_BITS(NX_BITS(v) ^ (NX_BITS(s) & NX_SIGN_BIT));
}

NX_INLINE int NX_(nx_finite)(NX_T x) { return NX_(nx_abs_bits)(x) < NX_INF; }

/* The exponent e of a finite x > 0, x = m 2^e with m in [1, 2). */
NX_INLINE int32_t NX_(nx_ilogb)(NX_T x) {
  int sub = NX_BITS(x) < NX_MIN_NORMAL_BITS;
  NX_T s = sub ? x * NX_SUB_SCALE : x;
  return (int32_t)(NX_BITS(s) >> NX_C(MBITS)) - NX_C(BIAS) -
         (sub ? NX_C(MBITS) : 0);
}

/* y 2^k rounded once, for k up to twice the span nx_scale takes. */
NX_INLINE NX_T NX_(nx_ldexp)(NX_T y, int32_t k) {
  int32_t k1 = k >> 1;
  return NX_(nx_scale)(NX_(nx_scale)(y, k1), k - k1);
}

/* a b - c d within 1.5 ulps: c d's rounding error, which an fma gives
   exactly, is added back (Kahan's algorithm; Jeannerod, Louvet and Muller,
   2013). */
NX_INLINE NX_T NX_(nx_det)(NX_T a, NX_T b, NX_T c, NX_T d) {
  NX_T w = c * d;
  NX_T e = NX_FMA(-c, d, w);
  return NX_FMA(a, b, -w) + e;
}

/* sqrt(x^2 + y^2) of finite x and y, scaled by a power of two so that the
   larger part is in [1, 2): within 1.25 ulps, no intermediate over- or
   underflowing where the result does not. */
NX_INLINE NX_T NX_(nx_hypot)(NX_T x, NX_T y) {
  NX_T ax = NX_(nx_abs_bits)(x), ay = NX_(nx_abs_bits)(y);
  NX_T a = ax > ay ? ax : ay, b = ax > ay ? ay : ax;
  if (a == 0) return a;
  int32_t k = NX_(nx_ilogb)(a);
  NX_T as = NX_(nx_scale)(a, -k), bs = NX_(nx_scale)(b, -k);
  return NX_(nx_scale)(NX_SQRT(NX_FMA(as, as, bs * bs)), k);
}

/* log |x + iy|: of the larger part a and the smaller b, log a where b is 0;
   half log1p of a^2 + b^2 - 1 where a is near 1; else
   log a + log1p((b/a)^2) / 2. inf where a part is, NaN where a part is
   NaN otherwise. Near 1, a^2 + b^2 - 1 is (a - 1)(a + 1) + b^2 from exact
   products: a - 1 is exact, a + 1 and each product a head and a tail, and
   the heads' sum exact, so that it errs by about u^2 of its terms and the
   result keeps its digits near the unit circle. */
NX_INLINE NX_T NX_(nx_log_abs)(NX_T x, NX_T y) {
  NX_T ax = NX_(nx_abs_bits)(x), ay = NX_(nx_abs_bits)(y);
  if (ax == NX_INF || ay == NX_INF) return NX_INF;
  if (ax != ax || ay != ay) return NX_NAN;
  NX_T a = ax > ay ? ax : ay, b = ax > ay ? ay : ax;
  if (b == 0) return NX_(nx_log)(a);
  if (a >= NX_HALF && a <= (NX_T)2) {
    NX_T am = a - NX_ONE;
    NX_(nx_pair) ap = NX_(nx_two_sum)(a, NX_ONE);
    NX_T h1 = am * ap.hi, h2 = b * b;
    NX_T l1 = NX_FMA(am, ap.hi, -h1), l2 = NX_FMA(b, b, -h2);
    NX_(nx_pair) h = NX_(nx_two_sum)(h1, h2);
    NX_T d = h.hi + (h.lo + NX_FMA(am, ap.lo, l1 + l2));
    return NX_HALF * NX_(nx_log1p)(d);
  }
  NX_T q = b / a;
  return NX_FMA(NX_HALF, NX_(nx_log1p)(q * q), NX_(nx_log)(a));
}

/* asinh t, odd: log1p(a + a^2 / (1 + sqrt(1 + a^2))) of a = |t|, and
   log a + ln2 past ASIN_BIG, where a^2 + 1 rounds to a^2. */
NX_INLINE NX_T NX_(nx_asinh_part)(NX_T t) {
  NX_T a = NX_(nx_abs_bits)(t);
  NX_T a2 = a * a;
  NX_T near = NX_(nx_log1p)(a + a2 / (NX_ONE + NX_SQRT(NX_ONE + a2)));
  NX_T far = NX_(nx_log)(a) + NX_C(LN2_HI) + NX_C(LN2_LO);
  return NX_(nx_copysign)(a < NX_C(ASIN_BIG) ? near : far, t);
}

/* e^x c, for x past EXP_NEAR up to the clamp CEXP_HI, past which the
   product overflows whatever c's finite non-zero value: with
   x/2 = k ln2 + r and m = e^r, (m c) m 2^2k, so that neither e^x nor its
   product with a small c overflows first. shift halves the result shift
   times. */
NX_INLINE NX_T NX_(nx_exp_times)(NX_T x, NX_T c, int32_t shift) {
  NX_T h = (x < NX_C(CEXP_HI) ? x : NX_C(CEXP_HI)) * NX_HALF;
  NX_(nx_exp_red) e = NX_(nx_exp_reduce)(h);
  NX_T m = NX_ONE + NX_FMA(e.r * e.r, NX_(nx_exp_q)(e.r), e.r);
  return NX_(nx_ldexp)(m * c * m, 2 * e.k - shift);
}

/* Exponentials */

/* e^z = e^x cis y. */
NX_INLINE NX_CT NX_Z(nx_exp_raw)(NX_T x, NX_T y) {
  if (y == 0) return NX_MK(NX_(nx_exp)(x), y);
  if (x != x) return NX_MK(NX_NAN, NX_NAN);
  if (!NX_(nx_finite)(y)) {
    if (x == NX_INF) return NX_MK(x, NX_NAN);
    if (x == -NX_INF) return NX_MK(0, NX_(nx_copysign)(0, y));
    return NX_MK(NX_NAN, NX_NAN);
  }
  NX_T c = NX_(nx_cos)(y), s = NX_(nx_sin)(y);
  if (x == NX_INF || x == -NX_INF) {
    NX_T m = x > 0 ? x : 0;
    return NX_MK(m * c, m * s);
  }
  if (x <= NX_C(EXP_NEAR)) {
    NX_T e = NX_(nx_exp)(x);
    return NX_MK(e * c, e * s);
  }
  return NX_MK(NX_(nx_exp_times)(x, c, 0), NX_(nx_exp_times)(x, s, 0));
}

/* 2^z = 2^x cis(y ln2), 2^x as 2^k m, k the integer nearest x. Below the
   trigonometric kinds' switch to Payne and Hanek, the phase y ln2 is p + t,
   ln2 in two parts and t the rounding error of p, and cis p turns by
   cis t. Past it, y ln2 is reduced by pi/2 as sin and cos reduce their
   operand, from the bits of 2 ln2 / pi: exactly enough for every finite
   y, as for theirs. */
NX_INLINE NX_CT NX_Z(nx_exp2_raw)(NX_T x, NX_T y) {
  if (y == 0) return NX_MK(NX_(nx_exp2)(x), y);
  if (x != x || !NX_(nx_finite)(y)) return NX_Z(nx_exp_raw)(x, y);
  NX_T p = y * NX_C(LN2), c, s;
  if (NX_(nx_trig_big)(p)) {
    NX_(nx_rem) r = NX_(nx_rem_pio2_big)(nx_two_ln2_over_pi, y);
    c = NX_(nx_cos_at)(p, r);
    s = NX_(nx_sin_at)(p, r);
  } else {
    NX_T t = NX_FMA(y, NX_C(LN2_LOW), NX_FMA(y, NX_C(LN2), -p));
    NX_T cp = NX_(nx_cos)(p), sp = NX_(nx_sin)(p);
    NX_T ct = NX_(nx_cos)(t), st = NX_(nx_sin)(t);
    c = NX_(nx_det)(cp, ct, sp, st);
    s = NX_(nx_det)(sp, ct, -cp, st);
  }
  if (!NX_(nx_finite)(x)) {
    NX_T m = x > 0 ? x : 0;
    return NX_MK(m * c, m * s);
  }
  NX_T v = x < NX_C(CEXP2_HI) ? x : NX_C(CEXP2_HI);
  v = v > NX_C(EXP2_LO) ? v : NX_C(EXP2_LO);
  NX_(nx_int) k = NX_(nx_nearest)(v + NX_SHIFT);
  NX_T m = NX_(nx_exp2_poly)(v - k.f);
  return NX_MK(NX_(nx_ldexp)(m * c, k.i), NX_(nx_ldexp)(m * s, k.i));
}

/* log z = log |z| + i arg z, arg z = atan2(y, x) with its special values. */
NX_INLINE NX_CT NX_Z(nx_log_raw)(NX_T x, NX_T y) {
  return NX_MK(NX_(nx_log_abs)(x, y), NX_(nx_atan2)(y, x));
}

/* log2 z = log z / ln2, 1/ln2 as a head and a tail; on the axes, the part
   type's log2 of the modulus. */
NX_INLINE NX_CT NX_Z(nx_log2_raw)(NX_T x, NX_T y) {
  NX_CT l = NX_Z(nx_log_raw)(x, y);
  NX_T re = NX_FMA(l.re, NX_C(LOG2E), l.re * NX_C(LOG2E_LO));
  NX_T im = NX_FMA(l.im, NX_C(LOG2E), l.im * NX_C(LOG2E_LO));
  NX_T ax = NX_(nx_abs_bits)(x), ay = NX_(nx_abs_bits)(y);
  re = y == 0 && ax == ax ? NX_(nx_log2)(ax) : re;
  re = x == 0 && NX_(nx_finite)(y) ? NX_(nx_log2)(ay) : re;
  return NX_MK(re, im);
}

/* Square root (Kahan, 1987): t = sqrt((|x| + |z|) / 2) is the part of the
   larger magnitude, y / 2t the other; z is scaled by an even power of two
   so that its larger part is in [1, 4), and y / 2t is taken of y itself. */
NX_INLINE NX_CT NX_Z(nx_sqrt_raw)(NX_T x, NX_T y) {
  NX_T ax = NX_(nx_abs_bits)(x), ay = NX_(nx_abs_bits)(y);
  if (ay == NX_INF) return NX_MK(NX_INF, y);
  if (x != x) return NX_MK(NX_NAN, NX_NAN);
  if (x == NX_INF)
    return NX_MK(x, y != y ? NX_NAN : NX_(nx_copysign)(0, y));
  if (x == -NX_INF)
    return NX_MK(y != y ? NX_NAN : 0, NX_(nx_copysign)(NX_INF, y));
  if (y != y) return NX_MK(NX_NAN, NX_NAN);
  if (y == 0)
    return x >= 0 ? NX_MK(NX_SQRT(ax), y)
                  : NX_MK(0, NX_(nx_copysign)(NX_SQRT(ax), y));
  NX_T a = ax > ay ? ax : ay;
  int32_t j = NX_(nx_ilogb)(a) >> 1;
  NX_T xs = NX_(nx_scale)(ax, -2 * j), ys = NX_(nx_scale)(ay, -2 * j);
  NX_T t = NX_(nx_scale)(NX_SQRT((xs + NX_(nx_hypot)(xs, ys)) * NX_HALF), j);
  NX_T u = ay / (t + t);
  return x >= 0 ? NX_MK(t, NX_(nx_copysign)(u, y))
                : NX_MK(u, NX_(nx_copysign)(t, y));
}

/* z^w = e^(w log z): w real or imaginary multiplies log z part by part, so
   that a zero part of w meets no infinite part of log z. 1 for w = 0, and
   the part type's pow on the positive real axis. */
NX_INLINE NX_CT NX_Z(nx_pow_raw)(NX_CT z, NX_CT w) {
  if (w.re == 0 && w.im == 0) return NX_MK(NX_ONE, 0);
  if (z.im == 0 && w.im == 0 && z.re > 0)
    return NX_MK(NX_(nx_pow)(z.re, w.re), NX_(nx_flip)(z.im, w.re));
  NX_CT l = NX_Z(nx_log_raw)(z.re, z.im);
  NX_T pr, pi;
  if (w.im == 0) {
    pr = w.re * l.re;
    pi = w.re * l.im;
  } else if (w.re == 0) {
    pr = -(w.im * l.im);
    pi = w.im * l.re;
  } else {
    pr = NX_FMA(w.re, l.re, -(w.im * l.im));
    pi = NX_FMA(w.re, l.im, w.im * l.re);
  }
  return NX_Z(nx_exp_raw)(pr, pi);
}

/* sign z = z / |z|: z itself at 0, and at an infinity its direction among
   the eight a pair of parts names. z is scaled by a power of two first, so
   that |z| is no subnormal. */
NX_INLINE NX_CT NX_Z(nx_sign_raw)(NX_T x, NX_T y) {
  NX_T ax = NX_(nx_abs_bits)(x), ay = NX_(nx_abs_bits)(y);
  if (ax != ax || ay != ay) return NX_MK(NX_NAN, NX_NAN);
  if (ax == NX_INF || ay == NX_INF) {
    NX_T both = NX_OF_BITS(NX_C(SQRT_HALF_BITS));
    NX_T re = ax == NX_INF ? (ay == NX_INF ? both : NX_ONE) : 0;
    NX_T im = ay == NX_INF ? (ax == NX_INF ? both : NX_ONE) : 0;
    return NX_MK(NX_(nx_copysign)(re, x), NX_(nx_copysign)(im, y));
  }
  if (ax == 0 && ay == 0) return NX_MK(x, y);
  int32_t k = NX_(nx_ilogb)(ax > ay ? ax : ay);
  NX_T xs = NX_(nx_scale)(x, -k), ys = NX_(nx_scale)(y, -k);
  NX_T h = NX_(nx_hypot)(xs, ys);
  return NX_MK(xs / h, ys / h);
}

/* Hyperbolic functions

   sinh z = sinh x cos y + i cosh x sin y and cosh z = cosh x cos y +
   i sinh x sin y. Past HYP_BIG, sinh |x| and cosh |x| are e^|x| / 2 to the
   type's precision, taken as exp_times does so that a small cos y or sin y
   keeps the product finite. */

NX_INLINE NX_CT NX_Z(nx_sinh_raw)(NX_T x, NX_T y) {
  if (y == 0) return NX_MK(NX_(nx_sinh)(x), y);
  if (x != x) return NX_MK(NX_NAN, NX_NAN);
  if (!NX_(nx_finite)(y)) {
    if (x == 0 || !NX_(nx_finite)(x)) return NX_MK(x, NX_NAN);
    return NX_MK(NX_NAN, NX_NAN);
  }
  NX_T c = NX_(nx_cos)(y), s = NX_(nx_sin)(y);
  NX_T ax = NX_(nx_abs_bits)(x);
  if (ax == NX_INF) return NX_MK(x * c, ax * s);
  if (ax <= NX_C(HYP_BIG))
    return NX_MK(NX_(nx_sinh)(x) * c, NX_(nx_cosh)(x) * s);
  return NX_MK(NX_(nx_flip)(NX_(nx_exp_times)(ax, c, 1), x),
               NX_(nx_exp_times)(ax, s, 1));
}

NX_INLINE NX_CT NX_Z(nx_cosh_raw)(NX_T x, NX_T y) {
  if (y == 0) return NX_MK(NX_(nx_cosh)(x), NX_(nx_flip)(y, x));
  if (x != x) return NX_MK(NX_NAN, NX_NAN);
  if (!NX_(nx_finite)(y)) {
    if (x == 0) return NX_MK(NX_NAN, NX_(nx_flip)(x, y));
    if (!NX_(nx_finite)(x)) return NX_MK(NX_INF, NX_NAN);
    return NX_MK(NX_NAN, NX_NAN);
  }
  NX_T c = NX_(nx_cos)(y), s = NX_(nx_sin)(y);
  NX_T ax = NX_(nx_abs_bits)(x);
  if (ax == NX_INF) return NX_MK(ax * c, x * s);
  if (ax <= NX_C(HYP_BIG))
    return NX_MK(NX_(nx_cosh)(x) * c, NX_(nx_sinh)(x) * s);
  return NX_MK(NX_(nx_exp_times)(ax, c, 1),
               NX_(nx_flip)(NX_(nx_exp_times)(ax, s, 1), x));
}

/* tanh z (Kahan, 1987): with t = tan y, b = 1 + t^2, s = sinh x and
   c = cosh x, (b c s + i t) / (1 + b s^2), which cancels nowhere. Past
   HYP_BIG, 1 rounds the real part, and the imaginary part is
   4 sin y cos y e^-2|x|, which C99 Annex G's ctanh(inf + iy) =
   1 + i0 sin(2y) continues. */
NX_INLINE NX_CT NX_Z(nx_tanh_raw)(NX_T x, NX_T y) {
  if (y == 0) return NX_MK(NX_(nx_tanh)(x), y);
  if (x != x) return NX_MK(NX_NAN, NX_NAN);
  NX_T ax = NX_(nx_abs_bits)(x);
  NX_T one = NX_(nx_copysign)(NX_ONE, x);
  if (ax == NX_INF && !NX_(nx_finite)(y))
    return NX_MK(one, NX_(nx_copysign)(0, y));
  if (!NX_(nx_finite)(y)) return NX_MK(NX_NAN, NX_NAN);
  if (ax > NX_C(HYP_BIG)) {
    NX_T e = NX_(nx_exp)(-ax);
    NX_T sc = NX_(nx_sin)(y) * NX_(nx_cos)(y);
    return NX_MK(one, (NX_T)4 * sc * e * e);
  }
  NX_T t = NX_(nx_tan)(y), b = NX_FMA(t, t, NX_ONE);
  NX_T s = NX_(nx_sinh)(x), c = NX_(nx_cosh)(x);
  NX_T d = NX_FMA(b * s, s, NX_ONE);
  return NX_MK(b * c * s / d, t / d);
}

/* Inverse functions

   asin and acos (Kahan, 1987), from A = sqrt(1 - z) and B = sqrt(1 + z):
   asin z = atan(x / Re(A B)) + i asinh(Im(conj(A) B)) and
   acos z = 2 atan(Re A / Re B) + i asinh(Im(conj(B) A)); each product's
   two terms have one sign, so neither cancels. Past ASIN_BIG, asin z is
   atan2(x, y) + i (log|z| + ln2) in the first quadrant, its leading terms
   at infinity. On the real axis within [-1, 1], the part type's asin and
   acos. */

NX_INLINE NX_CT NX_Z(nx_asin_raw)(NX_T x, NX_T y) {
  NX_T ax = NX_(nx_abs_bits)(x), ay = NX_(nx_abs_bits)(y);
  NX_T re, im;
  if (ax > NX_C(ASIN_BIG) || ay > NX_C(ASIN_BIG)) {
    re = NX_(nx_atan2)(ax, ay);
    im = NX_(nx_log_abs)(ax, ay) + NX_C(LN2_HI) + NX_C(LN2_LO);
  } else if (ax != ax || ay != ay) {
    return NX_MK(x == 0 ? x : NX_NAN, NX_NAN);
  } else if (ay == 0 && ax <= NX_ONE) {
    re = NX_(nx_asin)(ax);
    im = 0;
  } else {
    NX_CT a = NX_Z(nx_sqrt_raw)(NX_ONE - ax, -ay);
    NX_CT b = NX_Z(nx_sqrt_raw)(NX_ONE + ax, ay);
    re = NX_(nx_atan)(ax / NX_(nx_det)(a.re, b.re, a.im, b.im));
    im = NX_(nx_asinh_part)(NX_(nx_det)(a.re, b.im, a.im, b.re));
  }
  return NX_MK(NX_(nx_flip)(re, x), NX_(nx_flip)(im, y));
}

NX_INLINE NX_CT NX_Z(nx_acos_raw)(NX_T x, NX_T y) {
  NX_T ax = NX_(nx_abs_bits)(x), ay = NX_(nx_abs_bits)(y);
  NX_T re, im;
  if (ax > NX_C(ASIN_BIG) || ay > NX_C(ASIN_BIG)) {
    re = NX_(nx_atan2)(ay, x);
    im = -(NX_(nx_log_abs)(ax, ay) + NX_C(LN2_HI) + NX_C(LN2_LO));
  } else if (ax != ax || ay != ay) {
    return NX_MK(x == 0 ? NX_C(PIO2_HI) : NX_NAN, NX_NAN);
  } else if (ay == 0 && ax <= NX_ONE) {
    re = NX_(nx_acos)(x);
    im = -(NX_T)0;
  } else {
    NX_CT a = NX_Z(nx_sqrt_raw)(NX_ONE - x, -ay);
    NX_CT b = NX_Z(nx_sqrt_raw)(NX_ONE + x, ay);
    re = (NX_T)2 * NX_(nx_atan)(a.re / b.re);
    im = NX_(nx_asinh_part)(NX_(nx_det)(b.re, a.im, b.im, a.re));
  }
  return NX_MK(re, NX_(nx_flip)(im, y));
}

/* atanh z in the first quadrant (Kahan, 1987):
   log1p(4x / ((1 - x)^2 + y^2)) / 4 + i atan2(2y, (1 - x)(1 + x) - y^2) / 2,
   with x = 1 apart, where the real part is log((4 + y^2)^(1/4) / sqrt y);
   past ATANH_BIG, Re(1/z) + i pi/2. On the imaginary axis, i atan y.
   Kahan adds a tiny constant to y to spare x = 1 its own case; at y below
   that constant its real part loses its digits, so x = 1 has its case. */
NX_INLINE NX_CT NX_Z(nx_atanh_raw)(NX_T x, NX_T y) {
  NX_T ax = NX_(nx_abs_bits)(x), ay = NX_(nx_abs_bits)(y);
  NX_T re, im;
  if (ax == NX_INF || ay == NX_INF) {
    re = 0;
    im = ay != ay ? NX_NAN : NX_C(PIO2_HI);
  } else if (ax == 0) {
    re = 0;
    im = NX_(nx_atan)(ay);
  } else if (ax != ax || ay != ay) {
    return NX_MK(NX_NAN, NX_NAN);
  } else if (ax == NX_ONE && ay == 0) {
    re = NX_INF;
    im = 0;
  } else if (ax > NX_C(ATANH_BIG) || ay > NX_C(ATANH_BIG)) {
    NX_T m = ax > ay ? ax : ay;
    NX_T xs = ax / m, ys = ay / m;
    re = xs / NX_FMA(xs, xs, ys * ys) / m;
    im = NX_C(PIO2_HI);
  } else if (ax == NX_ONE) {
    re = NX_(nx_log)(NX_SQRT(NX_SQRT(NX_FMA(ay, ay, (NX_T)4))) / NX_SQRT(ay));
    im = NX_HALF * (NX_C(PIO2_HI) + NX_(nx_atan)(NX_HALF * ay) + NX_C(PIO2_LO));
  } else {
    NX_T u = NX_ONE - ax;
    re = (NX_T)0.25 * NX_(nx_log1p)((NX_T)4 * ax / NX_FMA(u, u, ay * ay));
    im = NX_HALF * NX_(nx_atan2)(ay + ay, NX_FMA(-ay, ay, u * (NX_ONE + ax)));
  }
  return NX_MK(NX_(nx_flip)(re, x), NX_(nx_flip)(im, y));
}

/* The kinds

   The trigonometric kinds are the hyperbolic ones turned by i, as C99
   Annex G defines them: sin z = -i sinh(iz), cos z = cosh(iz),
   tan z = -i tanh(iz) and atan z = -i atanh(iz), where iz = -y + ix and
   -i(a + ib) = b - ia; asinh turns asin back, and acosh turns acos. Every
   NaN part of an operand is made quiet first.

   C leaves a signalling NaN operand's behaviour undefined, and compilers
   use the room: clang folds asin's (ax > c || ay > c) into
   fmaxnm(ax, ay) > c, which is false where ax is infinite and ay a
   signalling NaN. An operand's NaN part has its quiet bit set, its sign and
   payload kept; the kinds read no NaN bit but the sign, so a quiet NaN and
   a number keep their bits. */

#define NX_TURN(v) (NX_MK((v).im, NX_(nx_neg)((v).re)))

/* x with its quiet bit set where it is NaN. */
NX_INLINE NX_T NX_(nx_unsignalled)(NX_T x) {
  return x != x ? NX_OF_BITS(NX_BITS(x) | ((NX_U)1 << (NX_C(MBITS) - 1))) : x;
}

NX_INLINE NX_CT NX_Z(nx_unsignalled)(NX_CT z) {
  return NX_MK(NX_(nx_unsignalled)(z.re), NX_(nx_unsignalled)(z.im));
}

/* The kind K of one operand, of K's raw function. */
#define NX_KIND1(K)                                                          \
  NX_INLINE NX_CT NX_Z(nx_##K)(NX_CT z) {                                    \
    NX_CT u = NX_Z(nx_unsignalled)(z);                                       \
    return NX_Z(nx_##K##_raw)(u.re, u.im);                                   \
  }

NX_KIND1(exp)
NX_KIND1(exp2)
NX_KIND1(log)
NX_KIND1(log2)
NX_KIND1(sqrt)
NX_KIND1(sign)
NX_KIND1(sinh)
NX_KIND1(cosh)
NX_KIND1(tanh)
NX_KIND1(asin)
NX_KIND1(acos)

NX_INLINE NX_CT NX_Z(nx_pow)(NX_CT z, NX_CT w) {
  NX_CT u = NX_Z(nx_unsignalled)(z), v = NX_Z(nx_unsignalled)(w);
  return NX_Z(nx_pow_raw)(u, v);
}

NX_INLINE NX_CT NX_Z(nx_sin)(NX_CT z) {
  NX_CT u = NX_Z(nx_unsignalled)(z);
  NX_CT v = NX_Z(nx_sinh_raw)(NX_(nx_neg)(u.im), u.re);
  return NX_TURN(v);
}

NX_INLINE NX_CT NX_Z(nx_cos)(NX_CT z) {
  NX_CT u = NX_Z(nx_unsignalled)(z);
  return NX_Z(nx_cosh_raw)(NX_(nx_neg)(u.im), u.re);
}

NX_INLINE NX_CT NX_Z(nx_tan)(NX_CT z) {
  NX_CT u = NX_Z(nx_unsignalled)(z);
  NX_CT v = NX_Z(nx_tanh_raw)(NX_(nx_neg)(u.im), u.re);
  return NX_TURN(v);
}

NX_INLINE NX_CT NX_Z(nx_atan)(NX_CT z) {
  NX_CT u = NX_Z(nx_unsignalled)(z);
  NX_CT v = NX_Z(nx_atanh_raw)(NX_(nx_neg)(u.im), u.re);
  return NX_TURN(v);
}

/* asinh z = i asin(-iz), -iz = y - ix and i(a + ib) = -b + ia. */
NX_INLINE NX_CT NX_Z(nx_asinh)(NX_CT z) {
  NX_CT u = NX_Z(nx_unsignalled)(z);
  NX_CT v = NX_Z(nx_asin_raw)(u.im, NX_(nx_neg)(u.re));
  return NX_MK(NX_(nx_neg)(v.im), v.re);
}

NX_INLINE NX_CT NX_Z(nx_atanh)(NX_CT z) {
  NX_CT u = NX_Z(nx_unsignalled)(z);
  return NX_Z(nx_atanh_raw)(u.re, u.im);
}

/* acosh z = i acos z above the real axis, where Im(acos z) <= 0 makes the
   real part non-negative, and its conjugate below: i(a + ib) = -b + ia. A
   NaN imaginary part beside a finite real one is no direction: NaN, as
   Annex G's cacosh(x + iNaN). */
NX_INLINE NX_CT NX_Z(nx_acosh)(NX_CT z) {
  NX_CT u = NX_Z(nx_unsignalled)(z);
  if (u.im != u.im && NX_(nx_finite)(u.re))
    return NX_MK(NX_NAN, NX_NAN);
  NX_CT v = NX_Z(nx_acos_raw)(u.re, NX_(nx_abs_bits)(u.im));
  return NX_MK(NX_(nx_neg)(v.im), NX_(nx_flip)(v.re, u.im));
}

#undef NX_KIND1
#undef NX_TURN
#undef NX_ONE
#undef NX_HALF
#undef NX_SIGN_BIT
