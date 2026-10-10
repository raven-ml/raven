/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Kinds: what each scalar kind of a program computes.

   One function per kind and compute type: nx_<kind>_<type>, the type f32,
   f64, i32, u32, i64, u64, c64 or c128 (complex, at the end). Every kernel
   library computes a kind through these functions, or through code that
   gives the same bits, so that a library's interpreter and its loops agree
   bit for bit and two libraries agree wherever the kind is exact. The
   float kinds are written once, in nx_kinds_real.h, which this header
   includes for f32 and f64; the pieces whose algorithm differs between the
   two are written here.

   Compute types. float16, bfloat16 and the float8 dtypes compute in f32
   through nx_dtype.h's codecs and round once on the store. 8- and 16-bit
   integers compute in the 32-bit type of their signedness, their operands
   sign- or zero-extended, and wrap on the store; bool computes in u32.

   Exact kinds give the IEEE or modular result. Transcendental kinds hold
   these bounds, in ulps: the distance in ordered-bit ranks from the
   correctly rounded result, -0 and +0 one rank apart.

     kind                                        f32   f64
     exp, exp2, sin, cos, asin, acos, atan,
       cosh, tanh, erf                            2     2
     expm1, log, log2, log1p                      1     2
     atan2, sinh                                  3     2
     tan, pow                                     4     2

   Their special values (C99 Annex F) and the signs of their zero results
   are exact. exp2 is exact at integers whose power is in the type, log2 at
   powers of two.

   NaN. A float kind with a NaN operand gives its first NaN operand, its
   bits unchanged; neg and abs set its sign bit as they do a number's. A
   NaN a kind makes from numbers, as sin of an infinity, is the quiet NaN of
   a clear sign bit. The hardware's choice would differ: of two NaN
   operands x86 returns one by their order, which a compiler sets per
   target and differently in vector and scalar code, and its own NaN has
   the sign bit set where Arm's has it clear. So a kind's NaN bits are the
   same everywhere, as its numbers are.

   Each transcendental is one polynomial in one evaluation order, with fma
   where it is written and nowhere else, so its bits, NaNs aside, are the
   same on every target that rounds as this header assumes:

   - C: -ffp-contract=off, no -ffast-math, subnormals kept.
   - CUDA: --fmad=false -prec-div=true -prec-sqrt=true -ftz=false.
   - Metal: -fno-fast-math -ffp-contract=off. Metal has no f64, and this
     header defines no f64 function there. Apple GPUs flush f32
     subnormals: where an operand or a result is subnormal, or an fma's
     exact product is below 2^-126, Metal's bits differ.

   Within an expression, every product that feeds a sum is an explicit fma,
   so contraction has nothing to fuse there; the flags keep a product held
   in a variable from fusing with a later sum. Division and the square root
   are the target's, correctly rounded under these flags: Metal's are
   checked by the Metal library's probe.

   The polynomials' coefficients are minimax fits, written by
   dev/nx2/test/kernel/gen/kinds.py with the fit's error.

   A vector library runs a kind's function on its lanes. sin, cos and tan
   branch to an integer reduction past NX_PIO2_BIG_*: a vector loop computes
   nx_sin_near_* (nx_cos_near_*, nx_tan_near_*) on every lane and redoes the
   few lanes past the switch with nx_sin_* (nx_cos_*, nx_tan_*). */

#ifndef NX_KINDS_H
#define NX_KINDS_H

#include "nx_dtype.h"

/* Portability */

/* A table the kinds read, in the memory each compiler's code reads: nvcc
   compiles a file once for the host and once for the device. */
#if defined(__METAL_VERSION__)
#define NX_TABLE constant
#elif defined(__CUDA_ARCH__)
#define NX_TABLE static __device__ const
#else
#define NX_TABLE static const
#endif

#ifdef __METAL_VERSION__
NX_INLINE float nx_fmaf(float a, float b, float c) { return metal::fma(a, b, c); }
NX_INLINE float nx_sqrtf(float x) { return metal::sqrt(x); }
#else
NX_INLINE float nx_fmaf(float a, float b, float c) { return fmaf(a, b, c); }
NX_INLINE float nx_sqrtf(float x) { return sqrtf(x); }
NX_INLINE double nx_fmad(double a, double b, double c) { return fma(a, b, c); }
NX_INLINE double nx_sqrtd(double x) { return sqrt(x); }
#endif

/* Integers

   Signed arithmetic runs in the unsigned type of its width, where C defines
   the wrap, and converts back. Idiv truncates toward zero and gives 0 for
   a zero divisor; Mod takes the dividend's sign and gives the dividend for
   a zero divisor, so a = b idiv(a, b) + mod(a, b) for every b; the least
   value divided by -1 wraps to itself. Pow by a negative exponent is 0 but
   for a base of 1 or -1. Floor, Ceil, Round and Trunc are the identity. */

#define NX_INT_KINDS(T, U, S)                                                \
  NX_INLINE T nx_add_##S(T a, T b) { return (T)((U)a + (U)b); }             \
  NX_INLINE T nx_sub_##S(T a, T b) { return (T)((U)a - (U)b); }             \
  NX_INLINE T nx_mul_##S(T a, T b) { return (T)((U)a * (U)b); }             \
  NX_INLINE T nx_fma_##S(T a, T b, T c) {                                    \
    return (T)((U)a * (U)b + (U)c);                                          \
  }                                                                          \
  NX_INLINE T nx_neg_##S(T a) { return (T)((U)0 - (U)a); }                   \
  NX_INLINE U nx_upow_##S(U b, U n) {                                        \
    U r = 1;                                                                 \
    for (; n != 0; n >>= 1) {                                                \
      if (n & 1) r *= b;                                                     \
      b *= b;                                                                \
    }                                                                        \
    return r;                                                                \
  }                                                                          \
  NX_INLINE T nx_maximum_##S(T a, T b) { return a < b ? b : a; }             \
  NX_INLINE T nx_minimum_##S(T a, T b) { return b < a ? b : a; }             \
  NX_INLINE T nx_and_##S(T a, T b) { return a & b; }                         \
  NX_INLINE T nx_or_##S(T a, T b) { return a | b; }                          \
  NX_INLINE T nx_xor_##S(T a, T b) { return a ^ b; }                         \
  NX_INLINE int nx_equal_##S(T a, T b) { return a == b; }                    \
  NX_INLINE int nx_not_equal_##S(T a, T b) { return a != b; }                \
  NX_INLINE int nx_less_##S(T a, T b) { return a < b; }                      \
  NX_INLINE int nx_less_equal_##S(T a, T b) { return a <= b; }               \
  NX_INLINE T nx_where_##S(int c, T a, T b) { return c ? a : b; }

#define NX_SIGNED_KINDS(T, U, S)                                             \
  NX_INT_KINDS(T, U, S)                                                      \
  NX_INLINE T nx_abs_##S(T a) { return a < 0 ? nx_neg_##S(a) : a; }          \
  NX_INLINE T nx_sign_##S(T a) { return (T)((a > 0) - (a < 0)); }            \
  NX_INLINE T nx_recip_##S(T a) { return a == 1 || a == -1 ? a : 0; }        \
  NX_INLINE T nx_idiv_##S(T a, T b) {                                        \
    return b == 0 ? 0 : (b == -1 ? nx_neg_##S(a) : a / b);                   \
  }                                                                          \
  NX_INLINE T nx_mod_##S(T a, T b) {                                         \
    return b == 0 ? a : (b == -1 ? 0 : a % b);                               \
  }                                                                          \
  NX_INLINE T nx_pow_##S(T a, T e) {                                         \
    if (e >= 0) return (T)nx_upow_##S((U)a, (U)e);                           \
    return a == 1 ? 1 : (a == -1 ? ((e & 1) ? -1 : 1) : 0);                  \
  }

#define NX_UNSIGNED_KINDS(T, S)                                              \
  NX_INT_KINDS(T, T, S)                                                      \
  NX_INLINE T nx_abs_##S(T a) { return a; }                                  \
  NX_INLINE T nx_sign_##S(T a) { return a != 0; }                            \
  NX_INLINE T nx_recip_##S(T a) { return a == 1; }                           \
  NX_INLINE T nx_idiv_##S(T a, T b) { return b == 0 ? 0 : a / b; }           \
  NX_INLINE T nx_mod_##S(T a, T b) { return b == 0 ? a : a % b; }            \
  NX_INLINE T nx_pow_##S(T a, T e) { return nx_upow_##S(a, e); }

NX_SIGNED_KINDS(int32_t, uint32_t, i32)
NX_UNSIGNED_KINDS(uint32_t, u32)
NX_SIGNED_KINDS(int64_t, uint64_t, i64)
NX_UNSIGNED_KINDS(uint64_t, u64)

/* Threefry-2x32 with 20 rounds (Salmon et al., Random123): the counter and
   the key are word pairs, the low word first, and so is the result. The
   rounds are written out, four between key injections, so that a loop of
   them is straight code a compiler vectorises. */
NX_INLINE uint32_t nx_rotl32(uint32_t x, int r) {
  return (x << r) | (x >> (32 - r));
}

#define NX_THREEFRY_ROUND(r)                                                 \
  x0 += x1;                                                                  \
  x1 = nx_rotl32(x1, r) ^ x0;
#define NX_THREEFRY_FOUR(r0, r1, r2, r3, ka, kb, s)                          \
  NX_THREEFRY_ROUND(r0)                                                      \
  NX_THREEFRY_ROUND(r1)                                                      \
  NX_THREEFRY_ROUND(r2)                                                      \
  NX_THREEFRY_ROUND(r3)                                                      \
  x0 += ka;                                                                  \
  x1 += kb + s##u;

NX_INLINE uint64_t nx_threefry_u64(uint64_t counter, uint64_t key) {
  uint32_t k0 = (uint32_t)key, k1 = (uint32_t)(key >> 32);
  uint32_t k2 = 0x1BD11BDAu ^ k0 ^ k1;
  uint32_t x0 = (uint32_t)counter + k0, x1 = (uint32_t)(counter >> 32) + k1;
  NX_THREEFRY_FOUR(13, 15, 26, 6, k1, k2, 1)
  NX_THREEFRY_FOUR(17, 29, 16, 24, k2, k0, 2)
  NX_THREEFRY_FOUR(13, 15, 26, 6, k0, k1, 3)
  NX_THREEFRY_FOUR(17, 29, 16, 24, k1, k2, 4)
  NX_THREEFRY_FOUR(13, 15, 26, 6, k2, k0, 5)
  return ((uint64_t)x1 << 32) | x0;
}

#undef NX_THREEFRY_ROUND
#undef NX_THREEFRY_FOUR

/* The bits of 2/pi after the point, behind one zero word, for the
   Payne-Hanek reductions. */
NX_TABLE uint32_t nx_two_over_pi[] = {
    0x00000000u, 0xA2F9836Eu, 0x4E441529u, 0xFC2757D1u, 0xF534DDC0u,
    0xDB629599u, 0x3C439041u, 0xFE5163ABu, 0xDEBBC561u, 0xB7246E3Au,
    0x424DD2E0u, 0x06492EEAu, 0x09D1921Cu, 0xFE1DEB1Cu, 0xB129A73Eu,
    0xE88235F5u, 0x2EBB4484u, 0xE99C7026u, 0xB45F7E41u, 0x3991D639u,
    0x835339F4u, 0x9C845F8Bu, 0xBDF9283Bu, 0x1FF897FFu, 0xDE05980Fu,
    0xEF2F118Bu, 0x5A0A6D1Fu, 0x6D367ECFu, 0x27CB09B7u, 0x4F463F66u,
    0x9E5FEA2Du, 0x7527BAC7u, 0xEBE5F17Bu, 0x3D0739F7u, 0x8A5292EAu,
    0x6BFB5FB1u, 0x1F8D5D08u, 0x56033046u, 0xFC7B6BABu, 0xF0CFBC20u,
    0x9AF4361Du};

/* The 32 bits of 2/pi from bit [g] on, bit 0 the first of the zero word. */
NX_INLINE uint32_t nx_two_over_pi_at(int32_t g) {
  int32_t w = g >> 5, sh = g & 31;
  uint64_t pair = ((uint64_t)nx_two_over_pi[w] << 32) | nx_two_over_pi[w + 1];
  return (uint32_t)((pair << sh) >> 32);
}

/* f32 */

#define NX_BIAS_F32 127
#define NX_MBITS_F32 23
#define NX_MMASK_F32 0x007FFFFFu
#define NX_ONE_BITS_F32 0x3F800000u
#define NX_SQRT_HALF_BITS_F32 0x3F3504F3u
#define NX_MIN_NORMAL_BITS_F32 0x00800000u
#define NX_SUB_SCALE_F32 0x1p23f
#define NX_INF_F32 nx_bits_float(0x7F800000u)
#define NX_NAN_F32 nx_bits_float(0x7FC00000u)
/* Adding 1.5 * 2^23 rounds a float below 2^22 to the nearest integer, ties
   to even; the sum's bits less the shifter's count it. From 2^23 on every
   float is an integer. */
#define NX_SHIFT_F32 0x1.8p23f
#define NX_INTEGRAL_F32 0x1p23f
/* 1/ln2, ln2 with a 16-bit head (k ln2_hi exact for |k| < 2^8), 1/ln2 with
   a 12-bit head, 2/3, 2/pi, pi/2 in three parts, pi. */
#define NX_LOG2E_F32 0x1.715476p+0f
#define NX_LOG2E_LO_F32 0x1.4ae0cp-26f
#define NX_LN2_HI_F32 0x1.62e4p-1f
#define NX_LN2_LO_F32 0x1.7f7d1cp-20f
#define NX_IVLN2_HI_F32 0x1.716p+0f
#define NX_IVLN2_LO_F32 -0x1.7135a8p-13f
#define NX_LOG2_HI_MASK_F32 0xFFFFF000u
#define NX_TWO_THIRDS_HI_F32 0x1.555556p-1f
#define NX_TWO_THIRDS_LO_F32 -0x1.555556p-26f
#define NX_TWO_OVER_PI_F32 0x1.45f306p-1f
#define NX_PIO2_HI_F32 0x1.921fb6p+0f
#define NX_PIO2_LO_F32 -0x1.777a5cp-25f
#define NX_PIO2_3_F32 -0x1.ee59dap-50f
#define NX_PI_HI_F32 0x1.921fb6p+1f
#define NX_PI_LO_F32 -0x1.777a5cp-24f
/* Clamps past which a result is saturated, and cutoffs */
#define NX_EXP_HI_F32 89.0f
#define NX_EXP_LO_F32 -104.0f
#define NX_EXP2_HI_F32 129.0f
#define NX_EXP2_LO_F32 -151.0f
#define NX_EXPM1_LO_F32 -26.0f
#define NX_EXPM1_KCAP_F32 64
#define NX_EXPM1_TINY_F32 0x1p-25f
#define NX_HYP_HI_F32 90.0f
#define NX_HYP_KCAP_F32 32
#define NX_POW_HI_F32 130.0f
#define NX_POW_LO_F32 -160.0f
#define NX_PIO2_BIG_F32 0x1p+15f
#define NX_TRIG_TINY_F32 0x1p-12f

typedef struct {
  float hi, lo;
} nx_pair_f32;

/* x = q pi/2 + hi + lo. */
typedef struct {
  float hi, lo;
  uint32_t q;
} nx_rem_f32;

/* The polynomials. Each is a minimax fit on the interval its kind reduces
   to, with its error relative to the function unless absolute is said. */

/* expm1 r = r + r^2 Q(r) on |r| <= ln2/2 + 2^-13, 2^-28.2 (expQ). */
NX_INLINE float nx_exp_q_f32(float r) {
  float q = 0x1.6ac6e2p-10f;
  q = nx_fmaf(q, r, 0x1.123ddcp-7f);
  q = nx_fmaf(q, r, 0x1.555858p-5f);
  q = nx_fmaf(q, r, 0x1.55548cp-3f);
  return nx_fmaf(q, r, 0x1.fffffcp-2f);
}

/* 2^r = 1 + r P(r) on |r| <= 1/2, 2^-26.1 (exp2P). */
NX_INLINE float nx_exp2_poly_f32(float r) {
  float p = 0x1.177586p-13f;
  p = nx_fmaf(p, r, 0x1.5ef2ep-10f);
  p = nx_fmaf(p, r, 0x1.3b6816p-7f);
  p = nx_fmaf(p, r, 0x1.c6af9ep-5f);
  p = nx_fmaf(p, r, 0x1.ebfb94p-3f);
  p = nx_fmaf(p, r, 0x1.62e43p-1f);
  return nx_fmaf(p, r, 1.0f);
}

/* log((1 + s)/(1 - s)) = 2s + s^3 T(s^2), s^2 <= 0.0295, 2^-25.6 (logT). */
NX_INLINE float nx_log_t_f32(float z) {
  float t = 0x1.ab4e0cp-3f;
  t = nx_fmaf(t, z, 0x1.257d1cp-2f);
  t = nx_fmaf(t, z, 0x1.9996acp-2f);
  return nx_fmaf(t, z, 0x1.555556p-1f);
}

/* T(z) = 2/3 + z T'(z), 2^-22.6 (logT'). */
NX_INLINE float nx_log_tp_f32(float z) {
  float tp = nx_fmaf(z, 0x1.d8279ap-3f, 0x1.2479cap-2f);
  return nx_fmaf(z, tp, 0x1.9999a4p-2f);
}

/* sin r = r + r^3 S(r^2), 2^-27.9 (sinS), and cos r = 1 - r^2/2 + r^4 C(r^2),
   2^-33 (cosC), on |r| <= pi/4: C where cosine is 1, S where it is 0, a
   lane's coefficients selected. */
NX_INLINE float nx_sincos_p_f32(float z, int cosine) {
  float p = nx_fmaf(z, cosine ? 0x1.99e80cp-16f : -0x1.9952fap-13f,
                    cosine ? -0x1.6c0c28p-10f : 0x1.110776p-7f);
  return nx_fmaf(z, p, cosine ? 0x1.55554ap-5f : -0x1.555546p-3f);
}

/* asin s = s + s^3 B(s^2) on |s| <= 1/2, 2^-27.6 (asinB). */
NX_INLINE float nx_asin_b_f32(float z) {
  float b = 0x1.595c8cp-5f;
  b = nx_fmaf(b, z, 0x1.8c3e2ap-6f);
  b = nx_fmaf(b, z, 0x1.747bbap-5f);
  b = nx_fmaf(b, z, 0x1.330204p-4f);
  return nx_fmaf(b, z, 0x1.5555c8p-3f);
}

NX_INLINE nx_rem_f32 nx_rem_pio2_cw_f32(float x);
NX_INLINE nx_rem_f32 nx_rem_pio2_big_f32(float x);

#define NX_T float
#define NX_U uint32_t
#define NX_(f) f##_f32
#define NX_C(c) NX_##c##_F32
#define NX_FMA nx_fmaf
#define NX_SQRT nx_sqrtf
#define NX_BITS nx_float_bits
#define NX_OF_BITS nx_bits_float
#define NX_ISNAN nx_float_nan
#include "nx_kinds_real.h"
#undef NX_T
#undef NX_U
#undef NX_
#undef NX_C
#undef NX_FMA
#undef NX_SQRT
#undef NX_BITS
#undef NX_OF_BITS
#undef NX_ISNAN

/* Cody and Waite: pi/2 in three parts of 24 bits, each product subtracted
   by an fma, for sin, cos and tan below the switch. The second subtraction rounds and
   its error is dropped: keeping it, as f64 does, costs 44% of sin's time
   and makes no kind meet a bound it misses. Within the bounds for
   |x| < NX_PIO2_BIG_F32. */
NX_INLINE nx_rem_f32 nx_rem_pio2_cw_f32(float x) {
  nx_rem_f32 r;
  float t = nx_fmaf(x, NX_TWO_OVER_PI_F32, NX_SHIFT_F32);
  float q = t - NX_SHIFT_F32;
  r.q = nx_float_bits(t) - nx_float_bits(NX_SHIFT_F32);
  float a = nx_fmaf(-q, NX_PIO2_HI_F32, x);
  float b = nx_fmaf(-q, NX_PIO2_LO_F32, a);
  r.hi = nx_fmaf(-q, NX_PIO2_3_F32, b);
  r.lo = nx_fmaf(-q, NX_PIO2_3_F32, b - r.hi);
  return r;
}

/* Payne and Hanek, for finite |x| >= NX_PIO2_BIG_F32. x = m 2^e with m an
   integer of 24 bits; the bits of 2/pi worth 4 or more against m 2^e
   leave x 2/pi unchanged mod 4, so a window W of 96 bits from there gives
   x 2/pi = m W 2^-94 mod 4. Bits 94 and 95 of m W are q, the 64 below the
   fraction, rounded to nearest so that it is signed. */
NX_INLINE nx_rem_f32 nx_rem_pio2_big_f32(float x) {
  uint32_t ix = nx_float_bits(x) & 0x7FFFFFFFu;
  int32_t e = (int32_t)(ix >> 23) - 150;
  uint64_t m = (ix & 0x007FFFFFu) | 0x00800000u;
  int32_t g = e + 30;
  uint64_t a0 = m * nx_two_over_pi_at(g + 64);
  uint64_t a1 = m * nx_two_over_pi_at(g + 32);
  uint64_t a2 = m * nx_two_over_pi_at(g);
  uint64_t t1 = (a0 >> 32) + a1;
  uint64_t t2 = (t1 >> 32) + a2;
  uint64_t f = (t2 << 34) | ((t1 & 0xFFFFFFFFu) << 2) | ((a0 & 0xFFFFFFFFu) >> 30);
  uint32_t q = (uint32_t)(t2 >> 30) + (uint32_t)(f >> 63);
  /* The signed fraction in three exact pieces, its value f 2^-64. */
  float c2 = (float)(uint32_t)(f >> 40) * 0x1p-24f;
  float c1 = (float)(uint32_t)((f >> 16) & 0xFFFFFFu) * 0x1p-48f;
  float c0 = (float)(uint32_t)(f & 0xFFFFu) * 0x1p-64f;
  /* f >> 40 has 24 bits, the top one the sign: taken away as 1. */
  c2 = (f >> 63) ? c2 - 0x1p0f : c2;
  nx_pair_f32 v = nx_two_sum_f32(c2, c1);
  float vlo = v.lo + c0;
  /* times pi/2 */
  float ph = v.hi * NX_PIO2_HI_F32;
  float pl = nx_fmaf(v.hi, NX_PIO2_HI_F32, -ph);
  pl = nx_fmaf(v.hi, NX_PIO2_LO_F32, nx_fmaf(vlo, NX_PIO2_HI_F32, pl));
  nx_rem_f32 r;
  r.hi = ph + pl;
  r.lo = pl - (r.hi - ph);
  uint32_t neg = nx_float_bits(x) >> 31;
  r.q = neg ? (uint32_t)0 - q : q;
  r.hi = neg ? -r.hi : r.hi;
  r.lo = neg ? -r.lo : r.lo;
  return r;
}

/* f32 mod is fmod, exact, by integer long division: one subtraction per bit
   of the operands' exponent difference. Metal's fmod rounds. */
NX_INLINE float nx_mod_f32(float x, float y) {
  uint32_t ux = nx_float_bits(x), uy = nx_float_bits(y) & 0x7FFFFFFFu;
  uint32_t sx = ux & 0x80000000u;
  ux &= 0x7FFFFFFFu;
  if (uy == 0 || ux >= 0x7F800000u || uy > 0x7F800000u)
    return nx_float_nan(x) ? x : (nx_float_nan(y) ? y : NX_NAN_F32);
  if (ux < uy) return x;
  /* Significands as integers with their leading bit at bit 23, exponents
     lowered for subnormals. */
  int32_t ex = (int32_t)(ux >> 23), ey = (int32_t)(uy >> 23);
  uint32_t mx = ex ? (ux & 0x007FFFFFu) | 0x00800000u : ux;
  uint32_t my = ey ? (uy & 0x007FFFFFu) | 0x00800000u : uy;
  ex = ex ? ex : 1;
  ey = ey ? ey : 1;
  for (; (mx & 0x00800000u) == 0; mx <<= 1) ex--;
  for (; (my & 0x00800000u) == 0; my <<= 1) ey--;
  for (; ex > ey; ex--) {
    mx = mx >= my ? mx - my : mx;
    mx <<= 1;
  }
  mx = mx >= my ? mx - my : mx;
  if (mx == 0) return nx_bits_float(sx);
  for (; (mx & 0x00800000u) == 0; mx <<= 1) ex--;
  uint32_t r = ex > 0 ? ((uint32_t)ex << 23) | (mx & 0x007FFFFFu)
                      : mx >> (1 - ex);
  return nx_bits_float(sx | r);
}

/* atan t = t + t^3 A(t^2) on |t| <= 1, relative error 2^-25.7 (atanA). */
NX_INLINE float nx_atan_kernel_f32(float t) {
  float z = t * t;
  float a = 0x1.7ed21ep-9f;
  a = nx_fmaf(a, z, -0x1.0c2c0cp-6f);
  a = nx_fmaf(a, z, 0x1.61fdd4p-5f);
  a = nx_fmaf(a, z, -0x1.3556b4p-4f);
  a = nx_fmaf(a, z, 0x1.b4e12ap-4f);
  a = nx_fmaf(a, z, -0x1.230adcp-3f);
  a = nx_fmaf(a, z, 0x1.9978f4p-3f);
  a = nx_fmaf(a, z, -0x1.5554dcp-2f);
  return nx_fmaf(t * z, a, t);
}

/* atan: |x| > 1 reduces to pi/2 - atan(1/|x|). */
NX_INLINE float nx_atan_f32(float x) {
  float a = nx_abs_bits_f32(x);
  int big = a > 1.0f;
  float p = nx_atan_kernel_f32(big ? 1.0f / a : a);
  float y = big ? NX_PIO2_HI_F32 + (NX_PIO2_LO_F32 - p) : p;
  return nx_nan1_f32(x, nx_copysign_f32(y, x));
}

/* erf

   Below 0.875, erf a = a (2/sqrt(pi) + z E'(z)), z = a^2, relative error
   2^-31.8 (erfE), 2/sqrt(pi) split in two. Above, erf a = 1 - exp(G(a) -
   a^2), G = log erfc a + a^2 on [0.875, 3.92], absolute error 2^-26.1
   (erfG); erf rounds to 1 past 3.92. */
NX_INLINE float nx_erf_f32(float x) {
  float a = nx_abs_bits_f32(x);
  float z = a * a;
  float e = 0x1.70a944p-14f;
  e = nx_fmaf(e, z, -0x1.aff4e8p-11f);
  e = nx_fmaf(e, z, 0x1.556604p-8f);
  e = nx_fmaf(e, z, -0x1.b81e62p-6f);
  e = nx_fmaf(e, z, 0x1.ce2ec2p-4f);
  e = nx_fmaf(e, z, -0x1.812746p-2f);
  /* below 2^-100 the tail is subnormal, which a flushing device drops */
  float tail = a < 0x1p-100f ? 0.0f : a * nx_fmaf(e, z, -0x1.f7ac92p-25f);
  float small = nx_fmaf(a, 0x1.20dd76p+0f, tail);
  float b = a > 3.92f ? 3.92f : a;
  float g = 0x1.b199b8p-20f;
  g = nx_fmaf(g, b, -0x1.7e7eeep-15f);
  g = nx_fmaf(g, b, 0x1.36bb9ap-11f);
  g = nx_fmaf(g, b, -0x1.3694e4p-8f);
  g = nx_fmaf(g, b, 0x1.afe034p-6f);
  g = nx_fmaf(g, b, -0x1.c26b9ep-4f);
  g = nx_fmaf(g, b, 0x1.78e122p-2f);
  g = nx_fmaf(g, b, -0x1.2151d6p+0f);
  g = nx_fmaf(g, b, 0x1.3a6ab6p-12f);
  float big = 1.0f - nx_exp_of_f32(nx_fmaf(-b, b, g));
  return nx_nan1_f32(x, nx_copysign_f32(a < 0.875f ? small : big, x));
}

/* atan2 */

/* atan2: the angle of (|x|, |y|) from atan of the smaller magnitude over
   the larger; by quadrant, added to or subtracted from 0, pi/2 or pi as a
   pair, then given y's sign. */
NX_INLINE float nx_atan2_f32(float y, float x) {
  float ax = nx_abs_bits_f32(x), ay = nx_abs_bits_f32(y);
  int swap = ay > ax;
  float num = swap ? ax : ay, den = swap ? ay : ax;
  float t = num / den;
  t = den == 0.0f ? 0.0f : t;
  t = ax == NX_INF_F32 && ay == NX_INF_F32 ? 1.0f : t;
  float p = nx_atan_kernel_f32(t);
  int xneg = nx_float_bits(x) >> 31;
  float base_hi = swap ? NX_PIO2_HI_F32 : (xneg ? NX_PI_HI_F32 : 0.0f);
  float base_lo = swap ? NX_PIO2_LO_F32 : (xneg ? NX_PI_LO_F32 : 0.0f);
  p = swap != xneg ? -p : p;
  nx_pair_f32 h = nx_two_sum_f32(base_hi, p);
  float v = h.hi + (h.lo + base_lo);
  v = nx_copysign_f32(v, y);
  return nx_nan2_f32(y, x, v);
}

#ifndef __METAL_VERSION__

/* f64 */

NX_INLINE int nx_double_nan(double d) {
  return (nx_double_bits(d) & 0x7FFFFFFFFFFFFFFFu) > 0x7FF0000000000000u;
}

#define NX_BIAS_F64 1023
#define NX_MBITS_F64 52
#define NX_MMASK_F64 0x000FFFFFFFFFFFFFu
#define NX_ONE_BITS_F64 0x3FF0000000000000u
#define NX_SQRT_HALF_BITS_F64 0x3FE6A09E667F3BCDu
#define NX_MIN_NORMAL_BITS_F64 0x0010000000000000u
#define NX_SUB_SCALE_F64 0x1p52
#define NX_INF_F64 nx_bits_double(0x7FF0000000000000u)
#define NX_NAN_F64 nx_bits_double(0x7FF8000000000000u)
#define NX_SHIFT_F64 0x1.8p52
#define NX_INTEGRAL_F64 0x1p52
/* As f32's; ln2's head has 42 bits (k ln2_hi exact for |k| < 2^11), 1/ln2's
   32. */
#define NX_LOG2E_F64 0x1.71547652b82fep+0
#define NX_LOG2E_LO_F64 0x1.777d0ffda0d24p-56
#define NX_LN2_HI_F64 0x1.62e42fefa38p-1
#define NX_LN2_LO_F64 0x1.ef35793c7673p-45
#define NX_IVLN2_HI_F64 0x1.71547652p+0
#define NX_IVLN2_LO_F64 0x1.705fc2eefa2p-33
#define NX_LOG2_HI_MASK_F64 0xFFFFFFFF00000000u
#define NX_TWO_THIRDS_HI_F64 0x1.5555555555555p-1
#define NX_TWO_THIRDS_LO_F64 0x1.5555555555555p-55
#define NX_TWO_OVER_PI_F64 0x1.45f306dc9c883p-1
#define NX_PIO2_HI_F64 0x1.921fb54442d18p+0
#define NX_PIO2_LO_F64 0x1.1a62633145c07p-54
#define NX_PIO2_3_F64 -0x1.f1976b7ed8fbcp-110
#define NX_PI_HI_F64 0x1.921fb54442d18p+1
#define NX_PI_LO_F64 0x1.1a62633145c07p-53
/* atan 1/2, pi/4 and atan 3/2 */
#define NX_ATAN_1_2_HI_F64 0x1.dac670561bb4fp-2
#define NX_ATAN_1_2_LO_F64 0x1.a2b7f222f65e2p-56
#define NX_PIO4_HI_F64 0x1.921fb54442d18p-1
#define NX_PIO4_LO_F64 0x1.1a62633145c07p-55
#define NX_ATAN_3_2_HI_F64 0x1.f730bd281f69bp-1
#define NX_ATAN_3_2_LO_F64 0x1.007887af0cbbdp-56
#define NX_EXP_HI_F64 710.0
#define NX_EXP_LO_F64 -746.0
#define NX_EXP2_HI_F64 1025.0
#define NX_EXP2_LO_F64 -1076.0
#define NX_EXPM1_LO_F64 -40.0
#define NX_EXPM1_KCAP_F64 128
#define NX_EXPM1_TINY_F64 0x1p-54
#define NX_HYP_HI_F64 712.0
#define NX_HYP_KCAP_F64 64
#define NX_POW_HI_F64 1030.0
#define NX_POW_LO_F64 -1080.0
#define NX_PIO2_BIG_F64 0x1p+28
#define NX_TRIG_TINY_F64 0x1p-27

typedef struct {
  double hi, lo;
} nx_pair_f64;

typedef struct {
  double hi, lo;
  uint32_t q;
} nx_rem_f64;

/* expm1 r = r + r^2 Q(r) on |r| <= ln2/2 + 2^-13, 2^-56.4 (expQ). */
NX_INLINE double nx_exp_q_f64(double r) {
  double q = 0x1.ad7f55323502cp-26;
  q = nx_fmad(q, r, 0x1.28ad86039e73p-22);
  q = nx_fmad(q, r, 0x1.71df2581c2cafp-19);
  q = nx_fmad(q, r, 0x1.a01999f89494ap-16);
  q = nx_fmad(q, r, 0x1.a01a012a34bddp-13);
  q = nx_fmad(q, r, 0x1.6c16c18430f75p-10);
  q = nx_fmad(q, r, 0x1.1111111127c73p-7);
  q = nx_fmad(q, r, 0x1.555555555085fp-5);
  q = nx_fmad(q, r, 0x1.55555555554fap-3);
  return nx_fmad(q, r, 0x1.000000000000ap-1);
}

/* 2^r = 1 + r P(r) on |r| <= 1/2, 2^-55.6 (exp2P). */
NX_INLINE double nx_exp2_poly_f64(double r) {
  double p = 0x1.c2cea03f4f3ap-36;
  p = nx_fmad(p, r, 0x1.ea014ee31fd31p-32);
  p = nx_fmad(p, r, 0x1.e4d06c7074e68p-28);
  p = nx_fmad(p, r, 0x1.b524dd55482cfp-24);
  p = nx_fmad(p, r, 0x1.62c021e26f2c5p-20);
  p = nx_fmad(p, r, 0x1.ffcbfc74d611dp-17);
  p = nx_fmad(p, r, 0x1.430912f883cf5p-13);
  p = nx_fmad(p, r, 0x1.5d87fe78a26c9p-10);
  p = nx_fmad(p, r, 0x1.3b2ab6fba4e22p-7);
  p = nx_fmad(p, r, 0x1.c6b08d704a0cfp-5);
  p = nx_fmad(p, r, 0x1.ebfbdff82c58fp-3);
  p = nx_fmad(p, r, 0x1.62e42fefa39efp-1);
  return nx_fmad(p, r, 1.0);
}

/* T(s^2) on s^2 <= 0.0295, 2^-54.6 (logT). */
NX_INLINE double nx_log_t_f64(double z) {
  double t = 0x1.0c1994438ce53p-3;
  t = nx_fmad(t, z, 0x1.0fbc6f8b23776p-3);
  t = nx_fmad(t, z, 0x1.3b1c4a71fa6cep-3);
  t = nx_fmad(t, z, 0x1.745cf89e46188p-3);
  t = nx_fmad(t, z, 0x1.c71c720256c1fp-3);
  t = nx_fmad(t, z, 0x1.2492492476379p-2);
  t = nx_fmad(t, z, 0x1.9999999999a3cp-2);
  return nx_fmad(t, z, 0x1.5555555555555p-1);
}

/* T'(z), 2^-55.1 (logT'). */
NX_INLINE double nx_log_tp_f64(double z) {
  double tp = 0x1.e0550027cf2a4p-4;
  tp = nx_fmad(z, tp, 0x1.df790206d23b4p-4);
  tp = nx_fmad(z, tp, 0x1.1118da70f5042p-3);
  tp = nx_fmad(z, tp, 0x1.3b1395780c99fp-3);
  tp = nx_fmad(z, tp, 0x1.745d177b75a35p-3);
  tp = nx_fmad(z, tp, 0x1.c71c71c6e999ap-3);
  tp = nx_fmad(z, tp, 0x1.2492492492525p-2);
  return nx_fmad(z, tp, 0x1.999999999999ap-2);
}

/* S(r^2), 2^-56.5 (sinS), and C(r^2), 2^-59.4 (cosC), on |r| <= pi/4. */
NX_INLINE double nx_sincos_p_f64(double z, int cosine) {
  double p = nx_fmad(z, cosine ? -0x1.8fa3958a5c52ap-37 : 0x1.5d8ee79c0623bp-33,
                     cosine ? 0x1.1ee9d5c91b1dfp-29 : -0x1.ae5e57812c7p-26);
  p = nx_fmad(z, p, cosine ? -0x1.27e4f7e80273p-22 : 0x1.71de356408ad5p-19);
  p = nx_fmad(z, p, cosine ? 0x1.a01a019c80966p-16 : -0x1.a01a019bf9b6p-13);
  p = nx_fmad(z, p, cosine ? -0x1.6c16c16c14f6cp-10 : 0x1.111111110f7bp-7);
  return nx_fmad(z, p, cosine ? 0x1.555555555554bp-5 : -0x1.5555555555548p-3);
}

/* B(s^2) on |s| <= 1/2, 2^-55.8 (asinB). */
NX_INLINE double nx_asin_b_f64(double z) {
  double b = 0x1.056cbe7839745p-5;
  b = nx_fmad(b, z, -0x1.09d10e2ce02f9p-6);
  b = nx_fmad(b, z, 0x1.3ff33c58c8071p-6);
  b = nx_fmad(b, z, 0x1.abd19915948edp-8);
  b = nx_fmad(b, z, 0x1.8ec2b52f17c83p-7);
  b = nx_fmad(b, z, 0x1.c6fdcd892b624p-7);
  b = nx_fmad(b, z, 0x1.1c6be0fccac8fp-6);
  b = nx_fmad(b, z, 0x1.6e89f44cb6e5ap-6);
  b = nx_fmad(b, z, 0x1.f1c72c38610a7p-6);
  b = nx_fmad(b, z, 0x1.6db6db427c841p-5);
  b = nx_fmad(b, z, 0x1.333333336e7cbp-4);
  return nx_fmad(b, z, 0x1.555555555539p-3);
}

NX_INLINE nx_rem_f64 nx_rem_pio2_cw_f64(double x);
NX_INLINE nx_rem_f64 nx_rem_pio2_big_f64(double x);

#define NX_T double
#define NX_U uint64_t
#define NX_(f) f##_f64
#define NX_C(c) NX_##c##_F64
#define NX_FMA nx_fmad
#define NX_SQRT nx_sqrtd
#define NX_BITS nx_double_bits
#define NX_OF_BITS nx_bits_double
#define NX_ISNAN nx_double_nan
#include "nx_kinds_real.h"
#undef NX_T
#undef NX_U
#undef NX_
#undef NX_C
#undef NX_FMA
#undef NX_SQRT
#undef NX_BITS
#undef NX_OF_BITS
#undef NX_ISNAN

/* Cody and Waite with pi/2 in three parts: x - q p1 is exact, the product
   q p2 and its subtraction are kept as pairs. Within the bounds for
   |x| < NX_PIO2_BIG_F64. */
NX_INLINE nx_rem_f64 nx_rem_pio2_cw_f64(double x) {
  nx_rem_f64 r;
  double t = nx_fmad(x, NX_TWO_OVER_PI_F64, NX_SHIFT_F64);
  nx_int_f64 q = nx_nearest_f64(t);
  r.q = (uint32_t)q.i;
  double a = nx_fmad(-q.f, NX_PIO2_HI_F64, x);
  double p = q.f * NX_PIO2_LO_F64;
  double pe = nx_fmad(q.f, NX_PIO2_LO_F64, -p);
  nx_pair_f64 b = nx_two_sum_f64(a, -p);
  double lo = nx_fmad(-q.f, NX_PIO2_3_F64, b.lo - pe);
  nx_pair_f64 h = nx_fast_two_sum_f64(b.hi, lo);
  r.hi = h.hi;
  r.lo = h.lo;
  return r;
}

/* Payne and Hanek, for finite |x| >= NX_PIO2_BIG_F64: as f32's, x = m 2^e
   with m of 53 bits and a window W of 192 bits, x 2/pi = m W 2^-190 mod 4;
   the fraction's top 128 bits are kept. */
NX_INLINE nx_rem_f64 nx_rem_pio2_big_f64(double x) {
  uint64_t ix = nx_double_bits(x) & 0x7FFFFFFFFFFFFFFFu;
  int32_t e = (int32_t)(ix >> 52) - 1075;
  uint64_t m = (ix & 0x000FFFFFFFFFFFFFu) | 0x0010000000000000u;
  int32_t g = e + 30;
  uint32_t w[6], p[8];
  for (int j = 0; j < 6; j++) w[j] = nx_two_over_pi_at(g + 32 * (5 - j));
  uint64_t ml = m & 0xFFFFFFFFu, mh = m >> 32, c = 0;
  for (int j = 0; j < 6; j++) {
    uint64_t t = ml * w[j] + c;
    p[j] = (uint32_t)t;
    c = t >> 32;
  }
  p[6] = (uint32_t)c;
  c = 0;
  for (int j = 0; j < 6; j++) {
    uint64_t t = mh * w[j] + p[j + 1] + c;
    p[j + 1] = (uint32_t)t;
    c = t >> 32;
  }
  p[7] = (uint32_t)c;
  uint64_t fh = ((uint64_t)(p[5] & 0x3FFFFFFFu) << 34) | ((uint64_t)p[4] << 2) |
                (p[3] >> 30);
  uint64_t fl = ((uint64_t)(p[3] & 0x3FFFFFFFu) << 34) | ((uint64_t)p[2] << 2) |
                (p[1] >> 30);
  uint32_t q = (p[5] >> 30) + (uint32_t)(fh >> 63);
  /* The signed fraction (fh, fl) 2^-128 in three exact pieces. */
  double c2 = (double)(int32_t)(uint32_t)(fh >> 32) * 0x1p-32;
  double c1 = (double)(uint32_t)fh * 0x1p-64;
  double c0 = (double)(fl >> 11) * 0x1p-117;
  nx_pair_f64 s = nx_two_sum_f64(c2, c1);
  nx_pair_f64 v = nx_fast_two_sum_f64(s.hi, s.lo + c0);
  double ph = v.hi * NX_PIO2_HI_F64;
  double pl = nx_fmad(v.hi, NX_PIO2_HI_F64, -ph);
  pl = nx_fmad(v.hi, NX_PIO2_LO_F64, nx_fmad(v.lo, NX_PIO2_HI_F64, pl));
  nx_pair_f64 h = nx_fast_two_sum_f64(ph, pl);
  nx_rem_f64 r;
  uint32_t neg = (uint32_t)(nx_double_bits(x) >> 63);
  r.q = neg ? (uint32_t)0 - q : q;
  r.hi = neg ? -h.hi : h.hi;
  r.lo = neg ? -h.lo : h.lo;
  return r;
}

NX_INLINE double nx_mod_f64(double a, double b) {
  return nx_nan2_f64(a, b, fmod(a, b));
}

/* atan t = t + t^3 A(t^2) on |t| <= 7/16, relative error 2^-57.4
   (atanA). */
NX_INLINE double nx_atan_kernel_f64(double t) {
  double z = t * t;
  double a = -0x1.0a7ed459d86b8p-6;
  a = nx_fmad(a, z, 0x1.2b17be1f3a52cp-5);
  a = nx_fmad(a, z, -0x1.97a089c9a7af5p-5);
  a = nx_fmad(a, z, 0x1.ddddadf2f06c5p-5);
  a = nx_fmad(a, z, -0x1.10d6017d18c7ap-4);
  a = nx_fmad(a, z, 0x1.3b0f205706eb3p-4);
  a = nx_fmad(a, z, -0x1.745cdba3cb3p-4);
  a = nx_fmad(a, z, 0x1.c71c6fdb0e4afp-4);
  a = nx_fmad(a, z, -0x1.2492491ff2fd8p-3);
  a = nx_fmad(a, z, 0x1.999999998e7dp-3);
  a = nx_fmad(a, z, -0x1.555555555550bp-2);
  return t * z * a;
}

/* atan a for a >= 0 as a pair: a = c + (a - c) / (1 + c a), c the nearest
   of 0, 1/2, 1, 3/2 and inf, so the quotient t is within 7/16;
   atan a = atan c + atan t. The quotient's rounding error and the
   argument's tail al, taken below 1, are kept in tl, which reaches atan
   through 1/(1 + t^2) ~ 1 - t^2 + t^4: a tail needs few bits. The selects
   are on the comparisons, which every compiler vectorises. */
NX_INLINE nx_pair_f64 nx_atan_pos_f64(double a, double al) {
  int b1 = a >= 0.4375, b2 = a >= 0.6875, b3 = a >= 1.1875, b4 = a >= 2.4375;
  double c = b3 ? 1.5 : (b2 ? 1.0 : 0.5);
  double num = b4 ? -1.0 : a - c;
  double den = b4 ? a : nx_fmad(c, a, 1.0);
  double rden = 1.0 / den;
  double t = b1 ? num * rden : a;
  /* d t / d a = (1 - t c) / den */
  double rest = nx_fmad(-t, den, num);
  double tl = b1 ? nx_fmad(al, nx_fmad(-t, c, 1.0), rest) * rden : al;
  tl = b4 ? rest * rden : tl;
  tl = a < NX_INF_F64 ? tl : 0.0;
  double ch = b4 ? NX_PIO2_HI_F64 : b3 ? NX_ATAN_3_2_HI_F64 :
              b2 ? NX_PIO4_HI_F64 : b1 ? NX_ATAN_1_2_HI_F64 : 0.0;
  double cl = b4 ? NX_PIO2_LO_F64 : b3 ? NX_ATAN_3_2_LO_F64 :
              b2 ? NX_PIO4_LO_F64 : b1 ? NX_ATAN_1_2_LO_F64 : 0.0;
  double p = nx_atan_kernel_f64(t);
  double z = t * t;
  double w = nx_fmad(z, z - 1.0, 1.0);
  nx_pair_f64 h = nx_two_sum_f64(ch, t);
  return nx_fast_two_sum_f64(h.hi, h.lo + (cl + nx_fmad(tl, w, p)));
}

NX_INLINE double nx_atan_f64(double x) {
  nx_pair_f64 p = nx_atan_pos_f64(nx_abs_bits_f64(x), 0.0);
  return nx_nan1_f64(x, nx_copysign_f64(p.hi, x));
}

/* erf

   Below 1, erf a = a E(a^2) with E = 2/sqrt(pi) + z E'(z), relative error
   2^-56.4 (erfE). From 1 to 5.95, erf a = 1 - exp(G(a) - a^2), G = log
   erfc a + a^2 a polynomial in u = a - 3, exact there; erf's error is
   erfc(a) times G's, which is 2^-54.4 of erfc(1) so weighted (erfG). erf
   rounds to 1 past 5.95. */
NX_INLINE double nx_erf_g_f64(double u) {
  double g = -0x1.95a43236bf463p-51;
  g = nx_fmad(g, u, 0x1.84f40b633be47p-48);
  g = nx_fmad(g, u, -0x1.b42f5b1a0ecb3p-49);
  g = nx_fmad(g, u, -0x1.032edab313b81p-44);
  g = nx_fmad(g, u, 0x1.6c1fd3ad3cfb9p-43);
  g = nx_fmad(g, u, -0x1.18b6602f9057p-40);
  g = nx_fmad(g, u, 0x1.2fbef43aa23fp-37);
  g = nx_fmad(g, u, -0x1.920d81e8b8d56p-35);
  g = nx_fmad(g, u, 0x1.9de264b728be9p-33);
  g = nx_fmad(g, u, -0x1.4d08d8bfa8366p-31);
  g = nx_fmad(g, u, 0x1.5e38ff053bea3p-31);
  g = nx_fmad(g, u, 0x1.898481c060d96p-27);
  g = nx_fmad(g, u, -0x1.2473a46b25eebp-23);
  g = nx_fmad(g, u, 0x1.1f1048857e7b9p-20);
  g = nx_fmad(g, u, -0x1.dd82a42ede083p-18);
  g = nx_fmad(g, u, 0x1.69656fd3df7fcp-15);
  g = nx_fmad(g, u, -0x1.01a07b2021c2fp-12);
  g = nx_fmad(g, u, 0x1.62c3204aff82fp-10);
  g = nx_fmad(g, u, -0x1.e5bee8dc34918p-8);
  g = nx_fmad(g, u, 0x1.5d06cc9bd2dc8p-5);
  g = nx_fmad(g, u, -0x1.370b35013bd13p-2);
  return nx_fmad(g, u, -0x1.b869b65a8e53cp+0);
}

NX_INLINE double nx_erf_f64(double x) {
  double a = nx_abs_bits_f64(x);
  double z = a * a;
  double e = 0x1.3bd95b2fcc4c8p-42;
  e = nx_fmad(e, z, -0x1.b3fe40ecb1d83p-38);
  e = nx_fmad(e, z, 0x1.9a0cfad36fb97p-34);
  e = nx_fmad(e, z, -0x1.517c99ae9a999p-30);
  e = nx_fmad(e, z, 0x1.fcbb28cc88fd2p-27);
  e = nx_fmad(e, z, -0x1.5f73c3747d06bp-23);
  e = nx_fmad(e, z, 0x1.b9e6c38c36437p-20);
  e = nx_fmad(e, z, -0x1.f4d25bf9c9292p-17);
  e = nx_fmad(e, z, 0x1.f9a326f7afd31p-14);
  e = nx_fmad(e, z, -0x1.c02db400361ccp-11);
  e = nx_fmad(e, z, 0x1.565bcd0e6a301p-8);
  e = nx_fmad(e, z, -0x1.b82ce31288b48p-6);
  e = nx_fmad(e, z, 0x1.ce2f21a042be2p-4);
  e = nx_fmad(e, z, -0x1.812746b0379e7p-2);
  double small = nx_fmad(a, 0x1.20dd750429b6dp+0,
                         a * nx_fmad(e, z, 0x1.1ae3a914fed8p-56));
  double b = a > 5.95 ? 5.95 : a;
  double g = nx_erf_g_f64(b - 3.0);
  /* exp(g - b^2) with b^2 and the difference kept as pairs */
  double b2 = b * b;
  double b2l = nx_fmad(b, b, -b2);
  nx_pair_f64 d = nx_two_sum_f64(g, -b2);
  double ex = nx_exp_of_f64(d.hi);
  double big = 1.0 - nx_fmad(ex, d.lo - b2l, ex);
  return nx_nan1_f64(x, nx_copysign_f64(a < 1.0 ? small : big, x));
}

/* atan2: as f32's, with the quotient's rounding error passed to atan as a
   tail. */
NX_INLINE double nx_atan2_f64(double y, double x) {
  double ax = nx_abs_bits_f64(x), ay = nx_abs_bits_f64(y);
  int swap = ay > ax;
  double num = swap ? ax : ay, den = swap ? ay : ax;
  /* 1/den overflows below 2^-1024: scaled, num/den is unchanged */
  double scale = den < 0x1p-960 ? 0x1p64 : 1.0;
  num *= scale;
  den *= scale;
  double rden = 1.0 / den;
  double t = num * rden;
  double tl = nx_fmad(-t, den, num) * rden;
  int degenerate = den == 0.0 || (ax == NX_INF_F64 && ay == NX_INF_F64) ||
                   den == NX_INF_F64;
  t = den == 0.0 ? 0.0 : t;
  t = ax == NX_INF_F64 && ay == NX_INF_F64 ? 1.0 : t;
  tl = degenerate ? 0.0 : tl;
  nx_pair_f64 p = nx_atan_pos_f64(t, tl);
  int xneg = (int)(nx_double_bits(x) >> 63);
  double base_hi = swap ? NX_PIO2_HI_F64 : (xneg ? NX_PI_HI_F64 : 0.0);
  double base_lo = swap ? NX_PIO2_LO_F64 : (xneg ? NX_PI_LO_F64 : 0.0);
  int minus = swap != xneg;
  double ph = minus ? -p.hi : p.hi, pl = minus ? -p.lo : p.lo;
  nx_pair_f64 h = nx_two_sum_f64(base_hi, ph);
  double v = h.hi + (h.lo + (base_lo + pl));
  v = nx_copysign_f64(v, y);
  return nx_nan2_f64(y, x, v);
}

NX_INLINE uint32_t nx_double_sign(double d) {
  return (uint32_t)(nx_double_bits(d) >> 63);
}

#endif /* __METAL_VERSION__ */

/* Complex

   complex64 and complex128 compute as pairs of f32 and f64, nx_c64 and
   nx_c128, the real part first as they are stored. u below is the part's
   unit roundoff, 2^-24 or 2^-53, and the error of a result z' against the
   exact z is normwise, |z' - z| / |z|; each bound holds where no part of
   the computation overflows or underflows.

   Add, Sub and Neg are the parts', each part following its kind's NaN
   rule, so a NaN part leaves the other part's result as it is. Mul is the
   textbook product, (ar br - ai bi) + (ar bi + ai br) i, each part one fma
   over the other product: within 2u (Jeannerod, Kornerup, Louvet and
   Muller, 2017).

   Fdiv is Smith's algorithm (1962): the divisor's larger part divides its
   smaller, so no intermediate overflows or underflows where the quotient's
   parts do not. The platform's own division (C's operator, __divdc3)
   scales instead and gives a NaN part where the quotient's is 0: (0 +
   2^-50 i) / (0 - 2^-1074 i) is -inf + NaN i in glibc's libgcc, -inf + 0 i
   here. Each product that feeds a sum is an fma. Within 4u. Recip is
   (1 + 0 i) / x so. A NaN part in an operand of Mul, Fdiv or Recip makes
   both parts of the result NaN, and every NaN part of their results is
   the quiet NaN of a clear sign bit: which NaN an operation propagates
   varies between its scalar and vector instructions, and a negated
   operand flips its sign.

   Order is the real part's, then the imaginary part's. Equal holds where
   both parts are equal, Not_equal where Equal does not, Less and
   Less_equal by the order. A number with a NaN part is a NaN: Equal, Less
   and Less_equal with it are false, and Maximum and Minimum give the first
   such operand, as the real kinds do. Maximum and Minimum order -0 below
   +0 in each part. */

#define NX_COMPLEX_KINDS(C, T, R, FMA, ISNAN, SIGN, NANC)                    \
  typedef struct {                                                           \
    T re, im;                                                                \
  } nx_##C;                                                                  \
                                                                             \
  NX_INLINE nx_##C nx_##C##_of(T re, T im) {                                 \
    nx_##C z;                                                                \
    z.re = re;                                                               \
    z.im = im;                                                               \
    return z;                                                                \
  }                                                                          \
                                                                             \
  NX_INLINE int nx_nan_##C(nx_##C a) { return ISNAN(a.re) || ISNAN(a.im); }  \
                                                                             \
  NX_INLINE nx_##C nx_neg_##C(nx_##C a) {                                    \
    return nx_##C##_of(nx_neg_##R(a.re), nx_neg_##R(a.im));                  \
  }                                                                          \
  NX_INLINE nx_##C nx_add_##C(nx_##C a, nx_##C b) {                          \
    return nx_##C##_of(nx_add_##R(a.re, b.re), nx_add_##R(a.im, b.im));      \
  }                                                                          \
  NX_INLINE nx_##C nx_sub_##C(nx_##C a, nx_##C b) {                          \
    return nx_##C##_of(nx_sub_##R(a.re, b.re), nx_sub_##R(a.im, b.im));      \
  }                                                                          \
  /* z with each NaN part the quiet NaN of a clear sign bit. */              \
  NX_INLINE nx_##C nx_quiet_##C(T re, T im) {                                \
    return nx_##C##_of(ISNAN(re) ? NANC : re, ISNAN(im) ? NANC : im);        \
  }                                                                          \
  NX_INLINE nx_##C nx_mul_##C(nx_##C a, nx_##C b) {                          \
    T ii = a.im * b.im, ir = a.im * b.re;                                    \
    return nx_quiet_##C(FMA(a.re, b.re, -ii), FMA(a.re, b.im, ir));          \
  }                                                                          \
                                                                             \
  /* Smith's algorithm with selects for its branch, which vectorise: p is   \
     the divisor's larger part, q the other, x and y the dividend's parts   \
     in the same roles. Where the real part is the larger, the quotient is  \
     ((x + r y) + (y - r x) i) / d, else ((x + r y) + (r x - y) i) / d. */  \
  NX_INLINE nx_##C nx_fdiv_##C(nx_##C a, nx_##C b) {                         \
    int big = nx_abs_bits_##R(b.re) >= nx_abs_bits_##R(b.im);                \
    T p = big ? b.re : b.im, q = big ? b.im : b.re;                          \
    T x = big ? a.re : a.im, y = big ? a.im : a.re;                          \
    T r = q / p, d = FMA(r, q, p);                                           \
    T im = big ? FMA(-r, x, y) : FMA(r, x, -y);                              \
    return nx_quiet_##C(FMA(r, y, x) / d, im / d);                           \
  }                                                                          \
  NX_INLINE nx_##C nx_recip_##C(nx_##C a) {                                  \
    return nx_fdiv_##C(nx_##C##_of((T)1, (T)0), a);                          \
  }                                                                          \
                                                                             \
  NX_INLINE int nx_equal_##C(nx_##C a, nx_##C b) {                           \
    return a.re == b.re && a.im == b.im;                                     \
  }                                                                          \
  NX_INLINE int nx_not_equal_##C(nx_##C a, nx_##C b) {                       \
    return !nx_equal_##C(a, b);                                              \
  }                                                                          \
  NX_INLINE int nx_less_##C(nx_##C a, nx_##C b) {                            \
    int n = nx_nan_##C(a) || nx_nan_##C(b);                                  \
    return !n && (a.re < b.re || (a.re == b.re && a.im < b.im));             \
  }                                                                          \
  NX_INLINE int nx_less_equal_##C(nx_##C a, nx_##C b) {                      \
    int n = nx_nan_##C(a) || nx_nan_##C(b);                                  \
    return !n && (a.re < b.re || (a.re == b.re && a.im <= b.im));            \
  }                                                                          \
                                                                             \
  /* 1 where x follows y in the order with -0 below +0, -1 where it         \
     precedes, 0 where they are one value; neither is NaN. */               \
  NX_INLINE int nx_order_##R(T x, T y) {                                     \
    int c = (x > y) - (x < y);                                               \
    return c != 0 ? c : (int)SIGN(y) - (int)SIGN(x);                         \
  }                                                                          \
  NX_INLINE int nx_order_##C(nx_##C a, nx_##C b) {                           \
    int c = nx_order_##R(a.re, b.re);                                        \
    return c != 0 ? c : nx_order_##R(a.im, b.im);                            \
  }                                                                          \
  NX_INLINE nx_##C nx_maximum_##C(nx_##C a, nx_##C b) {                      \
    if (nx_nan_##C(a)) return a;                                             \
    if (nx_nan_##C(b)) return b;                                             \
    return nx_order_##C(a, b) >= 0 ? a : b;                                  \
  }                                                                          \
  NX_INLINE nx_##C nx_minimum_##C(nx_##C a, nx_##C b) {                      \
    if (nx_nan_##C(a)) return a;                                             \
    if (nx_nan_##C(b)) return b;                                             \
    return nx_order_##C(a, b) <= 0 ? a : b;                                  \
  }                                                                          \
  NX_INLINE nx_##C nx_where_##C(int c, nx_##C a, nx_##C b) { return c ? a : b; }

NX_COMPLEX_KINDS(c64, float, f32, nx_fmaf, nx_float_nan, nx_float_sign,
                 NX_NAN_F32)
#ifndef __METAL_VERSION__
NX_COMPLEX_KINDS(c128, double, f64, nx_fmad, nx_double_nan, nx_double_sign,
                 NX_NAN_F64)
#endif

#endif /* NX_KINDS_H */
