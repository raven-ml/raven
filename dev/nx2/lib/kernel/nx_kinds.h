/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Kinds: what each scalar kind of a program computes.

   One function per kind and compute type: nx_<kind>_<type>, the type f32,
   f64, i32, u32, i64, u64, c64 or c128 (complex, at the end). A kernel
   library may compute a kind through these functions, or through code that
   gives the same bits; one that does so in its interpreter and its loops
   has them agree bit for bit. The float kinds are written once, in
   nx_kinds_real.h, which this header includes for f32 and f64; the pieces
   whose algorithm differs between the two are written here. The complex
   transcendental kinds are written once too, in nx_kinds_complex.h,
   included for c64 and c128.

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
   are exact.

   These functions also give exp2 exactly at integers whose power is in the
   type, and log2 at powers of two; a library computing them otherwise
   need not.

   NaN. A float kind with a NaN operand gives a NaN, but for pow(1, y) and
   pow(x, 0), which Annex F makes 1, and so does a kind that IEEE 754 or
   Annex F makes one from numbers, as sin of an infinity. Which NaN, its
   sign and its payload are unspecified: of two NaN operands x86 returns
   one by their order, which a compiler sets per target and differently in
   vector and scalar code, and its own NaN has the sign bit set where Arm's
   has it clear. neg and abs set a NaN's sign bit as they do a number's,
   its payload kept.

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
   are the target's, correctly rounded under these flags; on Metal, a
   library compiling this header probes that they are.

   The polynomials' coefficients are minimax fits, written by
   dev/nx2/test/kernel/gen/kinds.py with the fit's error.

   A vector library runs a kind's function on its lanes. sin, cos, tan and
   exp also have a near function, nx_sin_near_* and so on, which gives the
   kind's bits wherever a test is false: nx_trig_big_*, x past the integer
   reduction's switch, for the first three, nx_exp_big_* for exp. A vector
   loop computes the near function on runs with no lane past the test, and
   the kind on the others: the trigonometric kinds branch past the switch,
   and exp clamps and scales by two products. erf at f64 branches between
   two ranges, as its own comment says. */

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
NX_INLINE float nx_sqrtf(float x) { return sqrtf(x); }
NX_INLINE double nx_sqrtd(double x) { return sqrt(x); }
#endif

/* fma rounding once. A C library's fma may not: mingw-w64's fma and fmaf
   round twice at subnormal results and can lose a zero's sign, and OCaml's
   configure then leaves HAS_WORKING_FMA undefined. A unit built with the
   fma instruction (x86-64's -mfma) computes it in hardware. Elsewhere the
   double fma is the OCaml runtime's emulation, and the float fma rounds the
   exact product plus c to odd in double, which the rounding to float then
   rounds as once, since 53 >= 2 * 24 + 2: the product of two floats is
   exact in double, and TwoSum gives the sum's error. */
#if defined(__CUDACC__) || defined(__HIP__)
NX_INLINE float nx_fmaf(float a, float b, float c) { return fmaf(a, b, c); }
NX_INLINE double nx_fmad(double a, double b, double c) { return fma(a, b, c); }
#elif !defined(__METAL_VERSION__)
#include <caml/s.h>
#if defined(HAS_WORKING_FMA) || defined(__FMA__)
NX_INLINE float nx_fmaf(float a, float b, float c) { return fmaf(a, b, c); }
NX_INLINE double nx_fmad(double a, double b, double c) { return fma(a, b, c); }
#else
extern double caml_fma(double x, double y, double z);

NX_INLINE double nx_fmad(double a, double b, double c) {
  return caml_fma(a, b, c);
}

NX_INLINE float nx_fmaf(float a, float b, float c) {
  double p = (double)a * (double)b, z = (double)c;
  double s = p + z, d = s - p;
  double e = (p - (s - d)) + (z - d);
  if (e != 0 && isfinite(s)) {
    uint64_t u;
    memcpy(&u, &s, sizeof u);
    if ((u & 1) == 0) u += (e > 0) == (s > 0) ? 1 : (uint64_t)-1;
    memcpy(&s, &u, sizeof s);
  }
  return (float)s;
}
#endif
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
   them is straight code a compiler vectorises. NX_THREEFRY_SCHEDULE gives
   [FOUR] each four rounds' rotations, then the injection's two key words,
   as indices into the key's three, and its count: nx.cpu's rows run the
   same schedule on vectors of words (rows.c), whose rotation NX_ROTL32
   spells with shifts too. */
#define NX_THREEFRY_SCHEDULE(FOUR)                                           \
  FOUR(13, 15, 26, 6, 1, 2, 1u)                                              \
  FOUR(17, 29, 16, 24, 2, 0, 2u)                                             \
  FOUR(13, 15, 26, 6, 0, 1, 3u)                                              \
  FOUR(17, 29, 16, 24, 1, 2, 4u)                                             \
  FOUR(13, 15, 26, 6, 2, 0, 5u)

#define NX_ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))

#define NX_THREEFRY_ROUND(r)                                                 \
  x0 += x1;                                                                  \
  x1 = NX_ROTL32(x1, r) ^ x0;
#define NX_THREEFRY_FOUR(r0, r1, r2, r3, a, b, s)                            \
  NX_THREEFRY_ROUND(r0)                                                      \
  NX_THREEFRY_ROUND(r1)                                                      \
  NX_THREEFRY_ROUND(r2)                                                      \
  NX_THREEFRY_ROUND(r3)                                                      \
  x0 += k[a];                                                                \
  x1 += k[b] + s;

NX_INLINE uint64_t nx_threefry_u64(uint64_t counter, uint64_t key) {
  uint32_t k[3] = {(uint32_t)key, (uint32_t)(key >> 32), 0};
  k[2] = 0x1BD11BDAu ^ k[0] ^ k[1];
  uint32_t x0 = (uint32_t)counter + k[0], x1 = (uint32_t)(counter >> 32) + k[1];
  NX_THREEFRY_SCHEDULE(NX_THREEFRY_FOUR)
  return ((uint64_t)x1 << 32) | x0;
}

#undef NX_THREEFRY_ROUND
#undef NX_THREEFRY_FOUR

/* A table's address, in the memory NX_TABLE puts it. */
#if defined(__METAL_VERSION__)
#define NX_TABLE_PTR constant uint32_t *
#else
#define NX_TABLE_PTR const uint32_t *
#endif

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

/* The bits of 2 ln2 / pi after the point, behind one zero word, for the
   complex exp2's phase y ln2 reduced as the trigonometric kinds reduce
   theirs. */
NX_TABLE uint32_t nx_two_ln2_over_pi[] = {
    0x00000000u, 0x70F7263Du, 0xFA5A6EECu, 0xB608582Eu, 0x55FEF3DDu,
    0x8E13BECDu, 0x1D495C36u, 0x1D72013Au, 0xEA9F8FCAu, 0x76E866FCu,
    0x9B48BE48u, 0x15F5CB6Fu, 0xD8BEE211u, 0x0394DB30u, 0x20D6EB52u,
    0x30FFA9AEu, 0x90CBC735u, 0x51358CEDu, 0x67C0DB3Cu, 0x490D15BDu,
    0xD91A29F5u, 0x682402AFu, 0x5E8F9C23u, 0xBFF15CBCu, 0x71F05FEDu,
    0xFE0BB062u, 0x2A8C8F67u, 0x7309C45Au, 0x7502282Au, 0xEAD0C243u,
    0x5C5527BBu, 0x759F345Eu, 0x2D0628E8u, 0xB1A563A2u, 0x53E62918u,
    0xEA5FFDABu, 0xEAC0D479u, 0xCFEDB1B6u, 0xFC78B439u, 0xA333A25Fu,
    0xDC77BE31u};

/* The 32 bits of table t's number from bit [g] on, bit 0 the first of the
   zero word. */
NX_INLINE uint32_t nx_bits_at(NX_TABLE_PTR t, int32_t g) {
  int32_t w = g >> 5, sh = g & 31;
  uint64_t pair = ((uint64_t)t[w] << 32) | t[w + 1];
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
/* exp's near range: there |k| <= 124 (f64: 1021), so 2^k times 1 + em,
   within [1/2, 2], is normal */
#define NX_EXP_NEAR_F32 86.0f
#define NX_EXP2_HI_F32 129.0f
#define NX_EXP2_LO_F32 -151.0f
#define NX_EXPM1_LO_F32 -26.0f
#define NX_EXPM1_KCAP_F32 64
#define NX_EXPM1_TINY_F32 0x1p-25f
#define NX_HYP_HI_F32 90.0f
#define NX_HYP_KCAP_F32 32
/* tanh's split, where e^2a passes 3.5 and its error reaches the quotient
   shrunk by 0.62 or more, and where it rounds to 1: (p + 2) ln2 / 2
   rounded up, p the precision. */
#define NX_TANH_SMALL_F32 0.625f
#define NX_TANH_HI_F32 9.5f
#define NX_POW_HI_F32 130.0f
#define NX_POW_LO_F32 -160.0f
#define NX_PIO2_BIG_F32 0x1p+15f
#define NX_TRIG_TINY_F32 0x1p-12f
/* The complex kinds': ln2 rounded and its remainder; clamps past which
   e^x c and 2^x c overflow for every finite non-zero c; where sinh and
   cosh are e^|x|/2, asin and acos their terms at infinity, and atanh is
   past its overflow guard, sqrt of the largest float over 4. */
#define NX_LN2_F32 0x1.62e430p-1f
#define NX_LN2_LOW_F32 -0x1.05c610p-29f
#define NX_CEXP_HI_F32 194.0f
#define NX_CEXP2_HI_F32 280.0f
#define NX_HYP_BIG_F32 12.0f
#define NX_ASIN_BIG_F32 0x1p13f
#define NX_ATANH_BIG_F32 0x1p62f

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

/* tanh a = a + a^3 T(a^2) on a <= 0.625, 2^-27.7 (tanhT). */
NX_INLINE float nx_tanh_t_f32(float z) {
  float t = -0x1.750ce4p-8f;
  t = nx_fmaf(t, z, 0x1.51f8cep-6f);
  t = nx_fmaf(t, z, -0x1.b835a4p-5f);
  t = nx_fmaf(t, z, 0x1.1106f6p-3f);
  return nx_fmaf(t, z, -0x1.555532p-2f);
}

NX_INLINE nx_rem_f32 nx_rem_pio2_cw_f32(float x);
NX_INLINE nx_rem_f32 nx_rem_pio2_big_f32(NX_TABLE_PTR t, float x);

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

/* Payne and Hanek: x k reduced by pi/2, for finite |x| >= NX_PIO2_BIG_F32
   and t the bits of k 2/pi, below 1: nx_two_over_pi for k = 1. x = m 2^e
   with m an integer of 24 bits; the bits of t worth 4 or more against
   m 2^e leave x t unchanged mod 4, so a window W of 96 bits from there
   gives x t = m W 2^-94 mod 4. Bits 94 and 95 of m W are q, the 64 below
   the fraction, rounded to nearest so that it is signed. */
NX_INLINE nx_rem_f32 nx_rem_pio2_big_f32(NX_TABLE_PTR t, float x) {
  uint32_t ix = nx_float_bits(x) & 0x7FFFFFFFu;
  int32_t e = (int32_t)(ix >> 23) - 150;
  uint64_t m = (ix & 0x007FFFFFu) | 0x00800000u;
  int32_t g = e + 30;
  uint64_t a0 = m * nx_bits_at(t, g + 64);
  uint64_t a1 = m * nx_bits_at(t, g + 32);
  uint64_t a2 = m * nx_bits_at(t, g);
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
    return NX_NAN_F32;
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
  y = nx_copysign_f32(y, x);
  return nx_blend_f32(x != x, x, y);
}

/* erf

   Below 1, erf a = a (2/sqrt(pi) + z E'(z)), z = a^2, relative error
   2^-32.7 (erfE), 2/sqrt(pi) split in two. Above, erf a = 1 - Q(a - 5/2),
   Q erfc's fit on [1, 4], absolute error 2^-26.2 (erfQ); past 4, a is
   taken as 4, where 1 - Q rounds to 1. */
NX_INLINE float nx_erf_f32(float x) {
  float a = nx_abs_bits_f32(x);
  float z = a * a;
  float e = -0x1.252b1cp-17f;
  e = nx_fmaf(e, z, 0x1.cb3dccp-14f);
  e = nx_fmaf(e, z, -0x1.ba9afap-11f);
  e = nx_fmaf(e, z, 0x1.55fc8cp-8f);
  e = nx_fmaf(e, z, -0x1.b8260ep-6f);
  e = nx_fmaf(e, z, 0x1.ce2ee6p-4f);
  e = nx_fmaf(e, z, -0x1.812746p-2f);
  /* below 2^-100 the tail is subnormal, which a flushing device drops */
  float tail = a < 0x1p-100f ? 0.0f : a * nx_fmaf(e, z, -0x1.f7ac92p-25f);
  float small = nx_fmaf(a, 0x1.20dd76p+0f, tail);
  float u = (4.0f < a ? 4.0f : a) - 2.5f;
  float q = -0x1.e11c0ep-21f;
  q = nx_fmaf(q, u, -0x1.11733cp-21f);
  q = nx_fmaf(q, u, 0x1.088b0ep-16f);
  q = nx_fmaf(q, u, -0x1.28a63ap-16f);
  q = nx_fmaf(q, u, -0x1.54d0a6p-14f);
  q = nx_fmaf(q, u, 0x1.1b8868p-12f);
  q = nx_fmaf(q, u, -0x1.0f0962p-12f);
  q = nx_fmaf(q, u, -0x1.25a286p-11f);
  q = nx_fmaf(q, u, 0x1.6ea774p-9f);
  q = nx_fmaf(q, u, -0x1.906118p-8f);
  q = nx_fmaf(q, u, 0x1.1a8c54p-7f);
  q = nx_fmaf(q, u, -0x1.11ab44p-7f);
  q = nx_fmaf(q, u, 0x1.64e34ap-8f);
  q = nx_fmaf(q, u, -0x1.1d7fe4p-9f);
  q = nx_fmaf(q, u, 0x1.aab886p-12f);
  /* a NaN only where x is */
  float r = nx_copysign_f32(a < 1.0f ? small : 1.0f - q, x);
  return x != x ? x : r;
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
#define NX_EXP_NEAR_F64 708.0
#define NX_EXP2_HI_F64 1025.0
#define NX_EXP2_LO_F64 -1076.0
#define NX_EXPM1_LO_F64 -40.0
#define NX_EXPM1_KCAP_F64 128
#define NX_EXPM1_TINY_F64 0x1p-54
#define NX_HYP_HI_F64 712.0
#define NX_HYP_KCAP_F64 64
#define NX_TANH_SMALL_F64 0.625
#define NX_TANH_HI_F64 19.5
#define NX_POW_HI_F64 1030.0
#define NX_POW_LO_F64 -1080.0
#define NX_PIO2_BIG_F64 0x1p+28
#define NX_TRIG_TINY_F64 0x1p-27
#define NX_LN2_F64 0x1.62e42fefa39efp-1
#define NX_LN2_LOW_F64 0x1.abc9e3b39803fp-56
#define NX_CEXP_HI_F64 1456.0
#define NX_CEXP2_HI_F64 2101.0
#define NX_HYP_BIG_F64 22.0
#define NX_ASIN_BIG_F64 0x1p28
#define NX_ATANH_BIG_F64 0x1p510

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

/* T(a^2) on a <= 0.625, 2^-56.0 (tanhT). */
NX_INLINE double nx_tanh_t_f64(double z) {
  double t = -0x1.0da727a627179p-16;
  t = nx_fmad(t, z, 0x1.4391214300e67p-14);
  t = nx_fmad(t, z, -0x1.df5579a8b4a07p-13);
  t = nx_fmad(t, z, 0x1.336612e405191p-11);
  t = nx_fmad(t, z, -0x1.7d68a32fdd9b9p-10);
  t = nx_fmad(t, z, 0x1.d6cf280ecda0fp-9);
  t = nx_fmad(t, z, -0x1.226e162bf7b5p-7);
  t = nx_fmad(t, z, 0x1.664f477adac7ap-6);
  t = nx_fmad(t, z, -0x1.ba1ba1b5064afp-5);
  t = nx_fmad(t, z, 0x1.111111110accdp-3);
  return nx_fmad(t, z, -0x1.55555555554ffp-2);
}

NX_INLINE nx_rem_f64 nx_rem_pio2_cw_f64(double x);
NX_INLINE nx_rem_f64 nx_rem_pio2_big_f64(NX_TABLE_PTR t, double x);

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
   with m of 53 bits and a window W of 192 bits, x t = m W 2^-190 mod 4;
   the fraction's top 128 bits are kept. */
NX_INLINE nx_rem_f64 nx_rem_pio2_big_f64(NX_TABLE_PTR t, double x) {
  uint64_t ix = nx_double_bits(x) & 0x7FFFFFFFFFFFFFFFu;
  int32_t e = (int32_t)(ix >> 52) - 1075;
  uint64_t m = (ix & 0x000FFFFFFFFFFFFFu) | 0x0010000000000000u;
  int32_t g = e + 30;
  uint32_t w[6], p[8];
  for (int j = 0; j < 6; j++) w[j] = nx_bits_at(t, g + 32 * (5 - j));
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
  return fmod(a, b);
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
   2^-56.4 (erfE). To 2.5, erf a = 1 - Q(a - 7/4), Q erfc's fit on [1, 2.5],
   absolute error 2^-58.4 (erfQ). Past it, erf a = 1 - exp(G(a) - a^2),
   G = log erfc a + a^2 on [2.5, 5.95] in a - 17/4, erf's error 2^-62
   (erfG); erf rounds to 1 past 5.95. nx_erf_near_f64 and nx_erf_far_f64
   are the ranges either side of 2.5, which nx_erf_big_f64 tells apart: a
   vector loop computes one on every lane and redoes the others' lanes. */

/* Whether x is in the far range: past 2.5, infinities included. */
NX_INLINE int nx_erf_big_f64(double x) { return nx_abs_bits_f64(x) > 2.5; }

NX_INLINE double nx_erf_near_f64(double x) {
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
  double u = a - 1.75;
  double q = 0x1.1952d5a4cae0ep-27;
  q = nx_fmad(q, u, -0x1.28e112f89c851p-26);
  q = nx_fmad(q, u, -0x1.4fdfcb701a293p-24);
  q = nx_fmad(q, u, 0x1.5c6e3c15a59e3p-22);
  q = nx_fmad(q, u, 0x1.7545709555e68p-23);
  q = nx_fmad(q, u, -0x1.b74c8ca4aa58fp-19);
  q = nx_fmad(q, u, 0x1.367dce5756fd5p-18);
  q = nx_fmad(q, u, 0x1.294cd0fe59fcfp-16);
  q = nx_fmad(q, u, -0x1.20080dba8f773p-14);
  q = nx_fmad(q, u, 0x1.6d715d76f41ap-18);
  q = nx_fmad(q, u, 0x1.d041f3b63da76p-12);
  q = nx_fmad(q, u, -0x1.d4cf0b04a217cp-11);
  q = nx_fmad(q, u, -0x1.7ec1d228fe1f5p-11);
  q = nx_fmad(q, u, 0x1.8d10fb6c23dd8p-8);
  q = nx_fmad(q, u, -0x1.25b378c8cdfffp-7);
  q = nx_fmad(q, u, -0x1.b22257dd0a3ddp-8);
  q = nx_fmad(q, u, 0x1.8a0da543053a6p-5);
  q = nx_fmad(q, u, -0x1.7148c3d57c2d8p-4);
  q = nx_fmad(q, u, 0x1.7a4a8a2bdce2cp-4);
  q = nx_fmad(q, u, -0x1.b055303221016p-5);
  q = nx_fmad(q, u, 0x1.b4be201caa4b3p-7);
  /* a NaN only where x is */
  double r = nx_copysign_f64(a < 1.0 ? small : 1.0 - q, x);
  return x != x ? x : r;
}

NX_INLINE double nx_erf_far_f64(double x) {
  double a = nx_abs_bits_f64(x);
  double b = 5.95 < a ? 5.95 : a;
  double u = b - 4.25;
  double g = 0x1.7742963aff69cp-27;
  g = nx_fmad(g, u, 0x1.54fd3a02fb6fcp-30);
  g = nx_fmad(g, u, 0x1.52f34766557c1p-22);
  g = nx_fmad(g, u, -0x1.8a631a0b82da7p-20);
  g = nx_fmad(g, u, 0x1.71c1a68f670c6p-17);
  g = nx_fmad(g, u, -0x1.2f3790e7befacp-14);
  g = nx_fmad(g, u, 0x1.f7e642e7cc5b2p-12);
  g = nx_fmad(g, u, -0x1.abc91bf263532p-9);
  g = nx_fmad(g, u, 0x1.87d48af31b6e3p-6);
  g = nx_fmad(g, u, -0x1.ca4aec3d2de83p-3);
  g = nx_fmad(g, u, -0x1.05cb67754a622p+1);
  return nx_copysign_f64(1.0 - nx_exp_near_f64(g - b * b), x);
}

NX_INLINE double nx_erf_f64(double x) {
  return nx_erf_big_f64(x) ? nx_erf_far_f64(x) : nx_erf_near_f64(x);
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

   Add, Sub and Neg are the parts', so a NaN part leaves the other part's
   result as it is. Mul is the
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
   both parts of the result NaN.

   Order is the real part's, then the imaginary part's. Equal holds where
   both parts are equal, Not_equal where Equal does not, Less and
   Less_equal by the order. A number with a NaN part is a NaN: Equal, Less
   and Less_equal with it are false, and Maximum and Minimum give the first
   such operand, as the real kinds do. Maximum and Minimum order -0 below
   +0 in each part.

   The transcendental kinds, exp, exp2, log, log2, sqrt, sin, cos, tan,
   asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh and pow, are the
   principal values of C99 7.3 (ISO/IEC 9899), with the special values of
   its Annex G, G.6; sign is z / |z|. asinh, acosh and atanh have no float
   kind. nx_kinds_complex.h computes them from the part type's kinds,
   branching, so that a vector loop runs them on scalars. Their bounds,
   normwise in u where the result's modulus is a normal float:

     sqrt, sign                                3
     exp, exp2, log, log2                      4
     asin, acos, atan, asinh, acosh, atanh     5
     sin, cos, tan, sinh, cosh, tanh           6
     pow, z^w                                  3 + 3 |w log z|

   The bounds are measured, not proven: each kind's worst error at the
   goldens of test/kernel/gen/kinds.py and at ten times as many points of
   the same regions, branch cuts, poles and overflow edges among them, was
   at least 1.3u inside its bound (log2's 2.66u at c128, tan's 4.53u at
   c64). On the real axis, at a non-zero real part where
   the float kind is defined, the real part is the float kind's result;
   pow's is at a positive base and a real exponent.

   Where Annex G leaves a sign unspecified, each kind keeps f(conj z) =
   conj f(z) and its parity bit for bit, a NaN part's sign bit counting as
   its sign. A NaN's sign is unspecified, so where a result's sign comes
   from a NaN part, as sqrt(-inf + i NaN)'s infinite imaginary part's
   does, that sign is unspecified too, as Annex G allows. At operands of
   positive parts each gives:

     exp(-inf + i inf), exp(-inf + i NaN)      +0 + i0
     exp(inf + i inf), exp(inf + i NaN)        inf + i NaN
     sqrt(-inf + i NaN)                        NaN + i inf
     sinh(+0 + i inf), sinh(+0 + i NaN)        +0 + i NaN
     sinh(inf + i inf), sinh(inf + i NaN)      inf + i NaN
     cosh(+0 + i inf), cosh(+0 + i NaN),
       cosh(NaN + i0)                          NaN + i0
     cosh(inf + i inf)                         inf + i NaN
     tanh(inf + i inf), tanh(inf + i NaN)      1 + i0
     asin(inf + i NaN)                         NaN + i inf
     acos(inf + i NaN)                         NaN - i inf
     atan(inf + i NaN)                         pi/2 + i0
     asinh(NaN + i inf)                        inf + i NaN
     atanh(NaN + i inf)                        +0 + i pi/2 */

#define NX_COMPLEX_KINDS(C, T, R, FMA, ISNAN, SIGN)                          \
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
  NX_INLINE nx_##C nx_mul_##C(nx_##C a, nx_##C b) {                          \
    T ii = a.im * b.im, ir = a.im * b.re;                                    \
    return nx_##C##_of(FMA(a.re, b.re, -ii), FMA(a.re, b.im, ir));           \
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
    return nx_##C##_of(FMA(r, y, x) / d, im / d);                            \
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

NX_COMPLEX_KINDS(c64, float, f32, nx_fmaf, nx_float_nan, nx_float_sign)
#ifndef __METAL_VERSION__
NX_COMPLEX_KINDS(c128, double, f64, nx_fmad, nx_double_nan, nx_double_sign)
#endif

#define NX_T float
#define NX_U uint32_t
#define NX_CT nx_c64
#define NX_(f) f##_f32
#define NX_Z(f) f##_c64
#define NX_MK nx_c64_of
#define NX_C(c) NX_##c##_F32
#define NX_FMA nx_fmaf
#define NX_SQRT nx_sqrtf
#define NX_BITS nx_float_bits
#define NX_OF_BITS nx_bits_float
#include "nx_kinds_complex.h"
#undef NX_T
#undef NX_U
#undef NX_CT
#undef NX_
#undef NX_Z
#undef NX_MK
#undef NX_C
#undef NX_FMA
#undef NX_SQRT
#undef NX_BITS
#undef NX_OF_BITS

#ifndef __METAL_VERSION__
#define NX_T double
#define NX_U uint64_t
#define NX_CT nx_c128
#define NX_(f) f##_f64
#define NX_Z(f) f##_c128
#define NX_MK nx_c128_of
#define NX_C(c) NX_##c##_F64
#define NX_FMA nx_fmad
#define NX_SQRT nx_sqrtd
#define NX_BITS nx_double_bits
#define NX_OF_BITS nx_bits_double
#include "nx_kinds_complex.h"
#undef NX_T
#undef NX_U
#undef NX_CT
#undef NX_
#undef NX_Z
#undef NX_MK
#undef NX_C
#undef NX_FMA
#undef NX_SQRT
#undef NX_BITS
#undef NX_OF_BITS
#endif

#endif /* NX_KINDS_H */
