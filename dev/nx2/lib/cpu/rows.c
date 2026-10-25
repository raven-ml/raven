/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The rows of the kinds of one, two and three operands, and of fills,
   compiled once per target: as itself for base and as its copy rows_v3.c
   for v3, whose fused multiply-add is an instruction where base's x86-64 is
   a call.

   A row loads each operand's element into its compute type, computes the
   kind as nx_kinds.h does, and stores the result in the destination's
   dtype: 8- and 16-bit integers compute in the 32-bit type of their
   signedness and wrap on the store, booleans compute in uint32 from 0 or 1.
   A row whose destination is contiguous and whose operands are contiguous
   or one repeated element, as a broadcast operand or a program's constant
   is, takes a loop the compiler vectorises, the transcendental kinds'
   polynomials included. Rows exist at the carriers alone (cpu.h): apply.c
   runs a narrow or sub-byte dtype's kinds at its carrier's rows. */

#include "cpu.h"

#if !defined(NX_CPU_V3) || defined(__x86_64__)

#include <string.h>

#include "nx_kinds.h"

/* A kind's row, with every function it calls inlined: gcc otherwise calls
   a long kind, as erf, threefry or pow, out of its loop, which then runs
   one element at a time. */
#define ROW static __attribute__((flatten)) void

/* A row of [F] over one operand of the type [T], loaded by [LD]. */
#define UN(NAME, T, LD, F)                                                   \
  ROW NAME(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *x_,            \
           int64_t sx) {                                                     \
    T *d = (T *)d_;                                                          \
    const T *x = (const T *)x_;                                              \
    if (sd == 1 && sx == 1)                                                  \
      for (int64_t i = 0; i < n; i++) d[i] = F(LD(x[i]));                    \
    else if (sd == 1 && sx == 0) {                                           \
      T v = F(LD(x[0]));                                                     \
      for (int64_t i = 0; i < n; i++) d[i] = v;                              \
    } else                                                                   \
      for (int64_t i = 0; i < n; i++) d[i * sd] = F(LD(x[i * sx]));          \
  }

/* Elements of a piece of a row that splits its kind's range, a multiple
   of eight. */
#define PIECE 256

/* A kind with a near function, as nx_kinds.h says a vector loop computes
   it: a piece with no lane past the test [BIG] runs the near function on
   every lane, a piece with one runs the kind. Both loops vectorise but the
   trigonometric kinds', which branch. Each piece is tested before it is
   written, so a destination that is its operand reads every element
   first. */
#define NEAR(K, S, T, BIG)                                                   \
  ROW K##_##S(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *x_,         \
              int64_t sx) {                                                  \
    T *d = (T *)d_;                                                          \
    const T *x = (const T *)x_;                                              \
    if (sd == 1 && sx == 0) {                                                \
      T v = nx_##K##_##S(x[0]);                                              \
      for (int64_t i = 0; i < n; i++) d[i] = v;                              \
      return;                                                                \
    }                                                                        \
    if (sd != 1 || sx != 1) {                                                \
      for (int64_t i = 0; i < n; i++) d[i * sd] = nx_##K##_##S(x[i * sx]);   \
      return;                                                                \
    }                                                                        \
    for (int64_t b = 0; b < n; b += PIECE) {                                 \
      int64_t e = n - b < PIECE ? n : b + PIECE;                             \
      int big = 0;                                                           \
      for (int64_t i = b; i < e; i++) big |= nx_##BIG##_big_##S(x[i]);       \
      if (big)                                                               \
        for (int64_t i = b; i < e; i++) d[i] = nx_##K##_##S(x[i]);           \
      else                                                                   \
        for (int64_t i = b; i < e; i++) d[i] = nx_##K##_near_##S(x[i]);      \
    }                                                                        \
  }

/* erf at f64 as nx_kinds.h says a vector loop computes it: a piece runs
   the range most of its lanes are in, [NEAR] or [FAR] as [BIG] tells, on
   every lane, and the other range on the others' lanes, gathered into a
   run of their own; both loops vectorise. The gathering reads a flag per
   lane eight at a time and skips eight lanes of the majority at once. The
   others are computed first, from the operand, so that a destination that
   is the operand reads every element before it is written. [F] is the
   scalar kind. */
#define RANGED(NAME, T, BIG, NEAR, FAR, F)                                   \
  ROW NAME(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *x_,            \
           int64_t sx) {                                                     \
    T *d = (T *)d_;                                                          \
    const T *x = (const T *)x_;                                              \
    if (sd == 1 && sx == 0) {                                                \
      T v = F(x[0]);                                                         \
      for (int64_t i = 0; i < n; i++) d[i] = v;                              \
      return;                                                                \
    }                                                                        \
    if (sd != 1 || sx != 1) {                                                \
      for (int64_t i = 0; i < n; i++) d[i * sd] = F(x[i * sx]);              \
      return;                                                                \
    }                                                                        \
    uint8_t big[PIECE];                                                      \
    int idx[PIECE];                                                          \
    T in[PIECE], out[PIECE];                                                 \
    for (int64_t b = 0; b < n; b += PIECE) {                                 \
      int len = n - b < PIECE ? (int)(n - b) : PIECE;                        \
      const T *xb = x + b;                                                   \
      T *db = d + b;                                                         \
      int bigs = 0, m = 0;                                                   \
      for (int i = 0; i < len; i++) {                                        \
        big[i] = (uint8_t)BIG(xb[i]);                                        \
        bigs += big[i];                                                      \
      }                                                                      \
      uint8_t far = 2 * bigs > len;                                          \
      if (bigs != 0 && bigs != len) {                                        \
        uint64_t all = far ? 0x0101010101010101u : 0;                        \
        for (int i = len; i % 8 != 0; i++) big[i] = far;                     \
        for (int j = 0; j < len; j += 8) {                                   \
          uint64_t w;                                                        \
          memcpy(&w, big + j, 8);                                            \
          if (w == all) continue;                                            \
          for (int i = j; i < j + 8 && i < len; i++) {                       \
            idx[m] = i;                                                      \
            in[m] = xb[i];                                                   \
            m += big[i] != far;                                              \
          }                                                                  \
        }                                                                    \
      }                                                                      \
      if (far) {                                                             \
        for (int k = 0; k < m; k++) out[k] = NEAR(in[k]);                    \
        for (int i = 0; i < len; i++) db[i] = FAR(xb[i]);                    \
      } else {                                                               \
        for (int k = 0; k < m; k++) out[k] = FAR(in[k]);                     \
        for (int i = 0; i < len; i++) db[i] = NEAR(xb[i]);                   \
      }                                                                      \
      for (int k = 0; k < m; k++) db[idx[k]] = out[k];                       \
    }                                                                        \
  }

/* The kinds of one operand by nx_prog.h's code and nx_kinds.h's name: of
   floats, of floats with a near function, of integers. Each applies [X] to
   the code, the name and the arguments that follow [X]; NEAR_KINDS passes
   after the name the prefix of the near function's test. */
#define FLOAT_KINDS1(X, ...)                                                 \
  X(NEG, neg, __VA_ARGS__) X(RECIP, recip, __VA_ARGS__)                      \
  X(ABS, abs, __VA_ARGS__) X(SIGN, sign, __VA_ARGS__)                        \
  X(SQRT, sqrt, __VA_ARGS__) X(EXP2, exp2, __VA_ARGS__)                      \
  X(LOG, log, __VA_ARGS__) X(LOG2, log2, __VA_ARGS__)                        \
  X(LOG1P, log1p, __VA_ARGS__)                                               \
  X(EXPM1, expm1, __VA_ARGS__) X(ASIN, asin, __VA_ARGS__)                    \
  X(ACOS, acos, __VA_ARGS__) X(ATAN, atan, __VA_ARGS__)                      \
  X(SINH, sinh, __VA_ARGS__) X(COSH, cosh, __VA_ARGS__)                      \
  X(TANH, tanh, __VA_ARGS__) X(FLOOR, floor, __VA_ARGS__)                    \
  X(CEIL, ceil, __VA_ARGS__) X(ROUND, round, __VA_ARGS__)                    \
  X(TRUNC, trunc, __VA_ARGS__)
#define NEAR_KINDS(X, ...)                                                   \
  X(SIN, sin, trig, __VA_ARGS__) X(COS, cos, trig, __VA_ARGS__)              \
  X(TAN, tan, trig, __VA_ARGS__) X(EXP, exp, exp, __VA_ARGS__)
#define INT_KINDS1(X, ...)                                                   \
  X(NEG, neg, __VA_ARGS__) X(RECIP, recip, __VA_ARGS__)                      \
  X(ABS, abs, __VA_ARGS__) X(SIGN, sign, __VA_ARGS__)

#define FLOAT1(C, K, S, T) UN(K##_##S, T, , nx_##K##_##S)
#define NEAR1(C, K, B, S, T) NEAR(K, S, T, B)
#define INT1(C, K, D, T, CT, S) UN(K##_##D, T, (CT), nx_##K##_##S)

/* Floor, Ceil, Round and Trunc are the identity on integers: a row per
   width. */
#define IDENT(W, T) UN(ident_##W, T, , )

IDENT(1, uint8_t)
IDENT(2, uint16_t)
IDENT(4, uint32_t)
IDENT(8, uint64_t)

/* The loops of a row of [F] over two operands of the storage type [T],
   loaded by [LD] into the compute type: contiguous, one of them a repeated
   element, or any steps. */
#define LOOPS2(T, LD, F)                                                     \
  if (sd == 1 && sx == 1 && sy == 1)                                         \
    for (int64_t i = 0; i < n; i++) d[i] = F(LD(x[i]), LD(y[i]));            \
  else if (sd == 1 && sx == 1 && sy == 0) {                                  \
    T b = y[0];                                                              \
    for (int64_t i = 0; i < n; i++) d[i] = F(LD(x[i]), LD(b));               \
  } else if (sd == 1 && sx == 0 && sy == 1) {                                \
    T a = x[0];                                                              \
    for (int64_t i = 0; i < n; i++) d[i] = F(LD(a), LD(y[i]));               \
  } else                                                                     \
    for (int64_t i = 0; i < n; i++)                                          \
      d[i * sd] = F(LD(x[i * sx]), LD(y[i * sy]))

/* A row of [F] over two operands of the storage type [T], loaded by [LD]
   into the compute type, stored as [R]. */
#define BIN(NAME, T, R, LD, F)                                               \
  ROW NAME(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *x_,            \
           int64_t sx, const uint8_t *y_, int64_t sy) {                      \
    R *d = (R *)d_;                                                          \
    const T *x = (const T *)x_, *y = (const T *)y_;                          \
    LOOPS2(T, LD, F);                                                        \
  }

/* The loops of a row of [F] over three operands of the type [T], loaded by
   [LD]: contiguous, one of them a repeated element, or any steps. */
#define LOOPS3(T, LD, F)                                                     \
  if (sd == 1 && sa == 1 && sb == 1 && sc == 1)                              \
    for (int64_t i = 0; i < n; i++)                                          \
      d[i] = (T)F(LD(a[i]), LD(b[i]), LD(c[i]));                             \
  else if (sd == 1 && sa == 1 && sb == 1 && sc == 0) {                       \
    T v = c[0];                                                              \
    for (int64_t i = 0; i < n; i++) d[i] = (T)F(LD(a[i]), LD(b[i]), LD(v));  \
  } else if (sd == 1 && sa == 1 && sb == 0 && sc == 1) {                     \
    T v = b[0];                                                              \
    for (int64_t i = 0; i < n; i++) d[i] = (T)F(LD(a[i]), LD(v), LD(c[i]));  \
  } else if (sd == 1 && sa == 0 && sb == 1 && sc == 1) {                     \
    T v = a[0];                                                              \
    for (int64_t i = 0; i < n; i++) d[i] = (T)F(LD(v), LD(b[i]), LD(c[i]));  \
  } else                                                                     \
    for (int64_t i = 0; i < n; i++)                                          \
      d[i * sd] = (T)F(LD(a[i * sa]), LD(b[i * sb]), LD(c[i * sc]))

#define FMA(NAME, T, LD, F)                                                  \
  ROW NAME(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *a_,            \
           int64_t sa, const uint8_t *b_, int64_t sb,                        \
           const uint8_t *c_, int64_t sc) {                                  \
    T *d = (T *)d_;                                                          \
    const T *a = (const T *)a_, *b = (const T *)b_, *c = (const T *)c_;      \
    LOOPS3(T, LD, F);                                                        \
  }

/* The kinds every dtype of a class takes, at the compute suffix [S]. */
#define COMPARES(D, T, LD, S)                                                \
  BIN(equal_##D, T, uint8_t, LD, nx_equal_##S)                               \
  BIN(not_equal_##D, T, uint8_t, LD, nx_not_equal_##S)                       \
  BIN(less_##D, T, uint8_t, LD, nx_less_##S)                                 \
  BIN(less_equal_##D, T, uint8_t, LD, nx_less_equal_##S)                     \
  BIN(maximum_##D, T, T, LD, nx_maximum_##S)                                 \
  BIN(minimum_##D, T, T, LD, nx_minimum_##S)

/* Idiv and Mod of integers by a divisor 2^k repeated along the row, as
   Nx.Rng's are by 2, 2^31 and 2^32, shift and mask, as a compiler divides
   by such a constant: the kind's division is a divide instruction an
   element. A signed quotient rounds toward zero as the kind's does, a
   negative dividend taking 2^k - 1 more before its shift; the remainder is
   what the quotient leaves. Given [v] of the compute type [CT], whose
   unsigned type is [UT], QUO_ and REM_ are the quotient and the remainder,
   signed and unsigned; LOW is 2^k - 1, and NEG all ones for a negative
   [v], else 0. */
#define LOW(CT, UT) ((UT)b - 1)
#define NEG(CT, UT) ((UT)(v >> (8 * sizeof(CT) - 1)))
#define QUO_s(CT, UT) (v + (CT)(NEG(CT, UT) & LOW(CT, UT))) >> k
#define REM_s(CT, UT)                                                        \
  (CT)((UT)v - (((UT)v + (NEG(CT, UT) & LOW(CT, UT))) & ~LOW(CT, UT)))
#define QUO_u(CT, UT) v >> k
#define REM_u(CT, UT) v & LOW(CT, UT)

#define SHIFTED(NAME, T, CT, UT, F, E)                                       \
  ROW NAME(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *x_,            \
           int64_t sx, const uint8_t *y_, int64_t sy) {                      \
    T *d = (T *)d_;                                                          \
    const T *x = (const T *)x_, *y = (const T *)y_;                          \
    CT b = sy == 0 ? (CT)y[0] : 0;                                           \
    if (sd == 1 && sx == 1 && sy == 0 && b > 0 && (b & (b - 1)) == 0) {      \
      int k = 0;                                                             \
      while (((CT)1 << k) != b) k++;                                         \
      for (int64_t i = 0; i < n; i++) {                                      \
        CT v = (CT)x[i];                                                     \
        d[i] = (T)(E(CT, UT));                                               \
      }                                                                      \
      return;                                                                \
    }                                                                        \
    LOOPS2(T, (CT), F);                                                      \
  }

#define ARITH(D, T, LD, S)                                                   \
  COMPARES(D, T, LD, S)                                                      \
  BIN(add_##D, T, T, LD, nx_add_##S)                                         \
  BIN(sub_##D, T, T, LD, nx_sub_##S)                                         \
  BIN(mul_##D, T, T, LD, nx_mul_##S)                                         \
  BIN(pow_##D, T, T, LD, nx_pow_##S)                                         \
  FMA(fma_##D, T, LD, nx_fma_##S)

#define FLOATS(D, T, S)                                                      \
  COMPARES(D, T, , S)                                                        \
  BIN(add_##D, T, T, , nx_add_##S)                                           \
  BIN(sub_##D, T, T, , nx_sub_##S)                                           \
  BIN(mul_##D, T, T, , nx_mul_##S)                                           \
  BIN(fdiv_##D, T, T, , nx_fdiv_##S)                                         \
  BIN(mod_##D, T, T, , nx_mod_##S)                                           \
  BIN(pow_##D, T, T, , nx_pow_##S)                                           \
  FMA(fma_##D, T, , nx_fma_##S)                                              \
  BIN(atan2_##D, T, T, , nx_atan2_##S)                                       \
  FLOAT_KINDS1(FLOAT1, S, T)                                                 \
  NEAR_KINDS(NEAR1, S, T)

#define INTS(D, T, CT, UT, S, SIGN)                                          \
  ARITH(D, T, (CT), S)                                                       \
  INT_KINDS1(INT1, D, T, CT, S)                                              \
  SHIFTED(idiv_##D, T, CT, UT, nx_idiv_##S, QUO_##SIGN)                      \
  SHIFTED(mod_##D, T, CT, UT, nx_mod_##S, REM_##SIGN)                        \
  BIN(and_##D, T, T, (CT), nx_and_##S)                                       \
  BIN(or_##D, T, T, (CT), nx_or_##S)                                         \
  BIN(xor_##D, T, T, (CT), nx_xor_##S)

/* The kinds of one operand of complex numbers, as FLOAT_KINDS1's. */
#define COMPLEX_KINDS1(X, ...)                                               \
  X(NEG, neg, __VA_ARGS__) X(RECIP, recip, __VA_ARGS__)                      \
  X(SIGN, sign, __VA_ARGS__) X(SQRT, sqrt, __VA_ARGS__)                      \
  X(EXP, exp, __VA_ARGS__) X(EXP2, exp2, __VA_ARGS__)                        \
  X(LOG, log, __VA_ARGS__) X(LOG2, log2, __VA_ARGS__)                        \
  X(SIN, sin, __VA_ARGS__) X(COS, cos, __VA_ARGS__)                          \
  X(TAN, tan, __VA_ARGS__) X(ASIN, asin, __VA_ARGS__)                        \
  X(ACOS, acos, __VA_ARGS__) X(ATAN, atan, __VA_ARGS__)                      \
  X(SINH, sinh, __VA_ARGS__) X(COSH, cosh, __VA_ARGS__)                      \
  X(TANH, tanh, __VA_ARGS__) X(ASINH, asinh, __VA_ARGS__)                    \
  X(ACOSH, acosh, __VA_ARGS__) X(ATANH, atanh, __VA_ARGS__)

/* Complex numbers take the arithmetic kinds but Mod, and Fdiv. */
#define COMPLEX(D, T)                                                        \
  COMPARES(D, T, , D)                                                        \
  BIN(add_##D, T, T, , nx_add_##D)                                           \
  BIN(sub_##D, T, T, , nx_sub_##D)                                           \
  BIN(mul_##D, T, T, , nx_mul_##D)                                           \
  BIN(fdiv_##D, T, T, , nx_fdiv_##D)                                         \
  BIN(pow_##D, T, T, , nx_pow_##D)                                           \
  COMPLEX_KINDS1(FLOAT1, D, T)

/* A boolean is 1 where its byte is not zero. */
#define BOOL_LD(v) ((uint32_t)((v) != 0))

FLOATS(f32, float, f32)
FLOATS(f64, double, f64)
UN(erf_f32, float, , nx_erf_f32)
RANGED(erf_f64, double, nx_erf_big_f64, nx_erf_near_f64, nx_erf_far_f64,
       nx_erf_f64)
INTS(i8, int8_t, int32_t, uint32_t, i32, s)
INTS(i16, int16_t, int32_t, uint32_t, i32, s)
INTS(i32, int32_t, int32_t, uint32_t, i32, s)
INTS(i64, int64_t, int64_t, uint64_t, i64, s)
INTS(u8, uint8_t, uint32_t, uint32_t, u32, u)
INTS(u16, uint16_t, uint32_t, uint32_t, u32, u)
INTS(u32, uint32_t, uint32_t, uint32_t, u32, u)
INTS(u64, uint64_t, uint64_t, uint64_t, u64, u)
COMPLEX(c64, nx_c64)
COMPLEX(c128, nx_c128)
COMPARES(b, uint8_t, BOOL_LD, u32)
BIN(and_b, uint8_t, uint8_t, BOOL_LD, nx_and_u32)
BIN(or_b, uint8_t, uint8_t, BOOL_LD, nx_or_u32)
BIN(xor_b, uint8_t, uint8_t, BOOL_LD, nx_xor_u32)

/* Threefry's contiguous rows run on vectors of 32-bit words: TF_LANES
   elements' low words in one vector and their high words in another, as
   the rounds use them, two such blocks at a time so that their rounds
   overlap. A loop over uint64 elements left the shape to the compilers:
   clang took two elements a vector, and over word pairs gcc shuffled
   lanes at each element (fudge, one core: 1.40 and 1.93 ns an element
   under gcc, 2.18 and 1.18 under clang, against 1.13 and 1.06 here). */
#define TF_LANES (NX_VECTOR_BYTES / 4)
typedef uint32_t tf_words __attribute__((vector_size(NX_VECTOR_BYTES)));

#if TF_LANES == 8
#define TF_LOW 0, 2, 4, 6, 8, 10, 12, 14
#define TF_HIGH 1, 3, 5, 7, 9, 11, 13, 15
#define TF_ZIP0 0, 8, 1, 9, 2, 10, 3, 11
#define TF_ZIP1 4, 12, 5, 13, 6, 14, 7, 15
#else
#define TF_LOW 0, 2, 4, 6
#define TF_HIGH 1, 3, 5, 7
#define TF_ZIP0 0, 4, 1, 5
#define TF_ZIP1 2, 6, 3, 7
#endif

/* The low and high words of the TF_LANES elements at [p], or of the one
   at [p] in every lane where [s] is 0: a little-endian host lays each
   element's low word at the even index, its high word at the odd one. */
static inline void tf_load(const uint64_t *p, int s, tf_words *lo,
                           tf_words *hi) {
  if (s == 0) {
    *lo = (tf_words){0} + (uint32_t)p[0];
    *hi = (tf_words){0} + (uint32_t)(p[0] >> 32);
    return;
  }
  tf_words a, b;
  memcpy(&a, p, sizeof a);
  memcpy(&b, p + TF_LANES / 2, sizeof b);
  *lo = __builtin_shufflevector(a, b, TF_LOW);
  *hi = __builtin_shufflevector(a, b, TF_HIGH);
}

/* The key's three words of the elements at [p], as tf_load reads them. */
static inline void tf_key(const uint64_t *p, int s, tf_words *k) {
  tf_load(p, s, &k[0], &k[1]);
  k[2] = 0x1BD11BDAu ^ k[0] ^ k[1];
}

/* Stores TF_LANES elements of low words [lo] and high words [hi] at [p]. */
static inline void tf_store(uint64_t *p, tf_words lo, tf_words hi) {
  tf_words a = __builtin_shufflevector(lo, hi, TF_ZIP0);
  tf_words b = __builtin_shufflevector(lo, hi, TF_ZIP1);
  memcpy(p, &a, sizeof a);
  memcpy(p + TF_LANES / 2, &b, sizeof b);
}

/* nx_kinds.h's four rounds and injection on two blocks, (a0, a1) under key
   [ak] and (b0, b1) under [bk]. */
#define TF_ROUND(r)                                                          \
  a0 += a1;                                                                  \
  b0 += b1;                                                                  \
  a1 = NX_ROTL32(a1, r) ^ a0;                                                \
  b1 = NX_ROTL32(b1, r) ^ b0;
#define TF_FOUR(r0, r1, r2, r3, a, b, s)                                     \
  TF_ROUND(r0)                                                               \
  TF_ROUND(r1)                                                               \
  TF_ROUND(r2)                                                               \
  TF_ROUND(r3)                                                               \
  a0 += ak[a];                                                               \
  b0 += bk[a];                                                               \
  a1 += ak[b] + s;                                                           \
  b1 += bk[b] + s;

/* Two blocks at a time from element [i] while they fit in [n], the
   counters stepping [SX] elements and the keys [SY], 0 or 1: a key of
   step 0 is read once, into bk as well as ak, so that TF_FOUR reads both
   blocks' keys alike whatever the steps. */
#define THREEFRY_BLOCKS(SX, SY)                                              \
  {                                                                          \
    tf_words ak[3], bk[3];                                                   \
    if (SY == 0) {                                                           \
      tf_key(y, 0, ak);                                                      \
      tf_key(y, 0, bk);                                                      \
    }                                                                        \
    for (; i + 2 * TF_LANES <= n; i += 2 * TF_LANES) {                       \
      tf_words a0, a1, b0, b1;                                               \
      if (SY == 1) {                                                         \
        tf_key(y + i, 1, ak);                                                \
        tf_key(y + i + TF_LANES, 1, bk);                                     \
      }                                                                      \
      tf_load(x + i * SX, SX, &a0, &a1);                                     \
      tf_load(x + (i + TF_LANES) * SX, SX, &b0, &b1);                        \
      a0 += ak[0];                                                           \
      a1 += ak[1];                                                           \
      b0 += bk[0];                                                           \
      b1 += bk[1];                                                           \
      NX_THREEFRY_SCHEDULE(TF_FOUR)                                          \
      tf_store(d + i, a0, a1);                                               \
      tf_store(d + i + TF_LANES, b0, b1);                                    \
    }                                                                        \
  }

ROW threefry_u64(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *x_,
                 int64_t sx, const uint8_t *y_, int64_t sy) {
  uint64_t *d = (uint64_t *)d_;
  const uint64_t *x = (const uint64_t *)x_, *y = (const uint64_t *)y_;
  int64_t i = 0;
  if (sd == 1 && sx == 1 && sy == 1) THREEFRY_BLOCKS(1, 1)
  else if (sd == 1 && sx == 1 && sy == 0) THREEFRY_BLOCKS(1, 0)
  else if (sd == 1 && sx == 0 && sy == 1) THREEFRY_BLOCKS(0, 1)
  for (; i < n; i++) d[i * sd] = nx_threefry_u64(x[i * sx], y[i * sy]);
}

/* Where selects bits and a fill stores them: one row per width. */
/* A where of [w]-byte words selects by a mask, so that contiguous rows
   vectorise: a select of the form c ? x : y with a byte condition and wider
   words ran scalar on the M1 (where-f32-1M 86-126 us against 31-34 for a
   fill of its bytes). */
#define WHERE(W, T)                                                          \
  static void where_##W(int64_t n, uint8_t *d_, int64_t sd,                 \
                        const uint8_t *c, int64_t sc, const uint8_t *x_,     \
                        int64_t sx, const uint8_t *y_, int64_t sy) {         \
    T *d = (T *)d_;                                                          \
    const T *x = (const T *)x_, *y = (const T *)y_;                          \
    if (sd == 1 && sc == 1 && sx == 1 && sy == 1)                            \
      for (int64_t i = 0; i < n; i++) {                                      \
        T m = (T)0 - (T)(c[i] != 0);                                         \
        d[i] = (x[i] & m) | (y[i] & (T)~m);                                  \
      }                                                                      \
    else if (sd == 1 && sc == 1 && sx == 1 && sy == 0) {                     \
      T b = y[0];                                                            \
      for (int64_t i = 0; i < n; i++) {                                      \
        T m = (T)0 - (T)(c[i] != 0);                                         \
        d[i] = (x[i] & m) | (b & (T)~m);                                     \
      }                                                                      \
    } else if (sd == 1 && sc == 1 && sx == 0 && sy == 1) {                   \
      T a = x[0];                                                            \
      for (int64_t i = 0; i < n; i++) {                                      \
        T m = (T)0 - (T)(c[i] != 0);                                         \
        d[i] = (a & m) | (y[i] & (T)~m);                                     \
      }                                                                      \
    } else                                                                   \
      for (int64_t i = 0; i < n; i++)                                        \
        d[i * sd] = c[i * sc] ? x[i * sx] : y[i * sy];                       \
  }

typedef struct {
  uint64_t lo, hi;
} w16;

WHERE(1, uint8_t)
WHERE(2, uint16_t)
WHERE(4, uint32_t)
WHERE(8, uint64_t)

static void where_16(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *c,
                     int64_t sc, const uint8_t *x_, int64_t sx,
                     const uint8_t *y_, int64_t sy) {
  w16 *d = (w16 *)d_;
  const w16 *x = (const w16 *)x_, *y = (const w16 *)y_;
  for (int64_t i = 0; i < n; i++)
    d[i * sd] = c[i * sc] ? x[i * sx] : y[i * sy];
}

#define FILL(W, T)                                                           \
  static void fill_##W(int64_t n, uint8_t *d_, int64_t sd,                  \
                       const uint8_t *bits) {                               \
    T *d = (T *)d_;                                                          \
    T v;                                                                     \
    memcpy(&v, bits, sizeof v);                                              \
    if (sd == 1)                                                             \
      for (int64_t i = 0; i < n; i++) d[i] = v;                              \
    else                                                                     \
      for (int64_t i = 0; i < n; i++) d[i * sd] = v;                         \
  }

FILL(1, uint8_t)
FILL(2, uint16_t)
FILL(4, uint32_t)
FILL(8, uint64_t)
FILL(16, w16)

/* The tables, by kind then dtype: NULL where Nx_kernel.Prog refuses the
   kind at the dtype. */

#define CMP_ROWS(DT, D)                                                      \
  t->op2[NX_OP2_EQUAL][DT] = equal_##D;                                      \
  t->op2[NX_OP2_NOT_EQUAL][DT] = not_equal_##D;                              \
  t->op2[NX_OP2_LESS][DT] = less_##D;                                        \
  t->op2[NX_OP2_LESS_EQUAL][DT] = less_equal_##D;                            \
  t->op2[NX_OP2_MAXIMUM][DT] = maximum_##D;                                  \
  t->op2[NX_OP2_MINIMUM][DT] = minimum_##D

#define ARITH_ROWS(DT, D)                                                    \
  CMP_ROWS(DT, D);                                                           \
  t->op2[NX_OP2_ADD][DT] = add_##D;                                          \
  t->op2[NX_OP2_SUB][DT] = sub_##D;                                          \
  t->op2[NX_OP2_MUL][DT] = mul_##D;                                          \
  t->op2[NX_OP2_MOD][DT] = mod_##D;                                          \
  t->op2[NX_OP2_POW][DT] = pow_##D;                                          \
  t->fma[DT] = fma_##D

#define SET1(C, K, DT, D) t->op1[NX_OP1_##C][DT] = K##_##D;
#define SETN(C, K, B, DT, D) SET1(C, K, DT, D)

#define FLOAT_ROWS(DT, D)                                                    \
  ARITH_ROWS(DT, D);                                                         \
  t->op2[NX_OP2_FDIV][DT] = fdiv_##D;                                        \
  t->op2[NX_OP2_ATAN2][DT] = atan2_##D;                                      \
  FLOAT_KINDS1(SET1, DT, D)                                                  \
  NEAR_KINDS(SETN, DT, D)

#define INT_ROWS(DT, D, W)                                                   \
  ARITH_ROWS(DT, D);                                                         \
  t->op2[NX_OP2_IDIV][DT] = idiv_##D;                                        \
  t->op2[NX_OP2_AND][DT] = and_##D;                                          \
  t->op2[NX_OP2_OR][DT] = or_##D;                                            \
  t->op2[NX_OP2_XOR][DT] = xor_##D;                                          \
  INT_KINDS1(SET1, DT, D)                                                    \
  t->op1[NX_OP1_FLOOR][DT] = t->op1[NX_OP1_CEIL][DT] = ident_##W;            \
  t->op1[NX_OP1_ROUND][DT] = t->op1[NX_OP1_TRUNC][DT] = ident_##W

#define COMPLEX_ROWS(DT, D)                                                  \
  CMP_ROWS(DT, D);                                                           \
  t->op2[NX_OP2_ADD][DT] = add_##D;                                          \
  t->op2[NX_OP2_SUB][DT] = sub_##D;                                          \
  t->op2[NX_OP2_MUL][DT] = mul_##D;                                          \
  t->op2[NX_OP2_FDIV][DT] = fdiv_##D;                                        \
  t->op2[NX_OP2_POW][DT] = pow_##D;                                          \
  COMPLEX_KINDS1(SET1, DT, D)

static void set(nx_cpu_target *t) {
  COMPLEX_ROWS(NX_COMPLEX64, c64);
  COMPLEX_ROWS(NX_COMPLEX128, c128);
  FLOAT_ROWS(NX_FLOAT32, f32);
  FLOAT_ROWS(NX_FLOAT64, f64);
  t->op1[NX_OP1_ERF][NX_FLOAT32] = erf_f32;
  t->op1[NX_OP1_ERF][NX_FLOAT64] = erf_f64;
  INT_ROWS(NX_INT8, i8, 1);
  INT_ROWS(NX_INT16, i16, 2);
  INT_ROWS(NX_INT32, i32, 4);
  INT_ROWS(NX_INT64, i64, 8);
  INT_ROWS(NX_UINT8, u8, 1);
  INT_ROWS(NX_UINT16, u16, 2);
  INT_ROWS(NX_UINT32, u32, 4);
  INT_ROWS(NX_UINT64, u64, 8);
  CMP_ROWS(NX_BOOL, b);
  t->op2[NX_OP2_AND][NX_BOOL] = and_b;
  t->op2[NX_OP2_OR][NX_BOOL] = or_b;
  t->op2[NX_OP2_XOR][NX_BOOL] = xor_b;
  t->op2[NX_OP2_THREEFRY][NX_UINT64] = threefry_u64;
  nx_cpu_row3 where[] = {where_1, where_2, where_4, where_8, where_16};
  nx_cpu_row0 fills[] = {fill_1, fill_2, fill_4, fill_8, fill_16};
  for (int i = 0; i < 5; i++) {
    t->where[i] = where[i];
    t->fill[i] = fills[i];
  }
}

#if defined(NX_CPU_V3)
void nx_cpu_set_rows_v3(nx_cpu_target *t) { set(t); }
#else
void nx_cpu_set_rows_base(nx_cpu_target *t) { set(t); }
#endif

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_v3;

#endif
