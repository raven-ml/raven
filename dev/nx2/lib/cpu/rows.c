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
   runs a narrow or sub-byte dtype's kinds at its carrier's rows.

   nx_kinds.h's add, sub, mul, fdiv and fma of floats differ from the plain
   operation only in which NaN a NaN result is, and pinning it costs three
   compares and three selects an element, twice the plain operation's
   instructions. Their rows compute the plain operation and note a NaN,
   then run the kind over the row where they found one: a row whose
   destination shares no byte with an operand, which the kind can read
   again. */

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

/* Elements of a trigonometric row's piece. */
#define TRIG_PIECE 256

/* sin, cos and tan as nx_kinds.h says a vector loop computes them: a piece
   with no lane past the integer reduction's switch runs the kind below it
   on every lane, in a loop that vectorises; a piece with one runs the
   scalar kind, which branches. Each piece is tested before it is written,
   so a destination that is its operand reads every element first. */
#define TRIG(K, S, T)                                                        \
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
    for (int64_t b = 0; b < n; b += TRIG_PIECE) {                            \
      int64_t e = n - b < TRIG_PIECE ? n : b + TRIG_PIECE;                   \
      int big = 0;                                                           \
      for (int64_t i = b; i < e; i++) big |= nx_trig_big_##S(x[i]);          \
      if (big)                                                               \
        for (int64_t i = b; i < e; i++) d[i] = nx_##K##_##S(x[i]);           \
      else                                                                   \
        for (int64_t i = b; i < e; i++) d[i] = nx_##K##_near_##S(x[i]);      \
    }                                                                        \
  }

/* The kinds of one operand by nx_spec.h's code and nx_kinds.h's name: of
   floats, of floats past the trigonometric switch, of integers. Each
   applies [X] to the code, the name and the arguments that follow [X]. */
#define FLOAT_KINDS1(X, ...)                                                 \
  X(NEG, neg, __VA_ARGS__) X(RECIP, recip, __VA_ARGS__)                      \
  X(ABS, abs, __VA_ARGS__) X(SIGN, sign, __VA_ARGS__)                        \
  X(SQRT, sqrt, __VA_ARGS__) X(EXP, exp, __VA_ARGS__)                        \
  X(EXP2, exp2, __VA_ARGS__) X(LOG, log, __VA_ARGS__)                        \
  X(LOG2, log2, __VA_ARGS__) X(LOG1P, log1p, __VA_ARGS__)                    \
  X(EXPM1, expm1, __VA_ARGS__) X(ASIN, asin, __VA_ARGS__)                    \
  X(ACOS, acos, __VA_ARGS__) X(ATAN, atan, __VA_ARGS__)                      \
  X(SINH, sinh, __VA_ARGS__) X(COSH, cosh, __VA_ARGS__)                      \
  X(TANH, tanh, __VA_ARGS__) X(ERF, erf, __VA_ARGS__)                        \
  X(FLOOR, floor, __VA_ARGS__) X(CEIL, ceil, __VA_ARGS__)                    \
  X(ROUND, round, __VA_ARGS__) X(TRUNC, trunc, __VA_ARGS__)
#define TRIG_KINDS(X, ...)                                                   \
  X(SIN, sin, __VA_ARGS__) X(COS, cos, __VA_ARGS__) X(TAN, tan, __VA_ARGS__)
#define INT_KINDS1(X, ...)                                                   \
  X(NEG, neg, __VA_ARGS__) X(RECIP, recip, __VA_ARGS__)                      \
  X(ABS, abs, __VA_ARGS__) X(SIGN, sign, __VA_ARGS__)

#define FLOAT1(C, K, S, T) UN(K##_##S, T, , nx_##K##_##S)
#define TRIG1(C, K, S, T) TRIG(K, S, T)
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

/* A plain loop over [n] elements storing [E] into [d] and noting a NaN. */
#define NOTED(T, E)                                                          \
  for (int64_t i = 0; i < n; i++) {                                          \
    T r = E;                                                                 \
    d[i] = r;                                                                \
    nan |= r != r;                                                           \
  }

/* The row of the float kind [F] of two operands whose plain operation is
   [OP]. A row that notes a NaN computes again by [F] from its operands, so
   one whose destination is an operand (cpu.h: disjoint or identical) takes
   [F] alone. */
#define EXACT(NAME, T, OP, F)                                                \
  ROW NAME(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *x_,            \
           int64_t sx, const uint8_t *y_, int64_t sy) {                      \
    T *d = (T *)d_;                                                          \
    const T *x = (const T *)x_, *y = (const T *)y_;                          \
    int nan = 1;                                                             \
    if (sd == 1 && d_ != x_ && d_ != y_) {                                   \
      nan = 0;                                                               \
      if (sx == 1 && sy == 1) {                                              \
        NOTED(T, x[i] OP y[i])                                               \
      } else if (sx == 1 && sy == 0) {                                       \
        T b = y[0];                                                          \
        NOTED(T, x[i] OP b)                                                  \
      } else if (sx == 0 && sy == 1) {                                       \
        T a = x[0];                                                          \
        NOTED(T, a OP y[i])                                                  \
      } else                                                                 \
        nan = 1;                                                             \
    }                                                                        \
    if (!nan) return;                                                        \
    LOOPS2(T, , F);                                                          \
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

/* The row of the float kind [F], fma, whose plain operation is [FN]: the
   contiguous rows whose destination is no operand alone are noted. */
#define EXACT_FMA(NAME, T, FN, F)                                            \
  ROW NAME(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *a_,            \
           int64_t sa, const uint8_t *b_, int64_t sb,                        \
           const uint8_t *c_, int64_t sc) {                                  \
    T *d = (T *)d_;                                                          \
    const T *a = (const T *)a_, *b = (const T *)b_, *c = (const T *)c_;      \
    int nan = 1;                                                             \
    if (sd == 1 && sa == 1 && sb == 1 && sc == 1 && d_ != a_ && d_ != b_ &&  \
        d_ != c_) {                                                          \
      nan = 0;                                                               \
      NOTED(T, FN(a[i], b[i], c[i]))                                         \
    }                                                                        \
    if (!nan) return;                                                        \
    LOOPS3(T, , F);                                                          \
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

#define FLOATS(D, T, S, FN)                                                  \
  COMPARES(D, T, , S)                                                        \
  EXACT(add_##D, T, +, nx_add_##S)                                           \
  EXACT(sub_##D, T, -, nx_sub_##S)                                           \
  EXACT(mul_##D, T, *, nx_mul_##S)                                           \
  EXACT(fdiv_##D, T, /, nx_fdiv_##S)                                         \
  BIN(mod_##D, T, T, , nx_mod_##S)                                           \
  BIN(pow_##D, T, T, , nx_pow_##S)                                           \
  EXACT_FMA(fma_##D, T, FN, nx_fma_##S)                                      \
  BIN(atan2_##D, T, T, , nx_atan2_##S)                                       \
  FLOAT_KINDS1(FLOAT1, S, T)                                                 \
  TRIG_KINDS(TRIG1, S, T)

#define INTS(D, T, CT, UT, S, SIGN)                                          \
  ARITH(D, T, (CT), S)                                                       \
  INT_KINDS1(INT1, D, T, CT, S)                                              \
  SHIFTED(idiv_##D, T, CT, UT, nx_idiv_##S, QUO_##SIGN)                      \
  SHIFTED(mod_##D, T, CT, UT, nx_mod_##S, REM_##SIGN)                        \
  BIN(and_##D, T, T, (CT), nx_and_##S)                                       \
  BIN(or_##D, T, T, (CT), nx_or_##S)                                         \
  BIN(xor_##D, T, T, (CT), nx_xor_##S)

/* Complex numbers take the arithmetic kinds but Mod and Pow, and Fdiv. */
#define COMPLEX(D, T)                                                        \
  COMPARES(D, T, , D)                                                        \
  BIN(add_##D, T, T, , nx_add_##D)                                           \
  BIN(sub_##D, T, T, , nx_sub_##D)                                           \
  BIN(mul_##D, T, T, , nx_mul_##D)                                           \
  BIN(fdiv_##D, T, T, , nx_fdiv_##D)                                         \
  UN(neg_##D, T, , nx_neg_##D)                                               \
  UN(recip_##D, T, , nx_recip_##D)

/* A boolean is 1 where its byte is not zero. */
#define BOOL_LD(v) ((uint32_t)((v) != 0))

FLOATS(f32, float, f32, nx_fmaf)
FLOATS(f64, double, f64, nx_fmad)
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
BIN(threefry_u64, uint64_t, uint64_t, , nx_threefry_u64)

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

/* The tables, by kind then dtype: NULL where the case is declined. */

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

#define FLOAT_ROWS(DT, D)                                                    \
  ARITH_ROWS(DT, D);                                                         \
  t->op2[NX_OP2_FDIV][DT] = fdiv_##D;                                        \
  t->op2[NX_OP2_ATAN2][DT] = atan2_##D;                                      \
  FLOAT_KINDS1(SET1, DT, D)                                                  \
  TRIG_KINDS(SET1, DT, D)

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
  t->op1[NX_OP1_NEG][DT] = neg_##D;                                          \
  t->op1[NX_OP1_RECIP][DT] = recip_##D

static void set(nx_cpu_target *t) {
  COMPLEX_ROWS(NX_COMPLEX64, c64);
  COMPLEX_ROWS(NX_COMPLEX128, c128);
  FLOAT_ROWS(NX_FLOAT32, f32);
  FLOAT_ROWS(NX_FLOAT64, f64);
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
