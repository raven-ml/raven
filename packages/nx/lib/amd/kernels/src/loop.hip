/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Loops: a map, a reduction or a scan of a program, run over a tape the
   host lowers the program to (kernels.h, Loops), a kernel for each mode.

   A group of NX_LOOP_THREADS work-items first copies the words it uses
   into LDS (stage), then runs tiles of v elements a work-item. Each value
   the tape holds lives in planes of LDS, a 32-bit word an element each,
   one to four planes as its dtype is wide: element [i] of plane [p] is
   word p E + i, E the tile's elements, and a work-item's element v is
   i = v NX_LOOP_THREADS + its index. A work-item reads and writes its own
   elements only, so the tape runs without barriers. A work-item's
   elements' loop indices step evenly (lanes): past its first n elements
   none has one, whose loads read nothing and whose stores write nothing.

     stage words --> constants --> per tile: lanes --> body
     --> fold (reduce), chain (scan) or stores (map) --> epilogue's stores

   A value is held as its dtype's element exactly: a narrow float as its
   code, an integer of 16 bits or fewer sign- or zero-extended to 32, a
   boolean as its byte, so that Copy, Where and Bitcast move bits. The host
   lowers a kind at a narrow float to a decode, the float32 kind and an
   encode, and a kind at a narrow integer to its 32-bit kind and a wrap.

   A reduction's element is a term. A block's G work-items take G v
   consecutive terms a run of the tape, work-item t's element j the term
   t + G j. With G = 16 / v, or fewer where a block's terms fill fewer
   lanes, work-item t holds lanes t + G j of nx.cpu's lane order and folds
   them itself, its states in planes; with G of 16 or more (spread),
   work-item l below 16 folds lane l's terms of the run from the others'
   elements into its state, one a work-item past the planes. Lanes fold in
   increasing term order, then the lanes' tree, a lane of no term taking no
   part, then the blocks' binary tree, so the bits are nx.cpu's. Blocks
   fall into groups of a power of two of them, a subtree; an output of
   several groups leaves each group's state in the workspace, and the
   work-item that takes its ticket last merges them.

   A scan's unit is a chunk of a slice. S work-items load a chunk, S v
   consecutive terms a run, coalesced where its terms step by one; then
   the first of them chains them in order from the carry into the chunk:
   the slice's carry, or, past one chunk, the state the carries kernel
   left, the given carry then each earlier chunk's total folded in order.
   The bits are nx.cpu's.

   A Sum, Max or Min of one plain load at its carrier runs without a tape,
   its terms read straight into registers in the same orders (Plain
   reductions, Plain scans).

   Every function is inlined: the kernels make no call, and each runs the
   tape from one site. */

#include "device.h"
#include "dtypes.h"

/* nx_kernel.h's and nx_kinds.h's functions, inlined like this unit's. */
#pragma clang attribute push(__attribute__((always_inline)), apply_to = function)
#include "nx_kernel.h"
#include "nx_kinds.h"
#pragma clang attribute pop

#define TB NX_LOOP_THREADS

/* A kind's code, below 128 beside its compute type, falls in its arity's
   range (kernels.h). */
_Static_assert(NX_OP1_COUNT <= NX_LOOP_BINARY, "op1 codes reach op2's");
_Static_assert(NX_LOOP_BINARY + NX_OP2_COUNT <= NX_LOOP_TERNARY,
               "op2 codes reach op3's");
_Static_assert(NX_LOOP_TERNARY + NX_OP3_COUNT <= NX_LOOP_FMA_NARROW,
               "op3 codes reach the narrow Fmas'");
_Static_assert(NX_LOOP_FMA_BF16 < 128, "kind codes reach the compute type");

typedef uint64_t u64;

/* Words

   A group reads the words it uses, the header's NX_LOOP_USED, from LDS,
   where device.h's stage copies them once as it starts. */

static SHARED int64_t loop_words[NX_LOOP_WORDS];

DEVICE const u64 *loop_staged(void) {
  stage(loop_words, words()[NX_LOOP_USED], TB);
  return (const u64 *)loop_words;
}

/* Planes */

extern SHARED __attribute__((aligned(16))) uint32_t loop_planes[];

DEVICE uint32_t &word(int E, int p, int i) { return loop_planes[p * E + i]; }

DEVICE u64 get64(int E, int p, int i) {
  return word(E, p, i) | (u64)word(E, p + 1, i) << 32;
}

DEVICE void put64(int E, int p, int i, u64 x) {
  word(E, p, i) = (uint32_t)x;
  word(E, p + 1, i) = (uint32_t)(x >> 32);
}

template <class T> DEVICE T bits_as(u64 lo, u64 hi) {
  T x;
  u64 b[2] = {lo, hi};
  memcpy(&x, b, sizeof(T));
  return x;
}

template <class T> DEVICE T get(int E, int p, int i) {
  if constexpr (sizeof(T) == 4) return bits_as<T>(word(E, p, i), 0);
  else if constexpr (sizeof(T) == 8) return bits_as<T>(get64(E, p, i), 0);
  else return bits_as<T>(get64(E, p, i), get64(E, p + 2, i));
}

template <class T> DEVICE void put(int E, int p, int i, T x) {
  u64 b[2] = {0, 0};
  memcpy(b, &x, sizeof(T));
  if constexpr (sizeof(T) == 4) word(E, p, i) = (uint32_t)b[0];
  else if constexpr (sizeof(T) == 8) put64(E, p, i, b[0]);
  else {
    put64(E, p, i, b[0]);
    put64(E, p + 2, i, b[1]);
  }
}

/* A value of up to four words, as held: [lo] its words 0 and 1, [hi] its
   words 2 and 3. */
typedef struct {
  u64 lo, hi;
} held;

DEVICE held get_held(int E, int p, int i, int n) {
  held h = {word(E, p, i), 0};
  if (n > 1) h.lo |= (u64)word(E, p + 1, i) << 32;
  if (n > 2) h.hi = get64(E, p + 2, i);
  return h;
}

DEVICE void put_held(int E, int p, int i, int n, held h) {
  word(E, p, i) = (uint32_t)h.lo;
  if (n > 1) word(E, p + 1, i) = (uint32_t)(h.lo >> 32);
  if (n > 2) put64(E, p + 2, i, h.hi);
}

/* Each element of the calling work-item: its slot v and its index i.
   EACH runs them in turn; ALL unrolled, a slot past V doing nothing, so
   that their LDS reads are in flight together. */
#define EACH                                                                   \
  _Pragma("unroll 1") for (int v = 0, i = (int)item(); v < V; v++, i += TB)
#define ALL                                                                    \
  _Pragma("unroll") for (int v = 0, i = (int)item(); v < NX_LOOP_MOST;        \
                         v++, i += TB) if (v < V)

/* The loop indices of the calling work-item's elements: slot v's is
   L0 + v step for v below n; the others have none. */
typedef struct {
  int64_t L0;
  int64_t step;
  int n;
} lanes;

DEVICE int64_t index_of(lanes x, int v) { return x.L0 + (int64_t)v * x.step; }

/* The lanes of a work-item whose first element's index is [L0], stepping
   by [step], at most NX_LOOP_THREADS, those below [end] alone having one,
   of [V] elements. A division of 32 bits, where some but not all have
   one. */
DEVICE lanes lanes_of(int64_t L0, int64_t step, int64_t end, int V) {
  int64_t r = end - L0;
  int n = r <= 0                         ? 0
          : r > (int64_t)(V - 1) * step ? V
                                         : (int)(((uint32_t)r + (uint32_t)step - 1) /
                                                 (uint32_t)step);
  lanes x = {L0, step, n};
  return x;
}

/* Kinds

   One function per kind and compute type over a work-item's elements. A
   cheap kind's elements unrolled, their reads in flight together; a costly
   one's an element at a time, which keeps its code to one copy. */

#define K1(S, T, U, n)                                                         \
  DEVICE void k_##n##_##S(int E, int V, int d, int a, int, int) {              \
    T x[NX_LOOP_MOST] = {};                                                    \
    ALL x[v] = get<T>(E, a, i);                                                \
    ALL put<T>(E, d, i, nx_##n##_##S(x[v]));                                   \
  }
#define K2(S, T, U, n)                                                         \
  DEVICE void k_##n##_##S(int E, int V, int d, int a, int b, int) {            \
    T x[NX_LOOP_MOST] = {}, y[NX_LOOP_MOST] = {};                              \
    ALL {                                                                      \
      x[v] = get<T>(E, a, i);                                                  \
      y[v] = get<T>(E, b, i);                                                  \
    }                                                                          \
    ALL put<T>(E, d, i, nx_##n##_##S(x[v], y[v]));                             \
  }
#define KC(S, T, U, n)                                                         \
  DEVICE void k_##n##_##S(int E, int V, int d, int a, int b, int) {            \
    T x[NX_LOOP_MOST] = {}, y[NX_LOOP_MOST] = {};                              \
    ALL {                                                                      \
      x[v] = get<T>(E, a, i);                                                  \
      y[v] = get<T>(E, b, i);                                                  \
    }                                                                          \
    ALL put<uint32_t>(E, d, i, nx_##n##_##S(x[v], y[v]) != 0);                 \
  }
#define K3(S, T, U, n)                                                         \
  DEVICE void k_##n##_##S(int E, int V, int d, int a, int b, int c) {          \
    T x[NX_LOOP_MOST] = {}, y[NX_LOOP_MOST] = {}, z[NX_LOOP_MOST] = {};        \
    ALL {                                                                      \
      x[v] = get<T>(E, a, i);                                                  \
      y[v] = get<T>(E, b, i);                                                  \
      z[v] = get<T>(E, c, i);                                                  \
    }                                                                          \
    ALL put<T>(E, d, i, nx_##n##_##S(x[v], y[v], z[v]));                       \
  }
#define S1(S, T, U, n)                                                         \
  DEVICE void k_##n##_##S(int E, int V, int d, int a, int, int) {              \
    EACH put<T>(E, d, i, nx_##n##_##S(get<T>(E, a, i)));                       \
  }
#define S2(S, T, U, n)                                                         \
  DEVICE void k_##n##_##S(int E, int V, int d, int a, int b, int) {            \
    EACH put<T>(E, d, i, nx_##n##_##S(get<T>(E, a, i), get<T>(E, b, i)));      \
  }
#define SC(S, T, U, n)                                                         \
  DEVICE void k_##n##_##S(int E, int V, int d, int a, int b, int) {            \
    EACH put<uint32_t>(                                                        \
        E, d, i, nx_##n##_##S(get<T>(E, a, i), get<T>(E, b, i)) != 0);         \
  }

/* Each list in two: its cheap kinds (F), unrolled, then its costly ones
   (S). */
#define REAL1F(X, S, T)                                                        \
  X(S, T, NEG, neg) X(S, T, ABS, abs) X(S, T, SIGN, sign)                      \
  X(S, T, FLOOR, floor) X(S, T, CEIL, ceil) X(S, T, ROUND, round)              \
  X(S, T, TRUNC, trunc)
#define REAL1S(X, S, T)                                                        \
  X(S, T, RECIP, recip) X(S, T, SQRT, sqrt) X(S, T, EXP, exp)                  \
  X(S, T, EXP2, exp2) X(S, T, LOG, log) X(S, T, LOG2, log2)                    \
  X(S, T, LOG1P, log1p) X(S, T, EXPM1, expm1) X(S, T, SIN, sin)                \
  X(S, T, COS, cos) X(S, T, TAN, tan) X(S, T, ASIN, asin)                      \
  X(S, T, ACOS, acos) X(S, T, ATAN, atan) X(S, T, SINH, sinh)                  \
  X(S, T, COSH, cosh) X(S, T, TANH, tanh) X(S, T, ERF, erf)
#define REAL2F(X, S, T)                                                        \
  X(S, T, ADD, add) X(S, T, SUB, sub) X(S, T, MUL, mul)                        \
  X(S, T, MAXIMUM, maximum) X(S, T, MINIMUM, minimum)
#define REAL2S(X, S, T)                                                        \
  X(S, T, FDIV, fdiv) X(S, T, MOD, mod) X(S, T, POW, pow) X(S, T, ATAN2, atan2)
#define CPLX1(X, S, T)                                                         \
  X(S, T, NEG, neg) X(S, T, RECIP, recip) X(S, T, SIGN, sign)                  \
  X(S, T, SQRT, sqrt) X(S, T, EXP, exp) X(S, T, EXP2, exp2) X(S, T, LOG, log)  \
  X(S, T, LOG2, log2) X(S, T, SIN, sin) X(S, T, COS, cos) X(S, T, TAN, tan)    \
  X(S, T, ASIN, asin) X(S, T, ACOS, acos) X(S, T, ATAN, atan)                  \
  X(S, T, SINH, sinh) X(S, T, COSH, cosh) X(S, T, TANH, tanh)                  \
  X(S, T, ASINH, asinh) X(S, T, ACOSH, acosh) X(S, T, ATANH, atanh)
#define CPLX2(X, S, T)                                                         \
  X(S, T, ADD, add) X(S, T, SUB, sub) X(S, T, MUL, mul) X(S, T, FDIV, fdiv)    \
  X(S, T, POW, pow) X(S, T, MAXIMUM, maximum) X(S, T, MINIMUM, minimum)
#define INT1F(X, S, T) X(S, T, NEG, neg) X(S, T, ABS, abs) X(S, T, SIGN, sign)
#define INT1S(X, S, T) X(S, T, RECIP, recip)
#define INT2F(X, S, T)                                                         \
  X(S, T, ADD, add) X(S, T, SUB, sub) X(S, T, MUL, mul)                        \
  X(S, T, MAXIMUM, maximum) X(S, T, MINIMUM, minimum) X(S, T, AND, and)        \
  X(S, T, OR, or) X(S, T, XOR, xor)
#define INT2S(X, S, T) X(S, T, IDIV, idiv) X(S, T, MOD, mod) X(S, T, POW, pow)
#define COMPARE(X, S, T)                                                       \
  X(S, T, EQUAL, equal) X(S, T, NOT_EQUAL, not_equal) X(S, T, LESS, less)      \
  X(S, T, LESS_EQUAL, less_equal)

#define REAL(S, T)                                                             \
  REAL1F(K1, S, T) REAL1S(S1, S, T) REAL2F(K2, S, T) REAL2S(S2, S, T)          \
  COMPARE(KC, S, T) K3(S, T, FMA, fma)
#define INT(S, T)                                                              \
  INT1F(K1, S, T) INT1S(S1, S, T) INT2F(K2, S, T) INT2S(S2, S, T)              \
  COMPARE(KC, S, T) K3(S, T, FMA, fma)
#define CPLX(S, T) CPLX1(S1, S, T) CPLX2(S2, S, T) COMPARE(SC, S, T)

REAL(f32, float) REAL(f64, double) CPLX(c64, nx_c64) CPLX(c128, nx_c128)
INT(i32, int32_t) INT(u32, uint32_t) INT(i64, int64_t) INT(u64, u64)
S2(u64, u64, THREEFRY, threefry)

/* A kind's op: its compute type's code, then its own (kernels.h). */
#define CT_f32 NX_LOOP_F32
#define CT_f64 NX_LOOP_F64
#define CT_c64 NX_LOOP_C64
#define CT_c128 NX_LOOP_C128
#define CT_i32 NX_LOOP_I32
#define CT_u32 NX_LOOP_U32
#define CT_i64 NX_LOOP_I64
#define CT_u64 NX_LOOP_U64

/* Fma at a narrow float (NX_LOOP_FMA_NARROW, NX_LOOP_FMA_BF16). */
DEVICE void k_fma_narrow(int E, int V, int d, int a, int b, int c) {
  EACH put<float>(E, d, i,
                  nx_fma_narrow(get<float>(E, a, i), get<float>(E, b, i),
                                get<float>(E, c, i)));
}
DEVICE void k_fma_bf16(int E, int V, int d, int a, int b, int c) {
  EACH put<float>(E, d, i,
                  nx_fma_bf16(get<float>(E, a, i), get<float>(E, b, i),
                              get<float>(E, c, i)));
}

#define C1(S, T, U, n)                                                         \
  case (CT_##S << 7) + NX_OP1_##U: k_##n##_##S(E, V, d, a, b, c); return;
#define C2(S, T, U, n)                                                         \
  case (CT_##S << 7) + NX_LOOP_BINARY + NX_OP2_##U:                            \
    k_##n##_##S(E, V, d, a, b, c);                                             \
    return;
#define C3(S, T, U, n)                                                         \
  case (CT_##S << 7) + NX_LOOP_TERNARY + NX_OP3_##U:                           \
    k_##n##_##S(E, V, d, a, b, c);                                             \
    return;
#define D_REAL(S, T)                                                           \
  REAL1F(C1, S, T) REAL1S(C1, S, T) REAL2F(C2, S, T) REAL2S(C2, S, T)          \
  COMPARE(C2, S, T) C3(S, T, FMA, fma)
#define D_INT(S, T)                                                            \
  INT1F(C1, S, T) INT1S(C1, S, T) INT2F(C2, S, T) INT2S(C2, S, T)              \
  COMPARE(C2, S, T) C3(S, T, FMA, fma)
#define D_CPLX(S, T) CPLX1(C1, S, T) CPLX2(C2, S, T) COMPARE(C2, S, T)

/* The kind of op [op] (kernels.h): one switch over every op. The tape's
   runner is its one caller. */
DEVICE void kind(int op, int E, int V, int d, int a, int b, int c) {
  switch (op) {
    D_REAL(f32, float) D_REAL(f64, double) D_CPLX(c64, nx_c64)
    D_CPLX(c128, nx_c128) D_INT(i32, int32_t) D_INT(u32, uint32_t)
    D_INT(i64, int64_t) D_INT(u64, u64) C2(u64, u64, THREEFRY, threefry)
  case (NX_LOOP_F32 << 7) + NX_LOOP_FMA_NARROW:
    k_fma_narrow(E, V, d, a, b, c);
    return;
  case (NX_LOOP_F32 << 7) + NX_LOOP_FMA_BF16:
    k_fma_bf16(E, V, d, a, b, c);
    return;
  }
}

/* Dtypes

   [bits] of an element; the words its held value takes; its bits [r], the
   low bits of a word, as held: a signed integer sign-extended to 32 bits,
   the others as they are; and a held value's bits. */

constexpr int log2_of(int b) { return b <= 1 ? 0 : 1 + log2_of(b / 2); }
#define LOG_BITS(NAME, name, bits, kind) | (u64)log2_of(bits) << (3 * NX_##NAME)
constexpr u64 log_bits = 0 NX_DTYPES(LOG_BITS);
#undef LOG_BITS

DEVICE int bits_of(int dt) { return 1 << ((log_bits >> (3 * dt)) & 7); }

DEVICE int words_of(int dt) {
  int b = bits_of(dt);
  return b <= 32 ? 1 : b / 32;
}

DEVICE u64 of_raw(int dt, u64 r) {
  switch (dt) {
  case NX_INT16: return (uint32_t)(int32_t)(int16_t)r;
  case NX_INT8: return (uint32_t)(int32_t)(int8_t)r;
  case NX_INT4: return (uint32_t)((int32_t)((uint32_t)r << 28) >> 28);
  default: {
    int b = bits_of(dt);
    return b >= 64 ? r : r & ((1ull << b) - 1);
  }
  }
}

DEVICE u64 raw_of(int dt, u64 h) {
  int b = bits_of(dt);
  return b >= 64 ? h : h & ((1ull << b) - 1);
}

/* Casts */

DEVICE bool is_float(int dt) { return dt <= NX_FLOAT4_E2M1FN; }
DEVICE bool is_complex(int dt) {
  return dt == NX_COMPLEX64 || dt == NX_COMPLEX128;
}
DEVICE bool is_bool(int dt) { return dt == NX_BOOL || dt == NX_BIT; }

DEVICE float f32_of(uint32_t b) { return __builtin_bit_cast(float, b); }
DEVICE uint32_t bits_f32(float f) { return __builtin_bit_cast(uint32_t, f); }
DEVICE double f64_of(u64 b) { return __builtin_bit_cast(double, b); }
DEVICE u64 bits_f64(double f) { return __builtin_bit_cast(u64, f); }

/* A float's held [lo] of [dt] as a double, exactly. */
DEVICE double float_value(int dt, u64 lo) {
  if (dt == NX_FLOAT64) return f64_of(lo);
  if (dt == NX_FLOAT32) return f32_of((uint32_t)lo);
  return nx_bits_to_float(dt, (uint32_t)lo);
}

/* [x] held of the real dtype [from] cast into the real dtype [to], held,
   as Prog.Cast stores it: each value rounded once. A float32 that keeps
   its format keeps its bits. */
DEVICE held cast_real(int from, int to, u64 x) {
  held r = {0, 0};
  if (from == to) {
    r.lo = x;
    return r;
  }
  bool integer = !is_float(from);
  int64_t s = 0; /* an integer's value, a uint64's bits */
  if (is_bool(from)) s = x != 0;
  else if (from == NX_INT64 || from == NX_UINT64) s = (int64_t)x;
  else if (is_signed(from)) s = (int32_t)(uint32_t)x;
  else if (integer) s = (int64_t)(uint32_t)x;
  bool unsig = integer && !is_signed(from);
  double f = integer ? 0 : float_value(from, x);
  switch (to) {
  case NX_BOOL:
  case NX_BIT: r.lo = integer ? s != 0 : f != 0; return r;
  case NX_FLOAT64:
    r.lo = bits_f64(!integer ? f : unsig ? (double)(u64)s : (double)s);
    return r;
  case NX_FLOAT32:
    r.lo = bits_f32(!integer ? (float)f : unsig ? (float)(u64)s : (float)s);
    return r;
  }
  if (is_float(to)) {
    /* A float32 or narrower rounds once from its float32; a float64 or an
       integer from its double rounded to odd. */
    if (!integer && from != NX_FLOAT64) r.lo = narrow_code(to, (float)f);
    else if (!integer) r.lo = (uint32_t)nx_double_to_bits(to, f);
    else
      r.lo = (uint32_t)nx_double_to_bits(
          to, unsig ? nx_u64_odd((u64)s) : nx_i64_odd(s));
    return r;
  }
  /* Into an integer: modulo its width, a float toward zero, saturated. */
  r.lo = of_raw(to, integer ? (u64)s : (u64)nx_double_to_bits(to, f));
  return r;
}

DEVICE held cast(int from, int to, held x) {
  if (from == to) return x;
  if (!is_complex(from)) {
    held r = cast_real(from,
                       to == NX_COMPLEX64    ? NX_FLOAT32
                       : to == NX_COMPLEX128 ? NX_FLOAT64
                                             : to,
                       x.lo);
    if (to == NX_COMPLEX128) r.hi = 0;
    return r;
  }
  held r = {0, 0};
  double re, im;
  int part = from == NX_COMPLEX64 ? NX_FLOAT32 : NX_FLOAT64;
  u64 bre = from == NX_COMPLEX64 ? (uint32_t)x.lo : x.lo;
  if (from == NX_COMPLEX64) {
    re = f32_of((uint32_t)x.lo);
    im = f32_of((uint32_t)(x.lo >> 32));
  } else {
    re = f64_of(x.lo);
    im = f64_of(x.hi);
  }
  switch (to) {
  case NX_BOOL:
  case NX_BIT: r.lo = re != 0 || im != 0; return r;
  case NX_COMPLEX64:
    r.lo = bits_f32((float)re) | (u64)bits_f32((float)im) << 32;
    return r;
  case NX_COMPLEX128:
    r.lo = bits_f64(re);
    r.hi = bits_f64(im);
    return r;
  }
  return cast_real(part, to, bre);
}

/* [x] of [from] read in [to], of one width. */
DEVICE held bitcast(int from, int to, held x) {
  if (bits_of(from) == 128) return x;
  held r = {of_raw(to, raw_of(from, x.lo)), 0};
  return r;
}

/* Memory

   Arrays, the workspace and its pieces are addressed as global memory
   (device.h's GLOBAL). */

/* Whether [b]'s 64-bit elements lie on 8 bytes. A complex64 array is
   promised its components' 4 bytes alone: off 8 bytes, each of its
   elements takes two 4-byte accesses. */
DEVICE bool on8(const gbyte *b) { return ((uintptr_t)b & 7) == 0; }

/* The held value of [dt] at element [p] of [b]. */
DEVICE held read(const gbyte *b, int64_t p, int dt) {
  held h = {0, 0};
  switch (bits_of(dt)) {
  case 1: h.lo = (b[p >> 3] >> (p & 7)) & 1; break;
  case 4: h.lo = (b[p >> 1] >> ((p & 1) * 4)) & 15; break;
  case 8: h.lo = b[p]; break;
  case 16: h.lo = ((const GLOBAL uint16_t *)b)[p]; break;
  case 32: h.lo = ((const GLOBAL uint32_t *)b)[p]; break;
  case 64:
    if (on8(b)) h.lo = ((const GLOBAL u64 *)b)[p];
    else
      h.lo = ((const GLOBAL uint32_t *)b)[2 * p] |
             (u64)((const GLOBAL uint32_t *)b)[2 * p + 1] << 32;
    break;
  default:
    if (on8(b)) {
      h.lo = ((const GLOBAL u64 *)b)[2 * p];
      h.hi = ((const GLOBAL u64 *)b)[2 * p + 1];
    } else {
      const GLOBAL uint32_t *q = (const GLOBAL uint32_t *)b + 4 * p;
      h.lo = q[0] | (u64)q[1] << 32;
      h.hi = q[2] | (u64)q[3] << 32;
    }
    return h;
  }
  h.lo = of_raw(dt, h.lo);
  return h;
}

/* Stores [h] of [dt] at element [p] of [b]. */
DEVICE void write(gbyte *b, int64_t p, int dt, held h) {
  int n = bits_of(dt);
  switch (n) {
  case 1:
  case 4: store_bits(b, p, n, (uint32_t)h.lo); return;
  case 8: b[p] = (uint8_t)h.lo; return;
  case 16: ((GLOBAL uint16_t *)b)[p] = (uint16_t)h.lo; return;
  case 32: ((GLOBAL uint32_t *)b)[p] = (uint32_t)h.lo; return;
  case 64:
    if (on8(b)) {
      ((GLOBAL u64 *)b)[p] = h.lo;
      return;
    }
    ((GLOBAL uint32_t *)b)[2 * p] = (uint32_t)h.lo;
    ((GLOBAL uint32_t *)b)[2 * p + 1] = (uint32_t)(h.lo >> 32);
    return;
  default:
    if (on8(b)) {
      ((GLOBAL u64 *)b)[2 * p] = h.lo;
      ((GLOBAL u64 *)b)[2 * p + 1] = h.hi;
      return;
    }
    GLOBAL uint32_t *q = (GLOBAL uint32_t *)b + 4 * p;
    q[0] = (uint32_t)h.lo;
    q[1] = (uint32_t)(h.lo >> 32);
    q[2] = (uint32_t)h.hi;
    q[3] = (uint32_t)(h.hi >> 32);
  }
}

/* Walks */

/* Whether the loop's space is small (Axes, device.h). */
DEVICE bool small_loop(const u64 *w) { return wd(w, NX_LOOP_NUMEL) >> 32 == 0; }

/* The coordinate of [L] along loop axis [k], [L] becoming the index over
   the axes before it. */
DEVICE int64_t split_loop(const u64 *w, int64_t &L, int k, bool small) {
  const u64 *x = w + wd(w, NX_LOOP_SHAPE) + 2 * k;
  return split(L, (int64_t)wd(x, NX_AXIS_N), wd(x, NX_AXIS_DIV), small);
}

/* Index [L]'s coordinate along loop axis [a]. */
DEVICE int64_t coordinate(const u64 *w, int64_t L, int a) {
  const int rank = (int)wd(w, NX_LOOP_RANK);
  const bool small = small_loop(w);
  for (int k = rank - 1; k > a; k--) split_loop(w, L, k, small);
  return split_loop(w, L, a, small);
}

/* The position of a plain operand's element at loop index [L]: its
   descriptor [d], the first position then a stride per axis. */
DEVICE int64_t place(const u64 *w, const u64 *d, int64_t L) {
  const int rank = (int)wd(w, NX_LOOP_RANK);
  const bool small = small_loop(w);
  const int64_t *st = (const int64_t *)d + 2;
  int64_t at = (int64_t)wd(d, 1);
  if (rank == 0) return at;
  for (int k = rank - 1; k > 0; k--) at += split_loop(w, L, k, small) * st[k];
  return at + L * st[0];
}

/* The position a padded operand reads at loop index [L], or -1 where it
   reads its fill: per axis of the operand, the padded coordinate from the
   loop axes it reads less lo, where it lands on an element. */
DEVICE int64_t padded_place(const u64 *w, const u64 *d, int64_t L) {
  int rank = (int)wd(w, NX_LOOP_RANK), r = (int)wd(d, 4);
  const bool small = small_loop(w);
  int64_t at = (int64_t)wd(d, 1);
  for (int j = 0; j < r; j++) {
    const int64_t *a = (const int64_t *)d + 5 + NX_LOOP_PAD * j;
    int a1 = (int)(a[4] & 0xffff), a2 = (int)((a[4] >> 16) & 0xffff);
    int low = a2 == 0xffff || a1 < a2 ? a1 : a2;
    int64_t q = -a[0], l = L;
    for (int k = rank - 1; k >= low; k--) {
      const int64_t c = split_loop(w, l, k, small);
      if (k == a1) q += c * a[5];
      if (k == a2) q += c * a[6];
    }
    /* An axis without an interior takes no division. */
    int64_t s = a[1] + 1;
    if (s > 1) {
      if (q < 0 || q % s != 0) return -1;
      q /= s;
    }
    if (q < 0 || q >= a[2]) return -1;
    at += q * a[3];
  }
  return at;
}

DEVICE const u64 *operand(const u64 *w, int table, int k) {
  return w + wd(w, wd(w, table) + k);
}

/* The address of descriptor [d]'s buffer: its address word. */
DEVICE gbyte *buffer(const u64 *w, const u64 *d) {
  return (gbyte *)(uintptr_t)wd(w, NX_LOOP_HEADER + ((wd(d, 0) >> 16) & 0xffff));
}

/* Load [k]'s held element at loop index [L]. */
DEVICE held load_at(const u64 *w, int k, int64_t L) {
  const u64 *d = operand(w, NX_LOOP_LOADS, k);
  int dt = (int)(wd(d, 0) & 0xff);
  if ((wd(d, 0) >> 8) & 1) {
    int64_t p = padded_place(w, d, L);
    if (p < 0) {
      held f = {of_raw(dt, wd(d, 2)), wd(d, 3)};
      return f;
    }
    return read(buffer(w, d), p, dt);
  }
  return read(buffer(w, d), place(w, d, L), dt);
}

/* A plain operand's bytes along the work-item's elements, by its
   descriptor [d] and its elements' [bytes], for a loop of at most two
   axes: element v's byte steps from the first index's by the step's
   coordinates, a division a call, its byte at + v's sum of the steps,
   plus [wrap] each time the last axis's coordinate passes its extent. */
typedef struct {
  int64_t at, step, wrap, c, dc, e;
} walker;

DEVICE walker walker_of(const u64 *w, const u64 *d, lanes x, int bytes) {
  int rank = (int)wd(w, NX_LOOP_RANK);
  walker k;
  int64_t s0 = rank > 0 ? (int64_t)wd(d, 2) * bytes : 0;
  if (rank <= 1) {
    k.at = (int64_t)wd(d, 1) * bytes + x.L0 * s0;
    k.step = x.step * s0;
    k.wrap = 0;
    k.c = 0;
    k.dc = 0;
    k.e = 1;
    return k;
  }
  int64_t s1 = (int64_t)wd(d, 3) * bytes;
  const bool small = small_loop(w);
  int64_t r = x.L0, dr = x.step;
  k.c = split_loop(w, r, 1, small);
  k.dc = split_loop(w, dr, 1, small);
  k.e = (int64_t)wd(w, wd(w, NX_LOOP_SHAPE) + 2 + NX_AXIS_N);
  k.at = (int64_t)wd(d, 1) * bytes + r * s0 + k.c * s1;
  k.step = dr * s0 + k.dc * s1;
  k.wrap = s0 - k.e * s1;
  return k;
}

/* The walker's byte, then the next element's. */
DEVICE int64_t next(walker &k) {
  int64_t at = k.at;
  k.at += k.step;
  k.c += k.dc;
  if (k.c >= k.e) {
    k.c -= k.e;
    k.at += k.wrap;
  }
  return at;
}

/* Load [k]'s elements at the work-item's indices into plane [d], each in
   turn, an element of no index taking zeros: a padded load's, a load of
   fewer than 8 bits' or a loop's of more than two axes. */
DEVICE void load_each(const u64 *w, int E, int V, lanes x, int d, int k) {
  const u64 *desc = operand(w, NX_LOOP_LOADS, k);
  int n = words_of((int)(wd(desc, 0) & 0xff));
  EACH {
    held h = {0, 0};
    if (v < x.n) h = load_at(w, k, index_of(x, v));
    put_held(E, d, i, n, h);
  }
}

/* Load [k]'s elements at the work-item's indices into plane [d], an
   element of no index taking zeros. Along a walker, a word plane at a
   time, every element's read before any is stored, so that the reads are
   in flight together; elements of 32 bits or more as their 32-bit words,
   since a complex64's lie on 4 bytes alone. */
DEVICE void load(const u64 *w, int E, int V, lanes x, int d, int k) {
  const u64 *desc = operand(w, NX_LOOP_LOADS, k);
  int dt = (int)(wd(desc, 0) & 0xff), n = words_of(dt), bits = bits_of(dt);
  if (((wd(desc, 0) >> 8) & 1) || bits < 8 || wd(w, NX_LOOP_RANK) > 2) {
    load_each(w, E, V, x, d, k);
    return;
  }
  const gbyte *b = buffer(w, desc);
  walker q = walker_of(w, desc, x, bits >= 32 ? 4 * n : bits / 8);
  uint32_t y[NX_LOOP_MOST] = {};
  if (bits >= 32) {
    _Pragma("unroll 1") for (int j = 0; j < n; j++) {
      walker r = q;
      ALL {
        int64_t at = next(r) + 4 * j;
        y[v] = v < x.n ? *(const GLOBAL uint32_t *)(b + at) : 0;
      }
      ALL word(E, d + j, i) = y[v];
    }
    return;
  }
  if (bits == 16) {
    ALL {
      int64_t at = next(q);
      if (v < x.n) y[v] = *(const GLOBAL uint16_t *)(b + at);
    }
  } else {
    ALL {
      int64_t at = next(q);
      if (v < x.n) y[v] = b[at];
    }
  }
  ALL word(E, d, i) = (uint32_t)of_raw(dt, y[v]);
}

/* Stores plane [a]'s elements at the work-item's indices into destination
   [k], placed as load places them: along a walker as load reads them,
   elements of 8 bits or more of a loop of at most two axes, else each in
   turn. */
DEVICE void store(const u64 *w, int E, int V, lanes x, int a, int k) {
  const u64 *desc = operand(w, NX_LOOP_DSTS, k);
  int dt = (int)(wd(desc, 0) & 0xff), n = words_of(dt), bits = bits_of(dt);
  gbyte *b = buffer(w, desc);
  if (bits < 8 || wd(w, NX_LOOP_RANK) > 2) {
    EACH if (v < x.n) write(b, place(w, desc, index_of(x, v)), dt,
                            get_held(E, a, i, n));
    return;
  }
  walker q = walker_of(w, desc, x, bits >= 32 ? 4 * n : bits / 8);
  uint32_t y[NX_LOOP_MOST] = {};
  if (bits >= 32) {
    _Pragma("unroll 1") for (int j = 0; j < n; j++) {
      walker r = q;
      ALL y[v] = word(E, a + j, i);
      ALL {
        int64_t at = next(r) + 4 * j;
        if (v < x.n) *(GLOBAL uint32_t *)(b + at) = y[v];
      }
    }
    return;
  }
  ALL y[v] = word(E, a, i);
  if (bits == 16) {
    ALL {
      int64_t at = next(q);
      if (v < x.n) *(GLOBAL uint16_t *)(b + at) = (uint16_t)y[v];
    }
  } else {
    ALL {
      int64_t at = next(q);
      if (v < x.n) b[at] = (uint8_t)y[v];
    }
  }
}

/* The tape */

/* Instruction [op] of planes [d], [a], [b] and [c] over each element in
   turn: the runner's rarer instructions. */
DEVICE void each_op(const u64 *w, int E, int V, lanes x, int op, int d, int a,
                    int b, int c) {
  switch (op) {
  case NX_LOOP_COORD:
    EACH put<int64_t>(E, d, i, v < x.n ? coordinate(w, index_of(x, v), a) : 0);
    return;
  case NX_LOOP_CAST:
    EACH put_held(E, d, i, words_of(c),
                  cast(b, c, get_held(E, a, i, words_of(b))));
    return;
  case NX_LOOP_BITCAST:
    EACH put_held(E, d, i, words_of(c),
                  bitcast(b, c, get_held(E, a, i, words_of(b))));
    return;
  case NX_LOOP_DECODE:
    EACH put<float>(E, d, i, nx_bits_to_float(b, word(E, a, i)));
    return;
  case NX_LOOP_ENCODE:
    EACH word(E, d, i) = narrow_code(b, get<float>(E, a, i));
    return;
  case NX_LOOP_WRAP:
    EACH word(E, d, i) = (uint32_t)of_raw(b, word(E, a, i));
    return;
  }
}

/* Writes the prologue's constants into their planes, before any tile: a
   prologue holds NX_LOOP_CONST instructions alone. */
DEVICE void constants(const u64 *w, int E, int V) {
  int from = (int)wd(w, NX_LOOP_TAPE), to = from + (int)wd(w, NX_LOOP_PROLOGUE);
  for (int at = from; at < to; at += 3) {
    u64 t = wd(w, at);
    int d = (int)((t >> 16) & 0xfff), n = (int)((t >> 40) & 0xfff);
    held h = {wd(w, at + 1), wd(w, at + 2)};
    EACH put_held(E, d, i, n, h);
  }
}

/* Runs the tape's [n] words from [from]: its loads, stores and
   selections in line, the kinds through their switch. */
DEVICE void run(const u64 *w, int E, int V, lanes x, int from, int n) {
  for (int at = from; at < from + n; at++) {
    u64 t = wd(w, at);
    int op = (int)(t & 0xffff), d = (int)((t >> 16) & 0xfff);
    int a = (int)((t >> 28) & 0xfff), b = (int)((t >> 40) & 0xfff);
    int c = (int)((t >> 52) & 0xfff);
    if (op < NX_LOOP_LOAD) {
      kind(op, E, V, d, a, b, c);
      continue;
    }
    switch (op) {
    case NX_LOOP_LOAD: load(w, E, V, x, d, a); break;
    case NX_LOOP_STORE: store(w, E, V, x, a, b); break;
    case NX_LOOP_WHERE1: {
      uint32_t s[NX_LOOP_MOST] = {}, y[NX_LOOP_MOST] = {}, z[NX_LOOP_MOST] = {};
      ALL {
        s[v] = word(E, a, i);
        y[v] = word(E, b, i);
        z[v] = word(E, c, i);
      }
      ALL word(E, d, i) = s[v] != 0 ? y[v] : z[v];
      break;
    }
    case NX_LOOP_WHERE2:
    case NX_LOOP_WHERE4: {
      int m = op == NX_LOOP_WHERE2 ? 2 : 4;
      EACH put_held(E, d, i, m, get_held(E, word(E, a, i) != 0 ? b : c, i, m));
      break;
    }
    case NX_LOOP_TRUTH: {
      uint32_t y[NX_LOOP_MOST] = {};
      ALL y[v] = word(E, a, i);
      ALL word(E, d, i) = y[v] != 0;
      break;
    }
    case NX_LOOP_DECODE:
    case NX_LOOP_ENCODE: {
      if (b != NX_BFLOAT16 && b != NX_FLOAT16) {
        each_op(w, E, V, x, op, d, a, b, c);
        break;
      }
      uint32_t y[NX_LOOP_MOST] = {};
      ALL y[v] = word(E, a, i);
      if (op == NX_LOOP_DECODE && b == NX_BFLOAT16) {
        ALL put<float>(E, d, i, nx_bf16_to_float((uint16_t)y[v]));
      } else if (op == NX_LOOP_DECODE) {
        ALL put<float>(E, d, i, nx_f16_to_float((uint16_t)y[v]));
      } else if (b == NX_BFLOAT16) {
        ALL word(E, d, i) = nx_float_to_bf16(f32_of(y[v]));
      } else {
        ALL word(E, d, i) = nx_float_to_f16(f32_of(y[v]));
      }
      break;
    }
    default: each_op(w, E, V, x, op, d, a, b, c);
    }
  }
}

/* States

   A state of an output's terms, held in NX_LOOP_STATE planes or workspace
   words: its value in words 0 to 3, held at its compute type; a position
   or a second sum in 4 and 5; whether it holds a term in 6. A sum's lanes
   start from +0 and every lane takes part in the tree, as nx.cpu's do: a
   lane from +0 is never -0, so a lane of no term, which adds +0, changes
   no bit. Another monoid's lane, and Arg's, start from their first term,
   a lane of no term taking no part.

   Two sites that may combine the same pair of NaNs are the lanes' folds
   and their trees, and a scan's chain and its carries. Each site is a
   function of the dtypes and the extents alone, and each float combine
   chooses its NaN explicitly ([kept]): an instruction keeps its first
   operand's NaN, and the compiler may commute an add's or a multiply's
   operands. */

typedef struct {
  held v;
  u64 x;
  uint32_t has;
} st;

template <class T> DEVICE T as(held h) { return bits_as<T>(h.lo, h.hi); }

template <class T> DEVICE held of(T x) {
  u64 b[2] = {0, 0};
  memcpy(b, &x, sizeof(T));
  held h = {b[0], b[1]};
  return h;
}

/* A complex combine's result [r] ({!kept}): a part that is a NaN takes
   the first NaN of [a]'s parts then [b]'s. Maximum and Minimum choose
   theirs already. */
template <class T, class C> DEVICE C kept_c(C a, C b, C r) {
  T n = a.re != a.re ? a.re
        : a.im != a.im ? a.im
        : b.re != b.re ? b.re
                       : b.im;
  r.re = kept<T>(n, n, r.re);
  r.im = kept<T>(n, n, r.im);
  return r;
}

/* [a] op [b] at the compute type [ct], op NX_OP2_ADD, NX_OP2_MUL,
   NX_OP2_MAXIMUM or NX_OP2_MINIMUM, through [K] the NaN it keeps. */
#define ARITH(S, T, K)                                                         \
  {                                                                            \
    T x = as<T>(a), y = as<T>(b);                                              \
    switch (op) {                                                              \
    case NX_OP2_ADD: return of<T>(K(x, y, nx_add_##S(x, y)));                  \
    case NX_OP2_MUL: return of<T>(K(x, y, nx_mul_##S(x, y)));                  \
    case NX_OP2_MAXIMUM: return of<T>(nx_maximum_##S(x, y));                   \
    default: return of<T>(nx_minimum_##S(x, y));                               \
    }                                                                          \
  }

/* Integers keep their result. */
#define EXACT(x, y, r) (r)

DEVICE held arith(int ct, int op, held a, held b) {
  switch (ct) {
  case NX_LOOP_F32: ARITH(f32, float, kept<float>)
  case NX_LOOP_F64: ARITH(f64, double, kept<double>)
  case NX_LOOP_C64: ARITH(c64, nx_c64, (kept_c<float, nx_c64>))
  case NX_LOOP_C128: ARITH(c128, nx_c128, (kept_c<double, nx_c128>))
  case NX_LOOP_I32: ARITH(i32, int32_t, EXACT)
  case NX_LOOP_U32: ARITH(u32, uint32_t, EXACT)
  case NX_LOOP_I64: ARITH(i64, int64_t, EXACT)
  default: ARITH(u64, u64, EXACT)
  }
}

#undef ARITH
#undef EXACT

/* Arg's comparison of [a] and [b] at [ct]: bit 0 if [a] is a NaN, bit 1
   if [b] is, and, neither a NaN, bit 2 if [a] follows [b] in the dtype's
   order and bit 3 if [b] follows [a]. */
#define ORDER(S, T)                                                            \
  {                                                                            \
    T x = as<T>(a), y = as<T>(b);                                              \
    return (int)(x != x) | (int)(y != y) << 1 |                                \
           (int)(nx_order_##S(x, y) > 0) << 2 |                                \
           (int)(nx_order_##S(y, x) > 0) << 3;                                 \
  }
#define ORDER_C(S, T)                                                          \
  {                                                                            \
    T x = as<T>(a), y = as<T>(b);                                              \
    return (int)nx_nan_##S(x) | (int)nx_nan_##S(y) << 1 |                      \
           (int)(nx_order_##S(x, y) > 0) << 2 |                                \
           (int)(nx_order_##S(y, x) > 0) << 3;                                 \
  }
#define ORDER_I(T)                                                             \
  {                                                                            \
    T x = as<T>(a), y = as<T>(b);                                              \
    return (int)(x > y) << 2 | (int)(y > x) << 3;                              \
  }

DEVICE int order(int ct, held a, held b) {
  switch (ct) {
  case NX_LOOP_F32: ORDER(f32, float)
  case NX_LOOP_F64: ORDER(f64, double)
  case NX_LOOP_C64: ORDER_C(c64, nx_c64)
  case NX_LOOP_C128: ORDER_C(c128, nx_c128)
  case NX_LOOP_I32: ORDER_I(int32_t)
  case NX_LOOP_U32: ORDER_I(uint32_t)
  case NX_LOOP_I64: ORDER_I(int64_t)
  default: ORDER_I(u64)
  }
}

#undef ORDER
#undef ORDER_C
#undef ORDER_I

/* Arg's state [a] then [b]: the extreme, a NaN first, and of one value
   the earlier position. */
DEVICE st arg_merge(int rk, int ct, st a, st b) {
  if (!b.has) return a;
  bool take;
  if (!a.has) take = true;
  else {
    int o = order(ct, a.v, b.v);
    bool an = o & 1, bn = o & 2;
    bool b_first = rk == NX_LOOP_ARG_MAX ? o & 8 : o & 4;
    bool a_first = rk == NX_LOOP_ARG_MAX ? o & 4 : o & 8;
    if (an || bn) take = an && bn ? (int64_t)b.x < (int64_t)a.x : bn;
    else if (b_first) take = true;
    else if (a_first) take = false;
    else take = (int64_t)b.x < (int64_t)a.x;
  }
  return take ? b : a;
}

/* State [a] then state [b], of reduction [rk] at [ct]. */
DEVICE st merge(int rk, int ct, st a, st b) {
  switch (rk) {
  case NX_LOOP_SUM:
  case NX_LOOP_EXPS: a.v = arith(ct, NX_OP2_ADD, a.v, b.v); return a;
  case NX_LOOP_SQUARES: {
    held s = {a.x, 0}, t = {b.x, 0};
    a.v = arith(ct, NX_OP2_ADD, a.v, b.v);
    a.x = arith(ct, NX_OP2_ADD, s, t).lo;
    return a;
  }
  case NX_LOOP_ARG_MAX:
  case NX_LOOP_ARG_MIN: return arg_merge(rk, ct, a, b);
  }
  if (!b.has) return a;
  if (!a.has) return b;
  int op = rk == NX_LOOP_PROD  ? NX_OP2_MUL
           : rk == NX_LOOP_MAX ? NX_OP2_MAXIMUM
                               : NX_OP2_MINIMUM;
  a.v = arith(ct, op, a.v, b.v);
  return a;
}

/* Logsumexp's exp of [y], 0 below the cut (nx_amd.mli). */
DEVICE float ex(float y) { return y < -87.f ? 0.f : nx_exp_f32(y); }
DEVICE double ex(double y) { return y < -708. ? 0. : nx_exp_f64(y); }

/* A centered term [x] of center [c]: exp (x - c), or (x - c)^2 with the
   second sum x - c. */
template <class T> DEVICE st centered(int rk, held x, held c) {
  T d = as<T>(x) - as<T>(c);
  st b;
  b.has = 1;
  b.x = 0;
  if (rk == NX_LOOP_EXPS) b.v = of<T>(ex(d));
  else {
    b.v = of<T>(d * d);
    b.x = of<T>(d).lo;
  }
  return b;
}

/* State [a] then the term [x] at position [pos], of center [c]. */
DEVICE st fold(int rk, int ct, st a, held x, int64_t pos, held c) {
  st b = {x, (u64)pos, 1};
  if (rk == NX_LOOP_EXPS || rk == NX_LOOP_SQUARES)
    b = ct == NX_LOOP_F32 ? centered<float>(rk, x, c)
                          : centered<double>(rk, x, c);
  return merge(rk, ct, a, b);
}

DEVICE st empty(int rk) {
  st s = {{0, 0}, 0, 0};
  s.has = rk == NX_LOOP_SUM || rk == NX_LOOP_EXPS || rk == NX_LOOP_SQUARES;
  return s;
}

DEVICE st load_state(int E, int p, int i) {
  st s = {{get64(E, p, i), get64(E, p + 2, i)}, get64(E, p + 4, i),
          word(E, p + 6, i)};
  return s;
}

DEVICE void store_state(int E, int p, int i, st s) {
  put64(E, p, i, s.v.lo);
  put64(E, p + 2, i, s.v.hi);
  put64(E, p + 4, i, s.x);
  word(E, p + 6, i) = s.has;
}

/* A state in the workspace, or in LDS, NX_LOOP_STATE words [g] as in
   planes. */
template <class W> DEVICE u64 get_work64(W g) { return g[0] | (u64)g[1] << 32; }

template <class W> DEVICE void put_work64(W g, u64 x) {
  g[0] = (uint32_t)x;
  g[1] = (uint32_t)(x >> 32);
}

template <class W> DEVICE st load_work(W g) {
  st s = {{get_work64(g), get_work64(g + 2)}, get_work64(g + 4), g[6]};
  return s;
}

template <class W> DEVICE void store_work(W g, st s) {
  put_work64(g, s.v.lo);
  put_work64(g + 2, s.v.hi);
  put_work64(g + 4, s.x);
  g[6] = s.has;
}

/* A value at [ct] as its carrier holds it, an integer wrapped to it; and
   a carrier's held value at [ct], a boolean as 0 or 1. */
DEVICE held to_carrier(int carrier, int ct, held x) {
  if (ct >= NX_LOOP_I32) x.lo = of_raw(carrier, x.lo);
  return x;
}

DEVICE held of_carrier(int carrier, held h) {
  if (is_bool(carrier)) h.lo = h.lo != 0;
  return h;
}

/* The workspace's byte [field] of the header, then state [k] past it. */
DEVICE gbyte *work_byte(const u64 *w, int field) {
  return (gbyte *)(uintptr_t)wd(w, NX_LOOP_HEADER + wd(w, NX_LOOP_WORK)) +
         wd(w, field);
}

DEVICE GLOBAL uint32_t *work_at(const u64 *w, int field, int64_t k) {
  return (GLOBAL uint32_t *)work_byte(w, field) + k * NX_LOOP_STATE;
}

/* Stores [x] of the dtype [from] cast into destination [k] at its element
   [o] past its first. */
DEVICE void put_dst(const u64 *w, int k, int64_t o, int from, held x) {
  const u64 *d = operand(w, NX_LOOP_DSTS, k);
  int dt = (int)(wd(d, 0) & 0xff);
  write(buffer(w, d), (int64_t)wd(d, 1) + o, dt, cast(from, dt, x));
}

template <class T> DEVICE T log_(T x) {
  if constexpr (sizeof(T) == 4) return nx_log_f32(x);
  else return nx_log_f64(x);
}

template <class T> DEVICE T max_(T a, T b) {
  if constexpr (sizeof(T) == 4) return nx_maximum_f32(a, b);
  else return nx_maximum_f64(a, b);
}

/* Logsumexp's and Moments' finishes of output [o] of [n] terms from its
   state [s], by [fin], at the float T. */
template <class T>
DEVICE void finish_real(const u64 *w, int fin, int carrier, int64_t o,
                        int64_t n, st s) {
  GLOBAL T *centers = (GLOBAL T *)work_byte(w, NX_LOOP_CENTERS);
  T c = centers[o], v = as<T>(s.v);
  switch (fin) {
  case NX_LOOP_PUT_CENTER: centers[o] = v - v == 0 ? v : T(0); return;
  case NX_LOOP_PUT_MEAN: centers[o] = v / (T)n; return;
  case NX_LOOP_PUT_LOGSUMEXP:
    put_dst(w, 0, o, carrier, of<T>(T(c + log_(v))));
    return;
  default: {
    held sh = {s.x, 0};
    T tn = (T)n, d = as<T>(sh);
    T u = (v - d * (d / tn)) / tn;
    T var = c != c ? c : v == (T)INFINITY ? v : u < 0 ? T(0) : u;
    put_dst(w, 0, o, carrier, of<T>(c));
    put_dst(w, 1, o, carrier, of<T>(var));
  }
  }
}

/* What output [o] of [n] terms takes from its state [s], by [fin]. */
DEVICE void finish(const u64 *w, int rk, int ct, int fin, int carrier,
                   int64_t o, int64_t n, st s) {
  switch (fin) {
  case NX_LOOP_PUT:
    put_dst(w, 0, o, carrier, to_carrier(carrier, ct, s.v));
    if (rk == NX_LOOP_ARG_MAX || rk == NX_LOOP_ARG_MIN) {
      held p = {s.x, 0};
      put_dst(w, 1, o, NX_INT64, p);
    }
    return;
  case NX_LOOP_PUT_TOTAL: store_work(work_at(w, NX_LOOP_TOTALS, o), s); return;
  }
  if (ct == NX_LOOP_F32) finish_real<float>(w, fin, carrier, o, n, s);
  else finish_real<double>(w, fin, carrier, o, n, s);
}

/* Reductions */

/* Each element's term, of the term plane, folded into its state, its
   position its index past [row], its row's first. A sum adds every
   element's, an element of no index adding +0. */
DEVICE void fold_all(const u64 *w, int rk, int ct, int E, int V, lanes x,
                     int64_t row) {
  int tp = (int)wd(w, NX_LOOP_TERM);
  int sp = (int)wd(w, NX_LOOP_STATE_AT), cp = (int)wd(w, NX_LOOP_CENTER);
  bool c = rk == NX_LOOP_EXPS || rk == NX_LOOP_SQUARES;
  held z = {0, 0};
  EACH {
    if (rk == NX_LOOP_SUM) {
      held t = v < x.n ? get_held(E, tp, i, 4) : z;
      put_held(E, sp, i, 4, arith(ct, NX_OP2_ADD, get_held(E, sp, i, 4), t));
      continue;
    }
    if (v >= x.n) continue;
    st s = fold(rk, ct, load_state(E, sp, i), get_held(E, tp, i, 4),
                index_of(x, v) - row, c ? get_held(E, cp, i, 2) : z);
    store_state(E, sp, i, s);
  }
}

/* Element [i]'s state then element [j]'s, into [i]'s. */
DEVICE void merge_at(int rk, int ct, int E, int sp, int i, int j) {
  store_state(E, sp, i,
              merge(rk, ct, load_state(E, sp, i), load_state(E, sp, j)));
}

/* An output's group's state, element [i]'s: with one group, it goes
   out; with [m], it goes into the workspace as part [q], and whether the
   group is the last of the [m] to arrive is the answer. */
DEVICE bool put_group(const u64 *w, int rk, int ct, int fin, int carrier,
                      int E, int i, int64_t o, int64_t q, int64_t m,
                      int64_t n) {
  st s = load_state(E, (int)wd(w, NX_LOOP_STATE_AT), i);
  if (m == 1) {
    finish(w, rk, ct, fin, carrier, o, n, s);
    return false;
  }
  store_work(work_at(w, NX_LOOP_PARTIALS, o * m + q), s);
  uint32_t *tickets =
      (uint32_t *)(uintptr_t)wd(w, NX_LOOP_HEADER + wd(w, NX_LOOP_TICKETS));
  return last_alone(tickets + o, (uint32_t)m);
}

/* Output [o]'s [m] parts merged by the pairwise tree, in place, the
   group's work-items merging a level's pairs side by side, then sent out.
   Every work-item of the group calls it. */
DEVICE void put_parts(const u64 *w, int rk, int ct, int fin, int carrier,
                      int64_t o, int64_t m, int64_t n) {
  GLOBAL uint32_t *g = work_at(w, NX_LOOP_PARTIALS, o * m);
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  for (int64_t step = 1; step < m; step *= 2) {
    for (int64_t k = 2 * step * item(); k + step < m; k += 2 * step * TB)
      store_work(g + k * NX_LOOP_STATE,
                 merge(rk, ct, load_work(g + k * NX_LOOP_STATE),
                       load_work(g + (k + step) * NX_LOOP_STATE)));
    barrier();
  }
  if (item() == 0) finish(w, rk, ct, fin, carrier, o, n, load_work(g));
}

/* A run's terms of a block of [G] work-items, 16 or more, [R1] terms from
   the index [L0]: work-item l below 16 folds lane l's, terms l + 16 c, of
   work-item (l + 16 c) % G's element (l + 16 c) / G, those below [end],
   in order into its state past the planes, a term's position its index
   past [row]. Every work-item of the group calls it. */
DEVICE void fold_lanes(const u64 *w, int rk, int ct, int E, int G, int R1,
                       int64_t L0, int64_t end, int64_t row) {
  int l = (int)item() & (G - 1), at = (int)item() - l;
  int lg = __builtin_ctz((unsigned)G);
  if (G > NX_LOOP_LANES) barrier();
  int64_t r = end - L0 - l;
  int n = l >= NX_LOOP_LANES || r <= 0
              ? 0
              : (int)imin((int64_t)(R1 / NX_LOOP_LANES), (r + 15) / 16);
  int tp = (int)wd(w, NX_LOOP_TERM), sp = (int)wd(w, NX_LOOP_STATE_AT);
  int cp = (int)wd(w, NX_LOOP_CENTER);
  bool c = rk == NX_LOOP_EXPS || rk == NX_LOOP_SQUARES;
  held z = {0, 0};
  if (n > 0) {
    st s = load_state(TB, sp, (int)item());
    for (int j = 0; j < n; j++) {
      int k = l + 16 * j, i = (k >> lg) * TB + at + (k & (G - 1));
      s = fold(rk, ct, s, get_held(E, tp, i, 4), L0 - row + k,
               c ? get_held(E, cp, i, 2) : z);
    }
    store_state(TB, sp, (int)item(), s);
  }
  if (G > NX_LOOP_LANES) barrier();
}

/* A reduction: units of an output's group of [g] blocks, [per] units a
   round of a group of work-items. */
DEVICE void reduce_mode(const u64 *w, int E, int V) {
  int G = (int)((wd(w, NX_LOOP_MODE) >> 16) & 0x1ff), LB = TB / G, R1 = G * V;
  int64_t g = (int64_t)wd(w, NX_LOOP_GROUP), m = (int64_t)wd(w, NX_LOOP_GROUPS);
  int64_t O = (int64_t)wd(w, NX_LOOP_OUTPUTS), R = (int64_t)wd(w, NX_LOOP_TERMS);
  int64_t SEG = (int64_t)wd(w, NX_LOOP_SEGMENT);
  int64_t segs = (int64_t)wd(w, NX_LOOP_SEGMENTS), runs = (int64_t)wd(w, NX_LOOP_RUNS);
  int64_t per = LB / g, units = O * m, grid = (int64_t)wd(w, NX_LOOP_GRID);
  u64 r = wd(w, NX_LOOP_REDUCTION);
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff), fin = (int)((r >> 24) & 0xff);
  /* States: an element's, at [SE] = E, where a work-item holds lanes;
     else a work-item's, at SE = TB, past the planes. */
  int sp = (int)wd(w, NX_LOOP_STATE_AT), SE = G < NX_LOOP_LANES ? E : TB;
  int me = (int)item(), s = me / G, t = me % G;
  int body = (int)(wd(w, NX_LOOP_TAPE) + wd(w, NX_LOOP_PROLOGUE));
  int nbody = (int)wd(w, NX_LOOP_BODY);
  for (int64_t r0 = group_x() * per; r0 < units; r0 += grid * per) {
    /* A round's units side by side: slot s is block s / per of unit
       s % per. */
    int64_t k = s / per, u = r0 + s % per, o = u / m, q = u % m;
    int64_t b = q * g + k;
    int64_t start = (o % segs) * SEG, len = imin(SEG, R - start);
    {
      st e = empty(rk);
      int SV = SE / TB;
      for (int v = 0, i = me; v < SV; v++, i += TB) store_state(SE, sp, i, e);
    }
    int64_t row = o / segs * R, end = u < units ? row + start + len : 0;
    for (int64_t j = 0; j < runs; j++) {
      int64_t L0 = row + start + b * NX_LOOP_BLOCK + j * R1;
      lanes x = lanes_of(L0 + t, G, end, V);
      run(w, E, V, x, body, nbody);
      if (R1 <= NX_LOOP_LANES) fold_all(w, rk, ct, E, V, x, row);
      else fold_lanes(w, rk, ct, E, G, R1, L0, end, row);
    }
    barrier();
    /* The lanes' tree: lane l takes lane l + h for l below h; lanes past
       a run's R1 hold no term and take no part. */
    for (int h = (R1 < NX_LOOP_LANES ? R1 : NX_LOOP_LANES) / 2; h > 0;
         h /= 2) {
      if (h >= G)
        for (int v = 0; v < h / G; v++)
          merge_at(rk, ct, SE, sp, v * TB + me, (v + h / G) * TB + me);
      else if (t < h)
        merge_at(rk, ct, SE, sp, me, me + h);
      barrier();
    }
    /* The group's blocks by the pairwise tree, a slot of no block taking
       no part. */
    int64_t nb = (len + NX_LOOP_BLOCK - 1) / NX_LOOP_BLOCK;
    int64_t count = u < units ? imin(g, nb - q * g) : 0;
    for (int64_t step = 1; step < g; step *= 2) {
      if (t == 0 && k % (2 * step) == 0 && k + step < count)
        merge_at(rk, ct, SE, sp, me, me + (int)(step * per) * G);
      barrier();
    }
    /* A unit whose group arrives last at its output: the group merges
       the output's parts. */
    static SHARED bool last[TB];
    if (t == 0 && k == 0)
      last[s] =
          u < units && put_group(w, rk, ct, fin, carrier, SE, me, o, q, m, len);
    barrier();
    for (int p = 0; p < per; p++) {
      if (!last[p]) continue;
      int64_t op = (r0 + p) / m, at = (op % segs) * SEG;
      put_parts(w, rk, ct, fin, carrier, op, m, imin(SEG, R - at));
    }
    barrier();
  }
}

/* Plain reductions

   A Sum, Max or Min of one plain load at its carrier, its terms read
   straight into registers in nx.cpu's lane order, 16 loads a work-item in
   flight. A lane starts from the reduction's identity, which leaves the
   value it is folded into as it was (x + +0 of a lane from +0, max x
   -inf, min x +inf), so a term or a lane of none takes no part.

   Rows, where an output's terms lie side by side: sixteen work-items, a
   cell, take a block, work-item l holding lane l, and the lanes' tree is
   four permutes. A reduction of fewer blocks than a group's work-items
   runs one cell a group, its blocks spread over as many groups. A
   group's blocks of an output merge by the pairwise tree in LDS; an output of several groups leaves each group's value in the
   workspace, and the group that takes its ticket last merges them by the
   same tree, in LDS, in aligned chunks of NX_LOOP_PLAIN_CHUNK.

   Columns, where neighbouring outputs lie side by side: a group takes 32
   of them, a wave reading a row's 32 at once, work-item t of wave w
   holding lanes w and w + 8 of output t; it takes each of their blocks in
   turn, at most NX_LOOP_PLAIN_TALL, and merges them in LDS.

   Values are their bits in a word, of 4 or 8 bytes. The folds are typed;
   the walks, the trees and the merges are not, and combine values by
   plain_op. A float sum chooses its NaN as merge does ([kept]). */

/* The bits of the identity of the reduction [rk] at [ct]. */
DEVICE u64 plain_id(int ct, int rk) {
  if (rk == NX_LOOP_SUM) return 0;
  bool max = rk == NX_LOOP_MAX;
  switch (ct) {
  case NX_LOOP_F32: return max ? 0xff800000u : 0x7f800000u;
  case NX_LOOP_F64: return max ? 0xfff0000000000000ull : 0x7ff0000000000000ull;
  case NX_LOOP_I32: return max ? 0x80000000u : 0x7fffffffu;
  case NX_LOOP_U32: return max ? 0 : 0xffffffffu;
  case NX_LOOP_I64: return max ? 1ull << 63 : ~0ull >> 1;
  default: return max ? 0 : ~0ull;
  }
}

/* X(code, T, S, K) for each plain type: its compute type's code, its C
   type, nx_kinds.h's name of it and the NaN its sum keeps. */
#define EXACT(x, y, r) (r)
#define PLAIN_TYPES(X)                                                         \
  X(NX_LOOP_F32, float, f32, kept<float>)                                      \
  X(NX_LOOP_F64, double, f64, kept<double>)                                    \
  X(NX_LOOP_I32, int32_t, i32, EXACT) X(NX_LOOP_U32, uint32_t, u32, EXACT)     \
  X(NX_LOOP_I64, int64_t, i64, EXACT) X(NX_LOOP_U64, u64, u64, EXACT)

/* [a] then [b] by a Sum, a Max and a Min, and by the reduction [rk], at
   a plain type. */
#define PLAIN_AT(C, T, S, K)                                                   \
  DEVICE T plain_sum(T a, T b) { return K(a, b, nx_add_##S(a, b)); }           \
  DEVICE T plain_max(T a, T b) { return nx_maximum_##S(a, b); }                \
  DEVICE T plain_min(T a, T b) { return nx_minimum_##S(a, b); }                \
  DEVICE T plain_at(int rk, T a, T b) {                                        \
    switch (rk) {                                                              \
    case NX_LOOP_SUM: return plain_sum(a, b);                                  \
    case NX_LOOP_MAX: return plain_max(a, b);                                  \
    default: return plain_min(a, b);                                           \
    }                                                                          \
  }
PLAIN_TYPES(PLAIN_AT)
#undef PLAIN_AT

/* [a] then [b], bits, by the reduction [rk] at [ct]. */
DEVICE u64 plain_op(int ct, int rk, u64 a, u64 b) {
  switch (ct) {
#define CASE(C, T, S, K)                                                       \
  case C: return of<T>(plain_at(rk, as<T>(held{a, 0}), as<T>(held{b, 0}))).lo;
    PLAIN_TYPES(CASE)
#undef CASE
  }
  return 0;
}

/* Lane [l] + [h]'s [x] in lane l of the calling wave, for l + h below 32,
   by two permutes of LDS's crossbar. Every work-item of the wave calls
   it. */
DEVICE u64 lane_down(u64 x, int h) {
  int at = (int)(((item() & 31) + h) << 2);
  uint32_t lo = (uint32_t)__builtin_amdgcn_ds_bpermute(at, (int)(uint32_t)x);
  uint32_t hi =
      (uint32_t)__builtin_amdgcn_ds_bpermute(at, (int)(uint32_t)(x >> 32));
  return lo | (u64)hi << 32;
}

/* A term of the narrow float [DT] as float32, exactly. */
template <int DT> DEVICE float widened(uint16_t c) {
  if constexpr (DT == NX_BFLOAT16) return nx_bf16_to_float(c);
  else return nx_f16_to_float(c);
}

/* A term of the narrow float [DT] at [p], as float32, or of T. */
template <class T, int DT, class S> DEVICE T term_at(const GLOBAL S *p) {
  if constexpr (DT < 0) return *p;
  else return widened<DT>(*p);
}

/* Folds into the [W] lanes [s] (bits) their terms, lane c's of run j at
   p + j step + c gap, its 16 j + 8 c th of the work-item's [rem], a term
   past them reading the identity, in batches of 16 terms whose loads are
   all in flight, a batch wholly before [rem] without a test a term. A
   term is held as S: T itself, or the code of the narrow float [DT],
   which converts into T exactly. */
template <class T, int W, class S = T, int DT = -1>
DEVICE void fold_at(int rk, u64 (&s)[2], const GLOBAL S *p, int64_t step,
                    int64_t gap, int rem, u64 id) {
  constexpr int B = 16 / W;
  T a[W], e = as<T>(held{id, 0});
#pragma unroll
  for (int c = 0; c < W; c++) a[c] = as<T>(held{s[c], 0});
  for (int j = 0; 16 * j < rem; j += B) {
    T v[B][W];
    if (16 * (j + B - 1) + 8 * (W - 1) < rem) {
#pragma unroll
      for (int r = 0; r < B; r++)
#pragma unroll
        for (int c = 0; c < W; c++)
          v[r][c] = term_at<T, DT>(p + (j + r) * step + c * gap);
    } else {
#pragma unroll
      for (int r = 0; r < B; r++)
#pragma unroll
        for (int c = 0; c < W; c++)
          v[r][c] = 16 * (j + r) + 8 * c >= rem
                        ? e
                        : term_at<T, DT>(p + (j + r) * step + c * gap);
    }
#define FOLD(f)                                                                \
  _Pragma("unroll") for (int r = 0; r < B; r++)                                \
      _Pragma("unroll") for (int c = 0; c < W; c++) a[c] = f(a[c], v[r][c]);
    switch (rk) {
    case NX_LOOP_SUM: FOLD(plain_sum) break;
    case NX_LOOP_MAX: FOLD(plain_max) break;
    default: FOLD(plain_min)
    }
#undef FOLD
  }
#pragma unroll
  for (int c = 0; c < W; c++) s[c] = of<T>(a[c]).lo;
}

/* fold_at at [ct], of one lane (rows) or two (columns), its terms of
   the dtype [ld]: the carrier's, or bfloat16 or float16 into float32. */
DEVICE void plain_fold(int ct, int rk, int ld, bool columns, u64 (&s)[2],
                       const gbyte *p, int64_t step, int64_t gap, int rem) {
  u64 id = s[0];
#define NARROW(D)                                                              \
  if (ld == D) {                                                               \
    const GLOBAL uint16_t *q = (const GLOBAL uint16_t *)p;                     \
    if (columns) fold_at<float, 2, uint16_t, D>(rk, s, q, step, gap, rem, id); \
    else fold_at<float, 1, uint16_t, D>(rk, s, q, step, gap, rem, id);         \
    return;                                                                    \
  }
  NARROW(NX_BFLOAT16)
  NARROW(NX_FLOAT16)
#undef NARROW
  switch (ct) {
#define CASE(C, T, S, K)                                                       \
  case C:                                                                      \
    if (columns)                                                               \
      fold_at<T, 2>(rk, s, (const GLOBAL T *)p, step, gap, rem, id);           \
    else fold_at<T, 1>(rk, s, (const GLOBAL T *)p, step, gap, rem, id);        \
    return;
    PLAIN_TYPES(CASE)
#undef CASE
  }
}

/* The pairwise tree of the [n] values of [part], at its words [stride]
   apart, in the LDS [sm]; its value lands in part[0]. Every work-item of
   the group calls it. */
DEVICE void plain_merge(int ct, int rk, GLOBAL u64 *part, int64_t n,
                        u64 *sm) {
  int64_t stride = 1;
  while (n > 1) {
    for (int64_t c0 = 0; c0 < n; c0 += NX_LOOP_PLAIN_CHUNK) {
      int k = (int)imin(NX_LOOP_PLAIN_CHUNK, n - c0);
      for (int i = (int)item(); i < k; i += TB) sm[i] = part[(c0 + i) * stride];
      barrier();
      for (int h = 1; h < k; h *= 2) {
        for (int i = 2 * h * (int)item(); i + h < k; i += 2 * h * TB)
          sm[i] = plain_op(ct, rk, sm[i], sm[i + h]);
        barrier();
      }
      if (item() == 0) part[c0 * stride] = sm[0];
      barrier();
    }
    n = (n + NX_LOOP_PLAIN_CHUNK - 1) / NX_LOOP_PLAIN_CHUNK;
    stride *= NX_LOOP_PLAIN_CHUNK;
  }
}

/* Output [o]'s value [v], bits at its carrier, into its destination, or,
   a scan's chunk total, into the workspace's totals. */
DEVICE void plain_out(const u64 *w, int carrier, int64_t o, u64 v) {
  if (((wd(w, NX_LOOP_REDUCTION) >> 24) & 0xff) == NX_LOOP_PUT_TOTAL) {
    ((GLOBAL u64 *)work_byte(w, NX_LOOP_TOTALS))[o] = v;
    return;
  }
  put_dst(w, 0, o, carrier, held{v, 0});
}

/* A group's value [v] of output [o], its group [grp] of [m], held by the
   work-item where [mine]: sent out, or left in the workspace, the last
   group to arrive merging the [m] values. With [m] above 1 the whole
   group calls it. */
DEVICE void plain_put(const u64 *w, int ct, int rk, int carrier, int64_t o,
                      int64_t grp, int64_t m, bool mine, u64 v, u64 *sm) {
  if (m == 1) {
    if (mine) plain_out(w, carrier, o, v);
    return;
  }
  GLOBAL u64 *part = (GLOBAL u64 *)work_byte(w, NX_LOOP_PARTIALS) + o * m;
  if (mine) part[grp] = v;
  uint32_t *tickets =
      (uint32_t *)(uintptr_t)wd(w, NX_LOOP_HEADER + wd(w, NX_LOOP_TICKETS));
  if (!last_of(tickets + o, (uint32_t)m)) return;
  plain_merge(ct, rk, part, m, sm);
  if (item() == 0) plain_out(w, carrier, o, part[0]);
}

/* Rows: the [O] outputs' terms from [x], output o the segment o % segs
   of row o / segs, of [R] terms from x + (o / segs) so, segments of [SEG]
   terms. Of the group's [cells] at work, cell c takes block c / per of
   unit c % per of a round's [per], a unit [g] blocks of an output, [m]
   units an output; the other cells take none. */
DEVICE void plain_rows(const u64 *w, int ct, int rk, int ld, int carrier,
                       const gbyte *x, int es, int64_t O, int64_t R,
                       int64_t so, int64_t SEG, int64_t segs, int g,
                       int64_t m, int cells) {
  u64 *sm = (u64 *)loop_planes, id = plain_id(ct, rk);
  int cell = (int)item() >> 4, l = (int)item() & 15, per = cells / g;
  int k = cell / per;
  int64_t units = O * m, grid = (int64_t)wd(w, NX_LOOP_GRID);
  for (int64_t r0 = (int64_t)group_x() * per; r0 < units; r0 += grid * per) {
    int64_t u = r0 + cell % per, o = u / m, grp = u % m, b = grp * g + k;
    int64_t at = o % segs * SEG, n = imin(SEG, R - at);
    int64_t nb = (n + NX_LOOP_BLOCK - 1) / NX_LOOP_BLOCK;
    int len = u < units && k < g && b < nb
                  ? (int)imin(NX_LOOP_BLOCK, n - b * NX_LOOP_BLOCK)
                  : 0;
    u64 s[2] = {id, id};
    plain_fold(ct, rk, ld, false, s,
               x + (o / segs * so + at + b * NX_LOOP_BLOCK + l) * es, 16, 0,
               len - l);
    /* The lanes' tree: lane l below h taking lane l + h. */
    for (int h = 8; h > 0; h /= 2) {
      u64 y = lane_down(s[0], h);
      if (l < h) s[0] = plain_op(ct, rk, s[0], y);
    }
    /* The unit's blocks by the pairwise tree, a cell of no block taking
       no part. */
    int count = u < units ? (int)imin(g, nb - grp * g) : 0;
    if (l == 0) sm[cell] = s[0];
    barrier();
    for (int h = 1; h < g; h *= 2) {
      if (l == 0 && k % (2 * h) == 0 && k + h < count)
        sm[cell] = plain_op(ct, rk, sm[cell], sm[cell + h * per]);
      barrier();
    }
    u64 v = sm[cell];
    barrier();
    plain_put(w, ct, rk, carrier, o, grp, m, l == 0 && k == 0 && u < units, v,
              sm + TB / 16);
  }
}

/* Columns: the [O] columns' terms from [x], column o's term t at
   o + t S, of [R] terms in segments of [SEG], output o segs + the
   segment. A group takes a segment of 32 columns at a time. */
DEVICE void plain_columns(const u64 *w, int ct, int rk, int ld, int carrier,
                          const gbyte *x, int es, int64_t O, int64_t R,
                          int64_t S, int64_t SEG, int64_t segs) {
  u64 *sm = (u64 *)loop_planes, id = plain_id(ct, rk);
  int t = (int)item() & 31, lw = (int)item() >> 5;
  int64_t strips = (O + 31) / 32, grid = (int64_t)wd(w, NX_LOOP_GRID);
  /* sm[lw][t] holds the lanes' tree, then sm[8 + b][t] block b's value. */
  for (int64_t q = group_x(); q < strips * segs; q += grid) {
    int64_t o = q / segs * 32 + t, at = q % segs * SEG, n = imin(SEG, R - at);
    int nb = (int)((n + NX_LOOP_BLOCK - 1) / NX_LOOP_BLOCK);
    for (int b = 0; b < nb; b++) {
      int len = o < O ? (int)imin(NX_LOOP_BLOCK, n - (int64_t)b * NX_LOOP_BLOCK)
                      : 0;
      u64 s[2] = {id, id};
      plain_fold(ct, rk, ld, true, s,
                 x + (o + (at + (int64_t)b * NX_LOOP_BLOCK + lw) * S) * es,
                 16 * S, 8 * S, len - lw);
      /* The lanes' tree: lane lw takes lane lw + 8 in the work-item, then
         lanes below h take lanes h above them across the waves. */
      sm[32 * lw + t] = plain_op(ct, rk, s[0], s[1]);
      barrier();
      for (int h = 4; h > 0; h /= 2) {
        if (lw < h)
          sm[32 * lw + t] =
              plain_op(ct, rk, sm[32 * lw + t], sm[32 * (lw + h) + t]);
        barrier();
      }
      if (lw == 0) sm[32 * (8 + b) + t] = sm[t];
      barrier();
    }
    /* The blocks by the pairwise tree. */
    if (lw == 0 && o < O) {
      for (int h = 1; h < nb; h *= 2)
        for (int b = 0; b + h < nb; b += 2 * h)
          sm[32 * (8 + b) + t] =
              plain_op(ct, rk, sm[32 * (8 + b) + t], sm[32 * (8 + b + h) + t]);
      plain_out(w, carrier, o * segs + q % segs, sm[32 * 8 + t]);
    }
    barrier();
  }
}

/* The reduction's walk: rows where its terms step by one, else
   columns. Its outputs are its rows' segments. */
DEVICE void plain_mode(const u64 *w) {
  u64 r = wd(w, NX_LOOP_REDUCTION);
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff);
  const u64 *d = operand(w, NX_LOOP_LOADS, 0);
  int ld = (int)(wd(d, 0) & 0xff), es = bits_of(ld) / 8;
  const gbyte *x = buffer(w, d) + wd(d, 1) * es;
  int64_t O = (int64_t)wd(w, NX_LOOP_OUTPUTS), R = (int64_t)wd(w, NX_LOOP_TERMS);
  int64_t SEG = (int64_t)wd(w, NX_LOOP_SEGMENT);
  int64_t segs = (int64_t)wd(w, NX_LOOP_SEGMENTS);
  bool two = wd(w, NX_LOOP_RANK) == 2;
  int64_t so = two ? (int64_t)wd(d, 2) : 0, st = (int64_t)wd(d, 2 + two);
  if (st == 1)
    plain_rows(w, ct, rk, ld, carrier, x, es, O, R, so, SEG, segs,
               (int)wd(w, NX_LOOP_GROUP), (int64_t)wd(w, NX_LOOP_GROUPS),
               (int)wd(w, NX_LOOP_CELLS));
  else plain_columns(w, ct, rk, ld, carrier, x, es, O / segs, R, st, SEG, segs);
}

/* Scans */

/* The state a slice's scan starts from: its carry, or none. */
DEVICE st carried(const u64 *w, int rk, int carrier, int64_t slice) {
  st s = empty(rk);
  int64_t k = (int64_t)wd(w, NX_LOOP_CARRY);
  if (k >= 0) {
    s.v = of_carrier(carrier,
                     load_at(w, (int)k, slice * (int64_t)wd(w, NX_LOOP_TERMS)));
    s.has = 1;
  }
  return s;
}

/* Each slice's chunk totals, in place, become the states its chunks
   start from: its carry, then the totals of the chunks before folded in
   order. With NX_LOOP_GROUP 1, a work-item takes a slice and folds its
   totals where they lie; else a group takes a slice, loads TB totals at a
   time side by side into LDS, its first work-item folds them there, and
   the group stores them back. */
DEVICE void carries_mode(const u64 *w) {
  int64_t slices = (int64_t)wd(w, NX_LOOP_OUTPUTS);
  int64_t chunks = (int64_t)wd(w, NX_LOOP_SEGMENTS);
  int64_t grid = (int64_t)wd(w, NX_LOOP_GRID);
  u64 r = wd(w, NX_LOOP_REDUCTION);
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff);
  if (wd(w, NX_LOOP_GROUP) == 1) {
    for (int64_t slice = group_x() * TB + item(); slice < slices;
         slice += grid * TB) {
      st s = carried(w, rk, carrier, slice);
      for (int64_t c = 0; c < chunks; c++) {
        GLOBAL uint32_t *g = work_at(w, NX_LOOP_TOTALS, slice * chunks + c);
        st t = load_work(g);
        store_work(g, s);
        s = merge(rk, ct, s, t);
      }
    }
    return;
  }
  static SHARED uint32_t piece[TB * NX_LOOP_STATE];
  for (int64_t slice = group_x(); slice < slices; slice += grid) {
    st s = empty(rk);
    if (item() == 0) s = carried(w, rk, carrier, slice);
    for (int64_t c0 = 0; c0 < chunks; c0 += TB) {
      int n = (int)imin(TB, chunks - c0) * NX_LOOP_STATE;
      GLOBAL uint32_t *g = work_at(w, NX_LOOP_TOTALS, slice * chunks + c0);
      for (int i = (int)item(); i < n; i += TB) piece[i] = g[i];
      barrier();
      if (item() == 0)
        for (int i = 0; i < n; i += NX_LOOP_STATE) {
          st t = load_work(piece + i);
          store_work(piece + i, s);
          s = merge(rk, ct, s, t);
        }
      barrier();
      for (int i = (int)item(); i < n; i += TB) g[i] = piece[i];
      barrier();
    }
  }
}

/* Chunks: S = NX_LOOP_GROUP work-items take a chunk, TB / S chunks a
   round of a group. A run loads S v consecutive terms of each chunk,
   work-item l of a chunk's S holding terms l + S j; the chunk's first
   work-item then folds them in order from its state, each result, the
   state after it, into the result planes, its position after where Arg's;
   the epilogue stores them. The run's tape is the body, then the
   epilogue, from one site. */
DEVICE void scan_mode(const u64 *w, int E, int V) {
  int64_t units = (int64_t)wd(w, NX_LOOP_OUTPUTS), R = (int64_t)wd(w, NX_LOOP_TERMS);
  int64_t chunks = (int64_t)wd(w, NX_LOOP_SEGMENTS), CH = (int64_t)wd(w, NX_LOOP_SEGMENT);
  int64_t runs = (int64_t)wd(w, NX_LOOP_RUNS), grid = (int64_t)wd(w, NX_LOOP_GRID);
  int S = (int)wd(w, NX_LOOP_GROUP), per = TB / S;
  u64 r = wd(w, NX_LOOP_REDUCTION);
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff);
  int tp = (int)wd(w, NX_LOOP_TERM), rp = (int)wd(w, NX_LOOP_RESULT);
  int body = (int)(wd(w, NX_LOOP_TAPE) + wd(w, NX_LOOP_PROLOGUE));
  int nbody = (int)wd(w, NX_LOOP_BODY), nepi = (int)wd(w, NX_LOOP_EPILOGUE);
  int me = (int)item(), l = me % S, first_item = me - l;
  int64_t SV = (int64_t)S * V;
  held z = {0, 0};
  for (int64_t r0 = group_x() * per; r0 < units; r0 += grid * per) {
    int64_t u = r0 + me / S, slice = u / chunks, c = u % chunks;
    int64_t start = c * CH, len = u < units ? imin(CH, R - start) : 0;
    int64_t first = slice * R + start;
    st s = empty(rk);
    if (l == 0 && u < units)
      s = chunks > 1 ? load_work(work_at(w, NX_LOOP_TOTALS, u))
                     : carried(w, rk, carrier, slice);
    for (int64_t j = 0; j < runs; j++) {
      lanes x = lanes_of(first + j * SV + l, S, first + len, V);
      for (int phase = 0; phase < 2; phase++) {
        run(w, E, V, x, phase == 0 ? body : body + nbody,
            phase == 0 ? nbody : nepi);
        if (phase == 1) break;
        barrier();
        if (l == 0) {
          int n = (int)imax(0, imin(SV, len - j * SV));
          for (int o = 0; o < n; o++) {
            int i = (o / S) * TB + first_item + o % S;
            s = fold(rk, ct, s, get_held(E, tp, i, 4), start + j * SV + o, z);
            put_held(E, rp, i, 4, to_carrier(carrier, ct, s.v));
            put<int64_t>(E, rp + 4, i, (int64_t)s.x);
          }
        }
        barrier();
      }
    }
  }
}

/* Plain scans

   A Sum, Max or Min scan of one plain load at its carrier into one
   destination of that dtype, without a tape: a work-item folds a chunk's
   terms in order from the carry into it, storing each result. Past one
   chunk, a plain reduction first leaves each chunk's total in the
   workspace, NX_LOOP_TOTALS, and the carries kernel turns each slice's
   totals in place into the carries into its chunks. Values are their
   bits in a word, as plain reductions hold them.

   Where terms and results both step by one (NX_LOOP_TILED), a wave's 32
   chunks pass through a tile of LDS, the next 128 bytes of each at a
   time, so that each load and store reads or writes one chunk's run of
   words side by side (Tiles). Elsewhere a work-item streams its own
   chunk, 16 loads in flight; with terms that step by more than one, a
   wave's work-items take one chunk of 32 adjacent slices, whose terms
   lie side by side where the slices step by one. */

typedef uint32_t quad __attribute__((ext_vector_type(4)));

/* 16 values of T at [p], which lies on 16 bytes, read or written in
   16-byte accesses. */
template <class T> DEVICE void get16(const GLOBAL T *p, T (&v)[16]) {
#pragma unroll
  for (int r = 0; r < 16 * (int)sizeof(T) / 16; r++) {
    quad u = ((const GLOBAL quad *)p)[r];
    memcpy((char *)v + 16 * r, &u, 16);
  }
}

template <class T> DEVICE void put16(GLOBAL T *p, const T (&v)[16]) {
#pragma unroll
  for (int r = 0; r < 16 * (int)sizeof(T) / 16; r++) {
    quad u;
    memcpy(&u, (const char *)v + 16 * r, 16);
    ((GLOBAL quad *)p)[r] = u;
  }
}

/* Folds [n] terms at p + i st from [a], each result to y + i yst, 16 at
   a time, in 16-byte accesses where [vec]: both step by one from 16
   bytes. */
template <class T>
DEVICE void scan_at(int rk, T a, const GLOBAL T *p, int64_t st, GLOBAL T *y,
                    int64_t yst,
                    int n, bool vec) {
  for (int i = 0; i < n; i += 16) {
    T v[16], o[16];
    /* Strides opaque to the loop, so that the 16 terms' and results'
       offsets are computed in turn, not each held across it. */
    asm volatile("" : "+s"(st), "+s"(yst));
    bool whole = vec && i + 16 <= n;
    if (whole) get16(p + i, v);
    else
#pragma unroll
      for (int r = 0; r < 16; r++) v[r] = i + r < n ? p[(i + r) * st] : T(0);
#define STEP(f)                                                                \
  _Pragma("unroll") for (int r = 0; r < 16; r++) {                             \
    a = i + r < n ? f(a, v[r]) : a;                                            \
    o[r] = a;                                                                  \
  }
    switch (rk) {
    case NX_LOOP_SUM: STEP(plain_sum) break;
    case NX_LOOP_MAX: STEP(plain_max) break;
    default: STEP(plain_min)
    }
#undef STEP
    if (whole) put16(y + i, o);
    else
#pragma unroll
      for (int r = 0; r < 16; r++)
        if (i + r < n) y[(i + r) * yst] = o[r];
  }
}

/* Turns the [n] totals at [g] into the carries into their chunks, from
   [a]: each [a] then the totals before it, in order, 16 loads in
   flight. */
template <class T>
DEVICE void carries_at(int rk, T a, GLOBAL u64 *g, int64_t n) {
  for (int64_t i = 0; i < n; i += 16) {
    T v[16];
#pragma unroll
    for (int r = 0; r < 16; r++)
      v[r] = i + r < n ? as<T>(held{g[i + r], 0}) : T(0);
#pragma unroll
    for (int r = 0; r < 16; r++)
      if (i + r < n) {
        g[i + r] = of<T>(a).lo;
        a = plain_at(rk, a, v[r]);
      }
  }
}

DEVICE void plain_scan_run(int ct, int rk, u64 a, const gbyte *p, int64_t st,
                           gbyte *y, int64_t yst, int n) {
  bool vec = st == 1 && yst == 1 && (((uintptr_t)p | (uintptr_t)y) & 15) == 0;
  switch (ct) {
#define CASE(C, T, S, K)                                                       \
  case C:                                                                      \
    scan_at<T>(rk, as<T>(held{a, 0}), (const GLOBAL T *)p, st, (GLOBAL T *)y,  \
               yst, n, vec);                                                   \
    return;
    PLAIN_TYPES(CASE)
#undef CASE
  }
}

DEVICE void plain_carries_run(int ct, int rk, u64 a, GLOBAL u64 *g,
                              int64_t n) {
  switch (ct) {
#define CASE(C, T, S, K)                                                       \
  case C: carries_at<T>(rk, as<T>(held{a, 0}), g, n); return;
    PLAIN_TYPES(CASE)
#undef CASE
  }
}

/* The bits slice [slice]'s scan starts from: its carry, else the
   identity. */
DEVICE u64 plain_carry(const u64 *w, int ct, int rk, int carrier,
                       int64_t slice) {
  st s = carried(w, rk, carrier, slice);
  return s.has ? s.v.lo : plain_id(ct, rk);
}

/* Tiles

   A wave's tile is NX_LOOP_TILE bytes of LDS: row k holds the next 32
   words of the wave's chunk k, its work-item k's, 32 terms of 32 bits or
   16 of 64. A tile runs in three steps:

     fill   for each k, work-item l loads word l of chunk k's next 128
            bytes and puts it in row k: one load reads a run of words
     scan   work-item k folds row k, its chunk's terms in order, from its
            8 quads of words, writing each result over its term
     drain  for each k, work-item l stores word l of row k to chunk k's
            results

   The next tile's loads are issued before the scan, so a wave keeps a
   tile in flight. Quad q of row k lies at quad q ^ (k % 8) of the row, so
   that the fill's and the drain's words of a row fall on 32 banks, and
   the scan's quads of 8 rows a step on 8 distinct quads of banks, as a
   16-byte access of LDS takes them.

   A wave's LDS operations run in the order it issues them, so the fill's
   writes reach LDS before the scan reads them, the scan's write-back
   before the drain reads it, and the drain's reads before the next fill
   writes: wave_barrier keeps the compiler from moving them across a step,
   and a drain's store waits only on the read that gives its data. */

_Static_assert(NX_LOOP_TILE == 32 * NX_LOOP_WAVE * 4,
               "a tile is 32 words a work-item");

/* The word of word [j] of row [k]. */
DEVICE int tile_at(int k, int j) {
  return 32 * k + 4 * ((j >> 2) ^ (k & 7)) + (j & 3);
}

/* Work-item [k]'s [p]. */
DEVICE uintptr_t lane_of(uintptr_t p, int k) {
  uint32_t lo = __builtin_amdgcn_readlane((int)(uint32_t)p, k);
  uint32_t hi = __builtin_amdgcn_readlane((int)(uint32_t)(p >> 32), k);
  return lo | (uintptr_t)hi << 32;
}

/* Word l of each chunk k's next 32, at work-item k's [p], [left] of its
   bytes from there; 0 past them. A tile's addresses and bounds are read
   from their work-items in the tile's own step, so that the compiler
   holds none of them across the loop. */
DEVICE void tile_load(uint32_t (&v)[32], const gbyte *p, int left) {
  int l = (int)lane();
#pragma unroll
  for (int k = 0; k < 32; k++) {
    const GLOBAL uint32_t *q =
        (const GLOBAL uint32_t *)lane_of((uintptr_t)p, k) + l;
    v[k] = 4 * l < __builtin_amdgcn_readlane(left, k) ? *q : 0;
  }
}

/* Row k's word l to work-item k's [q], [left] of its bytes from there. */
DEVICE void tile_store(const uint32_t *tile, gbyte *q, int left) {
  int l = (int)lane();
#pragma unroll
  for (int k = 0; k < 32; k++) {
    GLOBAL uint32_t *r = (GLOBAL uint32_t *)lane_of((uintptr_t)q, k) + l;
    uint32_t x = tile[tile_at(k, l)];
    if (4 * l < __builtin_amdgcn_readlane(left, k)) *r = x;
  }
}

/* Folds the calling work-item's row of [tile], its first [n] terms of T,
   from [a], each result over its term: a half row, 4 quads, at a
   time. */
template <class T>
DEVICE T tile_fold(int rk, T a, uint32_t *tile, int n) {
  constexpr int W = sizeof(T) / 4, N = 16 / W;
  int l = (int)lane();
  quad *row = (quad *)(tile + 32 * l);
#pragma unroll
  for (int h = 0; h < 2; h++) {
    T v[N];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      quad u = row[(4 * h + q) ^ (l & 7)];
      memcpy((char *)v + 16 * q, &u, 16);
    }
    int m = n - h * N;
#define STEP(f)                                                                \
  _Pragma("unroll") for (int t = 0; t < N; t++) {                              \
    a = t < m ? f(a, v[t]) : a;                                                \
    v[t] = a;                                                                  \
  }
    switch (rk) {
    case NX_LOOP_SUM: STEP(plain_sum) break;
    case NX_LOOP_MAX: STEP(plain_max) break;
    default: STEP(plain_min)
    }
#undef STEP
#pragma unroll
    for (int q = 0; q < 4; q++) {
      quad u;
      memcpy(&u, (const char *)v + 16 * q, 16);
      row[(4 * h + q) ^ (l & 7)] = u;
    }
  }
  return a;
}

/* Scans each work-item's chunk of [nb] bytes at [p] from [a] into [q],
   through [tile]. Every work-item of the wave calls it. */
template <class T>
DEVICE void tile_scan(int rk, T a, const gbyte *p, gbyte *q, int nb,
                      uint32_t *tile) {
  int l = (int)lane();
  uint32_t v[32];
  tile_load(v, p, nb);
  for (int b = 0; __builtin_amdgcn_ballot_w32(b < nb); b += 128) {
#pragma unroll
    for (int k = 0; k < 32; k++) tile[tile_at(k, l)] = v[k];
    wave_barrier();
    tile_load(v, p + b + 128, nb - b - 128);
    a = tile_fold<T>(rk, a, tile,
                     (int)imax(0, imin(128, nb - b)) / (int)sizeof(T));
    wave_barrier();
    tile_store(tile, q + b, nb - b);
    wave_barrier();
  }
}

DEVICE void plain_tile_run(int ct, int rk, u64 a, const gbyte *p, gbyte *q,
                           int nb, uint32_t *tile) {
  switch (ct) {
#define CASE(C, T, S, K)                                                       \
  case C: tile_scan<T>(rk, as<T>(held{a, 0}), p, q, nb, tile); return;
    PLAIN_TYPES(CASE)
#undef CASE
  }
}

/* A scan's [O] slices of [R] terms, slice o's term t at x + o so + t st,
   its result at y + o yso + t yst, chunk c of slice o its unit
   u = o chunks + c, or, where terms step by more than one, its
   work-item's u = c O + o; unit u to work-item u % A of group u / A, [A]
   work-items a group, a multiple of 32. Tiled, wave i of a group takes
   tile i of LDS. */
DEVICE void plain_scan_mode(const u64 *w) {
  u64 r = wd(w, NX_LOOP_REDUCTION);
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff), es = bits_of(carrier) / 8;
  const u64 *d = operand(w, NX_LOOP_LOADS, 0);
  const u64 *e = operand(w, NX_LOOP_DSTS, 0);
  const gbyte *x = buffer(w, d) + wd(d, 1) * es;
  gbyte *y = buffer(w, e) + wd(e, 1) * es;
  int64_t O = (int64_t)wd(w, NX_LOOP_OUTPUTS), R = (int64_t)wd(w, NX_LOOP_TERMS);
  int64_t chunks = (int64_t)wd(w, NX_LOOP_SEGMENTS), units = O * chunks;
  int64_t grid = (int64_t)wd(w, NX_LOOP_GRID);
  int A = (int)wd(w, NX_LOOP_GROUP);
  bool two = wd(w, NX_LOOP_RANK) == 2, tiled = wd(w, NX_LOOP_TILED) != 0;
  int64_t so = two ? (int64_t)wd(d, 2) : 0, st = (int64_t)wd(d, 2 + two);
  int64_t yso = two ? (int64_t)wd(e, 2) : 0, yst = (int64_t)wd(e, 2 + two);
  bool across = st != 1;
  GLOBAL u64 *totals = (GLOBAL u64 *)work_byte(w, NX_LOOP_TOTALS);
  uint32_t *tile = loop_planes + wave() * (NX_LOOP_TILE / 4);
  if ((int)item() >= A) return;
  /* u0 is the wave's first unit, so every work-item of a wave runs the
     same rounds. */
  for (int64_t u0 = (int64_t)group_x() * A + wave() * NX_LOOP_WAVE; u0 < units;
       u0 += grid * A) {
    int64_t u = u0 + lane();
    int64_t o = across ? u % O : u / chunks, c = across ? u / O : u % chunks;
    int64_t at = c * NX_LOOP_CHUNK;
    int n = u < units ? (int)imin(NX_LOOP_CHUNK, R - at) : 0;
    u64 a = n == 0       ? 0
            : chunks > 1 ? totals[o * chunks + c]
                         : plain_carry(w, ct, rk, carrier, o);
    const gbyte *p = n == 0 ? x : x + (o * so + at * st) * es;
    gbyte *q = n == 0 ? y : y + (o * yso + at * yst) * es;
    if (tiled) plain_tile_run(ct, rk, a, p, q, n * es, tile);
    else plain_scan_run(ct, rk, a, p, st, q, yst, n);
  }
}

/* The carries of a plain scan: a work-item a slice. */
DEVICE void plain_carries_mode(const u64 *w) {
  u64 r = wd(w, NX_LOOP_REDUCTION);
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff);
  int64_t O = (int64_t)wd(w, NX_LOOP_OUTPUTS);
  int64_t chunks = (int64_t)wd(w, NX_LOOP_SEGMENTS);
  int64_t grid = (int64_t)wd(w, NX_LOOP_GRID);
  GLOBAL u64 *totals = (GLOBAL u64 *)work_byte(w, NX_LOOP_TOTALS);
  for (int64_t o = (int64_t)group_x() * TB + item(); o < O; o += grid * TB)
    plain_carries_run(ct, rk, plain_carry(w, ct, rk, carrier, o),
                      totals + o * chunks, chunks);
}

/* Networks: Logsumexp's and Moments' scans by nx_kernel.mli's odd/even
   network. Result g of a slice is its block, the pairwise tree of the 2^k
   terms ending at g, 2^k the lowest set bit of g + 1, combined after
   result g - 2^k where that is not -1. Any order of these combinations
   gives the network's bits; nx.cpu streams the terms in order.

   A run's element t holds position t of a slot of N = 2^n positions, a
   slice's terms from a multiple of N. A slice of at most TB terms takes
   a slot of its count rounded up to a power of two, E / N slices a run;
   a longer one takes runs of N = E terms in order, all one group's. A
   run builds each position's block in place, up by levels, then each
   result down by levels from the highest, its first results after the
   result that ends the run before it. A run's last block spans earlier
   runs: the group streams the runs' blocks through levels from n up, as
   nx.cpu streams terms, an even block that waits for its partner and the
   last result each, in LDS.

   A Logsumexp state is a maximum and a sum; a Moments state a mean, a sum
   of squared deviations and a count. A run's states lie in its result
   planes, a value a carrier's words. */

template <class T> struct Net {
  T m, s, n;
};

template <class T> DEVICE bool fin_(T x) { return x - x == 0; }

template <class T> DEVICE T lse_sum(T ma, T sa, T mb, T sb, T m) {
  bool hi = ma == m;
  T lo = hi ? mb : ma, sl = hi ? sb : sa, sh = hi ? sa : sb;
  T s = sh + sl * ex(lo - m);
  return fin_(m) ? s : m == -(T)INFINITY ? T(0) : m;
}

template <class T> DEVICE Net<T> join(int rk, const Net<T> &a, const Net<T> &b) {
  Net<T> r;
  if (rk == NX_LOOP_LOGSUMEXP) {
    r.m = max_(a.m, b.m);
    r.s = lse_sum(a.m, a.s, b.m, b.s, r.m);
    r.n = 0;
    return r;
  }
  T k = a.n + b.n, f = b.n / k, g = a.n * f, d = b.m - a.m;
  T t = a.m + d * f;
  r.m = fin_(a.m) && fin_(b.m) ? t : a.m + b.m;
  r.s = (a.s + b.s) + (d * d) * g;
  r.n = k;
  return r;
}

/* The state at element [t] of the planes from [p]. */
template <class T> DEVICE Net<T> net_get(int E, int p, int t) {
  const int k = sizeof(T) / 4;
  Net<T> a = {get<T>(E, p, t), get<T>(E, p + k, t), get<T>(E, p + 2 * k, t)};
  return a;
}

template <class T> DEVICE void net_put(int E, int p, int t, const Net<T> &a) {
  const int k = sizeof(T) / 4;
  put<T>(E, p, t, a.m);
  put<T>(E, p + k, t, a.s);
  put<T>(E, p + 2 * k, t, a.n);
}

/* The levels' states, a value's bits a word: the even block of level k
   at NET_HELD + k, the last result at NET_GOT + k, the result that ends
   the last run at NET_CARRY, and the slice's center. A run's index has
   at most 64 - 8 bits, N being at least TB. */
enum {
  NET_LEVELS = 64 - 8 + 1,
  NET_HELD = 0,
  NET_GOT = NET_LEVELS,
  NET_CARRY = 2 * NET_LEVELS,
  NET_STATES
};

static SHARED u64 net_levels[3 * NET_STATES];
static SHARED u64 net_center;

template <class T> DEVICE u64 word_bits(T x) {
  u64 b = 0;
  memcpy(&b, &x, sizeof(T));
  return b;
}

template <class T> DEVICE Net<T> level_at(int k) {
  const u64 *p = net_levels + 3 * k;
  Net<T> a = {bits_as<T>(p[0], 0), bits_as<T>(p[1], 0), bits_as<T>(p[2], 0)};
  return a;
}

template <class T> DEVICE void set_level(int k, const Net<T> &a) {
  u64 *p = net_levels + 3 * k;
  p[0] = word_bits(a.m);
  p[1] = word_bits(a.s);
  p[2] = word_bits(a.n);
}

/* Run [q]'s block [in] streamed through the levels: its result. */
template <class T> DEVICE Net<T> stream_run(int rk, int64_t q, Net<T> in) {
  int k = 0;
  for (; q & 1; k++, q >>= 1) in = join(rk, level_at<T>(NET_HELD + k), in);
  Net<T> e = q == 0 ? in : join(rk, level_at<T>(NET_GOT + k + 1), in);
  set_level(NET_HELD + k, in);
  for (int h = 0; h <= k; h++) set_level(NET_GOT + h, e);
  return e;
}

template <class T> DEVICE T finite_or_0(T y) { return fin_(y) ? y : T(0); }

/* The network's walk of run [j] of the run's slots from slice [s0], the
   terms in its plane NX_LOOP_TERM, each result into its result planes. */
template <class T>
DEVICE void network_run(const u64 *w, int rk, int E, int V, int n, int64_t s0,
                        int64_t j, lanes x) {
  int64_t units = (int64_t)wd(w, NX_LOOP_OUTPUTS), R = (int64_t)wd(w, NX_LOOP_TERMS);
  int tp = (int)wd(w, NX_LOOP_TERM), rp = (int)wd(w, NX_LOOP_RESULT);
  int N = 1 << n;
  bool lse = rk == NX_LOOP_LOGSUMEXP;
#define VALID(t) (s0 + ((t) >> n) < units && j * N + ((t) & (N - 1)) < R)
  /* Moments' center: its slot's first term, its slice's, where finite. */
#define CENTER(t)                                                              \
  (j > 0 ? bits_as<T>(net_center, 0) : finite_or_0(get<T>(E, tp, ((t) >> n) << n)))
  barrier();
  EACH {
    if (v >= x.n) continue;
    T y = get<T>(E, tp, i);
    Net<T> in;
    if (lse) {
      in.m = y;
      in.s = fin_(y) ? T(1) : y == -(T)INFINITY ? T(0) : y;
      in.n = 0;
    } else {
      in.m = y - CENTER(i);
      in.s = 0;
      in.n = 1;
    }
    net_put(E, rp, i, in);
  }
  if (!lse && j == 0 && item() == 0) net_center = word_bits(CENTER(0));
  for (int d = 0; d < n; d++) {
    barrier();
    for (int k = (int)item(); k < E >> (d + 1); k += TB) {
      int t = ((k + 1) << (d + 1)) - 1;
      if (VALID(t))
        net_put(E, rp, t, join(rk, net_get<T>(E, rp, t - (1 << d)),
                               net_get<T>(E, rp, t)));
    }
  }
  for (int d = n - 1; d >= 0; d--) {
    barrier();
    for (int k = (int)item(); k < E >> (d + 1); k += TB) {
      int t = (k << (d + 1)) + (1 << d) - 1;
      if (!VALID(t)) continue;
      if ((t & (N - 1)) >= 1 << d)
        net_put(E, rp, t, join(rk, net_get<T>(E, rp, t - (1 << d)),
                               net_get<T>(E, rp, t)));
      else if (j > 0)
        net_put(E, rp, t, join(rk, level_at<T>(NET_CARRY), net_get<T>(E, rp, t)));
    }
  }
  barrier();
  if (R > N && item() == 0 && VALID(N - 1)) {
    Net<T> e = stream_run(rk, j, net_get<T>(E, rp, N - 1));
    net_put(E, rp, N - 1, e);
    set_level(NET_CARRY, e);
  }
  barrier();
  EACH {
    if (v >= x.n) continue;
    Net<T> e = net_get<T>(E, rp, i);
    T r0, r1 = 0;
    if (lse) r0 = fin_(e.m) ? e.m + log_(e.s) : e.m;
    else {
      r0 = CENTER(i) + e.m;
      r1 = fin_(e.m) ? e.s / ((T)(j * N + (i & (N - 1))) + 1) : e.m - e.m;
    }
    put<T>(E, rp, i, r0);
    put<T>(E, rp + 4, i, r1);
  }
  /* Moments' results read the center from the term plane of their slot's
     first element, which its work-item overwrites in the next body: every
     read ends before any work-item leaves the run. */
  if (!lse) barrier();
#undef VALID
#undef CENTER
}

/* A unit is a run's slots of slices from [s0], its slices' runs in
   order. */
DEVICE void network_mode(const u64 *w, int E, int V) {
  int64_t units = (int64_t)wd(w, NX_LOOP_OUTPUTS), R = (int64_t)wd(w, NX_LOOP_TERMS);
  int64_t runs = (int64_t)wd(w, NX_LOOP_RUNS), grid = (int64_t)wd(w, NX_LOOP_GRID);
  int n = (int)wd(w, NX_LOOP_SEGMENT), N = 1 << n, Q = E >> n;
  int rk = (int)(wd(w, NX_LOOP_REDUCTION) & 0xff);
  bool f32 = ((wd(w, NX_LOOP_REDUCTION) >> 8) & 0xff) == NX_LOOP_F32;
  int body = (int)(wd(w, NX_LOOP_TAPE) + wd(w, NX_LOOP_PROLOGUE));
  int nbody = (int)wd(w, NX_LOOP_BODY), nepi = (int)wd(w, NX_LOOP_EPILOGUE);
  int l = (int)item();
  for (int64_t s0 = group_x() * (int64_t)Q; s0 < units; s0 += grid * Q)
    for (int64_t j = 0; j < runs; j++) {
      lanes x;
      if (N <= TB) {
        int64_t pos = j * N + (l & (N - 1));
        x = lanes_of((s0 + (l >> n)) * R + pos, (int64_t)(TB >> n) * R,
                     pos < R ? units * R : 0, V);
      } else
        x = lanes_of(s0 * R + j * N + l, TB, (s0 + 1) * R, V);
      for (int phase = 0; phase < 2; phase++) {
        run(w, E, V, x, phase == 0 ? body : body + nbody,
            phase == 0 ? nbody : nepi);
        if (phase == 1) break;
        if (f32) network_run<float>(w, rk, E, V, n, s0, j, x);
        else network_run<double>(w, rk, E, V, n, s0, j, x);
      }
    }
}

/* Maps: tiles of consecutive indices. */
DEVICE void map_mode(const u64 *w, int E, int V) {
  int64_t n = (int64_t)wd(w, NX_LOOP_NUMEL), grid = (int64_t)wd(w, NX_LOOP_GRID);
  int body = (int)(wd(w, NX_LOOP_TAPE) + wd(w, NX_LOOP_PROLOGUE));
  int nbody = (int)wd(w, NX_LOOP_BODY);
  for (int64_t base = group_x() * (int64_t)E; base < n; base += grid * E)
    run(w, E, V, lanes_of(base + item(), TB, n, V), body, nbody);
}

/* The kernels: a mode each, so that one mode's registers do not bound
   another's. Each stages its words, then writes its constants. */

#define LOOP(name, mode)                                                       \
  KERNEL SIZED(TB) void name(Words<NX_LOOP_WORDS>) {                           \
    const u64 *w = loop_staged();                                              \
    int V = (int)((wd(w, NX_LOOP_MODE) >> 8) & 0xff), E = V * TB;                  \
    constants(w, E, V);                                                        \
    mode(w, E, V);                                                             \
  }

LOOP(loop_map, map_mode)
LOOP(loop_reduce, reduce_mode)
LOOP(loop_scan, scan_mode)
LOOP(loop_network, network_mode)

KERNEL SIZED(TB) void loop_carries(Words<NX_LOOP_WORDS>) {
  carries_mode(loop_staged());
}

KERNEL SIZED(TB) void loop_plain(Words<NX_LOOP_WORDS>) {
  plain_mode(loop_staged());
}

KERNEL SIZED(TB) void loop_plain_scan(Words<NX_LOOP_WORDS>) {
  plain_scan_mode(loop_staged());
}

KERNEL SIZED(TB) void loop_plain_carries(Words<NX_LOOP_WORDS>) {
  plain_carries_mode(loop_staged());
}
