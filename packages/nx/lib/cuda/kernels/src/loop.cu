/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Loops: a map, a reduction or a scan of a program, run by one kernel over
   a tape the host lowers the program to (kernels.h, Loops).

   A group of NX_LOOP_THREADS threads runs tiles of v elements a thread.
   Each value the tape holds lives in planes of shared memory, a 32-bit
   word an element each, one to four planes as its dtype is wide: element
   [i] of plane [p] is word p E + i, E the tile's elements, and a thread's
   element v is i = v NX_LOOP_THREADS + its index. A thread reads and
   writes its own elements only, so the tape runs without barriers. A
   thread's elements' loop indices step evenly, held in registers (lanes):
   past its first n elements none has one, whose loads read nothing and
   whose stores write nothing.

     prologue (constants) --> per tile: lanes --> body --> fold or stores
     --> epilogue (a scan's stores)

   A value is held as its dtype's element exactly: a narrow float as its
   code, an integer of 16 bits or fewer sign- or zero-extended to 32, a
   boolean as its byte, so that Copy, Where and Bitcast move bits. The host
   lowers a kind at a narrow float to a decode, the float32 kind and an
   encode, and a kind at a narrow integer to its 32-bit kind and a wrap.

   A reduction's element is a term. A block's G threads take G v
   consecutive terms a run of the tape, thread t's element j the term
   t + G j. With G = 16 / v, or fewer where a block's terms fill fewer
   lanes, thread t holds lanes t + G j of nx.cpu's lane order and folds
   them itself, its states in planes; with G of 16 or more (spread: blocks
   of 128 terms or more, G doubled while a launch has few groups), thread
   l below 16 folds lane l's terms of the run from the others' elements
   into its state, one a thread past the planes. Lanes fold in increasing
   term order, then the lanes' tree, a lane of no term taking no part,
   then the blocks' binary tree, so the bits are nx.cpu's. Blocks fall
   into groups of a power of two of them, a subtree; an output of several
   groups leaves each group's state in the workspace, and the group that
   takes its ticket last merges them. A scan's thread folds one chunk in order from the carry into it:
   the slice's carry, or, past one chunk, the state a pass over the
   chunks' totals left, the given carry then each earlier chunk's total
   folded in order. A Sum, Max or Min scan along its destination's
   innermost axis is staged instead (Staged scans): the tape's terms go
   into the chunks' shared memory, scanned in nx.cuda's own order. */

#ifndef NX_CUDA_LOOP_CU
#define NX_CUDA_LOOP_CU

#include <type_traits>

#include "nx_kernel.h"
#include "words.cuh"

#define TB NX_LOOP_THREADS

namespace nx_loop {

/* Planes */

extern __shared__ __align__(16) uint32_t loop_planes[];

__device__ __forceinline__ uint32_t &word(int E, int p, int i) {
  return loop_planes[p * E + i];
}

template <class T> __device__ __forceinline__ T get(int E, int p, int i);
template <class T>
__device__ __forceinline__ void put(int E, int p, int i, T x);

__device__ __forceinline__ u64 get64(int E, int p, int i) {
  return word(E, p, i) | (u64)word(E, p + 1, i) << 32;
}

__device__ __forceinline__ void put64(int E, int p, int i, u64 x) {
  word(E, p, i) = (uint32_t)x;
  word(E, p + 1, i) = (uint32_t)(x >> 32);
}

template <> __device__ uint32_t get(int E, int p, int i) { return word(E, p, i); }
template <> __device__ int32_t get(int E, int p, int i) { return (int32_t)word(E, p, i); }
template <> __device__ float get(int E, int p, int i) { return __uint_as_float(word(E, p, i)); }
template <> __device__ u64 get(int E, int p, int i) { return get64(E, p, i); }
template <> __device__ int64_t get(int E, int p, int i) { return (int64_t)get64(E, p, i); }
template <> __device__ double get(int E, int p, int i) {
  return __longlong_as_double((long long)get64(E, p, i));
}
template <> __device__ nx_c64 get(int E, int p, int i) {
  return nx_c64_of(get<float>(E, p, i), get<float>(E, p + 1, i));
}
template <> __device__ nx_c128 get(int E, int p, int i) {
  return nx_c128_of(get<double>(E, p, i), get<double>(E, p + 2, i));
}

template <> __device__ void put(int E, int p, int i, uint32_t x) { word(E, p, i) = x; }
template <> __device__ void put(int E, int p, int i, int32_t x) { word(E, p, i) = (uint32_t)x; }
template <> __device__ void put(int E, int p, int i, float x) { word(E, p, i) = __float_as_uint(x); }
template <> __device__ void put(int E, int p, int i, u64 x) { put64(E, p, i, x); }
template <> __device__ void put(int E, int p, int i, int64_t x) { put64(E, p, i, (u64)x); }
template <> __device__ void put(int E, int p, int i, double x) {
  put64(E, p, i, (u64)__double_as_longlong(x));
}
template <> __device__ void put(int E, int p, int i, nx_c64 x) {
  put<float>(E, p, i, x.re);
  put<float>(E, p + 1, i, x.im);
}
template <> __device__ void put(int E, int p, int i, nx_c128 x) {
  put<double>(E, p, i, x.re);
  put<double>(E, p + 2, i, x.im);
}

/* A value of up to four words, as held: [lo] its words 0 and 1, [hi] its
   words 2 and 3. */
typedef struct {
  u64 lo, hi;
} held;

__device__ __forceinline__ held get_held(int E, int p, int i, int n) {
  held h = {word(E, p, i), 0};
  if (n > 1) h.lo |= (u64)word(E, p + 1, i) << 32;
  if (n > 2) h.hi = get64(E, p + 2, i);
  return h;
}

__device__ __forceinline__ void put_held(int E, int p, int i, int n, held h) {
  word(E, p, i) = (uint32_t)h.lo;
  if (n > 1) word(E, p + 1, i) = (uint32_t)(h.lo >> 32);
  if (n > 2) put64(E, p + 2, i, h.hi);
}

/* Each element of the calling thread: its slot v and its index i. EACH
   runs them in turn; ALL unrolled, a slot past V doing nothing, so that
   their shared-memory reads are in flight together. */
#define EACH                                                                   \
  _Pragma("unroll 1") for (int v = 0, i = threadIdx.x; v < V; v++, i += TB)
#define ALL                                                                    \
  _Pragma("unroll") for (int v = 0, i = threadIdx.x; v < NX_LOOP_MOST;        \
                         v++, i += TB) if (v < V)

/* The loop indices of the calling thread's elements: slot v's is
   L0 + v step for v below n; the others have none. */
typedef struct {
  int64_t L0;
  int step, n;
} lanes;

__device__ __forceinline__ int64_t index_of(lanes x, int v) {
  return x.L0 + (int64_t)v * x.step;
}

/* The lanes of a thread whose first element's index is [L0], stepping by
   [step], those below [end] alone having one, of [V] elements. */
__device__ __forceinline__ lanes lanes_of(int64_t L0, int step, int64_t end,
                                          int V) {
  int64_t r = end - L0;
  int n = r <= 0 ? 0 : (int)min((int64_t)V, (r + step - 1) / step);
  lanes x = {L0, step, n};
  return x;
}

/* Kinds

   One function per kind and compute type over a thread's elements, out of
   line, so that the cubin holds each kind once. A cheap kind's runs
   unrolled, its reads in flight together; a costly one's an element at a
   time, which keeps its code to one copy. */

#define K1(S, T, U, n)                                                         \
  __device__ __noinline__ void k_##n##_##S(int E, int V, int d, int a, int,    \
                                           int) {                              \
    T x[NX_LOOP_MOST] = {};                                                    \
    ALL x[v] = get<T>(E, a, i);                                                \
    ALL put<T>(E, d, i, nx_##n##_##S(x[v]));                                   \
  }
#define K2(S, T, U, n)                                                         \
  __device__ __noinline__ void k_##n##_##S(int E, int V, int d, int a, int b,  \
                                           int) {                              \
    T x[NX_LOOP_MOST] = {}, y[NX_LOOP_MOST] = {};                              \
    ALL {                                                                      \
      x[v] = get<T>(E, a, i);                                                  \
      y[v] = get<T>(E, b, i);                                                  \
    }                                                                          \
    ALL put<T>(E, d, i, nx_##n##_##S(x[v], y[v]));                             \
  }
#define KC(S, T, U, n)                                                         \
  __device__ __noinline__ void k_##n##_##S(int E, int V, int d, int a, int b,  \
                                           int) {                              \
    T x[NX_LOOP_MOST] = {}, y[NX_LOOP_MOST] = {};                              \
    ALL {                                                                      \
      x[v] = get<T>(E, a, i);                                                  \
      y[v] = get<T>(E, b, i);                                                  \
    }                                                                          \
    ALL put<uint32_t>(E, d, i, nx_##n##_##S(x[v], y[v]) != 0);                 \
  }
#define K3(S, T, U, n)                                                         \
  __device__ __noinline__ void k_##n##_##S(int E, int V, int d, int a, int b,  \
                                           int c) {                            \
    T x[NX_LOOP_MOST] = {}, y[NX_LOOP_MOST] = {}, z[NX_LOOP_MOST] = {};        \
    ALL {                                                                      \
      x[v] = get<T>(E, a, i);                                                  \
      y[v] = get<T>(E, b, i);                                                  \
      z[v] = get<T>(E, c, i);                                                  \
    }                                                                          \
    ALL put<T>(E, d, i, nx_##n##_##S(x[v], y[v], z[v]));                       \
  }
#define S1(S, T, U, n)                                                         \
  __device__ __noinline__ void k_##n##_##S(int E, int V, int d, int a, int,    \
                                           int) {                              \
    EACH put<T>(E, d, i, nx_##n##_##S(get<T>(E, a, i)));                       \
  }
#define S2(S, T, U, n)                                                         \
  __device__ __noinline__ void k_##n##_##S(int E, int V, int d, int a, int b,  \
                                           int) {                              \
    EACH put<T>(E, d, i, nx_##n##_##S(get<T>(E, a, i), get<T>(E, b, i)));      \
  }
#define SC(S, T, U, n)                                                         \
  __device__ __noinline__ void k_##n##_##S(int E, int V, int d, int a, int b,  \
                                           int) {                              \
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
__device__ __noinline__ void k_fma_narrow(int E, int V, int d, int a, int b,
                                          int c) {
  EACH put<float>(E, d, i,
                  nx_fma_narrow(get<float>(E, a, i), get<float>(E, b, i),
                                get<float>(E, c, i)));
}
__device__ __noinline__ void k_fma_bf16(int E, int V, int d, int a, int b,
                                        int c) {
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

/* The kind of op [op] (kernels.h): one switch over every op, a jump. */
__device__ __noinline__ void kind(int op, int E, int V, int d, int a, int b,
                                  int c) {
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

#undef C1
#undef C2
#undef C3
#undef K1
#undef K2
#undef KC
#undef K3
#undef S1
#undef S2
#undef SC

/* Dtypes

   [bits] of an element; the words its held value takes; its bits [r], the
   low bits of a word, as held: a signed integer sign-extended to 32 bits,
   the others as they are; and a held value's bits. */

/* log2 of each dtype's bits, three bits a code (nx_dtype.h's table). */
constexpr int log2_of(int b) { return b <= 1 ? 0 : 1 + log2_of(b / 2); }
#define LOG_BITS(NAME, name, bits, kind) | (u64)log2_of(bits) << (3 * NX_##NAME)
constexpr u64 log_bits = 0 NX_DTYPES(LOG_BITS);
#undef LOG_BITS

__device__ __forceinline__ int bits_of(int dt) {
  return 1 << ((log_bits >> (3 * dt)) & 7);
}

__device__ __forceinline__ int words_of(int dt) {
  int b = bits_of(dt);
  return b <= 32 ? 1 : b / 32;
}

__device__ u64 of_raw(int dt, u64 r) {
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

__device__ u64 raw_of(int dt, u64 h) {
  int b = bits_of(dt);
  return b >= 64 ? h : h & ((1ull << b) - 1);
}

/* Casts */

__device__ bool is_float(int dt) { return dt <= NX_FLOAT4_E2M1FN; }
__device__ bool is_complex(int dt) {
  return dt == NX_COMPLEX64 || dt == NX_COMPLEX128;
}
__device__ bool is_signed(int dt) {
  return dt == NX_INT64 || dt == NX_INT32 || dt == NX_INT16 || dt == NX_INT8 ||
         dt == NX_INT4;
}
__device__ bool is_bool(int dt) { return dt == NX_BOOL || dt == NX_BIT; }

/* A float's held [lo] of [dt] as a double, exactly. */
__device__ double float_value(int dt, u64 lo) {
  if (dt == NX_FLOAT64) return __longlong_as_double((long long)lo);
  if (dt == NX_FLOAT32) return __uint_as_float((uint32_t)lo);
  return nx_bits_to_float(dt, (uint32_t)lo);
}

/* [x] held of the real dtype [from] cast into the real dtype [to], held,
   as Prog.Cast stores it: each value rounded once. A float32 that keeps its
   format keeps its bits. */
__device__ __noinline__ held cast_real(int from, int to, u64 x) {
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
    r.lo = (u64)__double_as_longlong(
        !integer ? f : unsig ? (double)(u64)s : (double)s);
    return r;
  case NX_FLOAT32:
    r.lo = __float_as_uint(!integer ? (float)f
                           : unsig  ? (float)(u64)s
                                    : (float)s);
    return r;
  }
  if (is_float(to)) {
    /* A float32 or narrower rounds once from its float32; a float64 or an
       integer from its double rounded to odd. */
    if (!integer && from != NX_FLOAT64) r.lo = nx_narrow_code(to, (float)f);
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

__device__ __noinline__ held cast(int from, int to, held x) {
  if (from == to) return x;
  if (!is_complex(from)) {
    held r = cast_real(from, to == NX_COMPLEX64    ? NX_FLOAT32
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
    re = __uint_as_float((uint32_t)x.lo);
    im = __uint_as_float((uint32_t)(x.lo >> 32));
  } else {
    re = __longlong_as_double((long long)x.lo);
    im = __longlong_as_double((long long)x.hi);
  }
  switch (to) {
  case NX_BOOL:
  case NX_BIT: r.lo = re != 0 || im != 0; return r;
  case NX_COMPLEX64:
    r.lo = __float_as_uint((float)re) |
           (u64)__float_as_uint((float)im) << 32;
    return r;
  case NX_COMPLEX128:
    r.lo = (u64)__double_as_longlong(re);
    r.hi = (u64)__double_as_longlong(im);
    return r;
  }
  return cast_real(part, to, bre);
}

/* [x] of [from] read in [to], of one width. */
__device__ held bitcast(int from, int to, held x) {
  if (bits_of(from) == 128) return x;
  held r = {of_raw(to, raw_of(from, x.lo)), 0};
  return r;
}

/* Memory */

/* Whether [b]'s 64-bit elements lie on 8 bytes. A complex64 array is
   promised its components' 4 bytes alone, and its elements' alignment is
   its first byte's, the same for each element: off 8 bytes, each of its
   elements takes two 4-byte accesses. */
__device__ __forceinline__ bool on8(const uint8_t *b) {
  return ((u64)b & 7) == 0;
}

/* The held value of [dt] at element [p] of [b]. */
__device__ __noinline__ held read(const uint8_t *b, int64_t p, int dt) {
  held h = {0, 0};
  switch (bits_of(dt)) {
  case 1: h.lo = (b[p >> 3] >> (p & 7)) & 1; break;
  case 4: h.lo = (b[p >> 1] >> ((p & 1) * 4)) & 15; break;
  case 8: h.lo = b[p]; break;
  case 16: h.lo = ((const uint16_t *)b)[p]; break;
  case 32: h.lo = ((const uint32_t *)b)[p]; break;
  case 64:
    if (on8(b)) h.lo = ((const u64 *)b)[p];
    else
      h.lo = ((const uint32_t *)b)[2 * p] |
             (u64)((const uint32_t *)b)[2 * p + 1] << 32;
    break;
  default:
    h.lo = ((const u64 *)b)[2 * p];
    h.hi = ((const u64 *)b)[2 * p + 1];
    return h;
  }
  h.lo = of_raw(dt, h.lo);
  return h;
}

/* Stores [h] of [dt] at element [p] of [b], a sub-byte one into the word
   that holds it (nx_store_bits). */
__device__ __noinline__ void write(uint8_t *b, int64_t p, int dt, held h) {
  int n = bits_of(dt);
  switch (n) {
  case 1:
  case 4: nx_store_bits(b, p, n, (uint32_t)h.lo); return;
  case 8: b[p] = (uint8_t)h.lo; return;
  case 16: ((uint16_t *)b)[p] = (uint16_t)h.lo; return;
  case 32: ((uint32_t *)b)[p] = (uint32_t)h.lo; return;
  case 64:
    if (on8(b)) {
      ((u64 *)b)[p] = h.lo;
      return;
    }
    ((uint32_t *)b)[2 * p] = (uint32_t)h.lo;
    ((uint32_t *)b)[2 * p + 1] = (uint32_t)(h.lo >> 32);
    return;
  default:
    ((u64 *)b)[2 * p] = h.lo;
    ((u64 *)b)[2 * p + 1] = h.hi;
  }
}

/* Walks */

/* The loop's [rank] axes' extents and index [L]'s coordinate along axis
   [a]. */
__device__ __noinline__ int64_t coordinate(const u64 *w, int64_t L, int a) {
  int rank = (int)w[NX_LOOP_RANK];
  const u64 *ext = w + w[NX_LOOP_SHAPE];
  for (int k = rank - 1; k > a; k--) L /= (int64_t)ext[k];
  return L % (int64_t)ext[a];
}

/* The position of a plain operand's element at loop index [L]: its
   descriptor [d], the first position then a stride per axis. In 32 bits
   where the loop's indices fit them. */
__device__ __noinline__ int64_t place(const u64 *w, const u64 *d, int64_t L) {
  int rank = (int)w[NX_LOOP_RANK];
  const u64 *ext = w + w[NX_LOOP_SHAPE];
  const int64_t *st = (const int64_t *)d + 2;
  int64_t at = (int64_t)d[1];
  if (rank == 0) return at;
  if (w[NX_LOOP_NUMEL] >> 32 == 0) {
    uint32_t l = (uint32_t)L;
    for (int k = rank - 1; k > 0; k--) {
      uint32_t e = (uint32_t)ext[k], q = l / e;
      at += (int64_t)(l - q * e) * st[k];
      l = q;
    }
    return at + (int64_t)l * st[0];
  }
  for (int k = rank - 1; k > 0; k--) {
    int64_t e = (int64_t)ext[k], q = L / e;
    at += (L - q * e) * st[k];
    L = q;
  }
  return at + L * st[0];
}

/* The position a padded operand reads at loop index [L], or -1 where it
   reads its fill: per axis of the operand, the padded coordinate from the
   loop axes it reads less lo, where it lands on an element. Each axis
   takes its loop coordinates from one pass down the loop's axes, which
   keeps no coordinates beside it. */
__device__ __noinline__ int64_t padded_place(const u64 *w, const u64 *d, int64_t L) {
  int rank = (int)w[NX_LOOP_RANK], r = (int)d[4];
  bool narrow = w[NX_LOOP_NUMEL] >> 32 == 0;
  const u64 *ext = w + w[NX_LOOP_SHAPE];
  int64_t at = (int64_t)d[1];
  for (int j = 0; j < r; j++) {
    const int64_t *a = (const int64_t *)d + 5 + NX_LOOP_PAD * j;
    int a1 = (int)(a[4] & 0xffff), a2 = (int)((a[4] >> 16) & 0xffff);
    int low = a2 == 0xffff || a1 < a2 ? a1 : a2;
    int64_t q = -a[0], l = L;
    for (int k = rank - 1; k >= low; k--) {
      /* In 32 bits where the loop's indices fit them, as place. */
      int64_t e = (int64_t)ext[k];
      int64_t n = narrow ? (int64_t)((uint32_t)l / (uint32_t)e) : l / e;
      int64_t c = l - n * e;
      if (k == a1) q += c * a[5];
      if (k == a2) q += c * a[6];
      l = n;
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

__device__ __forceinline__ const u64 *operand(const u64 *w, int table, int k) {
  return w + w[w[table] + k];
}

__device__ __forceinline__ uint8_t *buffer(const u64 *w, const u64 *d) {
  return (uint8_t *)w[NX_LOOP_HEADER + ((d[0] >> 16) & 0xffff)];
}

/* Load [k]'s held element at loop index [L], which is an index. */
__device__ __noinline__ held load_at(const u64 *w, int k, int64_t L) {
  const u64 *d = operand(w, NX_LOOP_LOADS, k);
  int dt = (int)(d[0] & 0xff);
  if ((d[0] >> 8) & 1) {
    int64_t p = padded_place(w, d, L);
    if (p < 0) {
      held f = {of_raw(dt, d[2]), d[3]};
      return f;
    }
    return read(buffer(w, d), p, dt);
  }
  return read(buffer(w, d), place(w, d, L), dt);
}

/* A plain operand's bytes along the thread's elements, by its descriptor
   [d] and its elements' [bytes], for a loop of at most two axes: element
   v's position steps from the first index's by the step's coordinates, a
   division an instruction, its byte at + v's sum of the steps, plus
   [wrap] each time the last axis's coordinate passes its extent. */
typedef struct {
  int64_t at, step, wrap, c, dc, e;
} walker;

__device__ __forceinline__ walker walker_of(const u64 *w, const u64 *d,
                                            lanes x, int bytes) {
  int rank = (int)w[NX_LOOP_RANK];
  walker k;
  int64_t s0 = rank > 0 ? (int64_t)d[2] * bytes : 0;
  if (rank <= 1) {
    k.at = ((int64_t)d[1] * bytes) + x.L0 * s0;
    k.step = x.step * s0;
    k.wrap = 0;
    k.c = 0;
    k.dc = 0;
    k.e = 1;
    return k;
  }
  int64_t s1 = (int64_t)d[3] * bytes, e = (int64_t)w[w[NX_LOOP_SHAPE] + 1];
  int64_t r, dr;
  if (w[NX_LOOP_NUMEL] >> 32 == 0) {
    r = (uint32_t)x.L0 / (uint32_t)e;
    dr = (uint32_t)x.step / (uint32_t)e;
  } else {
    r = x.L0 / e;
    dr = x.step / e;
  }
  k.c = x.L0 - r * e;
  k.dc = x.step - dr * e;
  k.e = e;
  k.at = (int64_t)d[1] * bytes + r * s0 + k.c * s1;
  k.step = dr * s0 + k.dc * s1;
  k.wrap = s0 - e * s1;
  return k;
}

/* The walker's byte, then the next element's. */
__device__ __forceinline__ int64_t next(walker &k) {
  int64_t at = k.at;
  k.at += k.step;
  k.c += k.dc;
  if (k.c >= k.e) {
    k.c -= k.e;
    k.at += k.wrap;
  }
  return at;
}

/* Copies the 4 bytes at [b] + [at] to the shared byte [to], or, where
   not [in], zeros, asynchronously (cp.async), without a branch. */
__device__ __forceinline__ void copy4(unsigned to, const uint8_t *b, int64_t at,
                                      bool in) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(to),
               "l"(in ? b + at : b), "r"(in ? 4 : 0)
               : "memory");
}

/* Load [k]'s elements at the thread's indices into plane [d], a padded
   load's, a load of fewer than 8 bits' or a loop's of more than two axes:
   each element in turn, an element of no index taking zeros. */
__device__ __noinline__ void load_each(const u64 *w, int E, int V, lanes x,
                                       int d, int k) {
  const u64 *desc = operand(w, NX_LOOP_LOADS, k);
  int n = words_of((int)(desc[0] & 0xff));
  EACH {
    held h = {0, 0};
    if (v < x.n) h = load_at(w, k, index_of(x, v));
    put_held(E, d, i, n, h);
  }
}

/* Load [k]'s elements at the thread's indices into plane [d]. A load of
   words copies each word asynchronously (cp.async), so that every load of
   a tile is in flight at once until the tape's wait, an element of no
   index taking zeros; a narrower one reads its elements before it stores
   any. In line in the tape's runner, which then makes no call. */
__device__ __forceinline__ void load(const u64 *w, int E, int V, lanes x,
                                     int d, int k) {
  const u64 *desc = operand(w, NX_LOOP_LOADS, k);
  int dt = (int)(desc[0] & 0xff), n = words_of(dt), bits = bits_of(dt);
  if (((desc[0] >> 8) & 1) || bits < 8 || w[NX_LOOP_RANK] > 2) {
    load_each(w, E, V, x, d, k);
    return;
  }
  const uint8_t *b = buffer(w, desc);
  walker q = walker_of(w, desc, x, bits >= 32 ? 4 * n : bits / 8);
  if (bits >= 32) {
    /* An element of no index copies 0 bytes of [b], which fills its words
       with zeros: the copies take no branch. */
    unsigned to = (unsigned)__cvta_generic_to_shared(&word(E, d, threadIdx.x));
    if (n == 1) {
      ALL copy4(to + 4 * v * TB, b, next(q), v < x.n);
      return;
    }
#pragma unroll 1
    for (int j = 0; j < n; j++) {
      walker r = q;
      ALL copy4(to + 4 * (j * E + v * TB), b, next(r) + 4 * j, v < x.n);
    }
    return;
  }
  uint32_t y[NX_LOOP_MOST] = {};
  if (bits == 16) {
    ALL {
      int64_t at = next(q);
      if (v < x.n) y[v] = *(const uint16_t *)(b + at);
    }
  } else {
    ALL {
      int64_t at = next(q);
      if (v < x.n) y[v] = b[at];
    }
  }
  ALL word(E, d, i) = (uint32_t)of_raw(dt, y[v]);
}

/* Stores plane [a]'s elements at the thread's indices into destination
   [k], placed as load places them, each in turn. */
__device__ __noinline__ void store_each(const u64 *w, int E, int V, lanes x,
                                        int a, int k) {
  const u64 *desc = operand(w, NX_LOOP_DSTS, k);
  int dt = (int)(desc[0] & 0xff), n = words_of(dt);
  uint8_t *b = buffer(w, desc);
  EACH if (v < x.n)
      write(b, place(w, desc, index_of(x, v)), dt, get_held(E, a, i, n));
}

/* Stores plane [a]'s elements at the thread's indices into destination
   [k], placed as load places them: elements of 8 to 32 bits of a loop of
   at most two axes, each read before any is stored, in line; others by
   store_each. */
__device__ __forceinline__ void store(const u64 *w, int E, int V, lanes x,
                                      int a, int k) {
  const u64 *desc = operand(w, NX_LOOP_DSTS, k);
  int bits = bits_of((int)(desc[0] & 0xff));
  if (bits > 32 || bits < 8 || w[NX_LOOP_RANK] > 2) {
    store_each(w, E, V, x, a, k);
    return;
  }
  uint8_t *b = buffer(w, desc);
  walker q = walker_of(w, desc, x, bits / 8);
  uint32_t y[NX_LOOP_MOST] = {};
  ALL y[v] = word(E, a, i);
  if (bits == 32) {
    ALL {
      int64_t at = next(q);
      if (v < x.n) *(uint32_t *)(b + at) = y[v];
    }
  } else if (bits == 16) {
    ALL {
      int64_t at = next(q);
      if (v < x.n) *(uint16_t *)(b + at) = (uint16_t)y[v];
    }
  } else {
    ALL {
      int64_t at = next(q);
      if (v < x.n) b[at] = (uint8_t)y[v];
    }
  }
}

/* The tape */

/* Instruction [op] (kernels.h) of planes [d], [a], [b] and [c] over each
   element in turn: the runner's rarer instructions, out of its line. */
__device__ __noinline__ void each_op(const u64 *w, int E, int V, lanes x,
                                     int op, int d, int a, int b, int c,
                                     held h) {
  switch (op) {
  case NX_LOOP_COORD:
    EACH put<int64_t>(E, d, i, v < x.n ? coordinate(w, index_of(x, v), a) : 0);
    return;
  case NX_LOOP_CONST: EACH put_held(E, d, i, b, h); return;
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
    EACH word(E, d, i) = nx_narrow_code(b, get<float>(E, a, i));
    return;
  case NX_LOOP_WRAP:
    EACH word(E, d, i) = (uint32_t)of_raw(b, word(E, a, i));
    return;
  }
}

/* Runs the tape's [n] words from [from]. Its loads, stores, waits,
   selections and the narrow floats' codecs of 16 bits run in line,
   unrolled over the thread's elements; the kinds and rarer instructions
   are calls. */
__device__ __noinline__ void run(const u64 *w, int E, int V, lanes x, int from,
                                 int n) {
  /* The word and the end in one register, [pc]'s low and high halves: the
     calls leave too few registers for both, and an end spilled to local
     memory, whose lines the loads' copies evict, costs a trip to L2 an
     instruction. */
  for (uint32_t pc = (uint32_t)from | (uint32_t)(from + n) << 16;
       (pc & 0xffff) < pc >> 16; pc++) {
    int at = (int)(pc & 0xffff);
    u64 t = w[at];
    int op = (int)(t & 0xffff), d = (int)((t >> 16) & 0xfff);
    int a = (int)((t >> 28) & 0xfff), b = (int)((t >> 40) & 0xfff);
    int c = (int)((t >> 52) & 0xfff);
    if (op < NX_LOOP_LOAD) {
      kind(op, E, V, d, a, b, c);
      continue;
    }
    switch (op) {
    case NX_LOOP_LOAD: load(w, E, V, x, d, a); break;
    case NX_LOOP_WAIT: asm volatile("cp.async.wait_all;\n" ::: "memory"); break;
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
      int n = op == NX_LOOP_WHERE2 ? 2 : 4;
      EACH put_held(E, d, i, n, get_held(E, word(E, a, i) != 0 ? b : c, i, n));
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
        each_op(w, E, V, x, op, d, a, b, c, held{0, 0});
        break;
      }
      uint32_t y[NX_LOOP_MOST] = {};
      ALL y[v] = word(E, a, i);
      if (op == NX_LOOP_DECODE && b == NX_BFLOAT16) {
        ALL put<float>(E, d, i, nx_bf16_to_float((uint16_t)y[v]));
      } else if (op == NX_LOOP_DECODE) {
        ALL put<float>(E, d, i, nx_f16_to_float((uint16_t)y[v]));
      } else if (b == NX_BFLOAT16) {
        ALL word(E, d, i) = nx_float_to_bf16(__uint_as_float(y[v]));
      } else {
        ALL word(E, d, i) = nx_float_to_f16(__uint_as_float(y[v]));
      }
      break;
    }
    case NX_LOOP_CONST:
      each_op(w, E, V, x, op, d, a, b, c, held{w[at + 1], w[at + 2]});
      pc += 2;
      break;
    default: each_op(w, E, V, x, op, d, a, b, c, held{0, 0});
    }
  }
}

/* Reductions

   A state of an output's terms, held in NX_LOOP_STATE planes or workspace
   words: its value in words 0 to 3, held at its compute type; a position
   or a second sum in 4 and 5; whether it holds a term in 6. A sum's lanes
   start from +0 and every lane takes part in the tree, as nx.cpu's do: a
   lane from +0 is never -0, so a lane of no term, which adds +0, changes
   no bit. Another monoid's lane, and Arg's, start from their first term,
   a lane of no term taking no part.

   States are words: one walk serves every compute type, which only the
   few kinds a reduction combines states by read (arith, order). */

typedef struct {
  held v;
  u64 x;
  uint32_t has;
} st;

/* A held value at the compute type T, and back. */
template <class T> __device__ __forceinline__ T as(held h) {
  T x;
  u64 b[2] = {h.lo, h.hi};
  memcpy(&x, b, sizeof(T));
  return x;
}

template <class T> __device__ __forceinline__ held of(T x) {
  u64 b[2] = {0, 0};
  memcpy(b, &x, sizeof(T));
  held h = {b[0], b[1]};
  return h;
}

/* [a] op [b] at the compute type [ct], op NX_OP2_ADD, NX_OP2_MUL,
   NX_OP2_MAXIMUM or NX_OP2_MINIMUM. */
#define ARITH(S, T)                                                            \
  {                                                                            \
    T x = as<T>(a), y = as<T>(b);                                              \
    switch (op) {                                                              \
    case NX_OP2_ADD: return of<T>(nx_add_##S(x, y));                           \
    case NX_OP2_MUL: return of<T>(nx_mul_##S(x, y));                           \
    case NX_OP2_MAXIMUM: return of<T>(nx_maximum_##S(x, y));                   \
    default: return of<T>(nx_minimum_##S(x, y));                               \
    }                                                                          \
  }

__device__ __noinline__ held arith(int ct, int op, held a, held b) {
  switch (ct) {
  case NX_LOOP_F32: ARITH(f32, float)
  case NX_LOOP_F64: ARITH(f64, double)
  case NX_LOOP_C64: ARITH(c64, nx_c64)
  case NX_LOOP_C128: ARITH(c128, nx_c128)
  case NX_LOOP_I32: ARITH(i32, int32_t)
  case NX_LOOP_U32: ARITH(u32, uint32_t)
  case NX_LOOP_I64: ARITH(i64, int64_t)
  default: ARITH(u64, u64)
  }
}

#undef ARITH

/* Arg's comparison of [a] and [b] at [ct]: bit 0 if [a] is a NaN, bit 1 if
   [b] is, and, neither a NaN, bit 2 if [a] follows [b] in the dtype's
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

__device__ __noinline__ int order(int ct, held a, held b) {
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

/* The words a value at the compute type [ct] takes. */
__device__ int ct_words(int ct) {
  return ct == NX_LOOP_C128 ? 4
         : ct == NX_LOOP_F32 || ct == NX_LOOP_I32 || ct == NX_LOOP_U32 ? 1
                                                                       : 2;
}

/* Arg's state [a] then [b]: the extreme, a NaN first, and of one value
   the earlier position. */
__device__ st arg_merge(int rk, int ct, st a, st b) {
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
__device__ __noinline__ st merge(int rk, int ct, st a, st b) {
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

/* Logsumexp's exp of [y], 0 below the cut (nx_cpu.mli). */
__device__ float ex(float y) { return y < -87.f ? 0.f : nx_exp_f32(y); }
__device__ double ex(double y) { return y < -708. ? 0. : nx_exp_f64(y); }

/* A centered term [x] of center [c]: exp (x - c), or (x - c)^2 with the
   second sum x - c. */
template <class T> __device__ st centered(int rk, held x, held c) {
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
__device__ st fold(int rk, int ct, st a, held x, int64_t pos, held c) {
  st b = {x, (u64)pos, 1};
  if (rk == NX_LOOP_EXPS || rk == NX_LOOP_SQUARES)
    b = ct == NX_LOOP_F32 ? centered<float>(rk, x, c)
                          : centered<double>(rk, x, c);
  return merge(rk, ct, a, b);
}

__device__ st empty(int rk) {
  st s = {{0, 0}, 0, 0};
  s.has = rk == NX_LOOP_SUM || rk == NX_LOOP_EXPS || rk == NX_LOOP_SQUARES;
  return s;
}

__device__ st load_state(int E, int p, int i) {
  st s = {{get64(E, p, i), get64(E, p + 2, i)}, get64(E, p + 4, i),
          word(E, p + 6, i)};
  return s;
}

__device__ void store_state(int E, int p, int i, st s) {
  put64(E, p, i, s.v.lo);
  put64(E, p + 2, i, s.v.hi);
  put64(E, p + 4, i, s.x);
  word(E, p + 6, i) = s.has;
}

/* A state in the workspace, NX_LOOP_STATE words as in planes, read and
   written past the caches another group's writes miss. */
__device__ u64 ldcg64(const uint32_t *g) {
  return __ldcg(g) | (u64)__ldcg(g + 1) << 32;
}

__device__ void stcg64(uint32_t *g, u64 x) {
  __stcg(g, (uint32_t)x);
  __stcg(g + 1, (uint32_t)(x >> 32));
}

__device__ st load_work(const uint32_t *g) {
  st s = {{ldcg64(g), ldcg64(g + 2)}, ldcg64(g + 4), __ldcg(g + 6)};
  return s;
}

__device__ void store_work(uint32_t *g, st s) {
  stcg64(g, s.v.lo);
  stcg64(g + 2, s.v.hi);
  stcg64(g + 4, s.x);
  __stcg(g + 6, s.has);
}

/* A value at [ct] as its carrier holds it, an integer wrapped to it; and
   a carrier's held value at [ct], a boolean as 0 or 1. */
__device__ held to_carrier(int carrier, int ct, held x) {
  if (ct >= NX_LOOP_I32) x.lo = of_raw(carrier, x.lo);
  return x;
}

__device__ held of_carrier(int carrier, held h) {
  if (is_bool(carrier)) h.lo = h.lo != 0;
  return h;
}

/* The workspace's byte [field] of the header, then state [k] past it. */
__device__ __forceinline__ uint8_t *work_byte(const u64 *w, int field) {
  return (uint8_t *)w[NX_LOOP_HEADER + w[NX_LOOP_WORK]] + w[field];
}

__device__ __forceinline__ uint32_t *work_at(const u64 *w, int field,
                                             int64_t k) {
  return (uint32_t *)work_byte(w, field) + k * NX_LOOP_STATE;
}

/* Stores [x] of the dtype [from] cast into destination [k] at its element
   [o] past its first. */
__device__ __noinline__ void put_dst(const u64 *w, int k, int64_t o, int from, held x) {
  const u64 *d = operand(w, NX_LOOP_DSTS, k);
  int dt = (int)(d[0] & 0xff);
  write(buffer(w, d), (int64_t)d[1] + o, dt, cast(from, dt, x));
}

/* nx_kinds.h's log and maximum at the float T. */
template <class T> __device__ T log_(T x) {
  if constexpr (sizeof(T) == 4) return nx_log_f32(x);
  else return nx_log_f64(x);
}

template <class T> __device__ T max_(T a, T b) {
  if constexpr (sizeof(T) == 4) return nx_maximum_f32(a, b);
  else return nx_maximum_f64(a, b);
}

/* Logsumexp's and Moments' finishes of output [o] of [n] terms from its
   state [st], by [fin], at the float T. */
template <class T>
__device__ void finish_real(const u64 *w, int fin, int carrier, int64_t o,
                            int64_t n, st s) {
  T *centers = (T *)work_byte(w, NX_LOOP_CENTERS);
  T c = centers[o], v = as<T>(s.v);
  switch (fin) {
  case NX_LOOP_PUT_CENTER: centers[o] = v - v == 0 ? v : T(0); return;
  case NX_LOOP_PUT_MEAN: centers[o] = v / (T)n; return;
  case NX_LOOP_PUT_LOGSUMEXP: {
    put_dst(w, 0, o, carrier, of<T>(T(c + log_(v))));
    return;
  }
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
__device__ __noinline__ void finish(const u64 *w, int rk, int ct, int fin,
                                    int carrier, int64_t o, int64_t n,
                                    st s) {
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

/* Each element's state of the calling thread, from none. */
__device__ __noinline__ void init_all(int rk, int E, int V, int sp) {
  st e = empty(rk);
  EACH store_state(E, sp, i, e);
}

/* Each element's term, of the term plane, folded into its state. A sum
   adds every element's by the kind, an element of no index adding +0. */
__device__ __noinline__ void fold_all(const u64 *w, int rk, int ct, int E,
                                      int V, lanes x) {
  int tp = (int)w[NX_LOOP_TERM];
  int sp = (int)w[NX_LOOP_STATE_AT], cp = (int)w[NX_LOOP_CENTER];
  int64_t R = (int64_t)w[NX_LOOP_TERMS];
  if (rk == NX_LOOP_SUM) {
    int n = ct_words(ct);
    EACH if (v >= x.n) for (int j = 0; j < n; j++) word(E, tp + j, i) = 0;
    kind((ct << 7) + NX_LOOP_BINARY + NX_OP2_ADD, E, V, sp, sp, tp, 0);
    return;
  }
  bool c = rk == NX_LOOP_EXPS || rk == NX_LOOP_SQUARES;
  EACH {
    if (v >= x.n) continue;
    held z = {0, 0};
    st s = fold(rk, ct, load_state(E, sp, i), get_held(E, tp, i, 4),
                index_of(x, v) % R,
                c ? get_held(E, cp, i, 2) : z);
    store_state(E, sp, i, s);
  }
}

/* Element [i]'s state then element [j]'s, into [i]'s. */
__device__ __noinline__ void merge_at(int rk, int ct, int E, int sp, int i,
                                      int j) {
  store_state(E, sp, i,
              merge(rk, ct, load_state(E, sp, i), load_state(E, sp, j)));
}

/* An output's last group, or its one: its state goes out, or, with [m]
   groups, into the workspace, the last group to arrive merging the [m]
   states by the pairwise tree, in place, and sending it out. */
__device__ __noinline__ void put_group(const u64 *w, int rk, int ct, int fin,
                                       int carrier, int E, int i, int64_t o,
                                       int64_t q, int64_t m, int64_t n) {
  st s = load_state(E, (int)w[NX_LOOP_STATE_AT], i);
  if (m == 1) {
    finish(w, rk, ct, fin, carrier, o, n, s);
    return;
  }
  uint32_t *g = work_at(w, NX_LOOP_PARTIALS, o * m);
  store_work(g + q * NX_LOOP_STATE, s);
  __threadfence();
  uint32_t *tickets = (uint32_t *)w[NX_LOOP_HEADER + w[NX_LOOP_TICKETS]];
  if (atomicAdd(tickets + o, 1u) != (uint32_t)(m - 1)) return;
  tickets[o] = 0;
  __threadfence();
  for (int64_t step = 1; step < m; step *= 2)
    for (int64_t k = 0; k + step < m; k += 2 * step) {
      store_work(g + k * NX_LOOP_STATE,
                 merge(rk, ct, load_work(g + k * NX_LOOP_STATE),
                       load_work(g + (k + step) * NX_LOOP_STATE)));
    }
  finish(w, rk, ct, fin, carrier, o, n, load_work(g));
}

/* Defined among the plain reductions, whose folds it shares. */
__device__ __noinline__ void fold_lanes(const u64 *w, int rk, int ct, int E,
                                        int G, int R1, int64_t L0,
                                        int64_t end);

/* A reduction: units of an output's group of [g] blocks, [per] units a
   round of a group of threads. */
__device__ void reduce_mode(const u64 *w, int E, int V) {
  int G = (int)((w[NX_LOOP_MODE] >> 16) & 0x1ff), LB = TB / G, R1 = G * V;
  int64_t g = (int64_t)w[NX_LOOP_GROUP], m = (int64_t)w[NX_LOOP_GROUPS];
  int64_t O = (int64_t)w[NX_LOOP_OUTPUTS], R = (int64_t)w[NX_LOOP_TERMS];
  int64_t SEG = (int64_t)w[NX_LOOP_SEGMENT];
  int64_t segs = (int64_t)w[NX_LOOP_SEGMENTS], runs = (int64_t)w[NX_LOOP_RUNS];
  int64_t per = LB / g, units = O * m;
  u64 r = w[NX_LOOP_REDUCTION];
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff), fin = (int)((r >> 24) & 0xff);
  /* States: an element's, at [SE] = E, where a thread holds lanes; else a
     thread's, at SE = TB, past the planes. */
  int sp = (int)w[NX_LOOP_STATE_AT], SE = G < NX_LOOP_LANES ? E : TB;
  int s = threadIdx.x / G, t = threadIdx.x % G;
  int body = (int)(w[NX_LOOP_TAPE] + w[NX_LOOP_PROLOGUE]);
  for (int64_t r0 = blockIdx.x * per; r0 < units; r0 += gridDim.x * per) {
    /* A round's units side by side: slot s is block s / per of unit
       s % per, so that a warp reads one block of several outputs, which a
       reduction across rows finds side by side in memory. */
    int64_t k = s / per, u = r0 + s % per, o = u / m, q = u % m;
    int64_t b = q * g + k;
    int64_t start = (o % segs) * SEG, len = min(SEG, R - start);
    init_all(rk, SE, SE / TB, sp);
    int64_t end = u < units ? o / segs * R + start + len : 0;
    for (int64_t j = 0; j < runs; j++) {
      int64_t L0 = o / segs * R + start + b * NX_LOOP_BLOCK + j * R1;
      lanes x = lanes_of(L0 + t, G, end, V);
      run(w, E, V, x, body, (int)w[NX_LOOP_BODY]);
      if (R1 <= NX_LOOP_LANES) fold_all(w, rk, ct, E, V, x);
      else fold_lanes(w, rk, ct, E, G, R1, L0, end);
    }
    __syncthreads();
    /* The lanes' tree: lane l takes lane l + h for l below h; lanes past
       a run's R1 hold no term and take no part. */
    for (int h = min(NX_LOOP_LANES, R1) / 2; h > 0; h /= 2) {
      if (h >= G)
        for (int v = 0; v < h / G; v++)
          merge_at(rk, ct, SE, sp, v * TB + threadIdx.x,
                   (v + h / G) * TB + threadIdx.x);
      else if (t < h)
        merge_at(rk, ct, SE, sp, threadIdx.x, threadIdx.x + h);
      __syncthreads();
    }
    /* The group's blocks by the pairwise tree, a slot of no block taking
       no part. */
    int64_t nb = (len + NX_LOOP_BLOCK - 1) / NX_LOOP_BLOCK;
    int64_t count = u < units ? min(g, nb - q * g) : 0;
    for (int64_t step = 1; step < g; step *= 2) {
      if (t == 0 && k % (2 * step) == 0 && k + step < count)
        merge_at(rk, ct, SE, sp, threadIdx.x,
                 threadIdx.x + (int)(step * per) * G);
      __syncthreads();
    }
    if (t == 0 && k == 0 && u < units)
      put_group(w, rk, ct, fin, carrier, SE, threadIdx.x, o, q, m, len);
    __syncthreads();
  }
}

/* Plain reductions

   A Sum, Max or Min of one plain load at its carrier, its terms read
   straight into registers in nx.cpu's lane order, 16 loads a thread in
   flight. A lane starts from the reduction's identity, which leaves the
   value it is folded into as it was (x + +0 of a lane from +0, max x
   -inf, min x +inf), so a term or a lane of none takes no part.

   Rows, where an output's terms lie side by side: sixteen threads, a
   cell, take a block, thread l holding lane l, and the lanes' tree is
   four shuffles. A group's blocks of an output merge by the pairwise tree
   in shared memory; an output of several groups leaves each group's value
   in the workspace, and the group that takes its ticket last merges them
   by the same tree, in shared memory, in aligned chunks of
   NX_LOOP_PLAIN_CHUNK.

   Columns, where neighbouring outputs lie side by side: a group takes 32
   of them, a warp reading a row's 32 at once, thread t of warp w holding
   lanes w and w + 8 of output t; it takes each of their blocks in turn,
   at most NX_LOOP_PLAIN_TALL, and merges them in shared memory.

   Values are their bits in a word, of 4 or 8 bytes. The folds are typed;
   the walks, the trees and the merges are not, and combine values by
   plain_op. */

/* The bits of the identity of the reduction [rk] at [ct]. */
__device__ u64 plain_id(int ct, int rk) {
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

/* [a] then [b] by the reduction [rk], at T. */
template <class T> __device__ __forceinline__ T plain_at(int rk, T a, T b) {
  switch (rk) {
  case NX_LOOP_SUM: return k_add(a, b);
  case NX_LOOP_MAX: return k_maximum(a, b);
  default: return k_minimum(a, b);
  }
}

#define PLAIN_TYPES(X)                                                         \
  X(NX_LOOP_F32, float) X(NX_LOOP_F64, double) X(NX_LOOP_I32, int32_t)       \
  X(NX_LOOP_U32, uint32_t) X(NX_LOOP_I64, int64_t) X(NX_LOOP_U64, uint64_t)

/* [a] then [b], bits, by the reduction [rk] at [ct]. */
__device__ __noinline__ u64 plain_op(int ct, int rk, u64 a, u64 b) {
  switch (ct) {
#define CASE(C, T)                                                             \
  case C:                                                                      \
    return of<T>(plain_at<T>(rk, as<T>(held{a, 0}), as<T>(held{b, 0}))).lo;
    PLAIN_TYPES(CASE)
#undef CASE
  }
  return 0;
}

/* A value of T, a plain type, at element [i] of plane [p], and back. */
template <class T> __device__ __forceinline__ T get_plain(int E, int p, int i) {
  if constexpr (sizeof(T) == 4) return as<T>(held{word(E, p, i), 0});
  else return as<T>(held{get64(E, p, i), 0});
}

template <class T>
__device__ __forceinline__ void put_plain(int E, int p, int i, T x) {
  if constexpr (sizeof(T) == 4) word(E, p, i) = (uint32_t)of<T>(x).lo;
  else put64(E, p, i, of<T>(x).lo);
}

/* Whether [x] follows [y] in a plain type's order, -0 below +0, neither
   a NaN. */
template <class T> __device__ __forceinline__ bool follows(T x, T y) {
  if constexpr (std::is_same<T, float>::value) return nx_order_f32(x, y) > 0;
  else if constexpr (std::is_same<T, double>::value)
    return nx_order_f64(x, y) > 0;
  else return x > y;
}

/* fold_lanes' fold of [n] terms at the plain type T, as merge folds them:
   from the identity [id] or the state; Arg's first at [pos]. */
template <class T>
__device__ __forceinline__ void fold_typed(int rk, u64 id, int E, int lg,
                                           int tp, int cp, int sp, int at,
                                           int l, int n, int64_t pos) {
  int me = threadIdx.x, mask = (1 << lg) - 1;
  bool has = word(TB, sp + 6, me) != 0;
  T a = has ? get_plain<T>(TB, sp, me) : as<T>(held{id, 0});
#define AT(c) (((l + 16 * (c)) >> lg) * TB + at + ((l + 16 * (c)) & mask))
#define FOLD(f)                                                                \
  for (int c = 0; c < n; c++) a = f(a, get_plain<T>(E, tp, AT(c)));
  switch (rk) {
  case NX_LOOP_SUM: FOLD(k_add) break;
  case NX_LOOP_MAX: FOLD(k_maximum) break;
  case NX_LOOP_MIN: FOLD(k_minimum) break;
  case NX_LOOP_ARG_MAX:
  case NX_LOOP_ARG_MIN: {
    /* A later term of one value never takes the place. */
    bool max = rk == NX_LOOP_ARG_MAX;
    int64_t x = (int64_t)get64(TB, sp + 4, me);
    for (int c = 0; c < n; c++) {
      T b = get_plain<T>(E, tp, AT(c));
      bool an = has && a != a, bn = b != b;
      bool take = !has || (an || bn ? !an && bn
                           : max    ? follows(b, a)
                                    : follows(a, b));
      if (take) {
        a = b;
        x = pos + 16 * c;
        has = true;
      }
    }
    put64(TB, sp + 4, me, (u64)x);
    break;
  }
  default:
    if constexpr (std::is_floating_point<T>::value) {
      /* Exps and Squares, of a center. */
      T q = as<T>(held{get64(TB, sp + 4, me), 0});
      for (int c = 0; c < n; c++) {
        held x = get_held(E, tp, AT(c), sizeof(T) / 4);
        st b = centered<T>(rk, x, get_held(E, cp, AT(c), 2));
        a = k_add(a, as<T>(b.v));
        q = k_add(q, as<T>(held{b.x, 0}));
      }
      if (rk == NX_LOOP_SQUARES) put64(TB, sp + 4, me, of<T>(q).lo);
    }
  }
#undef FOLD
#undef AT
  if (n == 0) return;
  put_plain<T>(TB, sp, me, a);
  word(TB, sp + 6, me) = 1;
}

/* A run's terms of a block of [G] threads, 16 or more, [R1] terms from
   the index [L0]: thread l below 16 folds lane l's, terms l + 16 c, of
   thread (l + 16 c) % G's element (l + 16 c) / G, those below [end], in
   order into its state past the planes. A plain type's in line, the
   others' by fold. Every thread of the group calls it. */
__device__ __noinline__ void fold_lanes(const u64 *w, int rk, int ct, int E,
                                        int G, int R1, int64_t L0,
                                        int64_t end) {
  int l = threadIdx.x & (G - 1), at = threadIdx.x - l, lg = __ffs(G) - 1;
  if (G > NX_LOOP_LANES) __syncthreads();
  int64_t r = end - L0 - l, R = (int64_t)w[NX_LOOP_TERMS];
  int n = l >= NX_LOOP_LANES || r <= 0
              ? 0
              : (int)min((int64_t)(R1 / NX_LOOP_LANES), (r + 15) / 16);
  int tp = (int)w[NX_LOOP_TERM], sp = (int)w[NX_LOOP_STATE_AT];
  int cp = (int)w[NX_LOOP_CENTER];
  bool arg = rk == NX_LOOP_ARG_MAX || rk == NX_LOOP_ARG_MIN;
  bool typed =
      rk != NX_LOOP_PROD && ct != NX_LOOP_C64 && ct != NX_LOOP_C128;
  if (n > 0 && typed) {
    u64 id = rk == NX_LOOP_SUM || rk == NX_LOOP_MAX || rk == NX_LOOP_MIN
                 ? plain_id(ct, rk)
                 : 0;
    int64_t pos = arg ? (L0 + l) % R : 0;
    switch (ct) {
#define CASE(C, T)                                                             \
  case C: fold_typed<T>(rk, id, E, lg, tp, cp, sp, at, l, n, pos); break;
      PLAIN_TYPES(CASE)
#undef CASE
    }
  } else if (n > 0) {
    bool c = rk == NX_LOOP_EXPS || rk == NX_LOOP_SQUARES;
    int nw = ct_words(ct);
    held z = {0, 0};
    st s = load_state(TB, sp, threadIdx.x);
    for (int j = 0; j < n; j++) {
      int k = l + 16 * j, i = (k >> lg) * TB + at + (k & (G - 1));
      s = fold(rk, ct, s, get_held(E, tp, i, nw), (L0 + k) % R,
               c ? get_held(E, cp, i, 2) : z);
    }
    store_state(TB, sp, threadIdx.x, s);
  }
  if (G > NX_LOOP_LANES) __syncthreads();
}

/* Folds into the [W] lanes [s] (bits) their terms, lane c's of run j at
   p + j step + c gap, its 16 j + 8 c th of the thread's [rem], a term
   past them reading the identity, in batches of 16 terms whose loads are
   all in flight. A term is held as S: T itself, or the code of the
   narrow float [DT], which converts into T exactly. */
template <class T, int W, class S = T, int DT = -1>
__device__ void fold_at(int rk, u64 (&s)[2], const S *p, int64_t step,
                        int64_t gap, int rem, u64 id) {
  constexpr int B = 16 / W;
  T a[W], e = as<T>(held{id, 0});
#pragma unroll
  for (int c = 0; c < W; c++) a[c] = as<T>(held{s[c], 0});
  for (int j = 0; 16 * j < rem; j += B) {
    T v[B][W];
#pragma unroll
    for (int r = 0; r < B; r++)
#pragma unroll
      for (int c = 0; c < W; c++)
        if (16 * (j + r) + 8 * c >= rem) v[r][c] = e;
        else if constexpr (DT < 0) v[r][c] = __ldg(p + (j + r) * step + c * gap);
        else v[r][c] = nx_bits_to_float(DT, __ldg(p + (j + r) * step + c * gap));
#define FOLD(f)                                                                \
  _Pragma("unroll") for (int r = 0; r < B; r++)                                \
      _Pragma("unroll") for (int c = 0; c < W; c++) a[c] = f(a[c], v[r][c]);
    switch (rk) {
    case NX_LOOP_SUM: FOLD(k_add) break;
    case NX_LOOP_MAX: FOLD(k_maximum) break;
    default: FOLD(k_minimum)
    }
#undef FOLD
  }
#pragma unroll
  for (int c = 0; c < W; c++) s[c] = of<T>(a[c]).lo;
}

/* fold_at at [ct], of one lane (rows) or two (columns), its terms of
   the dtype [ld]: the carrier's, or bfloat16 or float16 into float32. */
__device__ __noinline__ void plain_fold(int ct, int rk, int ld, bool columns,
                                        u64 (&s)[2], const uint8_t *p,
                                        int64_t step, int64_t gap, int rem) {
  u64 id = s[0];
#define NARROW(D)                                                              \
  if (ld == D) {                                                               \
    const uint16_t *q = (const uint16_t *)p;                                   \
    if (columns) fold_at<float, 2, uint16_t, D>(rk, s, q, step, gap, rem, id); \
    else fold_at<float, 1, uint16_t, D>(rk, s, q, step, gap, rem, id);         \
    return;                                                                    \
  }
  NARROW(NX_BFLOAT16)
  NARROW(NX_FLOAT16)
#undef NARROW
  switch (ct) {
#define CASE(C, T)                                                             \
  case C:                                                                      \
    if (columns) fold_at<T, 2>(rk, s, (const T *)p, step, gap, rem, id);       \
    else fold_at<T, 1>(rk, s, (const T *)p, step, gap, rem, id);               \
    return;
    PLAIN_TYPES(CASE)
#undef CASE
  }
}

/* The pairwise tree of the [n] values of [part], at its words [stride]
   apart, in the shared [sm]; its value lands in part[0]. Every thread of
   the group calls it. */
__device__ __noinline__ void plain_merge(int ct, int rk, u64 *part,
                                         int64_t n, u64 *sm) {
  int64_t stride = 1;
  while (n > 1) {
    for (int64_t c0 = 0; c0 < n; c0 += NX_LOOP_PLAIN_CHUNK) {
      int k = (int)min((int64_t)NX_LOOP_PLAIN_CHUNK, n - c0);
      for (int i = threadIdx.x; i < k; i += TB)
        sm[i] = __ldcg(part + (c0 + i) * stride);
      __syncthreads();
      for (int h = 1; h < k; h *= 2) {
        for (int i = 2 * h * threadIdx.x; i + h < k; i += 2 * h * TB)
          sm[i] = plain_op(ct, rk, sm[i], sm[i + h]);
        __syncthreads();
      }
      if (threadIdx.x == 0) __stcg(part + c0 * stride, sm[0]);
      __syncthreads();
    }
    n = (n + NX_LOOP_PLAIN_CHUNK - 1) / NX_LOOP_PLAIN_CHUNK;
    stride *= NX_LOOP_PLAIN_CHUNK;
  }
}

/* Output [o]'s value [v], bits at [ct], into its destination, or, a
   scan's chunk total, into the workspace's totals. */
__device__ void plain_out(const u64 *w, int ct, int carrier, int64_t o,
                          u64 v) {
  if (((w[NX_LOOP_REDUCTION] >> 24) & 0xff) == NX_LOOP_PUT_TOTAL) {
    __stcg((u64 *)work_byte(w, NX_LOOP_TOTALS) + o, v);
    return;
  }
  const u64 *d = operand(w, NX_LOOP_DSTS, 0);
  if ((int)(d[0] & 0xff) != carrier) {
    put_dst(w, 0, o, carrier, held{v, 0});
    return;
  }
  uint8_t *y = buffer(w, d);
  int64_t at = (int64_t)d[1] + o;
  if (ct_words(ct) == 1) ((uint32_t *)y)[at] = (uint32_t)v;
  else ((u64 *)y)[at] = v;
}

/* A group's value [v] of output [o], its group [grp] of [m], held by the
   thread where [mine]: sent out, or left in the workspace, the last group
   to arrive merging the [m] values. With [m] above 1 the whole group calls
   it. */
__device__ __noinline__ void plain_put(const u64 *w, int ct, int rk,
                                       int carrier, int64_t o, int64_t grp,
                                       int64_t m, bool mine, u64 v, u64 *sm) {
  if (m == 1) {
    if (mine) plain_out(w, ct, carrier, o, v);
    return;
  }
  __shared__ int last;
  u64 *part = (u64 *)work_byte(w, NX_LOOP_PARTIALS) + o * m;
  if (mine) __stcg(part + grp, v);
  __threadfence();
  __syncthreads();
  uint32_t *tickets = (uint32_t *)w[NX_LOOP_HEADER + w[NX_LOOP_TICKETS]];
  if (threadIdx.x == 0) last = atomicAdd(tickets + o, 1u) == (uint32_t)(m - 1);
  __syncthreads();
  if (!last) return;
  if (threadIdx.x == 0) tickets[o] = 0;
  __threadfence();
  plain_merge(ct, rk, part, m, sm);
  if (threadIdx.x == 0) plain_out(w, ct, carrier, o, __ldcg(part));
}

/* Rows: the [O] outputs' terms from [x], output o the segment o % segs
   of row o / segs, of [R] terms from x + (o / segs) so, segments of [SEG]
   terms. Cell c takes block c / per of unit c % per of a round's [per], a
   unit [g] blocks of an output, [m] units an output. */
__device__ void plain_rows(const u64 *w, int ct, int rk, int ld, int carrier,
                           const uint8_t *x, int es, int64_t O, int64_t R,
                           int64_t so, int64_t SEG, int64_t segs, int g,
                           int64_t m) {
  u64 *sm = (u64 *)loop_planes, id = plain_id(ct, rk);
  int cell = threadIdx.x >> 4, l = threadIdx.x & 15, per = TB / 16 / g;
  int k = cell / per;
  int64_t units = O * m;
  for (int64_t r0 = (int64_t)blockIdx.x * per; r0 < units;
       r0 += (int64_t)gridDim.x * per) {
    int64_t u = r0 + cell % per, o = u / m, grp = u % m, b = grp * g + k;
    int64_t at = o % segs * SEG, n = min(SEG, R - at);
    int64_t nb = (n + NX_LOOP_BLOCK - 1) / NX_LOOP_BLOCK;
    int len = u < units && b < nb
                  ? (int)min((int64_t)NX_LOOP_BLOCK, n - b * NX_LOOP_BLOCK)
                  : 0;
    u64 s[2] = {id, id};
    plain_fold(ct, rk, ld, false, s,
               x + (o / segs * so + at + b * NX_LOOP_BLOCK + l) * es, 16, 0,
               len - l);
    /* The lanes' tree: lane l below h taking lane l + h. */
    for (int h = 8; h > 0; h /= 2) {
      u64 y = __shfl_down_sync(0xffffffff, s[0], h, 16);
      if (l < h) s[0] = plain_op(ct, rk, s[0], y);
    }
    /* The unit's blocks by the pairwise tree, a cell of no block taking
       no part. */
    int count = u < units ? (int)min((int64_t)g, nb - grp * g) : 0;
    if (l == 0) sm[cell] = s[0];
    __syncthreads();
    for (int h = 1; h < g; h *= 2) {
      if (l == 0 && k % (2 * h) == 0 && k + h < count)
        sm[cell] = plain_op(ct, rk, sm[cell], sm[cell + h * per]);
      __syncthreads();
    }
    u64 v = sm[cell];
    __syncthreads();
    plain_put(w, ct, rk, carrier, o, grp, m, l == 0 && k == 0 && u < units, v,
              sm + TB / 16);
  }
}

/* Columns: the [O] columns' terms from [x], column o's term t at
   o + t S, of [R] terms in segments of [SEG], output o segs + the
   segment. A group takes a segment of 32 columns at a time. */
__device__ void plain_columns(const u64 *w, int ct, int rk, int ld,
                              int carrier, const uint8_t *x, int es, int64_t O,
                              int64_t R,
                              int64_t S, int64_t SEG, int64_t segs) {
  u64 *sm = (u64 *)loop_planes, id = plain_id(ct, rk);
  int t = threadIdx.x & 31, lw = threadIdx.x >> 5;
  int64_t strips = (O + 31) / 32;
  /* sm[lw][t] holds the lanes' tree, then sm[8 + b][t] block b's value. */
  for (int64_t q = blockIdx.x; q < strips * segs; q += gridDim.x) {
    int64_t o = q / segs * 32 + t, at = q % segs * SEG, n = min(SEG, R - at);
    int nb = (int)((n + NX_LOOP_BLOCK - 1) / NX_LOOP_BLOCK);
    for (int b = 0; b < nb; b++) {
      int len = o < O ? (int)min((int64_t)NX_LOOP_BLOCK, n - (int64_t)b * NX_LOOP_BLOCK) : 0;
      u64 s[2] = {id, id};
      plain_fold(ct, rk, ld, true, s,
                 x + (o + (at + (int64_t)b * NX_LOOP_BLOCK + lw) * S) * es,
                 16 * S, 8 * S, len - lw);
      /* The lanes' tree: lane lw takes lane lw + 8 in the thread, then
         lanes below h take lanes h above them across the warps. */
      sm[32 * lw + t] = plain_op(ct, rk, s[0], s[1]);
      __syncthreads();
      for (int h = 4; h > 0; h /= 2) {
        if (lw < h)
          sm[32 * lw + t] = plain_op(ct, rk, sm[32 * lw + t], sm[32 * (lw + h) + t]);
        __syncthreads();
      }
      if (lw == 0) sm[32 * (8 + b) + t] = sm[t];
      __syncthreads();
    }
    /* The blocks by the pairwise tree. */
    if (lw == 0 && o < O) {
      for (int h = 1; h < nb; h *= 2)
        for (int b = 0; b + h < nb; b += 2 * h)
          sm[32 * (8 + b) + t] =
              plain_op(ct, rk, sm[32 * (8 + b) + t], sm[32 * (8 + b + h) + t]);
      plain_out(w, ct, carrier, o * segs + q % segs, sm[32 * 8 + t]);
    }
    __syncthreads();
  }
}

/* The reduction's walk: rows where its terms step by one, else
   columns. Its outputs are its rows' segments. */
__device__ __noinline__ void plain_mode(const u64 *w) {
  u64 r = w[NX_LOOP_REDUCTION];
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff);
  const u64 *d = operand(w, NX_LOOP_LOADS, 0);
  int ld = (int)(d[0] & 0xff), es = bits_of(ld) / 8;
  const uint8_t *x = buffer(w, d) + d[1] * es;
  int64_t O = (int64_t)w[NX_LOOP_OUTPUTS], R = (int64_t)w[NX_LOOP_TERMS];
  int64_t SEG = (int64_t)w[NX_LOOP_SEGMENT], segs = (int64_t)w[NX_LOOP_SEGMENTS];
  bool two = w[NX_LOOP_RANK] == 2;
  int64_t so = two ? (int64_t)d[2] : 0, st = (int64_t)d[2 + two];
  if (st == 1)
    plain_rows(w, ct, rk, ld, carrier, x, es, O, R, so, SEG, segs,
               (int)w[NX_LOOP_GROUP], (int64_t)w[NX_LOOP_GROUPS]);
  else plain_columns(w, ct, rk, ld, carrier, x, es, O / segs, R, st, SEG, segs);
}

/* Scans */

/* The state a slice's scan starts from: its carry, or none. */
__device__ st carried(const u64 *w, int rk, int carrier, int64_t slice) {
  st s = empty(rk);
  int64_t k = (int64_t)w[NX_LOOP_CARRY];
  if (k >= 0) {
    s.v = of_carrier(carrier,
                     load_at(w, (int)k, slice * (int64_t)w[NX_LOOP_TERMS]));
    s.has = 1;
  }
  return s;
}

/* Each slice's chunk totals, in place, become the states its chunks start
   from: its carry, then the totals of the chunks before folded in order. */
__device__ void carries_mode(const u64 *w) {
  int64_t slices = (int64_t)w[NX_LOOP_OUTPUTS];
  int64_t chunks = (int64_t)w[NX_LOOP_SEGMENTS];
  u64 r = w[NX_LOOP_REDUCTION];
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff);
  for (int64_t slice = (int64_t)blockIdx.x * TB + threadIdx.x; slice < slices;
       slice += (int64_t)gridDim.x * TB) {
    st s = carried(w, rk, carrier, slice);
    for (int64_t c = 0; c < chunks; c++) {
      uint32_t *g = work_at(w, NX_LOOP_TOTALS, slice * chunks + c);
      st t = load_work(g);
      store_work(g, s);
      s = merge(rk, ct, s, t);
    }
  }
}

/* A run of the thread's terms, in order: each result, the state after it,
   into the result planes, its position after where Arg's. */
__device__ __noinline__ void scan_run(const u64 *w, int rk, int ct,
                                      int carrier, int E, int V, lanes x) {
  int tp = (int)w[NX_LOOP_TERM];
  int sp = (int)w[NX_LOOP_STATE_AT], rp = (int)w[NX_LOOP_RESULT];
  int64_t R = (int64_t)w[NX_LOOP_TERMS];
  st s = load_state(E, sp, threadIdx.x);
  held z = {0, 0};
  EACH {
    if (v >= x.n) continue;
    s = fold(rk, ct, s, get_held(E, tp, i, 4), index_of(x, v) % R, z);
    put_held(E, rp, i, 4, to_carrier(carrier, ct, s.v));
    put<int64_t>(E, rp + 4, i, (int64_t)s.x);
  }
  store_state(E, sp, threadIdx.x, s);
}

/* A thread's chunk: one chunk a slice starts from the slice's carry;
   several from the state the carries pass left. */
__device__ void scan_mode(const u64 *w, int E, int V) {
  int64_t units = (int64_t)w[NX_LOOP_OUTPUTS], R = (int64_t)w[NX_LOOP_TERMS];
  int64_t chunks = (int64_t)w[NX_LOOP_SEGMENTS], CH = (int64_t)w[NX_LOOP_SEGMENT];
  int64_t runs = (int64_t)w[NX_LOOP_RUNS];
  u64 r = w[NX_LOOP_REDUCTION];
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff), sp = (int)w[NX_LOOP_STATE_AT];
  int body = (int)(w[NX_LOOP_TAPE] + w[NX_LOOP_PROLOGUE]);
  int nbody = (int)w[NX_LOOP_BODY], nepi = (int)w[NX_LOOP_EPILOGUE];
  for (int64_t r0 = (int64_t)blockIdx.x * TB; r0 < units;
       r0 += (int64_t)gridDim.x * TB) {
    int64_t u = r0 + threadIdx.x, slice = u / chunks, c = u % chunks;
    int64_t start = c * CH, len = u < units ? min(CH, R - start) : 0;
    st s = empty(rk);
    if (u < units)
      s = chunks > 1 ? load_work(work_at(w, NX_LOOP_TOTALS, u))
                     : carried(w, rk, carrier, slice);
    store_state(E, sp, threadIdx.x, s);
    for (int64_t j = 0; j < runs; j++) {
      int64_t first = slice * R + start;
      lanes x = lanes_of(first + j * V, 1, first + len, V);
      run(w, E, V, x, body, nbody);
      scan_run(w, rk, ct, carrier, E, V, x);
      run(w, E, V, x, body + nbody, nepi);
    }
  }
}

/* Plain scans

   A Sum, Max or Min scan of one plain load at its carrier into one
   destination of that dtype, without a tape: a thread folds a chunk's
   terms in order from the carry into it, 16 loads in flight, storing
   each result. Past one chunk, a plain reduction first leaves each
   chunk's total in the workspace, NX_LOOP_TOTALS, and the carries pass
   turns each slice's totals in place into the carries into its chunks.
   Values are their bits in a word, as plain reductions hold them. */

/* 16 values of T at [p], which lies on 16 bytes, read or written in
   16-byte accesses. */
template <class T> __device__ __forceinline__ void get16(const T *p, T (&v)[16]) {
#pragma unroll
  for (int r = 0; r < 16 * (int)sizeof(T) / 16; r++) {
    uint4 u = __ldg((const uint4 *)p + r);
    memcpy((char *)v + 16 * r, &u, 16);
  }
}

template <class T> __device__ __forceinline__ void put16(T *p, const T (&v)[16]) {
#pragma unroll
  for (int r = 0; r < 16 * (int)sizeof(T) / 16; r++) {
    uint4 u;
    memcpy(&u, (const char *)v + 16 * r, 16);
    ((uint4 *)p)[r] = u;
  }
}

/* Folds [n] terms at p + i st from [a], each result to y + i yst, 16 at
   a time, in 16-byte accesses where [vec]: both step by one from 16
   bytes. */
template <class T>
__device__ void scan_at(int rk, T a, const T *p, int64_t st, T *y,
                        int64_t yst, int n, bool vec) {
  for (int i = 0; i < n; i += 16) {
    T v[16], o[16];
    bool whole = vec && i + 16 <= n;
    if (whole) get16(p + i, v);
    else
#pragma unroll
      for (int r = 0; r < 16; r++) v[r] = i + r < n ? __ldg(p + (i + r) * st) : T(0);
#define STEP(f)                                                                \
  _Pragma("unroll") for (int r = 0; r < 16; r++) {                             \
    a = i + r < n ? f(a, v[r]) : a;                                            \
    o[r] = a;                                                                  \
  }
    switch (rk) {
    case NX_LOOP_SUM: STEP(k_add) break;
    case NX_LOOP_MAX: STEP(k_maximum) break;
    default: STEP(k_minimum)
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
   [a]: each the sum of [a] and the totals before it, in order, 16 loads
   in flight. */
template <class T> __device__ void carries_at(T a, u64 *g, int64_t n) {
  for (int64_t i = 0; i < n; i += 16) {
    T v[16];
#pragma unroll
    for (int r = 0; r < 16; r++)
      v[r] = i + r < n ? as<T>(held{__ldcg(g + i + r), 0}) : T(0);
#pragma unroll
    for (int r = 0; r < 16; r++)
      if (i + r < n) {
        __stcg(g + i + r, of<T>(a).lo);
        a = k_add(a, v[r]);
      }
  }
}

__device__ __noinline__ void plain_scan_run(int ct, int rk, u64 a,
                                            const uint8_t *p, int64_t st,
                                            uint8_t *y, int64_t yst, int n) {
  bool vec = st == 1 && yst == 1 && ((u64)p | (u64)y) % 16 == 0;
  switch (ct) {
#define CASE(C, T)                                                             \
  case C:                                                                      \
    scan_at<T>(rk, as<T>(held{a, 0}), (const T *)p, st, (T *)y, yst, n, vec);  \
    return;
    PLAIN_TYPES(CASE)
#undef CASE
  }
}

/* carries_at at [ct]; a Max or a Min, exact in any order and rarer, a
   total at a time. */
__device__ __noinline__ void plain_carries_run(int ct, int rk, u64 a, u64 *g,
                                               int64_t n) {
  if (rk != NX_LOOP_SUM) {
    for (int64_t i = 0; i < n; i++) {
      u64 t = __ldcg(g + i);
      __stcg(g + i, a);
      a = plain_op(ct, rk, a, t);
    }
    return;
  }
  switch (ct) {
#define CASE(C, T)                                                             \
  case C: carries_at<T>(as<T>(held{a, 0}), g, n); return;
    PLAIN_TYPES(CASE)
#undef CASE
  }
}

/* The bits slice [slice]'s scan starts from: its carry, else the
   identity. */
__device__ u64 plain_carry(const u64 *w, int ct, int rk, int carrier,
                           int64_t slice) {
  st s = carried(w, rk, carrier, slice);
  return s.has ? s.v.lo : plain_id(ct, rk);
}

/* A scan's [O] slices of [R] terms, slice o's term t at x + o so + t st,
   its result at y + o yso + t yst: unit u, chunk u % chunks of slice
   u / chunks, to thread u % A of group u / A, [A] threads a group. */
__device__ __noinline__ void plain_scan_mode(const u64 *w) {
  u64 r = w[NX_LOOP_REDUCTION];
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff), es = 4 * ct_words(ct);
  const u64 *d = operand(w, NX_LOOP_LOADS, 0);
  const u64 *e = operand(w, NX_LOOP_DSTS, 0);
  const uint8_t *x = buffer(w, d) + d[1] * es;
  uint8_t *y = buffer(w, e) + e[1] * es;
  int64_t O = (int64_t)w[NX_LOOP_OUTPUTS], R = (int64_t)w[NX_LOOP_TERMS];
  int64_t chunks = (int64_t)w[NX_LOOP_SEGMENTS];
  int A = (int)w[NX_LOOP_GROUP];
  bool two = w[NX_LOOP_RANK] == 2;
  int64_t so = two ? (int64_t)d[2] : 0, st = (int64_t)d[2 + two];
  int64_t yso = two ? (int64_t)e[2] : 0, yst = (int64_t)e[2 + two];
  u64 *totals = (u64 *)work_byte(w, NX_LOOP_TOTALS);
  if ((int)threadIdx.x >= A) return;
  for (int64_t u = (int64_t)blockIdx.x * A + threadIdx.x; u < O * chunks;
       u += (int64_t)gridDim.x * A) {
    int64_t o = u / chunks, at = u % chunks * NX_LOOP_CHUNK;
    u64 a = chunks > 1 ? __ldcg(totals + u) : plain_carry(w, ct, rk, carrier, o);
    plain_scan_run(ct, rk, a, x + (o * so + at * st) * es, st,
                   y + (o * yso + at * yst) * es, yst,
                   (int)min((int64_t)NX_LOOP_CHUNK, R - at));
  }
}

/* The carries pass of a plain scan: a thread a slice. */
__device__ __noinline__ void plain_carries_mode(const u64 *w) {
  u64 r = w[NX_LOOP_REDUCTION];
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff);
  int64_t O = (int64_t)w[NX_LOOP_OUTPUTS], chunks = (int64_t)w[NX_LOOP_SEGMENTS];
  u64 *totals = (u64 *)work_byte(w, NX_LOOP_TOTALS);
  for (int64_t o = (int64_t)blockIdx.x * TB + threadIdx.x; o < O;
       o += (int64_t)gridDim.x * TB)
    plain_carries_run(ct, rk, plain_carry(w, ct, rk, carrier, o),
                      totals + o * chunks, chunks);
}

/* Staged scans: a Sum, Max or Min scan whose results step by one, in
   nx.cuda's own order (nx_cuda.mli), in one pass. A group takes P =
   NX_LOOP_GROUP chunks at a time by a ticket, in order, into shared
   memory, so that each term is read once:

     tickets --> terms in --> totals --> carries --> scans --> results out

   The terms come in as copies of one plain load whose terms step by one,
   or from the tape's term plane: the chunks' loop indices follow each
   other, so the group's tiles walk them in turn and each thread copies
   its terms into their chunks. Either way a chunk holds the same terms,
   so a program's scan has the bits of its output's. A chunk's pass
   (chunk_pass) gives its total. The carry into a slice's chunk c is its
   carry, then each earlier chunk's total, folded in order. A group
   publishes each chunk's total, then its inclusive
   carry, the carry folded with the total, each behind a flag; the carry
   into its first chunk looks back from the chunk before it to the nearest
   inclusive carry, or the slice's start, and folds the totals between in
   order: whichever it finds, the fold is the same left fold, each step
   by carry_by, so the bits do not depend on the groups' timing. A group
   waits only for totals, which an earlier ticket's group publishes
   without waiting, so no group waits for one that cannot run. The last
   group to finish zeroes the flags and its counters, which the tickets'
   slot holds (a word a chunk, then the ticket and the count of groups
   done).

   A chunk's pass then scans it in place from its carry, and the group's
   threads copy the results out: in 16-byte pieces where the slices'
   results lie on 16 bytes, else an element at a time. */

/* [a] then [b] by the reduction RK at T. */
template <class T, int RK> __device__ __forceinline__ T plain_by(T a, T b) {
  if constexpr (RK == NX_LOOP_SUM) return k_add(a, b);
  else if constexpr (RK == NX_LOOP_MAX) return k_maximum(a, b);
  else return k_minimum(a, b);
}

/* plain_by along a slice's carries: a float64 sum whose [a] is a NaN is
   [a]. A float64 sum of two NaNs is the NaN its instruction names
   second, an order the compiler picks at each site, and a carry comes
   through the sites of the look-back's fold or those of the group's
   carries, as the groups' timing falls. A float32 sum's NaN is always
   the one canonical NaN. */
template <class T, int RK> __device__ __forceinline__ T carry_by(T a, T b) {
  if constexpr (RK == NX_LOOP_SUM && std::is_same<T, double>::value)
    if (a != a) return a;
  return plain_by<T, RK>(a, b);
}

/* Folds [n] totals of T at [s], which lies on 16 bytes, in place from
   [a] by carry_by: whole batches of 16 read and written in 16-byte
   accesses, the next batch's reads in flight while this one folds, then
   the rest one at a time. Few accesses keep the fold off the queue that
   the group's copies fill. */
template <class T, int RK> __device__ void scan_shared(T a, T *s, int n) {
  constexpr int Q = 16 * (int)sizeof(T) / 16;
  uint4 v[Q] = {}, u[Q] = {};
  int i = 0;
  if (n >= 16)
#pragma unroll
    for (int r = 0; r < Q; r++) v[r] = ((const uint4 *)s)[r];
#pragma unroll 1
  for (; i + 16 <= n; i += 16) {
    if (i + 32 <= n)
#pragma unroll
      for (int r = 0; r < Q; r++) u[r] = ((const uint4 *)(s + i + 16))[r];
    T t[16];
    memcpy(t, v, sizeof(t));
#pragma unroll
    for (int r = 0; r < 16; r++) {
      a = carry_by<T, RK>(a, t[r]);
      t[r] = a;
    }
    memcpy(v, t, sizeof(t));
#pragma unroll
    for (int r = 0; r < Q; r++) ((uint4 *)(s + i))[r] = v[r];
#pragma unroll
    for (int r = 0; r < Q; r++) v[r] = u[r];
  }
#pragma unroll 1
  for (; i < n; i++) {
    a = carry_by<T, RK>(a, s[i]);
    s[i] = a;
  }
}

/* A thread's run of a chunk: NX_LOOP_CHUNK / NX_LOOP_THREADS terms. */
#define RUN (NX_LOOP_CHUNK / NX_LOOP_THREADS)

/* A chunk's pass, every thread of the group together: thread t folds
   the run of terms 16 t to 16 t + 15 of the [n] at [s] in order; a warp's
   runs combine by an inclusive scan of the runs' folds, lane l taking
   lane l - d's for d = 1, 2, 4, 8, 16; each warp's total, its last run's
   scan, lands in [wt]. With [out], thread t then folds [carry], the
   totals of the warps before it in order, the scan of the runs before it
   in its warp, then its run's terms in order, each result in place. */
template <class T, int RK>
__device__ void chunk_pass(T *s, int n, bool out, T carry, u64 *wt, int t) {
  int lane = t & 31, wid = t >> 5;
  int lo = RUN * t, m = max(0, min(RUN, n - lo));
  /* A whole run in 16-byte reads: a thread's 16 words in one read each
     would fall in two banks a warp. */
  constexpr int Q = RUN * (int)sizeof(T) / 16;
  T v[RUN];
  if (m == RUN) {
    uint4 q[Q];
#pragma unroll
    for (int r = 0; r < Q; r++) q[r] = ((const uint4 *)(s + lo))[r];
    memcpy(v, q, sizeof(v));
  } else {
#pragma unroll
    for (int r = 0; r < RUN; r++) v[r] = r < m ? s[lo + r] : T(0);
  }
  T inc = v[0];
#pragma unroll
  for (int r = 1; r < RUN; r++)
    if (r < m) inc = plain_by<T, RK>(inc, v[r]);
  for (int d = 1; d < 32; d *= 2) {
    T z = __shfl_up_sync(0xffffffff, inc, d);
    if (lane >= d && m > 0) inc = plain_by<T, RK>(z, inc);
  }
  T before = __shfl_up_sync(0xffffffff, inc, 1);
  int last = min(31, (n - RUN * 32 * wid + RUN - 1) / RUN - 1);
  if (m > 0 && lane == last) wt[wid] = of<T>(inc).lo;
  __syncthreads();
  if (!out || m == 0) return;
  T a = carry;
  for (int k = 0; k < wid; k++) a = plain_by<T, RK>(a, as<T>(held{wt[k], 0}));
  if (lane > 0) a = plain_by<T, RK>(a, before);
#pragma unroll
  for (int r = 0; r < RUN; r++) v[r] = a = plain_by<T, RK>(a, v[r]);
  if (m == RUN) {
    uint4 q[Q];
    memcpy(q, v, sizeof(v));
#pragma unroll
    for (int r = 0; r < Q; r++) ((uint4 *)(s + lo))[r] = q[r];
  } else {
#pragma unroll
    for (int r = 0; r < RUN; r++)
      if (r < m) s[lo + r] = v[r];
  }
}

/* Bits at [ct] by the reduction [rk]: a chunk's pass (chunk_pass), [n]
   totals at [s] folded in place from [a] (scan_shared), and a carry
   folded with a total (carry_by). */
#define TYPED(F, ...)                                                          \
  switch (ct) {                                                                \
  case NX_LOOP_F32: F(float, __VA_ARGS__);                                     \
  case NX_LOOP_F64: F(double, __VA_ARGS__);                                    \
  case NX_LOOP_I32: F(int32_t, __VA_ARGS__);                                   \
  case NX_LOOP_U32: F(uint32_t, __VA_ARGS__);                                  \
  case NX_LOOP_I64: F(int64_t, __VA_ARGS__);                                   \
  default: F(uint64_t, __VA_ARGS__);                                           \
  }
#define BY_RK(T, call)                                                         \
  if (rk == NX_LOOP_SUM) call(T, NX_LOOP_SUM);                                 \
  else if (rk == NX_LOOP_MAX) call(T, NX_LOOP_MAX);                            \
  else call(T, NX_LOOP_MIN);

__device__ __noinline__ void staged_chunk(int ct, int rk, uint8_t *s, int n,
                                          bool out, u64 carry, u64 *wt,
                                          int t) {
#define CALL(T, RK)                                                            \
  return chunk_pass<T, RK>((T *)s, n, out, as<T>(held{carry, 0}), wt, t)
  TYPED(BY_RK, CALL)
#undef CALL
}

__device__ __noinline__ void staged_run(int ct, int rk, u64 a, uint8_t *s,
                                        int n) {
#define CALL(T, RK) return scan_shared<T, RK>(as<T>(held{a, 0}), (T *)s, n)
  TYPED(BY_RK, CALL)
#undef CALL
}

/* [a] then [b], bits, by carry_by at [ct]. */
__device__ __noinline__ u64 carry_op(int ct, int rk, u64 a, u64 b) {
#define CALL(T, RK)                                                            \
  return of<T>(carry_by<T, RK>(as<T>(held{a, 0}), as<T>(held{b, 0}))).lo
  TYPED(BY_RK, CALL)
#undef CALL
}

#undef TYPED
#undef BY_RK

/* The flags' values: a chunk's total, then its inclusive carry, there. */
#define STAGED_TOTAL 1u
#define STAGED_CARRY 2u

__device__ __forceinline__ uint32_t flag_of(const uint32_t *f) {
  return *(const volatile uint32_t *)f;
}

__device__ __forceinline__ void publish(u64 *at, u64 v, uint32_t *f,
                                        uint32_t state) {
  __stcg(at, v);
  __threadfence();
  *(volatile uint32_t *)f = state;
}

/* The carry into unit [u], chunk [c] of its slice, whose carry is
   [carry], the warp together: back from unit u - 1, 32 units at a time,
   to the nearest inclusive carry or the slice's start, then the totals
   after it folded in order, NX_LOOP_LOOK at a time through the shared
   [scratch] by staged_run, whose last result is their fold, the next
   ones' reads in flight meanwhile. */
__device__ u64 look_back(int ct, int rk, int64_t u, int64_t c, u64 carry,
                         const u64 *totals, const u64 *inclusive,
                         const uint32_t *flags, uint8_t *scratch) {
  int lane = threadIdx.x & 31;
  int64_t first = u - c, k = u, from = first;
  u64 a = carry;
  for (;;) {
    int64_t lo = max(first, k - 32), q = k - 1 - lane;
    uint32_t f = 0;
    if (q >= lo)
      while ((f = flag_of(flags + q)) == 0) {
      }
    unsigned found = __ballot_sync(0xffffffff, q >= lo && f == STAGED_CARRY);
    if (found) {
      int64_t at = k - 1 - (__ffs(found) - 1);
      __threadfence();
      a = __ldcg(inclusive + at);
      from = at + 1;
      break;
    }
    if (lo == first) break;
    k = lo;
  }
  __threadfence();
  int es = 4 * ct_words(ct);
  for (int64_t i0 = from; i0 < u; i0 += NX_LOOP_LOOK) {
    int m = (int)min((int64_t)NX_LOOP_LOOK, u - i0);
    u64 t[NX_LOOP_LOOK / 32];
#pragma unroll
    for (int j = 0; j < NX_LOOP_LOOK / 32; j++)
      t[j] = 32 * j + lane < m ? __ldcg(totals + i0 + 32 * j + lane) : 0;
#pragma unroll
    for (int j = 0; j < NX_LOOP_LOOK / 32; j++)
      if (es == 4) ((uint32_t *)scratch)[32 * j + lane] = (uint32_t)t[j];
      else ((u64 *)scratch)[32 * j + lane] = t[j];
    __syncwarp();
    if (lane == 0) staged_run(ct, rk, a, scratch, m);
    __syncwarp();
    a = es == 4 ? ((const uint32_t *)scratch)[m - 1] : ((const u64 *)scratch)[m - 1];
    __syncwarp();
  }
  return a;
}

/* Copies a chunk of [n] elements of [es] bytes in or out: 16-byte pieces
   where [vec], else elements, the group's threads in turn; in
   asynchronously (cp.async). */
__device__ __forceinline__ void chunk_in(uint8_t *s, const uint8_t *g, int n,
                                         int es, bool vec) {
  unsigned to = (unsigned)__cvta_generic_to_shared(s);
  int nb = n * es, whole = vec ? nb & ~15 : 0;
  for (int q = 16 * threadIdx.x; q < whole; q += 16 * TB)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(to + q),
                 "l"(g + q)
                 : "memory");
  for (int q = whole + es * threadIdx.x; q < nb; q += es * TB) {
    if (es == 4)
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(to + q),
                   "l"(g + q)
                   : "memory");
    else
      asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(to + q),
                   "l"(g + q)
                   : "memory");
  }
}

__device__ __forceinline__ void chunk_out(uint8_t *g, const uint8_t *s, int n,
                                          int es, bool vec) {
  int nb = n * es, whole = vec ? nb & ~15 : 0;
  for (int q = 16 * threadIdx.x; q < whole; q += 16 * TB)
    *(uint4 *)(g + q) = *(const uint4 *)(s + q);
  for (int q = whole + es * threadIdx.x; q < nb; q += es * TB)
    if (es == 4) *(uint32_t *)(g + q) = *(const uint32_t *)(s + q);
    else *(u64 *)(g + q) = *(const u64 *)(s + q);
}

/* The terms of unit [u]'s chunk, of slices of [R] terms. */
__device__ __forceinline__ int chunk_len(int64_t R, int64_t u, int64_t chunks) {
  return (int)min((int64_t)NX_LOOP_CHUNK, R - u % chunks * NX_LOOP_CHUNK);
}

/* The loop index of unit [u]'s first term. */
__device__ __forceinline__ int64_t chunk_first(int64_t R, int64_t u,
                                               int64_t chunks) {
  return u / chunks * R + u % chunks * NX_LOOP_CHUNK;
}

/* A staged scan's shared memory from its byte NX_LOOP_STAGE: its P
   chunks of [cb] bytes, then the look-back's totals, then the ticket. */
__device__ __forceinline__ uint8_t *stage_at(const u64 *w) {
  return (uint8_t *)loop_planes + w[NX_LOOP_STAGE];
}

__device__ __forceinline__ int64_t *ticket_at(const u64 *w, int cb) {
  return (int64_t *)(stage_at(w) + (int)w[NX_LOOP_GROUP] * cb +
                     8 * NX_LOOP_LOOK);
}

/* The bytes of a chunk of slices of [R] terms of [es] bytes, on 16. */
__device__ __forceinline__ int chunk_bytes(int64_t R, int es) {
  return ((int)min((int64_t)NX_LOOP_CHUNK, R) * es + 15) & ~15;
}

/* The tile from loop index [base] of the ticket's units, below [end]:
   each thread's terms copied from the term plane into their chunks. Its
   units' loop indices follow each other, so a term's place is the units'
   lengths from the first unit's first index. All it needs it reads from
   the parameters and the shared ticket. */
__device__ __noinline__ void staged_copy(const u64 *w, int64_t base,
                                         int64_t end) {
  int V = (int)((w[NX_LOOP_MODE] >> 8) & 0xff), E = V * TB;
  int tp = (int)w[NX_LOOP_TERM];
  int es = 4 * ct_words((int)((w[NX_LOOP_REDUCTION] >> 8) & 0xff));
  int64_t R = (int64_t)w[NX_LOOP_TERMS], chunks = (int64_t)w[NX_LOOP_SEGMENTS];
  int cb = chunk_bytes(R, es);
  uint8_t *s = stage_at(w);
  int64_t u0 = *ticket_at(w, cb);
  int k = 0, len = chunk_len(R, u0, chunks);
  int off = (int)(base - chunk_first(R, u0, chunks)) + threadIdx.x;
  EACH {
    if (base + i >= end) break;
    while (off >= len) {
      off -= len;
      len = chunk_len(R, u0 + ++k, chunks);
    }
    uint8_t *at = s + k * cb + off * es;
    if (es == 4) *(uint32_t *)at = word(E, tp, i);
    else *(u64 *)at = get64(E, tp, i);
    off += TB;
  }
}

/* What a call of staged_scan does: the whole scan of a plain load; or,
   for a tape's terms, take a ticket, false where none is left, or pass
   over the ticket's chunks once the tape has filled them. The tape runs
   from the kernel itself (staged_tape): a function that calls it changes
   how the runner's registers are allocated, and a caller of its own made
   every tape's loop slower. */
enum { STAGE_ALL, STAGE_TAKE, STAGE_PASS };

template <int WHAT> __device__ __noinline__ bool staged_scan(const u64 *w) {
  u64 r = w[NX_LOOP_REDUCTION];
  int rk = (int)(r & 0xff), ct = (int)((r >> 8) & 0xff);
  int carrier = (int)((r >> 16) & 0xff);
  int es = 4 * ct_words(ct);
  constexpr bool tape = WHAT != STAGE_ALL;
  const u64 *d = operand(w, NX_LOOP_LOADS, 0);
  const u64 *e = operand(w, NX_LOOP_DSTS, 0);
  uint8_t *y = buffer(w, e) + e[1] * es;
  int64_t O = (int64_t)w[NX_LOOP_OUTPUTS], R = (int64_t)w[NX_LOOP_TERMS];
  int cb = chunk_bytes(R, es);
  int64_t chunks = (int64_t)w[NX_LOOP_SEGMENTS], units = O * chunks;
  int P = (int)w[NX_LOOP_GROUP];
  /* The plain load's slices, and the destination's, which is
     C-contiguous: slice o's results from y + o R. */
  const uint8_t *x = tape ? nullptr : buffer(w, d) + d[1] * es;
  int64_t so = !tape && w[NX_LOOP_RANK] == 2 ? (int64_t)d[2] : 0;
  bool vin = (((u64)x | (u64)(so * es)) & 15) == 0;
  bool vout = (((u64)y | (u64)(O > 1 ? R * es : 0)) & 15) == 0;
  u64 *totals = (u64 *)work_byte(w, NX_LOOP_TOTALS);
  u64 *inclusive = (u64 *)work_byte(w, NX_LOOP_PARTIALS);
  uint32_t *flags = (uint32_t *)w[NX_LOOP_HEADER + w[NX_LOOP_TICKETS]];
  /* Shared memory: the tape's planes, then the chunks, then the
     look-back's totals, then the ticket, then a chunk's 8 warps' totals,
     its total and its carry, as bits. */
  uint8_t *s = stage_at(w), *scratch = s + P * cb;
  int64_t *ticket = ticket_at(w, cb);
  u64 *wt = (u64 *)(ticket + 2), *tot = wt + 8 * P, *carries = tot + P;
  /* A chunk's threads: a run each, in a power of two of whole warps. */
  int t = threadIdx.x, tc = 32;
  while (32 * RUN * (tc / 32) < (int)min((int64_t)NX_LOOP_CHUNK, R)) tc *= 2;
  for (;;) {
    if constexpr (WHAT != STAGE_PASS) {
      if (t == 0) *ticket = atomicAdd(flags + units, (uint32_t)P);
      __syncthreads();
    }
    int64_t u0 = *ticket;
    if (u0 >= units) break;
    if constexpr (WHAT == STAGE_TAKE) return true;
    if constexpr (!tape) {
      for (int k = 0; k < P && u0 + k < units; k++) {
        int64_t u = u0 + k, o = u / chunks, at = u % chunks * NX_LOOP_CHUNK;
        chunk_in(s + k * cb, x + (o * so + at) * es, chunk_len(R, u, chunks),
                 es, vin);
      }
      asm volatile("cp.async.wait_all;\n" ::: "memory");
    }
    __syncthreads();
    /* Totals: a chunk's warps' totals folded in order. The group passes
       over TB / tc chunks at once, tc threads a chunk. */
    for (int k0 = 0; k0 < P; k0 += TB / tc) {
      int k = k0 + t / tc;
      int n = k < P && u0 + k < units ? chunk_len(R, u0 + k, chunks) : 0;
      staged_chunk(ct, rk, s + k * cb, n, false, 0, wt + 8 * k, t % tc);
    }
    if (t < P && u0 + t < units && chunks > 1) {
      int n = chunk_len(R, u0 + t, chunks);
      u64 a = wt[8 * t];
      for (int q = 1; q < (n + 32 * RUN - 1) / (32 * RUN); q++)
        a = plain_op(ct, rk, a, wt[8 * t + q]);
      tot[t] = a;
      publish(totals + u0 + t, a, flags + u0 + t, STAGED_TOTAL);
    }
    __syncthreads();
    /* Carries: the first chunk's by looking back, the next ones' from it. */
    if (t < 32) {
      u64 a = 0;
      for (int k = 0; k < P && u0 + k < units; k++) {
        int64_t v = u0 + k, c = v % chunks;
        u64 carry = plain_carry(w, ct, rk, carrier, v / chunks);
        if (c == 0) a = carry;
        else if (k == 0)
          a = look_back(ct, rk, v, c, carry, totals, inclusive, flags,
                        scratch);
        else a = carry_op(ct, rk, a, tot[k - 1]);
        if (t == 0) {
          carries[k] = a;
          if (chunks > 1)
            publish(inclusive + v, carry_op(ct, rk, a, tot[k]), flags + v,
                    STAGED_CARRY);
        }
      }
    }
    __syncthreads();
    for (int k0 = 0; k0 < P; k0 += TB / tc) {
      int k = k0 + t / tc;
      int n = k < P && u0 + k < units ? chunk_len(R, u0 + k, chunks) : 0;
      staged_chunk(ct, rk, s + k * cb, n, true, n > 0 ? carries[k] : 0,
                   wt + 8 * k, t % tc);
    }
    __syncthreads();
    for (int k = 0; k < P && u0 + k < units; k++) {
      int64_t v = u0 + k, o = v / chunks, at = v % chunks * NX_LOOP_CHUNK;
      chunk_out(y + (o * R + at) * es, s + k * cb, chunk_len(R, v, chunks), es,
                vout);
    }
    __syncthreads();
    if constexpr (WHAT == STAGE_PASS) return true;
  }
  /* The last group out zeroes the flags and the counters. */
  __syncthreads();
  if (t == 0) {
    __threadfence();
    *ticket = atomicAdd(flags + units + 1, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (!*ticket) return false;
  __threadfence();
  for (int64_t i = t; i < units + 2; i += TB) flags[i] = 0;
  return false;
}

/* A staged scan whose terms the tape computes: each ticket's units' loop
   indices, which follow each other, by tiles, then the pass over its
   chunks. In line in the kernel. */
__device__ __forceinline__ void staged_tape(const u64 *w, int E, int V) {
  int body = (int)(w[NX_LOOP_TAPE] + w[NX_LOOP_PROLOGUE]);
  int nbody = (int)w[NX_LOOP_BODY];
  int64_t R = (int64_t)w[NX_LOOP_TERMS], chunks = (int64_t)w[NX_LOOP_SEGMENTS];
  int64_t units = (int64_t)w[NX_LOOP_OUTPUTS] * chunks;
  int es = 4 * ct_words((int)((w[NX_LOOP_REDUCTION] >> 8) & 0xff));
  while (staged_scan<STAGE_TAKE>(w)) {
    int64_t u0 = *ticket_at(w, chunk_bytes(R, es));
    int64_t last = min(u0 + (int64_t)w[NX_LOOP_GROUP], units) - 1;
    int64_t end = chunk_first(R, last, chunks) + chunk_len(R, last, chunks);
    for (int64_t base = chunk_first(R, u0, chunks); base < end; base += E) {
      run(w, E, V, lanes_of(base + threadIdx.x, TB, end, V), body, nbody);
      staged_copy(w, base, end);
    }
    staged_scan<STAGE_PASS>(w);
  }
}

/* Networks: Logsumexp's and Moments' scans by nx_kernel.mli's odd/even
   network, its states streamed in order through its levels, each level
   holding the even state that waits for its partner and its last result,
   as nx.cpu streams them (reduce_float.h), so the joins are nx.cpu's. A
   Logsumexp state is a maximum and a sum; a Moments state a mean, a sum
   of squared deviations and a count. A thread's levels lie in the
   workspace from byte NX_LOOP_PARTIALS, NX_LOOP_GROUPS of them, two
   states a level, level-major so that a warp's accesses are side by
   side. */

template <class T> struct Net {
  T m, s, n;
};

template <class T> __device__ bool fin_(T x) { return x - x == 0; }

template <class T>
__device__ T lse_sum(T ma, T sa, T mb, T sb, T m) {
  bool hi = ma == m;
  T lo = hi ? mb : ma, sl = hi ? sb : sa, sh = hi ? sa : sb;
  T s = sh + sl * ex(lo - m);
  return fin_(m) ? s : m == -(T)INFINITY ? T(0) : m;
}

template <class T>
__device__ Net<T> join(int rk, const Net<T> &a, const Net<T> &b) {
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

template <class T>
__device__ __noinline__ void network_mode(const u64 *w, int E, int V) {
  int64_t units = (int64_t)w[NX_LOOP_OUTPUTS], R = (int64_t)w[NX_LOOP_TERMS];
  int64_t runs = (int64_t)w[NX_LOOP_RUNS];
  int rk = (int)(w[NX_LOOP_REDUCTION] & 0xff);
  int tp = (int)w[NX_LOOP_TERM], rp = (int)w[NX_LOOP_RESULT];
  int body = (int)(w[NX_LOOP_TAPE] + w[NX_LOOP_PROLOGUE]);
  int nbody = (int)w[NX_LOOP_BODY], nepi = (int)w[NX_LOOP_EPILOGUE];
  int64_t nt = (int64_t)gridDim.x * TB;
  Net<T> *mine = (Net<T> *)work_byte(w, NX_LOOP_PARTIALS) +
                 (int64_t)blockIdx.x * TB + threadIdx.x;
#define HELD(k) mine[(2 * (int64_t)(k)) * nt]
#define GOT(k) mine[(2 * (int64_t)(k) + 1) * nt]
  for (int64_t r0 = (int64_t)blockIdx.x * TB; r0 < units; r0 += nt) {
    int64_t slice = r0 + threadIdx.x;
    T c = 0;
    for (int64_t j = 0; j < runs; j++) {
      int64_t end = slice < units ? (slice + 1) * R : 0;
      lanes x = lanes_of(slice * R + j * V, 1, end, V);
      run(w, E, V, x, body, nbody);
      EACH {
        if (v >= x.n) continue;
        int64_t at = j * V + v;
        T x = get<T>(E, tp, i);
        Net<T> in;
        if (rk == NX_LOOP_LOGSUMEXP) {
          in.m = x;
          in.s = fin_(x) ? T(1) : x == -(T)INFINITY ? T(0) : x;
          in.n = 0;
        } else {
          if (at == 0) c = fin_(x) ? x : T(0);
          in.m = x - c;
          in.s = 0;
          in.n = 1;
        }
        int k = 0;
        int64_t q = at;
        for (; q & 1; k++, q >>= 1) in = join(rk, HELD(k), in);
        GOT(k) = q == 0 ? in : join(rk, GOT(k + 1), in);
        HELD(k) = in;
        Net<T> e = GOT(k);
        for (int h = 0; h < k; h++) GOT(h) = e;
        T r0v, r1v = 0;
        if (rk == NX_LOOP_LOGSUMEXP) r0v = fin_(e.m) ? e.m + log_(e.s) : e.m;
        else {
          r0v = c + e.m;
          r1v = fin_(e.m) ? e.s / ((T)at + 1) : e.m - e.m;
        }
        put<T>(E, rp, i, r0v);
        put<T>(E, rp + 4, i, r1v);
      }
      run(w, E, V, x, body + nbody, nepi);
    }
  }
#undef HELD
#undef GOT
}

/* Maps: tiles of consecutive indices. */
__device__ void map_mode(const u64 *w, int E, int V) {
  int64_t n = (int64_t)w[NX_LOOP_NUMEL];
  int body = (int)(w[NX_LOOP_TAPE] + w[NX_LOOP_PROLOGUE]);
  int nbody = (int)w[NX_LOOP_BODY];
  for (int64_t base = (int64_t)blockIdx.x * E; base < n;
       base += (int64_t)gridDim.x * E) {
    run(w, E, V, lanes_of(base + threadIdx.x, TB, n, V), body, nbody);
  }
}

} // namespace nx_loop

/* The unit's later sources (kernels.cu) see none of this file's macros
   but TB and LOOP. */
#undef ALL
#undef EACH
#undef COMPARE
#undef CPLX
#undef CPLX1
#undef CPLX2
#undef CT_c128
#undef CT_c64
#undef CT_f32
#undef CT_f64
#undef CT_i32
#undef CT_i64
#undef CT_u32
#undef CT_u64
#undef D_CPLX
#undef D_INT
#undef D_REAL
#undef INT
#undef INT1F
#undef INT1S
#undef INT2F
#undef INT2S
#undef REAL
#undef REAL1F
#undef REAL1S
#undef REAL2F
#undef REAL2S
#undef STAGED_CARRY
#undef STAGED_TOTAL
#undef PLAIN_TYPES
#undef RUN

/* The kernel: two groups a multiprocessor, which keeps every function's
   registers within 128 without a spill; three would spill the tape's
   runner. A group first copies the parameters it uses, up to the tape's
   end, into shared memory past the planes, whose reads take a fraction of
   the time of the parameters' own. A warp copies 32 words, each read by
   all its threads at once: the constant cache serves a warp's reads of
   one word together, of 32 words one at a time. A launch whose planes
   leave no room for the copy reads the parameters themselves. */

#define LOOP(name, ...)                                                        \
  extern "C" __global__ void __launch_bounds__(TB, 2)                          \
      name(const __grid_constant__ loop_params p) {                            \
    using namespace nx_loop;                                                   \
    const u64 *w = (const u64 *)p.w;                                           \
    if ((int64_t)p.w[NX_LOOP_COPY] >= 0) {                                     \
      u64 *copy = (u64 *)((uint8_t *)loop_planes + p.w[NX_LOOP_COPY]);        \
      int used = (int)(p.w[NX_LOOP_TAPE] + p.w[NX_LOOP_PROLOGUE] +             \
                       p.w[NX_LOOP_BODY] + p.w[NX_LOOP_EPILOGUE]);             \
      int lane = threadIdx.x & 31;                                             \
      for (int at = threadIdx.x - lane; at < used; at += TB) {                 \
        u64 mine = 0;                                                          \
        _Pragma("unroll") for (int k = 0; k < 32; k++) {                       \
          u64 x = p.w[at + k];                                                 \
          if (lane == k) mine = x;                                             \
        }                                                                      \
        if (at + lane < used) copy[at + lane] = mine;                          \
      }                                                                        \
      __syncthreads();                                                         \
      w = copy;                                                                \
    }                                                                          \
    int V = (int)((w[NX_LOOP_MODE] >> 8) & 0xff), E = V * TB;                  \
    lanes none = {0, 0, 0};                                                    \
    if (w[NX_LOOP_PROLOGUE] > 0)                                               \
      run(w, E, V, none, (int)w[NX_LOOP_TAPE], (int)w[NX_LOOP_PROLOGUE]);      \
    switch (w[NX_LOOP_MODE] & 0xff) {                                          \
    case NX_LOOP_MAP: map_mode(w, E, V); return;                               \
    case NX_LOOP_REDUCE: reduce_mode(w, E, V); return;                         \
    case NX_LOOP_SCAN: scan_mode(w, E, V); return;                             \
    case NX_LOOP_CARRIES: carries_mode(w); return;                             \
    case NX_LOOP_PLAIN: plain_mode(w); return;                                 \
    case NX_LOOP_PLAIN_SCAN: plain_scan_mode(w); return;                       \
    case NX_LOOP_PLAIN_CARRIES: plain_carries_mode(w); return;                 \
    case NX_LOOP_STAGED_SCAN:                                                  \
      if (w[NX_LOOP_BODY] > 0) staged_tape(w, E, V);                           \
      else staged_scan<STAGE_ALL>(w);                                          \
      return;                                                                  \
    default:                                                                   \
      if (((w[NX_LOOP_REDUCTION] >> 8) & 0xff) == NX_LOOP_F32)                 \
        network_mode<float>(w, E, V);                                          \
      else                                                                     \
        network_mode<double>(w, E, V);                                         \
    }                                                                          \
  }

#endif
