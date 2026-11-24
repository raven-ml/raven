/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The tape: a program the host lowers to words (kernels.h, Loops), run
   over a group's tile, and what it runs with: the parameters, the
   planes, lanes, kinds, loads, stores and positions, over values.h's held
   values, casts and element reads and writes. loop.metal's modes run it,
   and fold.h folds its terms.

   A group of NX_LOOP_THREADS threads runs tiles of v elements a thread.
   Each value the tape holds lives in planes of threadgroup memory, a
   32-bit word an element each, one to four planes as its dtype is wide:
   element [i] of plane [p] is word p E + i, E the tile's elements, and a
   thread's element v is i = v NX_LOOP_THREADS + its index. A thread reads
   and writes its own elements only, so the tape runs without barriers. A
   thread's elements' loop indices step evenly (lanes): past its first n
   elements none has one, whose loads read nothing and whose stores write
   nothing. A map's thread t takes the indices t + NX_LOOP_THREADS j of
   its tile, so that a SIMD-group's loads of one element are side by
   side.

   The parameters are read in place: the header from the struct at
   buffer 0, the table (extents, divisors, descriptors and tape) from the
   struct where it fits, else from the constant tail it addresses. Each
   operand has a buffer of its own, whose address is its layout's lowest
   element, so that positions from it are not negative: below 2^32
   elements, a position is a 32-bit sum, its coordinates divided by the
   host's divisors, and only the address adds 64 bits. NX_LOOP_WIDE takes
   64-bit indices and positions throughout.

   A value is held as its dtype's element exactly: a narrow float as its
   code, an integer of 16 bits or fewer sign- or zero-extended to 32, a
   boolean as its byte, so that Copy, Where and Bitcast move bits. The host
   lowers a kind at a narrow float to a decode, the float32 kind and an
   encode, and a kind at a narrow integer to its 32-bit kind and a wrap.
   float64 and complex128 values only move: a load, a store, a Where, a
   cast into their own dtype. */

#ifndef NX_METAL_TAPE_H
#define NX_METAL_TAPE_H

#include "device.h"

#include "nx_kernel.h"
#include "nx_kinds.h"
#include "values.h"

using metal::atomic_uint;
using metal::mulhi;

#define TB NX_LOOP_THREADS

/* A kind's code, below 128 beside its compute type, falls in its arity's
   range; kernels.h's offsets are nx_kernel.h's. */
static_assert(NX_LOOP_UNARY == NX_OP1_NEG, "unary codes");
static_assert(NX_LOOP_COMPARE == NX_OP2_EQUAL, "compare codes");
static_assert(NX_OP1_COUNT <= NX_LOOP_BINARY, "op1 codes reach op2's");
static_assert(NX_LOOP_BINARY + NX_OP2_COUNT <= NX_LOOP_TERNARY,
              "op2 codes reach op3's");
static_assert(NX_LOOP_TERNARY + NX_OP3_COUNT <= NX_LOOP_FMA_NARROW,
              "op3 codes reach the narrow Fmas'");
static_assert(NX_LOOP_FMA_BF16 < 128, "kind codes reach the compute type");

/* Parameters */

struct loop_words {
  device uchar *buffer[NX_LOOP_BUFFERS];
  constant u64 *tail;
  u64 header[NX_LOOP_HELD - NX_LOOP_MODE];
  u64 held[NX_LOOP_HOLDS];
};

NX_FIELD(loop_words, buffer, NX_LOOP_BUFFER);
NX_FIELD(loop_words, tail, NX_LOOP_TAIL);
NX_FIELD(loop_words, header, NX_LOOP_MODE);
NX_FIELD(loop_words, held, NX_LOOP_HELD);
NX_SIZE(loop_words, NX_LOOP_WORDS);

/* A group's view of its loop: its parameters [w], their table [t], its
   planes [pl], a tile's elements [E], a thread's [V], the calling
   thread [me]; whether its groups take tiles of the walk's last two
   axes ([tiled], loop_map_tiled's and loop_map_direct_tiled's), and
   whether its 4 elements may read a padded load ([pads],
   loop_map_padded's, loop.ml's lean_loop), and whether a direct map's
   loads and destination are 32 bits wide, taken and given as they are,
   in quads ([b32], loop_map_direct32's and loop_map_direct32_tiled's),
   or 8 bits wide, in quads ([b8], loop_map_direct8's and
   loop_map_direct8_tiled's). */
struct loop {
  constant loop_words *w;
  constant u64 *t;
  threadgroup uint *pl;
  uint E, V, me;
  bool wide, tiled, pads, b32, b8;
};

/* The view of a kernel of [V] elements a thread, of 64-bit indices where
   [wide]: constants of each kernel, so that the compiler unrolls a
   thread's elements and drops the paths it does not take, and a kernel's
   registers are those of its own. */
__attribute__((always_inline)) static inline loop
loop_of(constant loop_words &w, threadgroup uint *pl, uint me, uint V,
        bool wide) {
  loop c;
  c.w = &w;
  c.pl = pl;
  c.me = me;
  c.V = V;
  c.E = V * TB;
  c.wide = wide;
  c.tiled = false;
  c.pads = false;
  c.b32 = false;
  c.b8 = false;
  c.t = w.header[NX_LOOP_TABLE - NX_LOOP_MODE] <= NX_LOOP_HOLDS ? w.held
                                                                 : w.tail;
  return c;
}

/* Header word [i]. */
__attribute__((always_inline)) static inline u64
hd(thread const loop &c, int i) {
  return c.w->header[i - NX_LOOP_MODE];
}

__attribute__((always_inline)) static inline device uchar *
buffer_at(thread const loop &c, int k) {
  return c.w->buffer[k];
}

/* Planes */

#define WORD(c, p, i) (c).pl[(p) * (c).E + (i)]

static inline u64 get64(thread const loop &c, uint p, uint i) {
  return u64(WORD(c, p, i)) | u64(WORD(c, p + 1, i)) << 32;
}

static inline void put64(thread const loop &c, uint p, uint i, u64 x) {
  WORD(c, p, i) = uint(x);
  WORD(c, p + 1, i) = uint(x >> 32);
}

template <class T> static T get(thread const loop &c, uint p, uint i);
template <class T> static void put(thread const loop &c, uint p, uint i, T x);

template <> float get<float>(thread const loop &c, uint p, uint i) {
  return as_type<float>(WORD(c, p, i));
}
template <> int get<int>(thread const loop &c, uint p, uint i) {
  return as_type<int>(WORD(c, p, i));
}
template <> uint get<uint>(thread const loop &c, uint p, uint i) {
  return WORD(c, p, i);
}
template <> long get<long>(thread const loop &c, uint p, uint i) {
  return long(get64(c, p, i));
}
template <> u64 get<u64>(thread const loop &c, uint p, uint i) {
  return get64(c, p, i);
}
template <> nx_c64 get<nx_c64>(thread const loop &c, uint p, uint i) {
  return nx_c64_of(as_type<float>(WORD(c, p, i)),
                   as_type<float>(WORD(c, p + 1, i)));
}

template <> void put<float>(thread const loop &c, uint p, uint i, float x) {
  WORD(c, p, i) = as_type<uint>(x);
}
template <> void put<int>(thread const loop &c, uint p, uint i, int x) {
  WORD(c, p, i) = as_type<uint>(x);
}
template <> void put<uint>(thread const loop &c, uint p, uint i, uint x) {
  WORD(c, p, i) = x;
}
template <> void put<long>(thread const loop &c, uint p, uint i, long x) {
  put64(c, p, i, u64(x));
}
template <> void put<u64>(thread const loop &c, uint p, uint i, u64 x) {
  put64(c, p, i, x);
}
template <> void put<nx_c64>(thread const loop &c, uint p, uint i, nx_c64 x) {
  WORD(c, p, i) = as_type<uint>(x.re);
  WORD(c, p + 1, i) = as_type<uint>(x.im);
}

static inline held get_held(thread const loop &c, uint p, uint i, uint n) {
  held h = held_of(WORD(c, p, i), 0);
  if (n > 1) h.lo |= u64(WORD(c, p + 1, i)) << 32;
  if (n > 2) h.hi = get64(c, p + 2, i);
  return h;
}

static inline void put_held(thread const loop &c, uint p, uint i, uint n,
                            held h) {
  WORD(c, p, i) = uint(h.lo);
  if (n > 1) WORD(c, p + 1, i) = uint(h.lo >> 32);
  if (n > 2) put64(c, p + 2, i, h.hi);
}

/* Each element of the calling thread: its slot v and its index i. EACH
   runs them in turn; ALL unrolled, a slot past V doing nothing, so that
   their reads are in flight together. */
#define EACH for (uint v = 0, i = c.me; v < c.V; v++, i += TB)
#define ALL                                                                    \
  _Pragma("unroll") for (uint v = 0, i = c.me; v < NX_LOOP_MOST;              \
                         v++, i += TB) if (v < c.V)

/* The loop indices of the calling thread's elements: slot v's is
   L0 + v step for v below n; the others have none. */
struct lanes {
  u64 L0, step;
  uint n;
  bool quad;
};

__attribute__((always_inline)) static inline u64
index_of(lanes x, uint v) {
  return x.L0 + u64(v) * x.step;
}

/* The lanes of a thread whose first element's index is [L0], stepping by
   [step], those below [end] alone having one, of [V] elements. */
__attribute__((always_inline)) static inline lanes
lanes_of(u64 L0, u64 step, u64 end, uint V) {
  lanes x;
  x.L0 = L0;
  x.step = step;
  x.quad = false;
  if (L0 >= end) x.n = 0;
  else {
    const u64 r = end - L0;
    x.n = r > u64(V - 1) * step ? V : uint((r + step - 1) / step);
  }
  return x;
}

/* lanes_of in 32-bit words, for a kernel that is never wide, whose count
   [end] is below 2^32: a thread [off] indices past its group's first,
   [base]. The thread's lanes come from the count left past [base], so
   that no word wraps; in 64-bit words the nibble cast of 64 Mi elements
   takes 379 us on an M1 Max, in these 251. A lane's index stays below
   2^32; a byte or bit place made from it is widened to 64 bits where the
   index times its width may pass 2^32. */
__attribute__((always_inline)) static inline lanes
lanes32_of(uint base, uint off, uint step, uint end, uint V) {
  lanes x;
  x.L0 = base + off;
  x.step = step;
  x.quad = false;
  const uint left = base < end ? end - base : 0;
  if (off >= left) x.n = 0;
  else {
    const uint r = left - off;
    x.n = r > (V - 1) * step ? V : (r + step - 1) / step;
  }
  return x;
}

/* Kinds

   One function per kind and compute type over a thread's elements. A
   cheap kind's elements unrolled, their reads in flight together; a costly
   one's an element at a time, which keeps its code to one copy. */

#define K1(S, T, U, n)                                                         \
  static inline void k_##n##_##S(thread const loop &c, uint d, uint a, uint,  \
                                 uint) {                                       \
    T x[NX_LOOP_MOST];                                                         \
    ALL x[v] = get<T>(c, a, i);                                                \
    ALL put<T>(c, d, i, nx_##n##_##S(x[v]));                                   \
  }
#define K2(S, T, U, n)                                                         \
  static inline void k_##n##_##S(thread const loop &c, uint d, uint a, uint b, \
                                 uint) {                                       \
    T x[NX_LOOP_MOST], y[NX_LOOP_MOST];                                        \
    ALL {                                                                      \
      x[v] = get<T>(c, a, i);                                                  \
      y[v] = get<T>(c, b, i);                                                  \
    }                                                                          \
    ALL put<T>(c, d, i, nx_##n##_##S(x[v], y[v]));                             \
  }
#define KC(S, T, U, n)                                                         \
  static inline void k_##n##_##S(thread const loop &c, uint d, uint a, uint b, \
                                 uint) {                                       \
    T x[NX_LOOP_MOST], y[NX_LOOP_MOST];                                        \
    ALL {                                                                      \
      x[v] = get<T>(c, a, i);                                                  \
      y[v] = get<T>(c, b, i);                                                  \
    }                                                                          \
    ALL put<uint>(c, d, i, nx_##n##_##S(x[v], y[v]) != 0 ? 1u : 0u);          \
  }
#define K3(S, T, U, n)                                                         \
  static inline void k_##n##_##S(thread const loop &c, uint d, uint a, uint b, \
                                 uint e) {                                     \
    T x[NX_LOOP_MOST], y[NX_LOOP_MOST], z[NX_LOOP_MOST];                       \
    ALL {                                                                      \
      x[v] = get<T>(c, a, i);                                                  \
      y[v] = get<T>(c, b, i);                                                  \
      z[v] = get<T>(c, e, i);                                                  \
    }                                                                          \
    ALL put<T>(c, d, i, nx_##n##_##S(x[v], y[v], z[v]));                       \
  }
#define S1(S, T, U, n)                                                         \
  static inline void k_##n##_##S(thread const loop &c, uint d, uint a, uint,  \
                                 uint) {                                       \
    EACH put<T>(c, d, i, nx_##n##_##S(get<T>(c, a, i)));                       \
  }
#define S2(S, T, U, n)                                                         \
  static inline void k_##n##_##S(thread const loop &c, uint d, uint a, uint b, \
                                 uint) {                                       \
    EACH put<T>(c, d, i, nx_##n##_##S(get<T>(c, a, i), get<T>(c, b, i)));      \
  }
#define SC(S, T, U, n)                                                         \
  static inline void k_##n##_##S(thread const loop &c, uint d, uint a, uint b, \
                                 uint) {                                       \
    EACH put<uint>(                                                            \
        c, d, i, nx_##n##_##S(get<T>(c, a, i), get<T>(c, b, i)) != 0 ? 1u : 0u); \
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

REAL(f32, float)
CPLX(c64, nx_c64)
INT(i32, int)
INT(u32, uint)
INT(i64, long)
INT(u64, u64)
S2(u64, u64, THREEFRY, threefry)

/* A kind's op: its compute type's code, then its own (kernels.h). */
#define CT_f32 NX_LOOP_F32
#define CT_c64 NX_LOOP_C64
#define CT_i32 NX_LOOP_I32
#define CT_u32 NX_LOOP_U32
#define CT_i64 NX_LOOP_I64
#define CT_u64 NX_LOOP_U64

/* Fma at a narrow float (NX_LOOP_FMA_NARROW, NX_LOOP_FMA_BF16). */
static inline void k_fma_narrow(thread const loop &c, uint d, uint a, uint b,
                                uint e) {
  EACH put<float>(c, d, i,
                  nx_fma_narrow(get<float>(c, a, i), get<float>(c, b, i),
                                get<float>(c, e, i)));
}
static inline void k_fma_bf16(thread const loop &c, uint d, uint a, uint b,
                              uint e) {
  EACH put<float>(c, d, i,
                  nx_fma_bf16(get<float>(c, a, i), get<float>(c, b, i),
                              get<float>(c, e, i)));
}

#define C1(S, T, U, n)                                                         \
  case (CT_##S << 7) + NX_OP1_##U: k_##n##_##S(c, d, a, b, e); return;
#define C2(S, T, U, n)                                                         \
  case (CT_##S << 7) + NX_LOOP_BINARY + NX_OP2_##U:                            \
    k_##n##_##S(c, d, a, b, e);                                                \
    return;
#define C3(S, T, U, n)                                                         \
  case (CT_##S << 7) + NX_LOOP_TERNARY + NX_OP3_##U:                           \
    k_##n##_##S(c, d, a, b, e);                                                \
    return;
#define D_REAL(S, T)                                                           \
  REAL1F(C1, S, T) REAL1S(C1, S, T) REAL2F(C2, S, T) REAL2S(C2, S, T)          \
  COMPARE(C2, S, T) C3(S, T, FMA, fma)
#define D_INT(S, T)                                                            \
  INT1F(C1, S, T) INT1S(C1, S, T) INT2F(C2, S, T) INT2S(C2, S, T)              \
  COMPARE(C2, S, T) C3(S, T, FMA, fma)
#define D_CPLX(S, T) CPLX1(C1, S, T) CPLX2(C2, S, T) COMPARE(C2, S, T)

/* The kinds of op [op] (kernels.h), in two switches. A word's kinds, at
   float32, int32 and uint32, are those a kernel of 4 elements a thread
   runs: with them, a load or a store of whole bytes of at most 32 bits, a
   selection of one word, a truth, a wrap, or a decode or an encode of
   bfloat16 or float16 (a lean tape, loop.ml). A tape of anything else
   runs at 1 a thread, so that the kernels of 4, which keep 4 elements'
   values live, hold no rare path's registers and a core holds 1024 of
   their threads. The tape's runner is their one caller. */
static inline void word_kind(thread const loop &c, int op, uint d, uint a,
                             uint b, uint e) {
  switch (op) {
    D_REAL(f32, float) D_INT(i32, int) D_INT(u32, uint)
  case (NX_LOOP_F32 << 7) + NX_LOOP_FMA_NARROW:
    k_fma_narrow(c, d, a, b, e);
    return;
  case (NX_LOOP_F32 << 7) + NX_LOOP_FMA_BF16:
    k_fma_bf16(c, d, a, b, e);
    return;
  }
}

static inline void kind(thread const loop &c, int op, uint d, uint a, uint b,
                        uint e) {
  switch (op) {
    D_CPLX(c64, nx_c64) D_INT(i64, long) D_INT(u64, u64)
    C2(u64, u64, THREEFRY, threefry)
  default: word_kind(c, op, d, a, b, e);
  }
}

/* Walks */

/* Index [L]'s coordinate along loop axis [a]. */
__attribute__((always_inline)) static inline u64
coordinate(thread const loop &c, u64 L, int a) {
  const int rank = int(hd(c, NX_LOOP_RANK));
  constant u64 *ext = c.t, *div = c.t + rank;
  if (!c.wide) {
    uint l = uint(L);
    for (int k = rank - 1; k > a; k--) l = divide(l, div[k]);
    return l - divide(l, div[a]) * uint(ext[a]);
  }
  for (int k = rank - 1; k > a; k--) L /= ext[k];
  return L % ext[a];
}

/* The position of a plain operand's element at loop index [L]: its
   descriptor [d], the first position then a stride per axis. In 32 bits,
   wrapping, where the loop's positions fit them: the position itself is
   below 2^32. */
__attribute__((always_inline)) static inline u64
place(thread const loop &c, constant u64 *d, u64 L) {
  const int rank = int(hd(c, NX_LOOP_RANK));
  constant u64 *ext = c.t, *div = c.t + rank, *st = d + 2;
  if (!c.wide) {
    uint at = uint(d[1]), l = uint(L);
    for (int k = rank - 1; k > 0; k--) {
      const uint q = divide(l, div[k]);
      at += (l - q * uint(ext[k])) * uint(st[k]);
      l = q;
    }
    return rank > 0 ? at + l * uint(st[0]) : at;
  }
  long at = long(d[1]);
  for (int k = rank - 1; k > 0; k--) {
    const u64 q = L / ext[k];
    at += long(L - q * ext[k]) * long(st[k]);
    L = q;
  }
  return u64(rank > 0 ? at + long(L) * long(st[0]) : at);
}

/* The position a padded operand reads at loop index [L], or -1 where it
   reads its fill: per axis of the operand, the padded coordinate from the
   loop axes it reads less lo, where it lands on an element. */
static inline long padded_place(thread const loop &c, constant u64 *d,
                                u64 L) {
  const int r = int(d[4]);
  if (!c.wide) {
    /* In 32 bits: where the loop is not wide, a padded coordinate's
       magnitude is below 2^31 (loop.ml, wide), so q, summed in uint with
       wrapping, is right as an int; the interior divides by the host's
       divisor. */
    uint at = uint(d[1]);
    for (int j = 0; j < r; j++) {
      constant u64 *a = d + 5 + NX_LOOP_PAD * j;
      const int a1 = int(a[4] & 0xffff), a2 = int((a[4] >> 16) & 0xffff);
      uint u = -uint(a[0]);
      if (a1 != 0xffff) u += uint(coordinate(c, L, a1)) * uint(a[5]);
      if (a2 != 0xffff) u += uint(coordinate(c, L, a2)) * uint(a[6]);
      if (int(u) < 0) return -1;
      if (a[1] != 0) {
        const uint q = divide(u, a[7]);
        if (u - q * (uint(a[1]) + 1) != 0) return -1;
        u = q;
      }
      if (u >= uint(a[2])) return -1;
      at += u * uint(a[3]);
    }
    return long(at);
  }
  long at = long(d[1]);
  for (int j = 0; j < r; j++) {
    constant long *a = (constant long *)d + 5 + NX_LOOP_PAD * j;
    const int a1 = int(a[4] & 0xffff), a2 = int((a[4] >> 16) & 0xffff);
    long q = -a[0];
    if (a1 != 0xffff) q += long(coordinate(c, L, a1)) * a[5];
    if (a2 != 0xffff) q += long(coordinate(c, L, a2)) * a[6];
    /* An axis without an interior takes no division. */
    const long s = a[1] + 1;
    if (s > 1) {
      if (q < 0 || q % s != 0) return -1;
      q /= s;
    }
    if (q < 0 || q >= a[2]) return -1;
    at += q * a[3];
  }
  return at;
}

static inline constant u64 *operand(thread const loop &c, int table, uint k) {
  return c.t + c.t[hd(c, table) + k];
}

/* Descriptor [d]'s buffer and the bit its position 0 lies past. */
static inline device uchar *buffer_of(thread const loop &c, constant u64 *d) {
  return buffer_at(c, int((d[0] >> 16) & 0xff));
}
static inline uint phase_of(constant u64 *d) { return uint((d[0] >> 24) & 31); }

/* Load [k]'s held element at loop index [L]. */
static inline held load_at(thread const loop &c, uint k, u64 L) {
  constant u64 *d = operand(c, NX_LOOP_LOADS, k);
  const int dt = int(d[0] & 0xff);
  if ((d[0] >> 8) & 1) {
    const long p = padded_place(c, d, L);
    if (p < 0) return held_of(of_raw(dt, d[2]), d[3]);
    return read(buffer_of(c, d), u64(p), phase_of(d), dt);
  }
  return read(buffer_of(c, d), place(c, d, L), phase_of(d), dt);
}

/* The word of [dt] of [r], the bits of an element of at most 32 bits, as
   held. */
static inline uint held_word(int dt, uint r) {
  switch (dt) {
  case NX_INT16: return uint(int(short(ushort(r))));
  case NX_INT8: return uint(int(char(uchar(r))));
  default: return r;
  }
}

/* Element [p] of [b], an array of T: a 32-bit index where the loop's
   positions fit one. */
#define AT(T, b, p) (c.wide ? ((T *)(b))[p] : ((T *)(b))[uint(p)])

/* Quads: a thread's 4 consecutive indices of one row of the walk's last
   axis (NX_LOOP_QUADS). An operand reads them from its element at the
   first, [p0], stepping by its last stride: 4 adjacent elements as one
   vector where they lie on its size, one element for all where the
   stride is 0. Positions are 32-bit: a map of quads is never wide. */

/* Whether [e], an element of [d]'s buffer, lies on [bytes]: by the
   buffer's address, which the parameters hold. */
static inline bool on(thread const loop &c, constant u64 *d, uint p0,
                      uint size, uint bytes) {
  const u64 a = ((constant u64 *)c.w)[NX_LOOP_BUFFER + ((d[0] >> 16) & 0xff)];
  return ((a + u64(p0) * size) & (bytes - 1)) == 0;
}

/* The last stride of plain operand [d]. */
static inline long last_stride(thread const loop &c, constant u64 *d) {
  return long(d[2 + hd(c, NX_LOOP_RANK) - 1]);
}

template <class T, class T4>
static inline void read_quad(thread const loop &c, constant u64 *d, lanes x,
                             thread uint *y) {
  device const T *e = (device const T *)buffer_of(c, d);
  const uint p0 = uint(place(c, d, x.L0));
  const long st = last_stride(c, d);
  if (st == 0) {
    const uint z = x.n > 0 ? uint(e[p0]) : 0u;
    for (uint v = 0; v < 4; v++) y[v] = z;
    return;
  }
  if (st == 1 && x.n == 4 && on(c, d, p0, sizeof(T), 4 * sizeof(T))) {
    const T4 q = *(device const T4 *)(e + p0);
    y[0] = uint(q.x);
    y[1] = uint(q.y);
    y[2] = uint(q.z);
    y[3] = uint(q.w);
    return;
  }
  for (uint v = 0; v < 4; v++)
    y[v] = v < x.n ? uint(e[p0 + uint(long(v) * st)]) : 0u;
}

template <class T, class T4>
static inline void write_quad(thread const loop &c, constant u64 *d, lanes x,
                              uint a) {
  device T *e = (device T *)buffer_of(c, d);
  const uint p0 = uint(place(c, d, x.L0));
  const long st = last_stride(c, d);
  const uint i = c.me;
  if (st == 1 && x.n == 4 && on(c, d, p0, sizeof(T), 4 * sizeof(T))) {
    *(device T4 *)(e + p0) =
        T4(T(WORD(c, a, i)), T(WORD(c, a, i + TB)), T(WORD(c, a, i + 2 * TB)),
           T(WORD(c, a, i + 3 * TB)));
    return;
  }
  for (uint v = 0; v < 4; v++)
    if (v < x.n) e[p0 + uint(long(v) * st)] = T(WORD(c, a, i + v * TB));
}

/* The positions of plain operand [d]'s elements at the thread's indices,
   into [at]. Where 4 lie in one row of the walk's last axis, the first's
   is placed and the others step from it by the last stride: in 32 bits,
   wrapping as [place] does, since a kernel of 4 elements a thread is
   never wide. */
static inline void positions(thread const loop &c, constant u64 *d, lanes x,
                             thread u64 *at) {
  const u64 p0 = place(c, d, x.L0);
  if (c.V == 1) {
    at[0] = p0;
    return;
  }
  const int rank = int(hd(c, NX_LOOP_RANK));
  const u64 span = u64(x.n > 0 ? x.n - 1 : 0) * x.step;
  if (rank > 0 && coordinate(c, x.L0, rank - 1) + span < c.t[rank - 1]) {
    const uint st = uint(x.step * d[2 + rank - 1]);
    ALL at[v] = uint(p0) + v * st;
    return;
  }
  ALL at[v] = v < x.n ? place(c, d, index_of(x, v)) : 0;
}

/* Padded quads

   Under quads a thread's 4 indices share every coordinate but the last.
   A map of quads with padded loads has NX_LOOP_COORDS: the thread
   writes its first index's coordinates once into group memory from
   word NX_LOOP_COORD_AT, axis k's at k TB + me, and its padded loads
   read them there (loop.ml refuses quads where they do not fit). Group
   memory, since a load's axes come from its descriptor at run time, so
   that an array of them in registers would go to the stack. A padded
   load takes, per axis of its operand, the padded coordinate at the
   first index and its step along the walk's last axis; then, per
   element, the bounds, the interior and the position, in 32 bits as
   padded_place. Its elements are one vector read where all 4 are
   elements, adjacent and on their size. */

static inline void quad_coords(thread const loop &c, u64 L0) {
  const int rank = int(hd(c, NX_LOOP_RANK));
  constant u64 *ext = c.t, *div = c.t + rank;
  threadgroup uint *co = c.pl + hd(c, NX_LOOP_COORD_AT) + c.me;
  uint l = uint(L0);
  for (int k = rank - 1; k > 0; k--) {
    const uint q = divide(l, div[k]);
    co[k * TB] = l - q * uint(ext[k]);
    l = q;
  }
  if (rank > 0) co[0] = l;
}

static inline uint quad_coord(thread const loop &c, int a) {
  return c.pl[hd(c, NX_LOOP_COORD_AT) + uint(a) * TB + c.me];
}

/* The positions of padded operand [d]'s elements at the thread's quad
   into [at], and whether each reads an element into [in]. */
static inline void padded_quad(thread const loop &c, constant u64 *d,
                               lanes x, thread uint *at, thread bool *in) {
  const int r = int(d[4]), last = int(hd(c, NX_LOOP_RANK)) - 1;
  ALL {
    at[v] = uint(d[1]);
    in[v] = v < x.n;
  }
  for (int j = 0; j < r; j++) {
    constant u64 *a = d + 5 + NX_LOOP_PAD * j;
    const int a1 = int(a[4] & 0xffff), a2 = int((a[4] >> 16) & 0xffff);
    uint u = -uint(a[0]), k = 0;
    if (a1 != 0xffff) {
      u += quad_coord(c, a1) * uint(a[5]);
      if (a1 == last) k += uint(a[5]);
    }
    if (a2 != 0xffff) {
      u += quad_coord(c, a2) * uint(a[6]);
      if (a2 == last) k += uint(a[6]);
    }
    ALL {
      uint w = u + v * k;
      bool ok = int(w) >= 0;
      if (ok && a[1] != 0) {
        const uint q = divide(w, a[7]);
        ok = w - q * (uint(a[1]) + 1) == 0;
        w = q;
      }
      in[v] = in[v] && ok && w < uint(a[2]);
      at[v] += w * uint(a[3]);
    }
  }
}

/* A padded load's elements of T at [at] where [in], else its [fill]'s
   bits, into [y]. */
template <class T, class T4>
static inline void padded_read(thread const loop &c, constant u64 *d,
                               thread const uint *at, thread const bool *in,
                               uint fill, thread uint *y) {
  device const T *e = (device const T *)buffer_of(c, d);
  bool whole = true;
  ALL whole = whole && in[v] && at[v] == at[0] + v;
  if (c.V == 4 && whole && on(c, d, at[0], sizeof(T), 4 * sizeof(T))) {
    const T4 q = *(device const T4 *)(e + at[0]);
    y[0] = uint(q.x);
    y[1] = uint(q.y);
    y[2] = uint(q.z);
    y[3] = uint(q.w);
    return;
  }
  ALL y[v] = in[v] ? uint(e[at[v]]) : fill;
}

/* Tiles

   A tiled map's group takes NX_LOOP_TILE indices along b by NX_LOOP_TILE
   consecutive along a, the walk's last two axes of extent past 1
   (kernels.h, NX_LOOP_TILE_A), and its thread 4 consecutive along a: row
   me / 8 of the tile, from column 4 (me mod 8). Tiles go along b
   2^NX_LOOP_TILE_RUN at a time, then along a, then along b again, then
   the outer axes: at 16 or 8 bits a staged load's line holds 2 or 4
   tiles' runs along b, which then run together. A plane's last run
   along b is whole, and its groups past the last tile return at once.

   A staged load, of stride 1 along b, is read before the tape: thread
   me reads the 4 elements along b from row 4 (me mod 8) at column me / 8,
   as one vector where they lie on its size, and writes their bits to its
   stage, column-major; one barrier; then the tape reads each of the
   thread's elements from the stage at its row and column. Each line of
   the load is then read whole by one group, where read by quads along a
   each element takes a line. Elements past the walk's extents read and
   stage zeros, and every thread stages and reaches the barrier. */

/* The stage of load descriptor [d], from 1, or 0. */
static inline uint stage_of(constant u64 *d) { return uint(d[0] >> 32) & 15; }

static inline threadgroup uint *stage_at(thread const loop &c, uint s) {
  return c.pl + hd(c, NX_LOOP_STAGE) + s * NX_LOOP_STAGE_WORDS;
}

/* [n] elements of T from [e] at [p0] into [y], one vector where 4 lie on
   their size. */
template <class T, class T4>
static inline void read_run(thread const loop &c, constant u64 *d, uint p0,
                            uint n, thread uint *y) {
  device const T *e = (device const T *)buffer_of(c, d);
  if (n == 4 && on(c, d, p0, sizeof(T), 4 * sizeof(T))) {
    const T4 q = *(device const T4 *)(e + p0);
    y[0] = uint(q.x);
    y[1] = uint(q.y);
    y[2] = uint(q.z);
    y[3] = uint(q.w);
    return;
  }
  for (uint v = 0; v < 4; v++) y[v] = v < n ? uint(e[p0 + v]) : 0u;
}

/* A group's tile: its corner's indices along b and a, the loop index
   of its plane's first element, and whether the group has a tile. */
struct tile {
  uint b0, a0, base;
  bool live;
};

/* The tile of group [gid], its number cut by the host's divisors
   (kernels.h, NX_LOOP_TILES). */
static inline tile tile_of(thread const loop &c, uint gid) {
  const uint ea = uint(hd(c, NX_LOOP_TILE_A));
  const uint eb = uint(hd(c, NX_LOOP_TILE_B));
  const uint lr = uint(hd(c, NX_LOOP_TILE_RUN));
  const uint ta = (ea + NX_LOOP_TILE - 1) / NX_LOOP_TILE;
  const uint tb = (eb + NX_LOOP_TILE - 1) / NX_LOOP_TILE;
  const uint plane = ta * (((tb + (1u << lr) - 1) >> lr) << lr);
  const uint g = uint(hd(c, NX_LOOP_FIRST)) + gid;
  const uint o = divide(g, hd(c, NX_LOOP_TILES));
  const uint r = g - o * plane;
  const uint s = r >> lr, w = r & ((1u << lr) - 1);
  const uint stripe = divide(s, hd(c, NX_LOOP_TILES_A));
  const uint bt = (stripe << lr) + w;
  tile t;
  t.a0 = NX_LOOP_TILE * (s - stripe * ta);
  t.b0 = NX_LOOP_TILE * bt;
  t.base = o * ea * eb;
  t.live = bt < tb;
  return t;
}

/* The position of plain operand [d]'s element at row [b] and column [a]
   of tile [t]: a and b are the walk's last two axes. */
static inline uint tile_place(thread const loop &c, constant u64 *d, tile t,
                              uint b, uint a) {
  const int rank = int(hd(c, NX_LOOP_RANK));
  const uint at = rank > 2 ? uint(place(c, d, t.base)) : uint(d[1]);
  return at + (t.b0 + b) * uint(d[rank]) + (t.a0 + a) * uint(d[rank + 1]);
}

/* Stages each staged load of tile [t]; then waits for the group. */
static inline void stage(thread const loop &c, tile t) {
  const u64 list = hd(c, NX_LOOP_STAGED);
  const uint na = uint(hd(c, NX_LOOP_TILE_A)) - t.a0;
  const uint nb = uint(hd(c, NX_LOOP_TILE_B)) - t.b0;
  const uint col = c.me / 8, row = 4 * (c.me % 8);
  for (uint s = 0; s < uint(list & 15); s++) {
    constant u64 *d =
        operand(c, NX_LOOP_LOADS, uint(list >> (4 + 8 * s)) & 0xff);
    uint y[4] = {0, 0, 0, 0};
    if (col < na && row < nb) {
      const uint p0 = tile_place(c, d, t, row, col);
      const uint n = metal::min(4u, nb - row);
      switch (bits_of(int(d[0] & 0xff))) {
      case 32: read_run<uint, uint4>(c, d, p0, n, y); break;
      case 16: read_run<ushort, ushort4>(c, d, p0, n, y); break;
      default: read_run<uchar, uchar4>(c, d, p0, n, y);
      }
    }
    threadgroup uint *at = stage_at(c, s) + col * (NX_LOOP_TILE + 1) + row;
    for (uint v = 0; v < 4; v++) at[v] = y[v];
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}

/* The lanes of the calling thread in tile [t]. */
static inline lanes tile_lanes(thread const loop &c, tile t) {
  const uint ea = uint(hd(c, NX_LOOP_TILE_A));
  const uint eb = uint(hd(c, NX_LOOP_TILE_B));
  const uint ii = c.me / 8, jj = 4 * (c.me % 8);
  lanes x;
  x.L0 = u64(t.base) + (t.b0 + ii) * ea + t.a0 + jj;
  x.step = 1;
  x.quad = false;
  x.n = t.b0 + ii < eb && t.a0 + jj < ea ? metal::min(4u, ea - t.a0 - jj)
                                         : 0;
  return x;
}

/* A staged load's bits at the thread's 4 indices, from stage [s]. */
static inline void staged_bits(thread const loop &c, uint s, thread uint *y) {
  threadgroup const uint *t = stage_at(c, s - 1);
  const uint ii = c.me / 8, jj = 4 * (c.me % 8);
  for (uint v = 0; v < 4; v++) y[v] = t[(jj + v) * (NX_LOOP_TILE + 1) + ii];
}

/* Load [k]'s elements at the thread's indices into plane [p], an element
   of no index taking zeros: every element's read before any is stored,
   so that the reads are in flight together. A plain operand of whole
   bytes in one word, the common case, reads its elements by their width
   alone, the descriptor read once. */
static inline void load(thread const loop &c, lanes x, uint p, uint k) {
  constant u64 *d = operand(c, NX_LOOP_LOADS, k);
  const int dt = int(d[0] & 0xff);
  const uint n = words_of(dt), bits = bits_of(dt);
  if (c.V != 4 && (n > 2 || ((d[0] >> 8) & 1))) {
    EACH put_held(c, p, i, n, v < x.n ? load_at(c, k, index_of(x, v))
                                      : held_of(0, 0));
    return;
  }
  if (c.pads && ((d[0] >> 8) & 1)) {
    /* A padded load of 8, 16 or 32 bits (loop.ml, lean_operand). */
    uint at[NX_LOOP_MOST], y[NX_LOOP_MOST];
    bool in[NX_LOOP_MOST];
    if (x.quad) padded_quad(c, d, x, at, in);
    else
      ALL {
        const long q = v < x.n ? padded_place(c, d, index_of(x, v)) : -1;
        in[v] = q >= 0;
        at[v] = uint(q);
      }
    const uint fill = uint(d[2]);
    switch (bits) {
    case 32: padded_read<uint, uint4>(c, d, at, in, fill, y); break;
    case 16: padded_read<ushort, ushort4>(c, d, at, in, fill, y); break;
    default: padded_read<uchar, uchar4>(c, d, at, in, fill, y);
    }
    ALL WORD(c, p, i) = held_word(dt, y[v]);
    return;
  }
  if (c.V == 4 || (bits >= 8 && bits <= 32)) {
    device const uchar *b = buffer_of(c, d);
    uint y[NX_LOOP_MOST];
    if (c.tiled && stage_of(d) != 0) {
      staged_bits(c, stage_of(d), y);
      ALL WORD(c, p, i) = held_word(dt, y[v]);
      return;
    }
    if (c.V == 4 && x.quad) {
      switch (bits) {
      case 32: read_quad<uint, uint4>(c, d, x, y); break;
      case 16: read_quad<ushort, ushort4>(c, d, x, y); break;
      default: read_quad<uchar, uchar4>(c, d, x, y);
      }
      ALL WORD(c, p, i) = held_word(dt, y[v]);
      return;
    }
    u64 at[NX_LOOP_MOST];
    positions(c, d, x, at);
    switch (bits) {
    case 32:
      ALL y[v] = v < x.n ? AT(device const uint, b, at[v]) : 0u;
      break;
    case 16:
      ALL y[v] = v < x.n ? uint(AT(device const ushort, b, at[v])) : 0u;
      break;
    default:
      ALL y[v] = v < x.n ? uint(AT(device const uchar, b, at[v])) : 0u;
    }
    ALL WORD(c, p, i) = held_word(dt, y[v]);
    return;
  }
  EACH put_held(c, p, i, n, v < x.n ? load_at(c, k, index_of(x, v))
                                    : held_of(0, 0));
}

/* Stores plane [a]'s elements at the thread's indices into destination
   [k], placed as load places them. */
static inline void store(thread const loop &c, lanes x, uint a, uint k) {
  constant u64 *d = operand(c, NX_LOOP_DSTS, k);
  const int dt = int(d[0] & 0xff);
  const uint n = words_of(dt), phase = phase_of(d);
  device uchar *b = buffer_of(c, d);
  const uint bits = bits_of(dt);
  if (c.V == 4 && x.quad) {
    switch (bits) {
    case 32: write_quad<uint, uint4>(c, d, x, a); return;
    case 16: write_quad<ushort, ushort4>(c, d, x, a); return;
    default: write_quad<uchar, uchar4>(c, d, x, a); return;
    }
  }
  switch (bits) {
  case 32:
    ALL if (v < x.n) AT(device uint, b, place(c, d, index_of(x, v))) =
        WORD(c, a, i);
    return;
  case 16:
    ALL if (v < x.n) AT(device ushort, b, place(c, d, index_of(x, v))) =
        ushort(WORD(c, a, i));
    return;
  case 8:
    ALL if (v < x.n) AT(device uchar, b, place(c, d, index_of(x, v))) =
        uchar(WORD(c, a, i));
    return;
  default:
    if (c.V != 4)
      EACH if (v < x.n) write(b, place(c, d, index_of(x, v)), phase, dt,
                              get_held(c, a, i, n));
  }
}

/* The tape */

/* Instruction [op] of planes [d], [a], [b] and [e] over each element in
   turn: the runner's rarer instructions. */
static inline void each_op(thread const loop &c, lanes x, int op, uint d,
                           uint a, uint b, uint e) {
  switch (op) {
  case NX_LOOP_COORD:
    EACH put<long>(c, d, i, v < x.n ? long(coordinate(c, index_of(x, v), a))
                                    : 0);
    return;
  case NX_LOOP_CAST:
    EACH put_held(c, d, i, words_of(e),
                  cast(b, e, get_held(c, a, i, words_of(b))));
    return;
  case NX_LOOP_BITCAST:
    EACH put_held(c, d, i, words_of(e),
                  bitcast(b, e, get_held(c, a, i, words_of(b))));
    return;
  case NX_LOOP_DECODE:
    EACH put<float>(c, d, i, narrow_value(b, WORD(c, a, i)));
    return;
  case NX_LOOP_ENCODE:
    EACH WORD(c, d, i) = narrow_code(b, get<float>(c, a, i));
    return;
  }
}

/* Writes the prologue's constants into their planes, before any tile: a
   prologue holds NX_LOOP_CONST instructions alone. */
static inline void constants(thread const loop &c) {
  const u64 from = hd(c, NX_LOOP_TAPE), to = from + hd(c, NX_LOOP_PROLOGUE);
  for (u64 at = from; at < to; at += 3) {
    const u64 t = c.t[at];
    const uint d = uint((t >> 16) & 0xfff), n = uint((t >> 40) & 0xfff);
    const held h = held_of(c.t[at + 1], c.t[at + 2]);
    EACH put_held(c, d, i, n, h);
  }
}

/* Runs the tape's [n] words from [from]: its loads, stores and selections
   in line, the kinds through their switch. */
static inline void run(thread const loop &c, lanes x, u64 from, u64 n) {
  for (u64 at = from; at < from + n; at++) {
    const u64 t = c.t[at];
    const int op = int(t & 0xffff);
    const uint d = uint((t >> 16) & 0xfff), a = uint((t >> 28) & 0xfff);
    const uint b = uint((t >> 40) & 0xfff), e = uint((t >> 52) & 0xfff);
    if (op < NX_LOOP_LOAD) {
      if (c.V == 4) word_kind(c, op, d, a, b, e);
      else kind(c, op, d, a, b, e);
      continue;
    }
    switch (op) {
    case NX_LOOP_LOAD: load(c, x, d, a); break;
    case NX_LOOP_STORE: store(c, x, a, b); break;
    case NX_LOOP_WHERE1: {
      uint s[NX_LOOP_MOST], y[NX_LOOP_MOST], z[NX_LOOP_MOST];
      ALL {
        s[v] = WORD(c, a, i);
        y[v] = WORD(c, b, i);
        z[v] = WORD(c, e, i);
      }
      ALL WORD(c, d, i) = s[v] != 0 ? y[v] : z[v];
      break;
    }
    case NX_LOOP_WHERE2:
    case NX_LOOP_WHERE4: {
      if (c.V == 4) break;
      const uint m = op == NX_LOOP_WHERE2 ? 2 : 4;
      EACH put_held(c, d, i, m, get_held(c, WORD(c, a, i) != 0 ? b : e, i, m));
      break;
    }
    case NX_LOOP_TRUTH: {
      uint y[NX_LOOP_MOST];
      ALL y[v] = WORD(c, a, i);
      ALL WORD(c, d, i) = y[v] != 0 ? 1u : 0u;
      break;
    }
    case NX_LOOP_DECODE:
    case NX_LOOP_ENCODE: {
      if (b != NX_BFLOAT16 && b != NX_FLOAT16) {
        if (c.V != 4) each_op(c, x, op, d, a, b, e);
        break;
      }
      uint y[NX_LOOP_MOST];
      ALL y[v] = WORD(c, a, i);
      if (op == NX_LOOP_DECODE && b == NX_BFLOAT16) {
        ALL put<float>(c, d, i, nx_bf16_to_float(ushort(y[v])));
      } else if (op == NX_LOOP_DECODE) {
        ALL put<float>(c, d, i, nx_metal_f16_to_float(ushort(y[v])));
      } else if (b == NX_BFLOAT16) {
        ALL WORD(c, d, i) = nx_float_to_bf16(as_type<float>(y[v]));
      } else {
        ALL WORD(c, d, i) = nx_metal_float_to_f16(as_type<float>(y[v]));
      }
      break;
    }
    case NX_LOOP_CAST:
      /* A cast into its own dtype of one word, as a term into its plane,
         copies. */
      if (b == e && words_of(int(b)) == 1) {
        uint y[NX_LOOP_MOST];
        ALL y[v] = WORD(c, a, i);
        ALL WORD(c, d, i) = y[v];
      } else if (c.V != 4)
        each_op(c, x, op, d, a, b, e);
      break;
    case NX_LOOP_WRAP: {
      uint y[NX_LOOP_MOST];
      ALL y[v] = WORD(c, a, i);
      ALL WORD(c, d, i) = uint(of_raw(int(b), y[v]));
      break;
    }
    default:
      if (c.V != 4) each_op(c, x, op, d, a, b, e);
    }
  }
}

/* The tape's body, after its prologue. */
static inline u64 body_at(thread const loop &c) {
  return hd(c, NX_LOOP_TAPE) + hd(c, NX_LOOP_PROLOGUE);
}

#endif
