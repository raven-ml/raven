/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Folds: the states of a reduction's and a scan's terms, their merges,
   and the trees, chains, carries and finishes over them, which fix a
   result's bits. loop.metal's modes fold the tape's terms (tape.h)
   through them. */

#ifndef NX_METAL_FOLD_H
#define NX_METAL_FOLD_H

#include "tape.h"

using metal::mem_flags;
using metal::simd_shuffle_down;
using metal::threadgroup_barrier;

/* States

   A state of an output's terms: its value in words 0 to 3, held at its
   compute type; a position or a second sum in 4 and 5; whether it holds a
   term in 6. A sum's lanes start from +0 and every lane takes part in the
   tree, as nx.cpu's do: a lane from +0 is never -0, so a lane of no term,
   which adds +0, changes no bit. Another monoid's lane, and Arg's, start
   from their first term, a lane of no term taking no part.

   Two sites that may combine the same pair of NaNs are the lanes' folds
   and their trees, and a scan's chain and its carries. Each site is a
   function of the dtypes and the extents alone, and each float combine
   chooses its NaN explicitly ([kept]): an instruction keeps its first
   operand's NaN, and the compiler may commute an add's or a multiply's
   operands. */

struct st {
  held v;
  u64 x;
  uint has;
};

/* A combine's result [r], or, where [r] is a NaN, the first NaN of [a]
   then [b]. */
static inline float kept(float a, float b, float r) {
  return r == r ? r : a != a ? a : b != b ? b : r;
}

/* A complex combine's result [r]: a part that is a NaN takes the first
   NaN of [a]'s parts then [b]'s. Maximum and Minimum choose theirs
   already. */
static inline nx_c64 kept_c(nx_c64 a, nx_c64 b, nx_c64 r) {
  const float n = a.re != a.re ? a.re
                  : a.im != a.im ? a.im
                  : b.re != b.re ? b.re
                                 : b.im;
  r.re = kept(n, n, r.re);
  r.im = kept(n, n, r.im);
  return r;
}

template <class T> T as(held h);
template <> float as<float>(held h) { return as_type<float>(uint(h.lo)); }
template <> int as<int>(held h) { return int(uint(h.lo)); }
template <> uint as<uint>(held h) { return uint(h.lo); }
template <> long as<long>(held h) { return long(h.lo); }
template <> u64 as<u64>(held h) { return h.lo; }
template <> nx_c64 as<nx_c64>(held h) {
  return nx_c64_of(as_type<float>(uint(h.lo)), as_type<float>(uint(h.lo >> 32)));
}

static inline held of(float x) { return held_of(as_type<uint>(x), 0); }
static inline held of(int x) { return held_of(uint(x), 0); }
static inline held of(uint x) { return held_of(x, 0); }
static inline held of(long x) { return held_of(u64(x), 0); }
static inline held of(u64 x) { return held_of(x, 0); }
static inline held of(nx_c64 x) {
  return held_of(u64(as_type<uint>(x.re)) | u64(as_type<uint>(x.im)) << 32, 0);
}

/* [a] op [b] at the compute type [ct], op NX_OP2_ADD, NX_OP2_MUL,
   NX_OP2_MAXIMUM or NX_OP2_MINIMUM, through [K] the NaN it keeps. */
#define ARITH(S, T, K)                                                         \
  {                                                                            \
    const T x = as<T>(a), y = as<T>(b);                                        \
    switch (op) {                                                              \
    case NX_OP2_ADD: return of(K(x, y, nx_add_##S(x, y)));                     \
    case NX_OP2_MUL: return of(K(x, y, nx_mul_##S(x, y)));                     \
    case NX_OP2_MAXIMUM: return of(nx_maximum_##S(x, y));                      \
    default: return of(nx_minimum_##S(x, y));                                 \
    }                                                                          \
  }

/* Integers keep their result. */
#define EXACT(x, y, r) (r)

/* [w]: [ct] is a word's, float32, int32 or uint32, as in a kernel of 4
   elements a thread (word_kind), whose registers the other types' paths
   then do not hold. */
static inline held arith(bool w, int ct, int op, held a, held b) {
  if (w) switch (ct) {
    case NX_LOOP_F32: ARITH(f32, float, kept)
    case NX_LOOP_I32: ARITH(i32, int, EXACT)
    default: ARITH(u32, uint, EXACT)
    }
  switch (ct) {
  case NX_LOOP_F32: ARITH(f32, float, kept)
  case NX_LOOP_C64: ARITH(c64, nx_c64, kept_c)
  case NX_LOOP_I32: ARITH(i32, int, EXACT)
  case NX_LOOP_U32: ARITH(u32, uint, EXACT)
  case NX_LOOP_I64: ARITH(i64, long, EXACT)
  default: ARITH(u64, u64, EXACT)
  }
}

#undef ARITH
#undef EXACT

/* Arg's comparison of [a] and [b] at [ct]: bit 0 if [a] is a NaN, bit 1
   if [b] is, and, neither a NaN, bit 2 if [a] follows [b] in the dtype's
   order and bit 3 if [b] follows [a]. */
static inline int order(bool w, int ct, held a, held b) {
  if (w && ct != NX_LOOP_F32) {
    if (ct == NX_LOOP_I32) {
      const int x = as<int>(a), y = as<int>(b);
      return int(x > y) << 2 | int(y > x) << 3;
    }
    const uint x = as<uint>(a), y = as<uint>(b);
    return int(x > y) << 2 | int(y > x) << 3;
  }
  switch (ct) {
  case NX_LOOP_F32: {
    const float x = as<float>(a), y = as<float>(b);
    return int(x != x) | int(y != y) << 1 | int(nx_order_f32(x, y) > 0) << 2 |
           int(nx_order_f32(y, x) > 0) << 3;
  }
  case NX_LOOP_C64: {
    const nx_c64 x = as<nx_c64>(a), y = as<nx_c64>(b);
    return int(nx_nan_c64(x)) | int(nx_nan_c64(y)) << 1 |
           int(nx_order_c64(x, y) > 0) << 2 | int(nx_order_c64(y, x) > 0) << 3;
  }
  case NX_LOOP_I32: {
    const int x = as<int>(a), y = as<int>(b);
    return int(x > y) << 2 | int(y > x) << 3;
  }
  case NX_LOOP_U32: {
    const uint x = as<uint>(a), y = as<uint>(b);
    return int(x > y) << 2 | int(y > x) << 3;
  }
  case NX_LOOP_I64: {
    const long x = as<long>(a), y = as<long>(b);
    return int(x > y) << 2 | int(y > x) << 3;
  }
  default: {
    const u64 x = as<u64>(a), y = as<u64>(b);
    return int(x > y) << 2 | int(y > x) << 3;
  }
  }
}

/* Arg's state [a] then [b]: the extreme, a NaN first, and of one value
   the earlier position. */
static inline st arg_merge(bool w, int rk, int ct, st a, st b) {
  if (!b.has) return a;
  bool take;
  if (!a.has) take = true;
  else {
    const int o = order(w, ct, a.v, b.v);
    const bool an = o & 1, bn = o & 2;
    const bool b_first = rk == NX_LOOP_ARG_MAX ? (o & 8) : (o & 4);
    const bool a_first = rk == NX_LOOP_ARG_MAX ? (o & 4) : (o & 8);
    if (an || bn) take = an && bn ? long(b.x) < long(a.x) : bn;
    else if (b_first) take = true;
    else if (a_first) take = false;
    else take = long(b.x) < long(a.x);
  }
  return take ? b : a;
}

/* State [a] then state [b], of reduction [rk] at [ct], a word's where
   [w] (arith). */
static inline st merge(bool w, int rk, int ct, st a, st b) {
  switch (rk) {
  case NX_LOOP_SUM:
  case NX_LOOP_EXPS: a.v = arith(w, ct, NX_OP2_ADD, a.v, b.v); return a;
  case NX_LOOP_SQUARES:
    a.v = arith(w, ct, NX_OP2_ADD, a.v, b.v);
    a.x = arith(w, ct, NX_OP2_ADD, held_of(a.x, 0), held_of(b.x, 0)).lo;
    return a;
  case NX_LOOP_ARG_MAX:
  case NX_LOOP_ARG_MIN: return arg_merge(w, rk, ct, a, b);
  }
  if (!b.has) return a;
  if (!a.has) return b;
  const int op = rk == NX_LOOP_PROD  ? NX_OP2_MUL
                 : rk == NX_LOOP_MAX ? NX_OP2_MAXIMUM
                                     : NX_OP2_MINIMUM;
  a.v = arith(w, ct, op, a.v, b.v);
  return a;
}

/* Logsumexp's exp of [y], 0 below the cut (nx_metal.mli). */
static inline float ex(float y) { return y < -87.f ? 0.f : nx_exp_f32(y); }

/* A centered term [x] of center [cc]: exp (x - cc), or (x - cc)^2 with the
   second sum x - cc. Float32 alone: logsumexp and moments compute in
   their carrier, which is float32 or a narrower float here. */
static inline st centered(int rk, held x, held cc) {
  const float d = as<float>(x) - as<float>(cc);
  st b;
  b.has = 1;
  b.x = 0;
  if (rk == NX_LOOP_EXPS) b.v = of(ex(d));
  else {
    b.v = of(d * d);
    b.x = of(d).lo;
  }
  return b;
}

/* State [a] then the term [x] at position [pos], of center [cc]. */
static inline st fold(bool w, int rk, int ct, st a, held x, u64 pos,
                      held cc) {
  st b;
  b.v = x;
  b.x = pos;
  b.has = 1;
  if (rk == NX_LOOP_EXPS || rk == NX_LOOP_SQUARES) b = centered(rk, x, cc);
  return merge(w, rk, ct, a, b);
}

/* Folds of a word's terms

   A kernel of 4 elements a thread folds a run's terms of a word's type,
   float32, int32 or uint32, through the reduction's and the type's
   switches once a run, its state's value in a register of its type: the
   same combines as fold's, in the same order, so the same bits. */

static inline float c_add(float a, float b) { return kept(a, b, nx_add_f32(a, b)); }
static inline int c_add(int a, int b) { return nx_add_i32(a, b); }
static inline uint c_add(uint a, uint b) { return nx_add_u32(a, b); }
static inline float c_mul(float a, float b) { return kept(a, b, nx_mul_f32(a, b)); }
static inline int c_mul(int a, int b) { return nx_mul_i32(a, b); }
static inline uint c_mul(uint a, uint b) { return nx_mul_u32(a, b); }
static inline float c_max(float a, float b) { return nx_maximum_f32(a, b); }
static inline int c_max(int a, int b) { return nx_maximum_i32(a, b); }
static inline uint c_max(uint a, uint b) { return nx_maximum_u32(a, b); }
static inline float c_min(float a, float b) { return nx_minimum_f32(a, b); }
static inline int c_min(int a, int b) { return nx_minimum_i32(a, b); }
static inline uint c_min(uint a, uint b) { return nx_minimum_u32(a, b); }

/* Arg's comparison, as order's. */
static inline int c_order(float x, float y) {
  return int(x != x) | int(y != y) << 1 | int(nx_order_f32(x, y) > 0) << 2 |
         int(nx_order_f32(y, x) > 0) << 3;
}
static inline int c_order(int x, int y) {
  return int(x > y) << 2 | int(y > x) << 3;
}
static inline int c_order(uint x, uint y) {
  return int(x > y) << 2 | int(y > x) << 3;
}

/* A monoid's fold of plane [tp]'s terms of lanes [x]: from the state's
   value where it holds a term ([has]), else from the first term. */
#define FOLD_MONOID(T, OP)                                                     \
  EACH if (v < x.n) {                                                          \
    const T t = get<T>(c, tp, i);                                              \
    a = has ? OP(a, t) : t;                                                    \
    has = 1;                                                                   \
  }

template <class T>
static inline st fold_words(thread const loop &c, int rk, st s, lanes x,
                            uint tp, uint cp, u64 row) {
  T a = as<T>(s.v);
  uint has = s.has;
  switch (rk) {
  case NX_LOOP_SUM: FOLD_MONOID(T, c_add) break;
  case NX_LOOP_PROD: FOLD_MONOID(T, c_mul) break;
  case NX_LOOP_MAX: FOLD_MONOID(T, c_max) break;
  case NX_LOOP_MIN: FOLD_MONOID(T, c_min) break;
  case NX_LOOP_EXPS:
  case NX_LOOP_SQUARES: {
    /* Float32 alone (centered). */
    float sx = as_type<float>(uint(s.x));
    EACH if (v < x.n) {
      const float d = get<float>(c, tp, i) - get<float>(c, cp, i);
      if (rk == NX_LOOP_EXPS) a = T(c_add(float(a), ex(d)));
      else {
        a = T(c_add(float(a), d * d));
        sx = c_add(sx, d);
      }
    }
    s.x = as_type<uint>(sx);
    break;
  }
  default: {
    /* Arg: the extreme, a NaN first, and of one value the earlier
       position (arg_merge). */
    u64 at = s.x;
    EACH if (v < x.n) {
      const T t = get<T>(c, tp, i);
      const u64 p = index_of(x, v) - row;
      bool take;
      if (!has) take = true;
      else {
        const int o = c_order(a, t);
        const bool an = o & 1, bn = o & 2;
        const bool b_first = rk == NX_LOOP_ARG_MAX ? (o & 8) : (o & 4);
        const bool a_first = rk == NX_LOOP_ARG_MAX ? (o & 4) : (o & 8);
        if (an || bn) take = an && bn ? long(p) < long(at) : bn;
        else if (b_first) take = true;
        else if (a_first) take = false;
        else take = long(p) < long(at);
      }
      if (take) {
        a = t;
        at = p;
      }
      has = 1;
    }
    s.x = at;
  }
  }
  s.v = of(a);
  s.has = has;
  return s;
}

#undef FOLD_MONOID

/* A chunk's chain of a word's terms, float32, int32 or uint32: [n] terms of
   plane [tp] in order, term e at slot (e / S) TB + first + e % S and
   position [p0] + e, each result, the state after its term in the
   carrier, into plane [rp], an Arg's position into [rp] + 4. The combines
   are fold's, in its order, through the reduction's switch once a run. */
#define CHAIN_MONOID(OP)                                                       \
  for (uint e = 0; e < n; e++) {                                               \
    const uint i = (e / S) * TB + first + e % S;                               \
    const T t = get<T>(c, tp, i);                                              \
    a = has ? OP(a, t) : t;                                                    \
    has = 1;                                                                   \
    WORD(c, rp, i) = uint(to_carrier(carrier, ct, of(a)).lo);              \
  }

template <class T>
static inline st chain_words(thread const loop &c, int rk, int ct, int carrier,
                             st s, uint n, uint S, uint first, uint tp,
                             uint rp, u64 p0) {
  T a = as<T>(s.v);
  uint has = s.has;
  switch (rk) {
  case NX_LOOP_SUM: CHAIN_MONOID(c_add) break;
  case NX_LOOP_PROD: CHAIN_MONOID(c_mul) break;
  case NX_LOOP_MAX: CHAIN_MONOID(c_max) break;
  case NX_LOOP_MIN: CHAIN_MONOID(c_min) break;
  default: {
    /* Arg, as fold_words': a later term is taken only where it orders
       first, or it is a NaN and the state is not. */
    u64 at = s.x;
    for (uint e = 0; e < n; e++) {
      const uint i = (e / S) * TB + first + e % S;
      const T t = get<T>(c, tp, i);
      const u64 p = p0 + e;
      bool take;
      if (!has) take = true;
      else {
        const int o = c_order(a, t);
        const bool an = o & 1, bn = o & 2;
        const bool b_first = rk == NX_LOOP_ARG_MAX ? (o & 8) : (o & 4);
        const bool a_first = rk == NX_LOOP_ARG_MAX ? (o & 4) : (o & 8);
        if (an || bn) take = an && bn ? long(p) < long(at) : bn;
        else if (b_first) take = true;
        else if (a_first) take = false;
        else take = long(p) < long(at);
      }
      if (take) {
        a = t;
        at = p;
      }
      has = 1;
      WORD(c, rp, i) = uint(to_carrier(carrier, ct, of(a)).lo);
      put<long>(c, rp + 4, i, long(at));
    }
    s.x = at;
  }
  }
  s.v = of(a);
  s.has = has;
  return s;
}

#undef CHAIN_MONOID

static inline st empty(int rk) {
  st s;
  s.v = held_of(0, 0);
  s.x = 0;
  s.has = rk == NX_LOOP_SUM || rk == NX_LOOP_EXPS || rk == NX_LOOP_SQUARES;
  return s;
}

/* A state in device or threadgroup memory: NX_LOOP_STATE words [g]. */
template <class W> static inline st load_work(W g) {
  st s;
  s.v = held_of(u64(g[0]) | u64(g[1]) << 32, u64(g[2]) | u64(g[3]) << 32);
  s.x = u64(g[4]) | u64(g[5]) << 32;
  s.has = g[6];
  return s;
}

template <class W> static inline void store_work(W g, st s) {
  g[0] = uint(s.v.lo);
  g[1] = uint(s.v.lo >> 32);
  g[2] = uint(s.v.hi);
  g[3] = uint(s.v.hi >> 32);
  g[4] = uint(s.x);
  g[5] = uint(s.x >> 32);
  g[6] = s.has;
}

/* State [s] of lane l + [h] in lane l of the calling SIMD-group. Every
   thread of the group calls it. */
static inline st lane_down(st s, ushort h) {
  st r;
  r.v.lo = u64(simd_shuffle_down(uint(s.v.lo), h)) |
           u64(simd_shuffle_down(uint(s.v.lo >> 32), h)) << 32;
  r.v.hi = u64(simd_shuffle_down(uint(s.v.hi), h)) |
           u64(simd_shuffle_down(uint(s.v.hi >> 32), h)) << 32;
  r.x = u64(simd_shuffle_down(uint(s.x), h)) |
        u64(simd_shuffle_down(uint(s.x >> 32), h)) << 32;
  r.has = simd_shuffle_down(s.has, h);
  return r;
}

/* A value at [ct] as its carrier holds it, an integer wrapped to it; and
   a carrier's held value at [ct], a boolean as 0 or 1. */
static inline held to_carrier(int carrier, int ct, held x) {
  if (ct >= NX_LOOP_I32) x.lo = of_raw(carrier, x.lo);
  return x;
}

static inline held of_carrier(int carrier, held h) {
  if (bool_dt(carrier)) h.lo = h.lo != 0;
  return h;
}

/* The workspace's byte [field] of the header, then state [k] past it. */
static inline device uchar *work_byte(thread const loop &c, int field) {
  return buffer_at(c, int(hd(c, NX_LOOP_WORK))) + hd(c, field);
}

static inline device uint *work_at(thread const loop &c, int field, u64 k) {
  return (device uint *)work_byte(c, field) + k * NX_LOOP_STATE;
}

/* Stores [x] of the dtype [from] cast into destination [k] at its element
   for output [o]: a destination is C-contiguous over the outputs. A kernel
   of 4 elements a thread stores whole bytes of at most 32 bits, of [from]
   or a float32's bfloat16 or float16 (loop.ml), as [cast] rounds them, so
   that it holds no other cast's registers. */
static inline void put_dst(thread const loop &c, uint k, u64 o, int from,
                           held x) {
  constant u64 *d = operand(c, NX_LOOP_DSTS, k);
  const int dt = int(d[0] & 0xff);
  device uchar *b = buffer_of(c, d);
  const u64 p = d[1] + o;
  if (c.V != 4) {
    write(b, p, phase_of(d), dt, cast(from, dt, x));
    return;
  }
  const uint w = uint(x.lo);
  switch (from == dt ? bits_of(dt) : 0) {
  case 8: b[p] = uchar(w); return;
  case 16: ((device ushort *)b)[p] = ushort(w); return;
  case 32: ((device uint *)b)[p] = w; return;
  }
  const float f = as_type<float>(w);
  ((device ushort *)b)[p] = ushort(dt == NX_BFLOAT16 ? nx_float_to_bf16(f)
                                                   : nx_metal_float_to_f16(f));
}

/* Stores the position [x] of an Arg's output [o] into its destination
   [k], an int64's. */
static inline void put_position(thread const loop &c, uint k, u64 o, u64 x) {
  constant u64 *d = operand(c, NX_LOOP_DSTS, k);
  device uint *q = (device uint *)buffer_of(c, d) + 2 * (d[1] + o);
  q[0] = uint(x);
  q[1] = uint(x >> 32);
}

/* The center of an output of [n] terms from its first pass's state [s]:
   Logsumexp's maximum, 0 where it is not finite, by NX_LOOP_PUT_CENTER,
   or Moments' mean, by NX_LOOP_PUT_MEAN. */
static inline float center_of(int fin, u64 n, st s) {
  const float v = as<float>(s.v);
  return fin == NX_LOOP_PUT_CENTER ? (v - v == 0 ? v : 0.0f) : v / float(n);
}

/* What output [o] of [n] terms of center [cc] takes from its second
   pass's state [s], by [fin]: Logsumexp, or Moments. */
static inline void put_centered(thread const loop &c, int fin, int carrier,
                                u64 o, u64 n, st s, float cc) {
  const float v = as<float>(s.v);
  if (fin == NX_LOOP_PUT_LOGSUMEXP) {
    put_dst(c, 0, o, carrier, of(cc + nx_log_f32(v)));
    return;
  }
  const float tn = float(n), d = as<float>(held_of(s.x, 0));
  const float u = (v - d * (d / tn)) / tn;
  const float var = cc != cc ? cc : v == INFINITY ? v : u < 0 ? 0.0f : u;
  put_dst(c, 0, o, carrier, of(cc));
  put_dst(c, 1, o, carrier, of(var));
}

/* What output [o] of [n] terms takes from its state [s], by [fin]. */
static inline void finish(thread const loop &c, int rk, int ct, int fin,
                          int carrier, u64 o, u64 n, st s) {
  switch (fin) {
  case NX_LOOP_PUT:
    put_dst(c, 0, o, carrier, to_carrier(carrier, ct, s.v));
    if (rk == NX_LOOP_ARG_MAX || rk == NX_LOOP_ARG_MIN)
      put_position(c, 1, o, s.x);
    return;
  case NX_LOOP_PUT_TOTAL: store_work(work_at(c, NX_LOOP_TOTALS, o), s); return;
  }
  /* Logsumexp's and Moments' finishes, at float32, through the centers
     in the workspace. */
  device float *centers = (device float *)work_byte(c, NX_LOOP_CENTERS);
  switch (fin) {
  case NX_LOOP_PUT_CENTER:
  case NX_LOOP_PUT_MEAN: centers[o] = center_of(fin, n, s); return;
  default: put_centered(c, fin, carrier, o, n, s, centers[o]);
  }
}

/* The reduction word's fields. */
struct reduction {
  int rk, ct, carrier, fin;
};

static inline reduction reduction_of(thread const loop &c) {
  const u64 r = hd(c, NX_LOOP_REDUCTION);
  reduction x;
  x.rk = int(r & 0xff);
  x.ct = int((r >> 8) & 0xff);
  x.carrier = int((r >> 16) & 0xff);
  x.fin = int((r >> 24) & 0xff);
  return x;
}

/* Trees */

/* The lanes' tree over [span] lanes, a power of two, of which the calling
   thread is lane [l] with its state [s]: lane l takes lane l + h for l
   below h, from h = span / 2, and lane 0 ends with the span's state. A
   lane of no term takes no part ([has]); a sum's lanes all do. Every
   thread of the SIMD-group calls it. It and block_tree are inlined into
   every kernel that calls them, so that one kernel's code does not depend
   on how many others call them. */
__attribute__((always_inline)) static inline st
lane_tree(bool w, reduction r, st s, uint span, uint l) {
  for (ushort h = ushort(span / 2); h > 0; h /= 2) {
    const st other = lane_down(s, h);
    if (l < h) s = merge(w, r.rk, r.ct, s, other);
  }
  return s;
}

/* The pairwise tree of the group's units of [g] blocks, g a power of two,
   in [scratch]: the calling thread's slot [s] holds block [j] of its unit,
   the state [x] of the slot's lane 0 ([l]); slot s takes slot s + step
   where 2 step divides j, a block at or past the unit's [count] taking no
   part. Gives slot s's state, the unit's in its first slot. Every thread
   of the group calls it. */
__attribute__((always_inline)) static inline st
block_tree(bool w, reduction r, threadgroup uint *scratch, st x, uint s,
           u64 j, uint l, u64 g, u64 count) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (l == 0) store_work(scratch + s * NX_LOOP_STATE, x);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (u64 step = 1; step < g; step *= 2) {
    if (l == 0 && j % (2 * step) == 0 && j + step < count)
      store_work(scratch + s * NX_LOOP_STATE,
                 merge(w, r.rk, r.ct, load_work(scratch + s * NX_LOOP_STATE),
                       load_work(scratch + (s + step) * NX_LOOP_STATE)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return load_work(scratch + s * NX_LOOP_STATE);
}

/* Output [o]'s [m] states at [g], merged by the pairwise tree in place,
   the group's threads merging a level's pairs side by side, then sent
   out. Every thread of the group calls it; the states are read through
   coherent pointers, after the ticket or the launch that wrote them. */
static inline void put_parts(thread const loop &c, reduction r,
                             coherent(device) device uint *g, u64 o, u64 m,
                             u64 n) {
  for (u64 step = 1; step < m; step *= 2) {
    for (u64 k = 2 * step * c.me; k + step < m; k += 2 * step * TB)
      store_work(g + k * NX_LOOP_STATE,
                 merge(c.V == 4, r.rk, r.ct, load_work(g + k * NX_LOOP_STATE),
                       load_work(g + (k + step) * NX_LOOP_STATE)));
    threadgroup_barrier(mem_flags::mem_device);
  }
  if (c.me == 0) finish(c, r.rk, r.ct, r.fin, r.carrier, o, n, load_work(g));
}

/* Carries */

/* The state a slice's scan starts from: its carry, or none. */
static inline st carried(thread const loop &c, reduction r, u64 slice) {
  st s = empty(r.rk);
  const long k = long(hd(c, NX_LOOP_CARRY));
  if (k >= 0) {
    s.v = of_carrier(r.carrier,
                     load_at(c, uint(k), slice * hd(c, NX_LOOP_TERMS)));
    s.has = 1;
  }
  return s;
}

#endif
