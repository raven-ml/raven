/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Loops: a map, a reduction or a scan of a program, run over a tape the
   host lowers the program to (kernels.h, Loops), a kernel for each mode.
   tape.h runs the tape over a group's tiles; fold.h holds the states,
   merges, trees, chains and finishes that fix a fold's bits.

     constants --> per tile: lanes --> body
     --> fold (reduce), chain (scan) or stores (map) --> epilogue's stores

   A reduction's element is a term. A group holds 16 block slots of 16
   threads, thread l of a slot holding lane l of nx.cpu's lane order: in a
   run it takes its lane's next v terms, terms l + 16 j of the block, and
   folds them in order into its state, in registers. Then the lanes' tree,
   by SIMD shuffles, a lane of no term taking no part, then the blocks'
   binary tree in threadgroup memory, so the bits are nx.cpu's. A group's
   slots hold 16 / g units of g blocks, g a power of two, a subtree; an
   output of several groups leaves each group's state in the workspace,
   and the group that takes its ticket last merges them, or, where the
   output's groups run as several launches (NX_LOOP_APART), loop_merge
   does.

   A scan's unit is a chunk of a slice. S threads load a chunk, S v
   consecutive terms a run, coalesced where its terms step by one; then
   the first of them chains them in order from the carry into the chunk:
   the slice's carry, or, past one chunk, the state the carries kernel
   left, the given carry then each earlier chunk's total folded in order.
   The bits are nx.cpu's. */

#include "fold.h"
#include "tape.h"

using metal::min;

/* Reductions */

/* A thread's state between launches of its group's runs, by its group
   within the launch, past [parts] states from NX_LOOP_PARTIALS: a
   launch takes a batch of groups whole, the same in each of its
   pieces (loop.ml, stepped). */
static inline device uint *lane_at(thread const loop &c, u64 parts,
                                   uint gid) {
  return (device uint *)(work_byte(c, NX_LOOP_PARTIALS)) +
         (parts + u64(gid) * TB + c.me) * NX_LOOP_STATE;
}

/* Direct reductions: a Sum, Max, Min or Arg whose tape loads one plain
   operand of whole bytes and takes its element as the term, as it is, as
   a truth or decoded from bfloat16 or float16 (kernels.h, NX_LOOP_DIRECT),
   runs as loop_reduce_direct, its lanes' terms in registers with no tape
   and no planes: a lane reads DIRECT_AHEAD terms at once, then folds them
   in order as fold_words does. Its trees hold three words, not fold.h's
   states: the value at the carrier's type, an Arg's position and whether
   it holds a term, merged as merge does. So its bits are the tape's. Its
   loop is never wide: its counts and positions fit 32 bits. */
#define DIRECT_AHEAD 16

/* Lane terms [e0] to [e0] + K - 1 of plain operand [d], an array of E,
   into [y], those below [n] alone unless [whole]: term e at loop index
   [L] + NX_LOOP_LANES e, at position [p0] + e [step] where [along], else
   placed. */
template <class E, bool whole, uint K>
__attribute__((always_inline)) static inline void
direct_loads(thread const loop &c, constant u64 *d, uint e0, uint n, u64 L,
             bool along, uint p0, uint step, thread uint *y) {
  device const E *b = (device const E *)buffer_of(c, d);
  if (along) {
#pragma unroll
    for (uint j = 0; j < K; j++) {
      const uint e = e0 + j;
      y[j] = whole || e < n ? uint(b[p0 + e * step]) : 0u;
    }
    return;
  }
#pragma unroll
  for (uint j = 0; j < K; j++) {
    const uint e = e0 + j;
    y[j] = whole || e < n
               ? uint(b[uint(place(c, d, L + NX_LOOP_LANES * e))])
               : 0u;
  }
}

/* Whether an Arg of [rk] takes term [b] at [bt] over its state [a] at
   [at], each holding a term where [ah] and [bh]: arg_merge's choice. */
template <class T>
static inline bool arg_takes(int rk, T a, uint at, uint ah, T b, uint bt,
                             uint bh) {
  if (!bh) return false;
  if (!ah) return true;
  const int o = c_order(a, b);
  const bool an = o & 1, bn = o & 2;
  const bool b_first = rk == NX_LOOP_ARG_MAX ? (o & 8) : (o & 4);
  const bool a_first = rk == NX_LOOP_ARG_MAX ? (o & 4) : (o & 8);
  if (an || bn) return an && bn ? bt < at : bn;
  if (b_first) return true;
  if (a_first) return false;
  return bt < at;
}

/* A key of [t] for an extreme's fold in a lane, [up] for Max and Arg
   Max: a float's order key (nx_key, -0 below +0), a NaN's past every
   other value's, above where [up], below otherwise. A lane's positions
   rise, so its next term takes the state's place only where its key
   passes the state's: nx_maximum's and nx_minimum's choice, the first
   NaN kept, and arg_takes' for a later position. A term then costs one
   comparison of integers. */
static inline uint extreme_key(float t, bool up) {
  return t != t ? (up ? ~0u : 0u) : nx_key_f32(t);
}
static inline uint extreme_key(int t, bool up) {
  return uint(t) ^ 0x80000000u;
}
static inline uint extreme_key(uint t, bool up) { return t; }

/* Lane terms [e0] to [e0] + K - 1, those below [n] unless [whole], each
   taken by [how], folded into [a], [at] and [has] in order, as fold_words
   folds them. An Arg's position is its term's index past the row, [base]
   + 16 e. Where [plain], a float's terms are folded by plain adds and
   comparisons, and [bad] set where a term is a NaN (direct_pass). */
template <class T, class E, bool whole, bool plain, uint K>
__attribute__((always_inline)) static inline void
direct_fold(thread const loop &c, int rk, constant u64 *d, int dt, uint how,
            uint e0, uint n, u64 L, bool along, uint p0, uint step, uint base,
            thread T &a, thread uint &at, thread uint &has,
            thread bool &bad) {
  uint y[K];
  direct_loads<E, whole, K>(c, d, e0, n, L, along, p0, step, y);
  /* held_word's words, by one switch a chunk rather than one a term. */
  switch (dt) {
  case NX_INT16:
#pragma unroll
    for (uint j = 0; j < K; j++) y[j] = uint(int(short(ushort(y[j]))));
    break;
  case NX_INT8:
#pragma unroll
    for (uint j = 0; j < K; j++) y[j] = uint(int(char(uchar(y[j]))));
    break;
  }
  /* A bfloat16's bits shifted are its float's but at a NaN, which
     direct_pass refolds. */
  switch (how) {
  case NX_DIRECT_TRUTH:
#pragma unroll
    for (uint j = 0; j < K; j++) y[j] = y[j] != 0 ? 1u : 0u;
    break;
  case NX_DIRECT_BF16:
#pragma unroll
    for (uint j = 0; j < K; j++)
      y[j] = plain ? y[j] << 16
                   : as_type<uint>(nx_bf16_to_float(ushort(y[j])));
    break;
  case NX_DIRECT_F16:
#pragma unroll
    for (uint j = 0; j < K; j++)
      y[j] = as_type<uint>(nx_metal_f16_to_float(ushort(y[j])));
    break;
  }
  if (rk == NX_LOOP_SUM) {
#pragma unroll
    for (uint j = 0; j < K; j++) {
      if (!whole && e0 + j >= n) break;
      const T t = as<T>(held_of(y[j], 0));
      a = plain ? a + t : c_add(a, t);
    }
    return;
  }
  /* An extreme's, or an Arg's: a term takes the state's place where its
     key passes the state's (extreme_key), or, where [plain], where it
     compares past the state. A state of no term takes the chunk's first,
     which then meets itself and stays. */
  const bool up = rk == NX_LOOP_MAX || rk == NX_LOOP_ARG_MAX;
  if (!has) {
    a = as<T>(held_of(y[0], 0));
    at = base + NX_LOOP_LANES * e0;
    has = 1;
  }
  if (plain) {
#pragma unroll
    for (uint j = 0; j < K; j++) {
      if (!whole && e0 + j >= n) break;
      const T t = as<T>(held_of(y[j], 0));
      bad = bad || t != t;
      if (up ? t > a : t < a) {
        a = t;
        at = base + NX_LOOP_LANES * (e0 + j);
      }
    }
    return;
  }
  uint ka = extreme_key(a, up);
#pragma unroll
  for (uint j = 0; j < K; j++) {
    if (!whole && e0 + j >= n) break;
    const T t = as<T>(held_of(y[j], 0));
    const uint kt = extreme_key(t, up);
    if (up ? kt > ka : kt < ka) {
      a = t;
      ka = kt;
      at = base + NX_LOOP_LANES * (e0 + j);
    }
  }
}

/* The lane's [n] terms of load [d], from loop index [L] at its row's
   index [row], NX_LOOP_LANES apart, of E, after state [s]: whole chunks
   of K, whose loads and folds test no count, then the rest. Where
   [plain], [bad] is set where its result may differ from the tape's
   (row_terms). */
template <class T, class E, bool plain, uint K>
__attribute__((always_inline)) static inline st
direct_pass(thread const loop &c, int rk, st s, constant u64 *d, int dt,
            uint how, u64 L, uint n, u64 row, bool along, uint p0, uint step,
            thread bool &bad) {
  T a = as<T>(s.v);
  uint at = uint(s.x), has = s.has;
  const uint base = uint(L - row), whole = n / K * K;
  for (uint e0 = 0; e0 < whole; e0 += K)
    direct_fold<T, E, true, plain, K>(c, rk, d, dt, how, e0, n, L, along, p0,
                                      step, base, a, at, has, bad);
  if (whole < n)
    direct_fold<T, E, false, plain, K>(c, rk, d, dt, how, whole, n, L, along,
                                       p0, step, base, a, at, has, bad);
  if (plain) bad = bad || (rk == NX_LOOP_SUM ? a != a : a == 0);
  s.v = of(a);
  s.x = at;
  s.has = has;
  return s;
}

template <class T, class E>
static inline st direct_run(thread const loop &c, int rk, st s,
                            constant u64 *d, int dt, uint how, u64 L, uint n,
                            u64 row, bool along, uint p0, uint step) {
  bool bad = false;
  return direct_pass<T, E, false, DIRECT_AHEAD>(c, rk, s, d, dt, how, L, n,
                                                row, along, p0, step, bad);
}

/* State [s] then the lane's [n] terms of load [d], from loop index [L],
   stepping by NX_LOOP_LANES, each taken by [how]; an Arg's position is
   its index past [row]. */
template <class T>
static inline st direct_terms(thread const loop &c, int rk, st s,
                              constant u64 *d, uint how, u64 L, uint n,
                              u64 row) {
  const int dt = int(d[0] & 0xff);
  const int rank = int(hd(c, NX_LOOP_RANK));
  const uint p0 = uint(place(c, d, L));
  const bool along =
      rank > 0 &&
      coordinate(c, L, rank - 1) + NX_LOOP_LANES * u64(n > 0 ? n - 1 : 0) <
          c.t[rank - 1];
  const uint step = rank > 0 ? uint(NX_LOOP_LANES * d[2 + rank - 1]) : 0;
  switch (bits_of(dt)) {
  case 32:
    return direct_run<T, uint>(c, rk, s, d, dt, how, L, n, row, along, p0,
                               step);
  case 16:
    return direct_run<T, ushort>(c, rk, s, d, dt, how, L, n, row, along, p0,
                                 step);
  default:
    return direct_run<T, uchar>(c, rk, s, d, dt, how, L, n, row, along, p0,
                                step);
  }
}

/* How a reduction's kernel takes its terms: from the tape, or, where
   [direct], from registers (direct_terms); where [centered], those of a
   Logsumexp's or Moments' second pass (centered_terms); and, where
   [fixed], at the reduction [rk] of a float32 carrier, from a load of
   dtype [dt], an array of E whose element is taken by [how], along its
   row (row_terms). A fixed kernel's code holds only its reduction's and
   its dtype's paths, and the generic kernels hold no centered pass, so
   that their registers do not bound one another. */
struct by_tape {
  enum { direct = 0, fixed = 0, centered = 0, rk = 0, dt = 0, how = 0 };
  typedef uint E;
};
struct by_registers {
  enum { direct = 1, fixed = 0, centered = 0, rk = 0, dt = 0, how = 0 };
  typedef uint E;
};
struct by_centered {
  enum { direct = 1, fixed = 0, centered = 1, rk = 0, dt = 0, how = 0 };
  typedef uint E;
};
template <int R, int D, int H, class L> struct by_row {
  enum {
    direct = 1,
    fixed = 1,
    centered = R == NX_LOOP_EXPS || R == NX_LOOP_SQUARES,
    rk = R,
    dt = D,
    how = H
  };
  typedef L E;
};

/* State [s] then the lane's [n] terms of load [d] at the fixed kernel F's
   facts, from loop index [L] at its row's index [row]; the loop's last
   axis holds the row's terms, so a lane's terms step along it.

   A float's terms first fold by plain adds and comparisons. A plain add
   differs from c_add only where an operand is a NaN, which a sum then
   keeps, and a plain comparison from the keys' order only at a NaN, and
   between zeros, and subnormals where Metal's comparisons flush them, all
   of which compare equal to 0. So where no term is a NaN and an extreme's
   result does not compare equal to 0, or a sum's is no NaN, the plain
   fold has the tape's bits; elsewhere the lane folds its terms again
   through c_add and the keys, a term at a time, so that the refold's
   registers stay below the plain fold's. A float16's decode is software,
   whose cost a plain fold would not save: its terms fold once, through
   the keys.

   An extreme's fold reads ROW_EXTREME_AHEAD terms at once: at
   DIRECT_AHEAD, its registers let a group of its pipeline 640 threads,
   not 1024, and argmax 8192^2 takes 1.3 times as long. */
#define ROW_EXTREME_AHEAD 8

template <class F>
static inline st row_terms(thread const loop &c, st s, constant u64 *d,
                           u64 L, uint n, u64 row) {
  const int rank = int(hd(c, NX_LOOP_RANK));
  const uint p0 = uint(place(c, d, L));
  const uint step = uint(NX_LOOP_LANES * d[2 + rank - 1]);
  typedef typename F::E E;
  const uint K = F::rk == NX_LOOP_SUM ? DIRECT_AHEAD : ROW_EXTREME_AHEAD;
  bool bad = false;
  if (F::how == NX_DIRECT_F16)
    return direct_pass<float, E, false, K>(c, F::rk, s, d, F::dt, F::how, L,
                                           n, row, true, p0, step, bad);
  const st r = direct_pass<float, E, true, K>(c, F::rk, s, d, F::dt, F::how,
                                              L, n, row, true, p0, step, bad);
  if (!bad) return r;
  return direct_pass<float, E, false, 1>(c, F::rk, s, d, F::dt, F::how, L, n,
                                         row, true, p0, step, bad);
}

/* Logsumexp's and Moments' second pass in registers

   A centered term is exp (x - cc), or (x - cc)^2 with the second sum
   x - cc (fold.h, centered), of the lane's term x at float32 and its
   output's center cc. A lane folds its terms first by plain adds. A
   plain add differs from c_add only where an operand is a NaN, and a sum
   that meets a NaN ends a NaN, so where neither sum ends a NaN the plain
   fold has the tape's bits; elsewhere the lane folds its terms again
   through c_add, a term at a time. A float16's decode is software, whose
   cost a plain fold would not save: its terms fold once, through c_add.
   A lane reads CENTERED_AHEAD terms at once: an exp's registers are
   many. */
#define CENTERED_AHEAD 8

/* Lane terms [e0] to [e0] + K - 1 of load [d], those below [n] unless
   [whole], each taken by [how], their centered terms of [cc] folded into
   [a] and [sx] in order, by plain adds where [plain]. */
template <class E, bool whole, bool plain, uint K>
__attribute__((always_inline)) static inline void
centered_fold(thread const loop &c, int rk, constant u64 *d, uint how,
              uint e0, uint n, u64 L, bool along, uint p0, uint step,
              float cc, thread float &a, thread float &sx) {
  uint y[K];
  direct_loads<E, whole, K>(c, d, e0, n, L, along, p0, step, y);
  switch (how) {
  case NX_DIRECT_BF16:
#pragma unroll
    for (uint j = 0; j < K; j++)
      y[j] = plain ? y[j] << 16
                   : as_type<uint>(nx_bf16_to_float(ushort(y[j])));
    break;
  case NX_DIRECT_F16:
#pragma unroll
    for (uint j = 0; j < K; j++)
      y[j] = as_type<uint>(nx_metal_f16_to_float(ushort(y[j])));
    break;
  }
#pragma unroll
  for (uint j = 0; j < K; j++) {
    if (!whole && e0 + j >= n) break;
    const float dv = as_type<float>(y[j]) - cc;
    if (rk == NX_LOOP_EXPS) {
      const float t = ex(dv);
      a = plain ? a + t : c_add(a, t);
    } else {
      const float t = dv * dv;
      a = plain ? a + t : c_add(a, t);
      sx = plain ? sx + dv : c_add(sx, dv);
    }
  }
}

/* State [s] then the centered terms of the lane's [n] terms of load [d],
   from loop index [L], of E: whole chunks of K, then the rest. */
template <class E, bool plain, uint K>
static inline st centered_pass(thread const loop &c, int rk, st s,
                               constant u64 *d, uint how, u64 L, uint n,
                               bool along, uint p0, uint step, float cc) {
  float a = as<float>(s.v), sx = as_type<float>(uint(s.x));
  const uint whole = n / K * K;
  for (uint e0 = 0; e0 < whole; e0 += K)
    centered_fold<E, true, plain, K>(c, rk, d, how, e0, n, L, along, p0,
                                     step, cc, a, sx);
  if (whole < n)
    centered_fold<E, false, plain, K>(c, rk, d, how, whole, n, L, along, p0,
                                      step, cc, a, sx);
  s.v = of(a);
  s.x = as_type<uint>(sx);
  return s;
}

template <class E>
static inline st centered_run(thread const loop &c, int rk, st s,
                              constant u64 *d, uint how, u64 L, uint n,
                              bool along, uint p0, uint step, float cc) {
  if (how == NX_DIRECT_F16)
    return centered_pass<E, false, CENTERED_AHEAD>(c, rk, s, d, how, L, n,
                                                   along, p0, step, cc);
  const st r = centered_pass<E, true, CENTERED_AHEAD>(c, rk, s, d, how, L, n,
                                                      along, p0, step, cc);
  const float a = as<float>(r.v), sx = as_type<float>(uint(r.x));
  if (a == a && sx == sx) return r;
  return centered_pass<E, false, 1>(c, rk, s, d, how, L, n, along, p0, step,
                                    cc);
}

/* State [s] then the centered terms of [cc] of the lane's [n] terms of
   load [d], from loop index [L], stepping by NX_LOOP_LANES, each taken
   by [how]: a float32's, or a bfloat16's or float16's decoded. */
static inline st centered_terms(thread const loop &c, int rk, st s,
                                constant u64 *d, uint how, u64 L, uint n,
                                float cc) {
  const int dt = int(d[0] & 0xff);
  const int rank = int(hd(c, NX_LOOP_RANK));
  const uint p0 = uint(place(c, d, L));
  const bool along =
      rank > 0 &&
      coordinate(c, L, rank - 1) + NX_LOOP_LANES * u64(n > 0 ? n - 1 : 0) <
          c.t[rank - 1];
  const uint step = rank > 0 ? uint(NX_LOOP_LANES * d[2 + rank - 1]) : 0;
  if (bits_of(dt) == 32)
    return centered_run<uint>(c, rk, s, d, how, L, n, along, p0, step, cc);
  return centered_run<ushort>(c, rk, s, d, how, L, n, along, p0, step, cc);
}

/* State [a] then state [b] of a direct reduction [rk], as merge's: a
   Moments' second sum in the position's word, as its float32's bits. */
template <class T>
static inline void direct_merge(int rk, thread T &a, thread uint &at,
                                thread uint &ah, T b, uint bt, uint bh) {
  switch (rk) {
  case NX_LOOP_SUM:
  case NX_LOOP_EXPS: a = c_add(a, b); return;
  case NX_LOOP_SQUARES:
    a = c_add(a, b);
    at = as_type<uint>(c_add(as_type<float>(at), as_type<float>(bt)));
    return;
  case NX_LOOP_MAX:
  case NX_LOOP_MIN:
    if (!bh) return;
    a = !ah ? b : rk == NX_LOOP_MAX ? c_max(a, b) : c_min(a, b);
    ah = 1;
    return;
  }
  if (arg_takes(rk, a, at, ah, b, bt, bh)) {
    a = b;
    at = bt;
    ah = 1;
  }
}

/* lane_tree's, of a direct reduction: a Sum's lanes pass their values
   alone, and Moments' second pass its two sums. */
template <class T>
__attribute__((always_inline)) static inline st
direct_lanes(int rk, st s, uint span, uint l) {
  T a = as<T>(s.v);
  uint at = uint(s.x), ah = s.has;
  const bool sum = rk == NX_LOOP_SUM || rk == NX_LOOP_EXPS;
  const bool sums = sum || rk == NX_LOOP_SQUARES;
  for (ushort h = ushort(span / 2); h > 0; h /= 2) {
    const T b = simd_shuffle_down(a, h);
    const uint bt = sum ? 0u : simd_shuffle_down(at, h);
    const uint bh = sums ? 1u : simd_shuffle_down(ah, h);
    if (l < h) direct_merge(rk, a, at, ah, b, bt, bh);
  }
  s.v = of(a);
  s.x = at;
  s.has = ah;
  return s;
}

/* block_tree's, of a direct reduction, its slots' states 3 words each in
   [scratch]: the value's bits, the position and whether it holds a term.
   No plane lies under the scratch. */
template <class T>
__attribute__((always_inline)) static inline st
direct_blocks(int rk, threadgroup uint *scratch, st x, uint s, u64 j, uint l,
              u64 g, u64 count) {
  threadgroup uint *me = scratch + 3 * s;
  if (l == 0) {
    me[0] = uint(x.v.lo);
    me[1] = uint(x.x);
    me[2] = x.has;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (u64 step = 1; step < g; step *= 2) {
    if (l == 0 && j % (2 * step) == 0 && j + step < count) {
      T a = as<T>(held_of(me[0], 0));
      uint at = me[1], ah = me[2];
      threadgroup uint *b = me + 3 * step;
      direct_merge(rk, a, at, ah, as<T>(held_of(b[0], 0)), b[1], b[2]);
      me[0] = uint(of(a).lo);
      me[1] = at;
      me[2] = ah;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  x.v = held_of(me[0], 0);
  x.x = me[1];
  x.has = me[2];
  return x;
}

/* The trees of a direct reduction [r], at its carrier's word type. */
static inline st direct_lane_tree(reduction r, st s, uint span, uint l) {
  switch (r.ct) {
  case NX_LOOP_F32: return direct_lanes<float>(r.rk, s, span, l);
  case NX_LOOP_I32: return direct_lanes<int>(r.rk, s, span, l);
  default: return direct_lanes<uint>(r.rk, s, span, l);
  }
}

static inline st direct_block_tree(reduction r, threadgroup uint *scratch,
                                   st x, uint s, u64 j, uint l, u64 g,
                                   u64 count) {
  switch (r.ct) {
  case NX_LOOP_F32:
    return direct_blocks<float>(r.rk, scratch, x, s, j, l, g, count);
  case NX_LOOP_I32:
    return direct_blocks<int>(r.rk, scratch, x, s, j, l, g, count);
  default: return direct_blocks<uint>(r.rk, scratch, x, s, j, l, g, count);
  }
}

/* Logsumexp's and Moments' first pass in the launch of their second
   (NX_LOOP_ONCE): where each output's blocks lie in one group, the
   group folds the center's terms, Max's or Sum's, by the direct trees,
   then each output's center, as finish would put it, reaches its
   threads: through the scratch's words from ONCE_CENTERS, a slot each,
   behind a barrier, where the output has several blocks, else by
   shuffles over its span. Each lane then reads its terms again, from
   cache. The scratch's slots' states lie below ONCE_CENTERS. */
#define ONCE_CENTERS (3 * 16)
static_assert(ONCE_CENTERS + 16 <= 16 * NX_LOOP_STATE,
              "a group's centers fit its scratch");

/* The row kernel of the first pass of F's, a centered row's. */
template <class F> struct center_row {
  typedef by_row<F::rk == NX_LOOP_EXPS ? NX_LOOP_MAX : NX_LOOP_SUM, F::dt,
                 F::how, typename F::E>
      t;
};

/* The center of the calling lane's output of [len] terms, of which the
   lane holds its [n] from loop index [L], its output's row at [row]:
   the lane is lane [l] of its span [span] in slot [s], block [j] of its
   unit's [g], [count] of which hold terms. Every thread of the group
   calls it. */
template <class F>
static inline float once_center(thread const loop &c, reduction r,
                                threadgroup uint *scratch, constant u64 *d,
                                uint how, u64 L, uint n, u64 row, u64 len,
                                uint span, uint l, uint s, u64 j, u64 g,
                                u64 count) {
  const int rc = r.rk == NX_LOOP_EXPS ? NX_LOOP_MAX : NX_LOOP_SUM;
  const int fc = r.rk == NX_LOOP_EXPS ? NX_LOOP_PUT_CENTER : NX_LOOP_PUT_MEAN;
  st x = empty(rc);
  if (F::fixed) x = row_terms<typename center_row<F>::t>(c, x, d, L, n, row);
  else x = direct_terms<float>(c, rc, x, d, how, L, n, row);
  x = direct_lanes<float>(rc, x, span, l);
  if (g == 1) {
    /* From lane 0 of the span to its lanes, doubling. */
    float cc = center_of(fc, len, x);
    for (ushort h = 1; h < span; h *= 2) {
      const float up = metal::simd_shuffle_up(cc, h);
      if (l >= h) cc = up;
    }
    return cc;
  }
  x = direct_blocks<float>(rc, scratch, x, s, j, l, g, count);
  threadgroup float *centers = (threadgroup float *)(scratch + ONCE_CENTERS);
  if (l == 0 && j == 0) centers[s] = center_of(fc, len, x);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return centers[s - j];
}

/* A reduction's group: 16 slots of NX_LOOP_LANES threads, slot s block
   s % g of unit s / g, a unit an output's g blocks; this group's units
   from (NX_LOOP_FIRST + group) 16 / g. An output of at most 8 terms
   takes a span of the least power of two lanes at or past them, a slot
   16 / span such outputs, and the lanes' tree from half its span: the
   lanes past its terms would take no part. A launch takes NX_LOOP_RUNS
   runs of each block, from run NX_LOOP_FROM in pieces (NX_LOOP_PIECES):
   past the first, a thread starts from the state an earlier launch left,
   and it leaves its own where runs remain. Its terms come from the tape,
   or, where [direct], from registers. */
template <class F>
static inline void reduce_mode(thread const loop &c, threadgroup uint *scratch,
                               uint gid) {
  const bool direct = F::direct;
  reduction r = reduction_of(c);
  if (F::fixed) {
    r.rk = F::rk;
    r.ct = NX_LOOP_F32;
  }
  const u64 g = hd(c, NX_LOOP_GROUP), m = hd(c, NX_LOOP_GROUPS);
  const u64 O = hd(c, NX_LOOP_OUTPUTS), R = hd(c, NX_LOOP_TERMS);
  const u64 SEG = hd(c, NX_LOOP_SEGMENT), segs = hd(c, NX_LOOP_SEGMENTS);
  const u64 mode = hd(c, NX_LOOP_MODE);
  const bool apart = (mode & NX_LOOP_APART) != 0;
  const bool pieces = (mode & NX_LOOP_PIECES) != 0;
  const u64 runs = hd(c, NX_LOOP_RUNS);
  const u64 from = pieces ? hd(c, NX_LOOP_FROM) : 0;
  const uint span = uint(hd(c, NX_LOOP_SPAN));
  const u64 per = NX_LOOP_LANES / g, units = O * m;
  const uint s = c.me / NX_LOOP_LANES, l = c.me % span;
  const u64 u = ((hd(c, NX_LOOP_FIRST) + gid) * per + s / g) *
                    (NX_LOOP_LANES / span) +
                c.me % NX_LOOP_LANES / span,
            j = s % g;
  const u64 o = u / m, q = u % m, b = q * g + j;
  const u64 start = (o % segs) * SEG, len = min(SEG, R - min(R, start));
  const u64 row = o / segs * R, end = u < units ? row + start + len : 0;
  const uint tp = uint(hd(c, NX_LOOP_TERM)), cp = uint(hd(c, NX_LOOP_CENTER));
  const bool centered = r.rk == NX_LOOP_EXPS || r.rk == NX_LOOP_SQUARES;
  const u64 body = body_at(c), nbody = hd(c, NX_LOOP_BODY);
  const u64 R1 = u64(NX_LOOP_LANES) * c.V;
  const u64 kept = m > 1 ? O * m : 0;

  /* The group's blocks of the unit, [count] of which hold terms. */
  const u64 nb = (len + NX_LOOP_BLOCK - 1) / NX_LOOP_BLOCK;
  const u64 count = u < units ? min(g, nb - min(nb, q * g)) : 0;

  /* The lane's terms, run by run from run [from], folded in order. A
     centered pass's terms deviate from [cc], its output's center. */
  st state = empty(r.rk);
  if (pieces && from > 0) state = load_work(lane_at(c, kept, gid));
  const u64 L = row + start + b * NX_LOOP_BLOCK + from * R1 + l;
  const bool once = F::centered && (mode & NX_LOOP_ONCE) != 0;
  float cc = 0.0f;
  if (direct) {
    /* The lane's terms of these runs, NX_LOOP_LANES apart, below [end]. */
    const uint n =
        L < end ? uint(min(runs * c.V,
                           (end - L + NX_LOOP_LANES - 1) / NX_LOOP_LANES))
                : 0;
    const u64 f = hd(c, NX_LOOP_DIRECT);
    constant u64 *d =
        operand(c, NX_LOOP_LOADS, uint(f >> NX_DIRECT_OPERANDS) & 3);
    const uint how = uint(f >> (NX_DIRECT_OPERANDS + 2)) & 3;
    if (F::centered) {
      if (once)
        cc = once_center<F>(c, r, scratch, d, how, L, n, row, len, span, l, s,
                            j, g, count);
      else if (u < units)
        cc = ((device const float *)work_byte(c, NX_LOOP_CENTERS))[o];
      if (F::fixed) {
        const int rank = int(hd(c, NX_LOOP_RANK));
        state = centered_run<typename F::E>(
            c, r.rk, state, d, F::how, L, n, true, uint(place(c, d, L)),
            uint(NX_LOOP_LANES * d[2 + rank - 1]), cc);
      } else state = centered_terms(c, r.rk, state, d, how, L, n, cc);
    } else if (F::fixed) state = row_terms<F>(c, state, d, L, n, row);
    else switch (r.ct) {
    case NX_LOOP_F32:
      state = direct_terms<float>(c, r.rk, state, d, how, L, n, row);
      break;
    case NX_LOOP_I32:
      state = direct_terms<int>(c, r.rk, state, d, how, L, n, row);
      break;
    default: state = direct_terms<uint>(c, r.rk, state, d, how, L, n, row);
    }
  } else
  for (u64 k = 0; k < runs; k++) {
    const u64 L0 = L + k * R1;
    const lanes x = lanes_of(L0, NX_LOOP_LANES, end, c.V);
    run(c, x, body, nbody);
    if (c.V == 4) {
      switch (r.ct) {
      case NX_LOOP_F32:
        state = fold_words<float>(c, r.rk, state, x, tp, cp, row);
        break;
      case NX_LOOP_I32:
        state = fold_words<int>(c, r.rk, state, x, tp, cp, row);
        break;
      default: state = fold_words<uint>(c, r.rk, state, x, tp, cp, row);
      }
      continue;
    }
    EACH if (v < x.n) state = fold(false, r.rk, r.ct, state,
                                   get_held(c, tp, i, 4),
                                   index_of(x, v) - row,
                                   centered ? get_held(c, cp, i, 2)
                                            : held_of(0, 0));
  }
  /* Runs remain past this launch's: its last ends before the block's
     terms do. */
  if (pieces && (from + runs) * R1 < min(min(SEG, R), u64(NX_LOOP_BLOCK))) {
    store_work(lane_at(c, kept, gid), state);
    return;
  }

  state = F::centered ? direct_lanes<float>(r.rk, state, span, l)
          : direct    ? direct_lane_tree(r, state, span, l)
                      : lane_tree(c.V == 4, r, state, span, l);

  /* An output of one block: its state, in its first lane. */
  if (g == 1) {
    if (l != 0 || u >= units) return;
    if (once) put_centered(c, r.fin, r.carrier, o, len, state, cc);
    else finish(c, r.rk, r.ct, r.fin, r.carrier, o, len, state);
    return;
  }

  /* The group's blocks by the pairwise tree. The scratch lies over the
     planes: every thread's runs end before a state goes in. */
  const st mine =
      F::centered ? direct_blocks<float>(r.rk, scratch, state, s, j, l, g,
                                         count)
      : direct    ? direct_block_tree(r, scratch, state, s, j, l, g, count)
                  : block_tree(c.V == 4, r, scratch, state, s, j, l, g, count);

  /* A unit's state, in its first slot: out where the output has one
     group; else a part in the workspace, and the group that takes the
     output's ticket last merges its parts. Several groups an output hold
     one unit, of 16 blocks. */
  const bool first = l == 0 && j == 0 && u < units;
  if (m == 1) {
    if (!first) return;
    if (once) put_centered(c, r.fin, r.carrier, o, len, mine, cc);
    else finish(c, r.rk, r.ct, r.fin, r.carrier, o, len, mine);
    return;
  }
  /* Written through a coherent pointer, so that the fence before the
     ticket makes the part visible to the group that merges it. */
  coherent(device) device uint *parts =
      (coherent(device) device uint *)work_at(c, NX_LOOP_PARTIALS, o * m);
  if (first) store_work(parts + q * NX_LOOP_STATE, mine);
  if (apart || u >= units) return;
  device atomic_uint *tickets =
      (device atomic_uint *)buffer_at(c, int(hd(c, NX_LOOP_TICKETS)));
  const u64 o0 = hd(c, NX_LOOP_FIRST) / m;
  if (!nx_last(tickets + (o - o0), uint(m), c.me, scratch + 16 * NX_LOOP_STATE))
    return;
  put_parts(c, r, parts, o, m, len);
}

/* Outputs from NX_LOOP_FIRST, a group each: its NX_LOOP_GROUPS parts
   merged and sent out. */
static inline void merge_mode(thread const loop &c, uint gid) {
  const reduction r = reduction_of(c);
  const u64 m = hd(c, NX_LOOP_GROUPS), R = hd(c, NX_LOOP_TERMS);
  const u64 SEG = hd(c, NX_LOOP_SEGMENT), segs = hd(c, NX_LOOP_SEGMENTS);
  const u64 o = hd(c, NX_LOOP_FIRST) + gid;
  const u64 start = (o % segs) * SEG, len = min(SEG, R - min(R, start));
  put_parts(c, r,
            (coherent(device) device uint *)work_at(c, NX_LOOP_PARTIALS, o * m),
            o, m, len);
}

/* Scans */

/* Each slice's chunk totals, in place, become the states its chunks start
   from: its carry, then the totals of the chunks before folded in order.
   A thread takes a slice, from NX_LOOP_FIRST, and its NX_LOOP_RUNS chunks
   from chunk NX_LOOP_SEGMENT: it starts from the state an earlier launch
   left at the slice's running state, from NX_LOOP_PARTIALS, and leaves
   its own there where chunks remain. Totals are read CARRIES_AHEAD at a
   time, so that their reads are in flight together. */
#define CARRIES_AHEAD 4

static inline void carries_mode(thread const loop &c, uint gid) {
  const reduction r = reduction_of(c);
  const u64 slices = hd(c, NX_LOOP_OUTPUTS), chunks = hd(c, NX_LOOP_SEGMENTS);
  const u64 k0 = hd(c, NX_LOOP_SEGMENT);
  const u64 k1 = min(chunks, k0 + hd(c, NX_LOOP_RUNS));
  const u64 slice = (hd(c, NX_LOOP_FIRST) + gid) * TB + c.me;
  if (slice >= slices) return;
  device uint *running = work_at(c, NX_LOOP_PARTIALS, slice);
  st s = k0 == 0 ? carried(c, r, slice) : load_work(running);
  device uint *g = work_at(c, NX_LOOP_TOTALS, slice * chunks);
  for (u64 k = k0; k < k1; k += CARRIES_AHEAD) {
    const uint n = uint(min(u64(CARRIES_AHEAD), k1 - k));
    st t[CARRIES_AHEAD];
#pragma unroll
    for (uint j = 0; j < CARRIES_AHEAD; j++)
      if (j < n) t[j] = load_work(g + (k + j) * NX_LOOP_STATE);
#pragma unroll
    for (uint j = 0; j < CARRIES_AHEAD; j++)
      if (j < n) {
        store_work(g + (k + j) * NX_LOOP_STATE, s);
        s = merge(c.V == 4, r.rk, r.ct, s, t[j]);
      }
  }
  if (k1 < chunks) store_work(running, s);
}

/* Chunks: S = NX_LOOP_GROUP threads take a chunk, NX_LOOP_THREADS / S
   chunks a group, from (NX_LOOP_FIRST + group) NX_LOOP_THREADS / S. A run
   loads S v consecutive terms of each chunk, thread l of a chunk's S
   holding terms l + S j; the chunk's first thread then folds them in
   order from its state, each result, the state after it, into the result
   planes, its position after where Arg's, in the terms' word type where
   they compute in one (chain_words); the epilogue stores them. A launch
   takes NX_LOOP_RUNS runs of each chunk from its term
   NX_LOOP_SEGMENT: past the chunk's first term it starts from the state
   an earlier launch left in the chunk's total, and it leaves its own
   there where terms remain. */
static inline void scan_mode(thread const loop &c, uint gid) {
  const reduction r = reduction_of(c);
  const u64 units = hd(c, NX_LOOP_OUTPUTS), R = hd(c, NX_LOOP_TERMS);
  const u64 chunks = hd(c, NX_LOOP_SEGMENTS), CH = NX_LOOP_CHUNK;
  const u64 runs = hd(c, NX_LOOP_RUNS), T0 = hd(c, NX_LOOP_SEGMENT);
  const uint S = uint(hd(c, NX_LOOP_GROUP)), per = TB / S;
  const uint tp = uint(hd(c, NX_LOOP_TERM)), rp = uint(hd(c, NX_LOOP_RESULT));
  const u64 body = body_at(c), nbody = hd(c, NX_LOOP_BODY);
  const u64 nepi = hd(c, NX_LOOP_EPILOGUE);
  const uint l = c.me % S, first_item = c.me - l;
  const u64 SV = u64(S) * c.V;
  const u64 u = (hd(c, NX_LOOP_FIRST) + gid) * per + c.me / S;
  const u64 slice = u / chunks, k = u % chunks;
  const u64 start = k * CH, len = u < units ? min(CH, R - start) : 0;
  const u64 first = slice * R + start;
  st s = empty(r.rk);
  if (l == 0 && T0 < len)
    s = chunks > 1 || T0 > 0 ? load_work(work_at(c, NX_LOOP_TOTALS, u))
                             : carried(c, r, slice);
  for (u64 j = 0; j < runs; j++) {
    const u64 at = T0 + j * SV;
    const lanes x = lanes_of(first + at + l, S, first + len, c.V);
    run(c, x, body, nbody);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool word = r.ct == NX_LOOP_F32 || r.ct == NX_LOOP_I32 ||
                      r.ct == NX_LOOP_U32;
    if (l == 0 && word) {
      const uint n = uint(min(SV, len - min(len, at)));
      switch (r.ct) {
      case NX_LOOP_F32:
        s = chain_words<float>(c, r.rk, r.ct, r.carrier, s, n, S, first_item,
                                tp, rp, start + at);
        break;
      case NX_LOOP_I32:
        s = chain_words<int>(c, r.rk, r.ct, r.carrier, s, n, S, first_item,
                              tp, rp, start + at);
        break;
      default:
        s = chain_words<uint>(c, r.rk, r.ct, r.carrier, s, n, S, first_item,
                               tp, rp, start + at);
      }
    } else if (l == 0) {
      const uint n = uint(min(SV, len - min(len, at)));
      for (uint e = 0; e < n; e++) {
        const uint i = (e / S) * TB + first_item + e % S;
        s = fold(c.V == 4, r.rk, r.ct, s, get_held(c, tp, i, 4), start + at + e,
                 held_of(0, 0));
        put_held(c, rp, i, 4, to_carrier(r.carrier, r.ct, s.v));
        put<long>(c, rp + 4, i, long(s.x));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    run(c, x, body + nbody, nepi);
  }
  if (l == 0 && T0 + runs * SV < len)
    store_work(work_at(c, NX_LOOP_TOTALS, u), s);
}

/* Networks: Logsumexp's and Moments' scans by nx_kernel.mli's odd/even
   network. Its result at term g is R(g) = R(g - 2^k) + B(g, k), where 2^k
   is the lowest set bit of g + 1, B(g, k) the pairwise tree of the 2^k
   terms ending at g and + the join; no other association gives its bits.
   A Logsumexp state is a maximum and a sum; a Moments state a mean, a sum
   of squared deviations and a count.

   A slice is cut into aligned blocks of NX_LOOP_SEGMENT terms,
   NX_LOOP_SEGMENTS of them, each NX_LOOP_RUNS tiles of T = S v terms, S =
   NX_LOOP_GROUP. A group takes one block of each of W = NX_LOOP_THREADS
   / S slices side by side: group (NX_LOOP_FIRST + gid) takes block
   (NX_LOOP_FIRST + gid) % blocks of slices from ((NX_LOOP_FIRST + gid) /
   blocks) W. Thread me takes slice me % W and, of each tile, terms me / W
   + S j, so that neighbouring threads load neighbouring slices. A tile:

     terms   the tape's, into the term plane; their states into the
             result planes 1 to 3 (planes 0 and 4 take the results);
     up      level l joins slot e - 2^l into slot e where 2^(l+1)
             divides e + 1: slot e holds B(e, k), slot T - 1 the tile's
             block;
     end     the tile's block streamed through a network over the block's
             tile blocks, whose result at tile t is R(tT + T - 1): slot
             T - 1 takes it. A reach to the block's start takes the
             block's prefix P = R'(b - 1), and the end of a block's last
             tile, which reaches past it, the chain's R'(b);
     down    level k, from log2 T - 1 down: slot e with lowest bit 2^k
             of e + 1 takes R(e - 2^k) + B(e, k), R(tT - 1) the last
             tile's end before slot 0;
     results then the epilogue's stores.

   Under NX_LOOP_UP, a group streams its tiles' blocks into its block's
   total alone and leaves it in the workspace from byte NX_LOOP_TOTALS, a
   slice's blocks in order; loop_carries then turns each slice's totals
   into the R' at each block's end (net_chain), which the down groups
   read. Moments' center, the slice's term 0 made finite or 0, comes
   first from a map of it into the workspace from byte NX_LOOP_CENTERS,
   so that a group runs the tape once a tile.

   The block's tile network, two states a level, and the last two tiles'
   ends lie in threadgroup memory past the planes, from plane
   NX_LOOP_CENTER. */

struct net {
  float m, s, n;
};

static inline bool fin_(float x) { return x - x == 0; }

static inline float lse_sum(float ma, float sa, float mb, float sb, float m) {
  const bool hi = ma == m;
  const float lo = hi ? mb : ma, sl = hi ? sb : sa, sh = hi ? sa : sb;
  const float s = sh + sl * ex(lo - m);
  return fin_(m) ? s : m == -INFINITY ? 0.0f : m;
}

static inline net join(int rk, net a, net b) {
  net r;
  if (rk == NX_LOOP_LOGSUMEXP) {
    r.m = nx_maximum_f32(a.m, b.m);
    r.s = lse_sum(a.m, a.s, b.m, b.s, r.m);
    r.n = 0;
    return r;
  }
  const float k = a.n + b.n, f = b.n / k, g = a.n * f, d = b.m - a.m;
  const float t = a.m + d * f;
  r.m = fin_(a.m) && fin_(b.m) ? t : a.m + b.m;
  r.s = (a.s + b.s) + (d * d) * g;
  r.n = k;
  return r;
}

static inline net net_at(thread const loop &c, uint p, uint i) {
  net r;
  r.m = get<float>(c, p, i);
  r.s = get<float>(c, p + 1, i);
  r.n = get<float>(c, p + 2, i);
  return r;
}

static inline void net_put(thread const loop &c, uint p, uint i, net x) {
  put<float>(c, p, i, x.m);
  put<float>(c, p + 1, i, x.s);
  put<float>(c, p + 2, i, x.n);
}

/* A slice's words of threadgroup memory: the tile network's levels and
   two tiles' ends. */
static inline uint net_words(uint levels) { return 3 * (2 * levels + 2); }

static inline net net_get(threadgroup const float *a, uint k) {
  net r;
  r.m = a[3 * k];
  r.s = a[3 * k + 1];
  r.n = a[3 * k + 2];
  return r;
}

static inline void net_set(threadgroup float *a, uint k, net x) {
  a[3 * k] = x.m;
  a[3 * k + 1] = x.s;
  a[3 * k + 2] = x.n;
}

/* The levels of a network over [n] tile or block ends: one per bit of
   the greatest end's number, and one above. */
static inline uint net_levels(u64 n) {
  uint b = 0;
  for (; n > 0; n >>= 1) b++;
  return b + 1;
}

/* Streams the end [in] of the block's tile [j] through the levels at
   [lev]: [k] the set low bits of [j] it joins. Gives B(j, k). */
static inline net net_up(int rk, threadgroup float *lev, u64 j, thread uint &k,
                         net in) {
  k = 0;
  for (u64 q = j; q & 1; k++, q >>= 1) in = join(rk, net_get(lev, 2 * k), in);
  net_set(lev, 2 * k, in);
  return in;
}

static inline void network_mode(thread const loop &c, threadgroup float *sc,
                                uint gid) {
  const u64 units = hd(c, NX_LOOP_OUTPUTS), R = hd(c, NX_LOOP_TERMS);
  const u64 BL = hd(c, NX_LOOP_SEGMENT), blocks = hd(c, NX_LOOP_SEGMENTS);
  const u64 tiles = hd(c, NX_LOOP_RUNS);
  const uint L = net_levels(tiles);
  const int rk = int(hd(c, NX_LOOP_REDUCTION) & 0xff);
  const bool lse = rk == NX_LOOP_LOGSUMEXP;
  const bool up = (hd(c, NX_LOOP_MODE) & NX_LOOP_UP) != 0;
  const uint tp = uint(hd(c, NX_LOOP_TERM)), rp = uint(hd(c, NX_LOOP_RESULT));
  const uint sp = rp + 1;
  const u64 body = body_at(c), nbody = hd(c, NX_LOOP_BODY);
  const u64 nepi = hd(c, NX_LOOP_EPILOGUE);
  const uint S = uint(hd(c, NX_LOOP_GROUP)), W = TB / S, T = S * c.V;
  const uint w = c.me % W, l = c.me / W;
  const u64 group = hd(c, NX_LOOP_FIRST) + gid, b = group % blocks;
  const u64 slice = group / blocks * W + w;
  const bool live = slice < units;
  const u64 base = slice * R, end = live ? base + R : 0;
  const u64 T0 = b * BL, t0 = T0 / T;
  threadgroup float *mine = sc + w * net_words(L);
  device float *centers = (device float *)work_byte(c, NX_LOOP_CENTERS);
  device net *ends = (device net *)work_byte(c, NX_LOOP_TOTALS) + slice * blocks;
  /* The slot of a slice's tile term e. */
#define SLOT(e) (((e) / S) * TB + ((e) % S) * W + w)

  /* The block's prefix, as the last tile's end before its first. */
  net P = {0, 0, 0};
  if (!up && b > 0 && live && l == 0) {
    P = ends[b - 1];
    net_set(mine, 2 * L + ((t0 - 1) & 1), P);
  }

  const float c0 = !lse && live ? centers[slice] : 0.0f;
  const float cc = fin_(c0) ? c0 : 0.0f;

  for (u64 j = 0; j < tiles; j++) {
    const u64 at0 = T0 + j * T, t = t0 + j;
    const uint nv = live && at0 < R ? uint(min(u64(T), R - at0)) : 0;
    const lanes x = lanes_of(base + at0 + l, S, end, c.V);
    run(c, x, body, nbody);
    EACH {
      if (v >= x.n) continue;
      const float y = get<float>(c, tp, i);
      net in;
      if (lse) {
        in.m = y;
        in.s = fin_(y) ? 1.0f : y == -INFINITY ? 0.0f : y;
        in.n = 0;
      } else {
        in.m = y - cc;
        in.s = 0;
        in.n = 1;
      }
      net_put(c, sp, i, in);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < T; h <<= 1) {
      for (uint m = l; m < T / (2 * h); m += S) {
        const uint e = 2 * h * m + 2 * h - 1;
        if (e < nv)
          net_put(c, sp, SLOT(e),
                  join(rk, net_at(c, sp, SLOT(e - h)), net_at(c, sp, SLOT(e))));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* The block's total, from its last tile: a partial block's is never
       read. */
    if (up) {
      if (l == 0 && nv == T) {
        uint k;
        const net in = net_up(rk, mine, j, k, net_at(c, sp, SLOT(T - 1)));
        if (j + 1 == tiles) ends[b] = in;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      continue;
    }

    /* The tile's end: R(t T + T - 1) = R(t T + T - 1 - 2^k T) + B, with
       the R before the block P and the block's last end the chain's; ends
       alternate, so that the next tile's end does not overwrite the one
       this tile's down levels read. */
    if (l == 0 && nv == T) {
      uint k;
      const net in = net_up(rk, mine, j, k, net_at(c, sp, SLOT(T - 1)));
      const bool first = (j >> k) == 0;
      net e = first ? (b > 0 ? join(rk, P, in) : in)
                    : join(rk, net_get(mine, 2 * (k + 1) + 1), in);
      if (first && b > 0 && j + 1 == tiles) e = ends[b];
      net_set(mine, 2 * k + 1, e);
      for (uint g = 0; g < k; g++) net_set(mine, 2 * g + 1, e);
      net_set(mine, 2 * L + (t & 1), e);
      net_put(c, sp, SLOT(T - 1), e);
    }
    for (uint h = T / 2; h >= 1; h >>= 1) {
      for (uint m = l; m < T / (2 * h); m += S) {
        const uint e = h * (2 * m + 1) - 1;
        if (e >= nv) continue;
        const net own = net_at(c, sp, SLOT(e));
        net r = own;
        if (e + 1 != h) r = join(rk, net_at(c, sp, SLOT(e - h)), own);
        else if (t > 0)
          r = join(rk, net_get(mine, 2 * L + ((t - 1) & 1)), own);
        net_put(c, sp, SLOT(e), r);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    EACH {
      if (v >= x.n) continue;
      const u64 at = at0 + l + u64(S) * v;
      const net e = net_at(c, sp, i);
      float r0, r1 = 0;
      if (lse) r0 = fin_(e.m) ? e.m + nx_log_f32(e.s) : e.m;
      else {
        r0 = cc + e.m;
        r1 = fin_(e.m) ? e.s / (float(at) + 1) : e.m - e.m;
      }
      put<float>(c, rp, i, r0);
      put<float>(c, rp + 4, i, r1);
    }
    run(c, x, body + nbody, nepi);
  }
#undef SLOT
}

/* Each slice's block totals, in place, become the network's results at
   their blocks' ends, R'(b) = R(b BL + BL - 1): a thread a slice, from
   NX_LOOP_FIRST, its NX_LOOP_RUNS blocks from block NX_LOOP_SEGMENT,
   streamed as network_mode streams a block's tile ends. The stream's
   levels lie in the workspace from byte NX_LOOP_PARTIALS, two states a
   level a slice, so that a launch takes up where the last left them.
   Totals are read CARRIES_AHEAD at a time. */
static inline void net_chain(thread const loop &c, uint gid) {
  const int rk = int(hd(c, NX_LOOP_REDUCTION) & 0xff);
  const u64 slices = hd(c, NX_LOOP_OUTPUTS), blocks = hd(c, NX_LOOP_SEGMENTS);
  const u64 k0 = hd(c, NX_LOOP_SEGMENT);
  const u64 k1 = min(blocks, k0 + hd(c, NX_LOOP_RUNS));
  const u64 slice = (hd(c, NX_LOOP_FIRST) + gid) * TB + c.me;
  if (slice >= slices) return;
  const uint L = net_levels(blocks);
  device net *g = (device net *)work_byte(c, NX_LOOP_TOTALS) + slice * blocks;
  device net *lev = (device net *)work_byte(c, NX_LOOP_PARTIALS) + slice * 2 * L;
  for (u64 k = k0; k < k1; k += CARRIES_AHEAD) {
    const uint n = uint(min(u64(CARRIES_AHEAD), k1 - k));
    net t[CARRIES_AHEAD];
#pragma unroll
    for (uint a = 0; a < CARRIES_AHEAD; a++)
      if (a < n) t[a] = g[k + a];
#pragma unroll
    for (uint a = 0; a < CARRIES_AHEAD; a++) {
      if (a >= n) continue;
      net in = t[a];
      uint d = 0;
      u64 q = k + a;
      for (; q & 1; d++, q >>= 1) in = join(rk, lev[2 * d], in);
      const net e = q == 0 ? in : join(rk, lev[2 * (d + 1) + 1], in);
      lev[2 * d] = in;
      lev[2 * d + 1] = e;
      for (uint u = 0; u < d; u++) lev[2 * u + 1] = e;
      g[k + a] = e;
    }
  }
}

/* Maps: tiles of consecutive indices, from NX_LOOP_FIRST, a thread's
   interleaved across the group or, at 4 a thread, its quad; or in
   loop_map_tiled, tiles of the walk's last two axes (tape.h, Tiles). */
static inline void map_mode(thread const loop &c, uint gid) {
  const u64 n = hd(c, NX_LOOP_NUMEL);
  const u64 base = (hd(c, NX_LOOP_FIRST) + gid) * c.E;
  const u64 mode = hd(c, NX_LOOP_MODE);
  if (c.tiled) {
    const tile t = tile_of(c, gid);
    if (!t.live) return;
    stage(c, t);
    lanes x = tile_lanes(c, t);
    x.quad = (mode & NX_LOOP_QUADS) != 0;
    run(c, x, body_at(c), hd(c, NX_LOOP_BODY));
    return;
  }
  if (c.V == 4 && (mode & NX_LOOP_QUADS) != 0) {
    lanes x = lanes_of(base + 4 * c.me, 1, n, 4);
    x.quad = true;
    if (c.pads && (mode & NX_LOOP_COORDS) != 0) quad_coords(c, x.L0);
    run(c, x, body_at(c), hd(c, NX_LOOP_BODY));
    return;
  }
  run(c, lanes_of(base + c.me, TB, n, c.V), body_at(c), hd(c, NX_LOOP_BODY));
}

/* The kernels: a mode each, so that one mode's registers do not bound
   another's, at 4 and at 1 element a thread, and at 1 of 64-bit indices
   (loop_of). Each writes its constants first. A reduction's scratch, its
   slots' 16 states then the ticket's flag, takes its planes' memory once
   its runs are done. loop_merge and loop_carries hold no planes: one
   kernel each, of the width the header names.

   A kernel states no bound on its groups' threads, so that the limit
   Metal gives its pipeline is the one its registers set, which the host
   reads as the threads a core holds of it. Its groups have
   NX_LOOP_THREADS threads. */

#define LOOP_KERNEL(name)                                                      \
  [[kernel]]                                                                   \
  void name(constant loop_words &w [[buffer(0)]],                              \
            threadgroup uint *pl [[threadgroup(0)]],                           \
            uint gid [[threadgroup_position_in_grid]],                         \
            uint me [[thread_index_in_threadgroup]])

#define LOOP_KERNELS(name, mode)                                               \
  LOOP_KERNEL(name##_v4) { mode(loop_of(w, pl, me, 4, false), pl, gid); }      \
  LOOP_KERNEL(name##_v1) { mode(loop_of(w, pl, me, 1, false), pl, gid); }      \
  LOOP_KERNEL(name##_wide) { mode(loop_of(w, pl, me, 1, true), pl, gid); }

static inline bool wide_of(constant loop_words &w) {
  return (w.header[0] & NX_LOOP_WIDE) != 0;
}

static inline void mapped(thread const loop &c, threadgroup uint *, uint gid) {
  constants(c);
  map_mode(c, gid);
}

static inline void reduced(thread const loop &c, threadgroup uint *pl,
                           uint gid) {
  constants(c);
  reduce_mode<by_tape>(c, pl, gid);
}

static inline void scanned(thread const loop &c, threadgroup uint *, uint gid) {
  constants(c);
  scan_mode(c, gid);
}

static inline void networked(thread const loop &c, threadgroup uint *pl,
                             uint gid) {
  constants(c);
  network_mode(
      c, (threadgroup float *)(pl + c.E * uint(hd(c, NX_LOOP_CENTER))), gid);
}

LOOP_KERNELS(loop_map, mapped)

/* A map's kernels of tiles and of padded loads, at 4 elements a thread:
   their paths cost no other map. */
LOOP_KERNEL(loop_map_tiled) {
  loop c = loop_of(w, pl, me, 4, false);
  c.tiled = true;
  mapped(c, pl, gid);
}

LOOP_KERNEL(loop_map_padded) {
  loop c = loop_of(w, pl, me, 4, false);
  c.pads = true;
  mapped(c, pl, gid);
}

LOOP_KERNELS(loop_reduce, reduced)
LOOP_KERNELS(loop_scan, scanned)
LOOP_KERNELS(loop_network, networked)

/* A direct reduction holds no planes and no constants: its scratch alone.
   A Logsumexp's or Moments' second pass, with its first where it runs in
   one launch, takes a kernel of its own. */
LOOP_KERNEL(loop_reduce_direct) {
  reduce_mode<by_registers>(loop_of(w, pl, me, 4, false), pl, gid);
}

LOOP_KERNEL(loop_reduce_centered) {
  reduce_mode<by_centered>(loop_of(w, pl, me, 4, false), pl, gid);
}

/* A row's kernels (kernels.h, NX_METAL_ROW_KERNELS). */
#define ROW_KERNEL(X, name, rk, dt, how, E)                                    \
  LOOP_KERNEL(name) {                                                          \
    reduce_mode<by_row<rk, dt, how, E>>(loop_of(w, pl, me, 4, false), pl,      \
                                        gid);                                  \
  }
NX_METAL_ROW_KERNELS(ROW_KERNEL, 0)

LOOP_KERNEL(loop_merge) {
  merge_mode(loop_of(w, pl, me, 1, wide_of(w)), gid);
}

LOOP_KERNEL(loop_carries) {
  const loop c = loop_of(w, pl, me, 1, wide_of(w));
  const int rk = int(hd(c, NX_LOOP_REDUCTION) & 0xff);
  if (rk == NX_LOOP_LOGSUMEXP || rk == NX_LOOP_MOMENTS) net_chain(c, gid);
  else carries_mode(c, gid);
}
