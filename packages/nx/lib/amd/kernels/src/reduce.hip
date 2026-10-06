/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* reduce and arg_reduce: each element of the result folds a row, the input's
   elements over the reduced axes in C order, as nx.cpu folds it
   (cpu/nx_c_fold.c). REDUCE(sum, float32) is the module of sum at float32;
   ARG_REDUCE(max, float32) is argmax's, which writes the position of the
   row's extreme instead of its value.

   Sums start from +0, so that a sum of nothing or of -0 terms is +0, and
   floats narrower than binary32 accumulate in binary32 and round once.
   Integers sum and multiply in 64 bits, so they wrap. Booleans have max and
   min, the or and and of their 0 and 1.

   An extreme carries its element's position: of two equal elements, or of
   two NaNs, the earlier wins, so the fold is nx.cpu's whatever its
   association: the first NaN, else the first element with the extreme's
   bits, +0 above -0.

   A row is folded by [lanes] threads, a power of two up to THREADS, each
   walking the row from its lane by [lanes], then by a tree in the
   workgroup's memory. A workgroup holds THREADS / lanes rows at a time and
   walks the rows by the grid's. With [parts] above 1, a row's elements are
   split into [parts] runs of [chunk] and each run's fold goes to the
   partials [pv] (the accumulators) and [pi] (the positions), which [f]
   then folds into the result, rows of [parts] partials.

   - [c (dst, src, pv, pi, rows, len, lanes, parts, chunk, groups)]: row [r]
     at [src + r * len], its elements adjacent.
   - [s (dst, src, pv, pi, rmeta)]: row [r]'s element [j] at [off] plus the
     strided place of [r] in [oext] and of [j] in [rext].
   - [f (dst, pv, pi, rows, parts, lanes, groups)]. */

#include "common.h"

struct rmeta {
  i64 rows, len, lanes, parts, chunk, groups;
  i64 orank, rrank, off;
  i64 oext[MAX_RANK], ostr[MAX_RANK];
  i64 rext[MAX_RANK], rstr[MAX_RANK];
};

/* Where the element of C-order index [i] lies in extents [ext]. The outer
   axis takes what is left of [i] without a division. */
DEVICE i64 place(i64 i, i64 rank, const i64 *ext, const i64 *str) {
  if (rank == 0) return 0;
  i64 o = 0;
  for (i64 d = rank - 1; d > 0; d--) {
    i64 e = ext[d];
    o += (i % e) * str[d];
    i /= e;
  }
  return o + i * str[0];
}

DEVICE void barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

/* A fold's running value: its accumulator and, for an extreme, the position
   of the element it holds. */
template <class A> struct run {
  A v;
  i64 i;
};

/* The extremes of a compute type, which seed max and min: any element of a
   row displaces them, by value or by position. */
template <class A> DEVICE A lowest();
template <class A> DEVICE A highest();
template <> DEVICE float lowest<float>() { return -__builtin_inff(); }
template <> DEVICE float highest<float>() { return __builtin_inff(); }
template <> DEVICE double lowest<double>() { return -__builtin_inf(); }
template <> DEVICE double highest<double>() { return __builtin_inf(); }
template <> DEVICE int64_t lowest<int64_t>() { return INT64_MIN; }
template <> DEVICE int64_t highest<int64_t>() { return INT64_MAX; }
template <> DEVICE uint64_t lowest<uint64_t>() { return 0; }
template <> DEVICE uint64_t highest<uint64_t>() { return UINT64_MAX; }
template <> DEVICE uint8_t lowest<uint8_t>() { return 0; }
template <> DEVICE uint8_t highest<uint8_t>() { return UINT8_MAX; }

/* Does [v] order above [b], as IEEE 754 maximum takes them: NaN above every
   number, +0 above -0. Strict: an equal element or a second NaN does not. */
template <class T> DEVICE bool above(T v, T b) { return v > b; }
template <class T> DEVICE bool below(T v, T b) { return v < b; }

template <class T> DEVICE bool fabove(T v, T b) {
  return v > b ||
         (!(v < b) && ((v != v && b == b) ||
                       (v == b && __builtin_signbit(b) && !__builtin_signbit(v))));
}

template <class T> DEVICE bool fbelow(T v, T b) {
  return v < b ||
         (!(v > b) && ((v != v && b == b) ||
                       (v == b && __builtin_signbit(v) && !__builtin_signbit(b))));
}

template <> DEVICE bool above(float v, float b) { return fabove(v, b); }
template <> DEVICE bool above(double v, double b) { return fabove(v, b); }
template <> DEVICE bool below(float v, float b) { return fbelow(v, b); }
template <> DEVICE bool below(double v, double b) { return fbelow(v, b); }

DEVICE float add(float a, float b) { return a + b; }
DEVICE double add(double a, double b) { return a + b; }
DEVICE int64_t add(int64_t a, int64_t b) {
  return (int64_t)((uint64_t)a + (uint64_t)b);
}
DEVICE uint64_t add(uint64_t a, uint64_t b) { return a + b; }
DEVICE float mul(float a, float b) { return a * b; }
DEVICE double mul(double a, double b) { return a * b; }
DEVICE int64_t mul(int64_t a, int64_t b) {
  return (int64_t)((uint64_t)a * (uint64_t)b);
}
DEVICE uint64_t mul(uint64_t a, uint64_t b) { return a * b; }

/* A kind: its accumulator, whether it carries positions, its seed, and
   [join a b], which folds [b] into [a]. */
namespace kind {

template <class D> struct sum {
  typedef typename D::compute acc;
  static const bool indexed = false;
  static DEVICE acc seed() { return 0; }
  static DEVICE void join(run<acc> &a, run<acc> b) { a.v = add(a.v, b.v); }
};

template <class D> struct prod {
  typedef typename D::compute acc;
  static const bool indexed = false;
  static DEVICE acc seed() { return 1; }
  static DEVICE void join(run<acc> &a, run<acc> b) { a.v = mul(a.v, b.v); }
};

template <class D> struct max {
  typedef typename D::compute acc;
  static const bool indexed = true;
  static DEVICE acc seed() { return lowest<acc>(); }
  static DEVICE void join(run<acc> &a, run<acc> b) {
    if (above(b.v, a.v) || (!above(a.v, b.v) && b.i < a.i)) a = b;
  }
};

template <class D> struct min {
  typedef typename D::compute acc;
  static const bool indexed = true;
  static DEVICE acc seed() { return highest<acc>(); }
  static DEVICE void join(run<acc> &a, run<acc> b) {
    if (below(b.v, a.v) || (!below(a.v, b.v) && b.i < a.i)) a = b;
  }
};

} // namespace kind

/* What a row's fold writes: its value in the dtype, or its position. */
template <class D> struct value {
  typedef typename D::storage out;
  template <class A> static DEVICE out of(run<A> a) { return store<D>(a.v); }
};

struct position {
  typedef i64 out;
  template <class A> static DEVICE out of(run<A> a) { return a.i; }
};

/* Where a fold reads: the input's elements, in the row templates below, or
   the partials of a first pass. */
template <class K, class D, class R> struct elements {
  const typename D::storage *src;
  R rows;
  DEVICE i64 row(i64 r) const { return rows.row(r); }
  DEVICE run<typename K::acc> at(i64 base, i64 j) const {
    return {D::load(src[base + rows.elem(j)]), j};
  }
};

template <class K> struct partials {
  const typename K::acc *pv;
  const i64 *pi;
  i64 parts;
  DEVICE i64 row(i64 r) const { return r * parts; }
  DEVICE run<typename K::acc> at(i64 base, i64 j) const {
    return {pv[base + j], K::indexed ? pi[base + j] : 0};
  }
};

struct adjacent {
  i64 len;
  DEVICE i64 row(i64 r) const { return r * len; }
  DEVICE i64 elem(i64 j) const { return j; }
};

struct placed {
  const rmeta *m;
  DEVICE i64 row(i64 r) const {
    return m->off + place(r, m->orank, m->oext, m->ostr);
  }
  DEVICE i64 elem(i64 j) const { return place(j, m->rrank, m->rext, m->rstr); }
};

/* [a] of each of the [lanes] threads from lane 0 folded, in lane 0. */
template <class K>
DEVICE run<typename K::acc> tree(run<typename K::acc> a, int t, int lane,
                                 int lanes) {
  __attribute__((shared)) static typename K::acc vs[THREADS];
  __attribute__((shared)) static i64 is[THREADS];
  vs[t] = a.v;
  if (K::indexed) is[t] = a.i;
  barrier();
  for (int s = lanes >> 1; s > 0; s >>= 1) {
    if (lane < s) {
      run<typename K::acc> x = {vs[t], K::indexed ? is[t] : 0};
      K::join(x, {vs[t + s], K::indexed ? is[t + s] : 0});
      vs[t] = x.v;
      if (K::indexed) is[t] = x.i;
    }
    barrier();
  }
  return {vs[t], K::indexed ? is[t] : 0};
}

/* Each row's fold, or with [parts] above 1 each run's, from source [S]. The
   bounds of the walk are the workgroup's, so that all its threads meet at
   every barrier. */
template <class K, class O, class S>
DEVICE void fold(typename O::out *dst, typename K::acc *pv, i64 *pi, S src,
                 i64 rows, i64 len, i64 lanes, i64 parts, i64 chunk,
                 i64 groups) {
  int t = __builtin_amdgcn_workitem_id_x();
  int shift = __builtin_ctzll(lanes);
  int lane = t & (lanes - 1);
  i64 per = THREADS >> shift;
  i64 items = rows * parts;
  for (i64 b = (i64)__builtin_amdgcn_workgroup_id_x() * per; b < items;
       b += groups * per) {
    i64 it = b + (t >> shift);
    run<typename K::acc> a = {K::seed(), INT64_MAX};
    if (it < items) {
      i64 r = parts == 1 ? it : it / parts;
      i64 lo = (it - r * parts) * chunk;
      i64 hi = lo + chunk < len ? lo + chunk : len;
      i64 base = src.row(r);
      for (i64 j = lo + lane; j < hi; j += lanes) K::join(a, src.at(base, j));
    }
    if (lanes > 1) a = tree<K>(a, t, lane, (int)lanes);
    if (it >= items || lane != 0) continue;
    if (parts == 1) {
      dst[it] = O::of(a);
    } else {
      pv[it] = a.v;
      if (K::indexed) pi[it] = a.i;
    }
  }
}

#define REDUCE_MODULE(K, D, O)                                                 \
  KERNEL void c(O::out *dst, const D::storage *src, K::acc *pv, i64 *pi,       \
                i64 rows, i64 len, i64 lanes, i64 parts, i64 chunk,            \
                i64 groups) {                                                  \
    fold<K, O>(dst, pv, pi, elements<K, D, adjacent>{src, {len}}, rows, len,   \
               lanes, parts, chunk, groups);                                   \
  }                                                                            \
  KERNEL void s(O::out *dst, const D::storage *src, K::acc *pv, i64 *pi,       \
                rmeta m) {                                                     \
    fold<K, O>(dst, pv, pi, elements<K, D, placed>{src, {&m}}, m.rows, m.len,  \
               m.lanes, m.parts, m.chunk, m.groups);                           \
  }                                                                            \
  KERNEL void f(O::out *dst, const K::acc *pv, const i64 *pi, i64 rows,        \
                i64 parts, i64 lanes, i64 groups) {                            \
    fold<K, O>(dst, (K::acc *)0, (i64 *)0, partials<K>{pv, pi, parts}, rows,   \
               parts, lanes, 1, parts, groups);                                \
  }

#define REDUCE(k, D)                                                           \
  typedef kind::k<D> K_;                                                       \
  typedef value<D> O_;                                                         \
  REDUCE_MODULE(K_, D, O_)

#define ARG_REDUCE(k, D)                                                       \
  typedef kind::k<D> K_;                                                       \
  REDUCE_MODULE(K_, D, position)
