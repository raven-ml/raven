/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* scatter: updates combined into a copy of the operand at the positions their
   indices select along an axis, as nx.cpu combines them (cpu/nx_c_move.c).
   An update whose index lies outside [0, len) is dropped and touches nothing.

   The destination already holds the operand; its positions' scratch [pa] and
   [pb], 64-bit words, holds what the updates decide there, one launch's runs
   apart, which order their atomics:
   - SCATTER_SET(uint32_t), keyed by width: with unique positions [u] writes
     each update; otherwise [c] sets each position's [pa] to -1, [w] raises
     it to the greatest row-major index of the position's updates, and [v]
     writes the update of that index, so the last wins.
   - SCATTER_ADD(float32): [i] starts each position's sum from +0 plus its
     element, at the compute type, and clears its mark in [pb]; [a] adds each
     update atomically and marks its position; [s] stores each marked sum,
     rounded once. Integers add in 64 bits and wrap; a boolean sum is their
     or.
   - SCATTER_EXTREME(max, float32) and (min, ...): each element and update
     becomes the sort key that orders like it, complemented for min, every
     NaN the width's greatest key. [i] starts each position's key [pa] from
     its element's, and [pb], the least row-major index of its NaN updates,
     from the greatest; [k] raises the key by each update's atomically; [n]
     lowers [pb] by each NaN update's index where the key is NaN's and the
     element is no NaN; [s] stores the bits whose key won, the element's NaN
     where it is one and otherwise the first NaN update's.

   An update j, at the index c of extents [ext], has its index at [ioff +
   sum_d c_d istr[d]] and its value at [uoff + sum_d c_d ustr[d]]; its
   position is [sum_d c'_d dstr[d]], c' being c with the index along [axis]. */

#include "common.h"

struct umeta {
  i64 n, rank, groups, axis, len, ioff, uoff;
  i64 ext[MAX_RANK], istr[MAX_RANK], ustr[MAX_RANK], dstr[MAX_RANK];
};

/* Update [j]: the offsets of its value and of its position, which is -1
   where its index is outside [0, len). */
struct update {
  i64 value, position;
};

DEVICE update locate(const umeta &m, const i64 *idx, i64 j) {
  i64 r = j, io = m.ioff, uo = m.uoff, po = 0;
  for (i64 d = m.rank - 1; d >= 0; d--) {
    i64 c = r % m.ext[d];
    r /= m.ext[d];
    io += c * m.istr[d];
    uo += c * m.ustr[d];
    if (d != m.axis) po += c * m.dstr[d];
  }
  i64 k = idx[io];
  if (k < 0 || k >= m.len) return {uo, -1};
  return {uo, po + k * m.dstr[m.axis]};
}

#define FOR_UPDATES(m, j)                                                      \
  for (i64 j = first(); j < (m).n; j += (m).groups * THREADS)

/* Positions: [n] of them, walked by the grid. */
#define FOR_POSITIONS(n, groups, p)                                            \
  for (i64 p = first(); p < (n); p += (groups) * THREADS)

/* Set */

#define SCATTER_SET(T)                                                         \
  KERNEL void c(i64 *pa, i64 n, i64 groups) {                                  \
    FOR_POSITIONS(n, groups, p) pa[p] = -1;                                    \
  }                                                                            \
  KERNEL void u(T *dst, const i64 *idx, const T *up, umeta m) {                \
    FOR_UPDATES(m, j) {                                                        \
      update e = locate(m, idx, j);                                            \
      if (e.position >= 0) dst[e.position] = up[e.value];                      \
    }                                                                          \
  }                                                                            \
  KERNEL void w(i64 *pa, const i64 *idx, umeta m) {                            \
    FOR_UPDATES(m, j) {                                                        \
      update e = locate(m, idx, j);                                            \
      if (e.position >= 0)                                                     \
        __hip_atomic_fetch_max(pa + e.position, j, __ATOMIC_RELAXED,           \
                               __HIP_MEMORY_SCOPE_AGENT);                      \
    }                                                                          \
  }                                                                            \
  KERNEL void v(T *dst, const i64 *pa, const i64 *idx, const T *up, umeta m) { \
    FOR_UPDATES(m, j) {                                                        \
      update e = locate(m, idx, j);                                            \
      if (e.position >= 0 && pa[e.position] == j) dst[e.position] = up[e.value]; \
    }                                                                          \
  }

/* Add */

template <class A> DEVICE void add_atomic(A *a, A v) {
  __hip_atomic_fetch_add(a, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

/* The accumulator of a sum at a compute type: booleans sum as 64-bit
   integers, whose stored value is whether they are 0. */
template <class C> struct accumulator { typedef C t; };
template <> struct accumulator<uint8_t> { typedef uint64_t t; };

template <class D>
DEVICE typename D::storage settle(typename accumulator<typename D::compute>::t a) {
  if (D::cat == BOOL) return store<D>(a != 0);
  return store<D>((typename D::compute)a);
}

#define SCATTER_ADD(D)                                                         \
  typedef accumulator<D::compute>::t A_;                                       \
  KERNEL void i(const D::storage *dst, A_ *pa, i64 *pb, i64 n, i64 groups) {   \
    FOR_POSITIONS(n, groups, p) {                                              \
      pa[p] = add((A_)0, (A_)D::load(dst[p]));                                 \
      pb[p] = 0;                                                               \
    }                                                                          \
  }                                                                            \
  KERNEL void a(A_ *pa, i64 *pb, const i64 *idx, const D::storage *up,         \
                umeta m) {                                                     \
    FOR_UPDATES(m, j) {                                                        \
      update e = locate(m, idx, j);                                            \
      if (e.position < 0) continue;                                            \
      add_atomic(pa + e.position, (A_)D::load(up[e.value]));                   \
      pb[e.position] = 1;                                                      \
    }                                                                          \
  }                                                                            \
  KERNEL void s(D::storage *dst, const A_ *pa, const i64 *pb, i64 n,           \
                i64 groups) {                                                  \
    FOR_POSITIONS(n, groups, p) if (pb[p]) dst[p] = settle<D>(pa[p]);          \
  }

/* Max and min */

/* The key of element [x] of [D] at the storage width, which orders like it,
   every NaN the greatest; complemented within the width for min, a NaN's
   staying the greatest. */
template <class D> DEVICE bool is_nan(typename D::storage x) {
  if (D::cat != FLOAT) return false;
  typename D::compute v = D::load(x);
  return v != v;
}

template <class D> struct keys {
  typedef typename D::storage S;
  enum { W = sizeof(S) * 8 };
  static DEVICE uint64_t all() {
    return W == 64 ? ~(uint64_t)0 : (((uint64_t)1 << W) - 1);
  }
  static DEVICE uint64_t sign() { return (uint64_t)1 << (W - 1); }
  static DEVICE uint64_t bits(S x) {
    uint64_t b = 0;
    __builtin_memcpy(&b, &x, sizeof(S));
    return b;
  }
  static DEVICE S of_bits(uint64_t b) {
    S x;
    __builtin_memcpy(&x, &b, sizeof(S));
    return x;
  }
  static DEVICE uint64_t up(S x) {
    if (is_nan<D>(x)) return all();
    uint64_t b = bits(x);
    if (D::cat == FLOAT) return b & sign() ? ~b & all() : b | sign();
    if (D::cat == SINT) return b ^ sign();
    return b;
  }
  static DEVICE S down(uint64_t k) {
    if (D::cat == FLOAT) return of_bits(k & sign() ? k & ~sign() : ~k & all());
    if (D::cat == SINT) return of_bits(k ^ sign());
    return of_bits(k);
  }
};

/* The key a mode raises: the element's for max, its complement within the
   width for min, a float NaN's staying the greatest. */
struct max_order {
  template <class D> static DEVICE uint64_t flip(uint64_t k) { return k; }
};

struct min_order {
  template <class D> static DEVICE uint64_t flip(uint64_t k) {
    uint64_t all = keys<D>::all();
    return D::cat == FLOAT && k == all ? k : ~k & all;
  }
};

#define SCATTER_EXTREME(order, D)                                              \
  typedef keys<D> K_;                                                          \
  typedef order##_order O_;                                                    \
  KERNEL void i(const D::storage *dst, uint64_t *pa, i64 *pb, i64 n,           \
                i64 groups) {                                                  \
    FOR_POSITIONS(n, groups, p) {                                              \
      pa[p] = O_::template flip<D>(K_::up(dst[p]));                             \
      pb[p] = INT64_MAX;                                                       \
    }                                                                          \
  }                                                                            \
  KERNEL void k(uint64_t *pa, const i64 *idx, const D::storage *up, umeta m) { \
    FOR_UPDATES(m, j) {                                                        \
      update e = locate(m, idx, j);                                            \
      if (e.position < 0) continue;                                            \
      __hip_atomic_fetch_max(pa + e.position,                                  \
                             O_::template flip<D>(K_::up(up[e.value])),         \
                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);      \
    }                                                                          \
  }                                                                            \
  KERNEL void n(const D::storage *dst, const uint64_t *pa, i64 *pb,            \
                const i64 *idx, const D::storage *up, umeta m) {               \
    FOR_UPDATES(m, j) {                                                        \
      update e = locate(m, idx, j);                                            \
      if (e.position < 0 || pa[e.position] != K_::all()) continue;             \
      if (is_nan<D>(dst[e.position]) || !is_nan<D>(up[e.value])) continue;     \
      __hip_atomic_fetch_min(pb + e.position, j, __ATOMIC_RELAXED,             \
                             __HIP_MEMORY_SCOPE_AGENT);                        \
    }                                                                          \
  }                                                                            \
  KERNEL void s(D::storage *dst, const uint64_t *pa, const i64 *pb,            \
                const i64 *idx, const D::storage *up, umeta m, i64 positions,  \
                i64 groups) {                                                  \
    (void)idx;                                                                 \
    FOR_POSITIONS(positions, groups, p) {                                      \
      uint64_t key = pa[p];                                                    \
      if (key == K_::all() && D::cat == FLOAT) {                               \
        if (pb[p] == INT64_MAX) continue;                                      \
        i64 r = pb[p], o = m.uoff;                                             \
        for (i64 d = m.rank - 1; d >= 0; d--) {                                \
          o += (r % m.ext[d]) * m.ustr[d];                                     \
          r /= m.ext[d];                                                       \
        }                                                                      \
        dst[p] = up[o];                                                        \
      } else {                                                                 \
        dst[p] = K_::down(O_::template flip<D>(key));                           \
      }                                                                        \
    }                                                                          \
  }
