/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* scan: the running sum, product, max or min of each row of an operand, the
   elements along the axis, as nx.cpu computes it (cpu/nx_c_fold.c).
   SCAN(sum, float32) defines the kernels of cumsum at float32.

   A running value starts from the kind's seed, so that a running sum starts
   from +0 and its first element of -0 is +0, and is stored at each element:
   floats narrower than binary32 run in binary32, each stored value rounded
   once. Integers add and multiply in 64 bits, so they wrap. A running max or
   min is IEEE 754 maximum or minimum with the earlier operand's NaN kept,
   which is associative, so the first NaN on stays whatever the grouping, and
   -0 orders below +0.

   Row r's element j lies at [base(r) + j * step], base(r) the place of r in
   the rows' extents [oext] at strides [ostr], for the operand and for the
   result alike.
   - [t (dst, x, pv, smeta)]: a thread runs along each row.
   - [w (dst, x, pv, smeta)]: a workgroup runs along each run of [chunk]
     elements of a row, [parts] runs to a row, in chunks of THREADS * PER
     elements: each thread takes PER adjacent ones, the workgroup scans the
     threads' totals, and a carry crosses chunks. With [parts] above 1, the
     run starts from the fold of the runs before it, whose totals [pv] holds,
     [parts] to a row, as reduce's first pass writes them. */

#include "common.h"

enum { PER = 4 };

struct smeta {
  i64 rows, len, parts, chunk, groups, orank;
  i64 xoff, xstep, dstep;
  i64 oext[MAX_RANK], xostr[MAX_RANK], dostr[MAX_RANK];
};

/* A kind: its seed and the step [op a b] from a running value [a] past an
   element or a running value after it, [b]. */
namespace kind {

template <class T> struct sum {
  static DEVICE T seed() { return 0; }
  static DEVICE T op(T a, T b) { return add(a, b); }
};

template <class T> struct prod {
  static DEVICE T seed() { return 1; }
  static DEVICE T op(T a, T b) { return mul(a, b); }
};

template <class T> struct max {
  static DEVICE T seed() { return lowest<T>(); }
  static DEVICE T op(T a, T b) { return maximum(a, b); }
};

template <class T> struct min {
  static DEVICE T seed() { return highest<T>(); }
  static DEVICE T op(T a, T b) { return minimum(a, b); }
};

} // namespace kind

/* Where row [r] starts in a view of the rows' extents at strides [str]. */
DEVICE i64 row(const smeta &m, const i64 *str, i64 r) {
  i64 o = 0;
  for (i64 d = m.orank - 1; d >= 0; d--) {
    o += (r % m.oext[d]) * str[d];
    r /= m.oext[d];
  }
  return o;
}

template <class K, class D>
DEVICE void along_rows(typename D::storage *dst, const typename D::storage *x,
                       const smeta &m) {
  typedef typename D::compute T;
  for (i64 r = first(); r < m.rows; r += m.groups * THREADS) {
    const typename D::storage *xr = x + m.xoff + row(m, m.xostr, r);
    typename D::storage *dr = dst + row(m, m.dostr, r);
    T a = K::seed();
    for (i64 j = 0; j < m.len; j++) {
      a = K::op(a, D::load(xr[j * m.xstep]));
      dr[j * m.dstep] = store<D>(a);
    }
  }
}

/* The fold of the [n] values [v] in order, by the workgroup, in lane 0. The
   tree pairs neighbours, so each step joins a run with the run after it. */
template <class K, class T>
DEVICE T fold_in_order(const T *v, i64 n, T *s, int t) {
  /* Thread t folds its slice of [v] in order, then the slices join. */
  i64 per = (n + THREADS - 1) / THREADS;
  T a = K::seed();
  for (i64 i = t * per; i < (t + 1) * per && i < n; i++) a = K::op(a, v[i]);
  s[t] = a;
  barrier();
  for (int d = 1; d < THREADS; d <<= 1) {
    if (t % (2 * d) == 0) s[t] = K::op(s[t], s[t + d]);
    barrier();
  }
  T total = s[0];
  barrier();
  return total;
}

template <class K, class D>
DEVICE void along_runs(typename D::storage *dst, const typename D::storage *x,
                       const typename D::compute *pv, const smeta &m) {
  typedef typename D::compute T;
  __attribute__((shared)) static T s[THREADS];
  int t = __builtin_amdgcn_workitem_id_x();
  i64 items = m.rows * m.parts;
  for (i64 it = __builtin_amdgcn_workgroup_id_x(); it < items; it += m.groups) {
    i64 r = m.parts == 1 ? it : it / m.parts;
    i64 part = it - r * m.parts;
    i64 lo = part * m.chunk;
    i64 hi = lo + m.chunk < m.len ? lo + m.chunk : m.len;
    const typename D::storage *xr = x + m.xoff + row(m, m.xostr, r);
    typename D::storage *dr = dst + row(m, m.dostr, r);
    T carry =
        part == 0 ? K::seed() : fold_in_order<K>(pv + r * m.parts, part, s, t);
    for (i64 c = lo; c < hi; c += THREADS * PER) {
      /* Thread t's elements, from j0, and their total. */
      i64 j0 = c + (i64)t * PER;
      T v[PER];
      T own = K::seed();
      for (int e = 0; e < PER; e++) {
        i64 j = j0 + e;
        v[e] = j < hi ? D::load(xr[j * m.xstep]) : K::seed();
        own = K::op(own, v[e]);
      }
      /* The threads' totals scanned in order: s[t] holds those of threads 0
         to t. */
      s[t] = own;
      barrier();
      for (int d = 1; d < THREADS; d <<= 1) {
        T before = t >= d ? s[t - d] : K::seed();
        barrier();
        if (t >= d) s[t] = K::op(before, s[t]);
        barrier();
      }
      T a = K::op(carry, t > 0 ? s[t - 1] : K::seed());
      for (int e = 0; e < PER; e++) {
        i64 j = j0 + e;
        a = K::op(a, v[e]);
        if (j < hi) dr[j * m.dstep] = store<D>(a);
      }
      carry = K::op(carry, s[THREADS - 1]);
      barrier();
    }
  }
}

#define SCAN(k, D)                                                             \
  typedef kind::k<D::compute> K_;                                              \
  KERNEL void NAME(t)(D::storage *dst, const D::storage *x,                    \
                      const D::compute *pv, smeta m) {                         \
    along_rows<K_, D>(dst, x, m);                                              \
  }                                                                            \
  KERNEL void NAME(w)(D::storage *dst, const D::storage *x,                    \
                      const D::compute *pv, smeta m) {                         \
    along_runs<K_, D>(dst, x, pv, m);                                          \
  }
