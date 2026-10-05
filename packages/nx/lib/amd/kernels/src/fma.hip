/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* fma: a * b + c at each index, as nx.cpu computes it (cpu/nx_c_map.c):
   rounded once in the compute type, so the narrow floats take binary32's and
   round again on the store, and modular on integers. A NaN result is the first
   NaN operand, or else the operation's own. FMA(float32) is the module of fma
   at float32. */

#include "common.h"

template <class T> DEVICE T nan3(T a, T b, T c, T r) {
  r = c != c ? c : r;
  r = b != b ? b : r;
  return a != a ? a : r;
}

DEVICE float fma(float a, float b, float c) {
  return nan3(a, b, c, __builtin_fmaf(a, b, c));
}
DEVICE double fma(double a, double b, double c) {
  return nan3(a, b, c, __builtin_fma(a, b, c));
}
DEVICE int64_t fma(int64_t a, int64_t b, int64_t c) {
  return (int64_t)((uint64_t)a * (uint64_t)b + (uint64_t)c);
}
DEVICE uint64_t fma(uint64_t a, uint64_t b, uint64_t c) { return a * b + c; }

template <class D, class I>
DEVICE void map3(typename D::storage *dst, const typename D::storage *a,
                 const typename D::storage *b, const typename D::storage *c,
                 i64 n, i64 groups, I index) {
  for (i64 i = first(); i < n; i += groups * THREADS)
    dst[i] = store<D>(fma(D::load(a[index.at(0, i)]),
                          D::load(b[index.at(1, i)]),
                          D::load(c[index.at(2, i)])));
}

#define FMA(D)                                                                 \
  KERNEL void c(D::storage *dst, const D::storage *a, const D::storage *b,     \
                const D::storage *c, i64 n, i64 groups) {                      \
    map3<D>(dst, a, b, c, n, groups, contiguous());                            \
  }                                                                            \
  KERNEL void s(D::storage *dst, const D::storage *a, const D::storage *b,     \
                const D::storage *c, meta m) {                                 \
    map3<D>(dst, a, b, c, m.n, m.groups, strided{&m});                         \
  }
