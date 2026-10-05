/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* compare: a comparison of the elements of two operands at each index, to a
   boolean, in the operands' compute type. COMPARE(less, float32) is the module
   of less at float32. */

#include "common.h"

namespace kind {

#define COMPARISON(k, op)                                                      \
  struct k {                                                                   \
    template <class T> static DEVICE uint8_t f(T a, T b) { return a op b; }    \
  };

COMPARISON(equal, ==)
COMPARISON(not_equal, !=)
COMPARISON(less, <)
COMPARISON(less_equal, <=)

} // namespace kind

template <class K, class D, class I>
DEVICE void compare(uint8_t *dst, const typename D::storage *a,
                    const typename D::storage *b, i64 n, i64 groups, I index) {
  for (i64 i = first(); i < n; i += groups * THREADS)
    dst[i] = K::f(D::load(a[index.at(0, i)]), D::load(b[index.at(1, i)]));
}

#define COMPARE(K, D)                                                          \
  KERNEL void c(uint8_t *dst, const D::storage *a, const D::storage *b, i64 n, \
                i64 groups) {                                                  \
    compare<kind::K, D>(dst, a, b, n, groups, contiguous());                   \
  }                                                                            \
  KERNEL void s(uint8_t *dst, const D::storage *a, const D::storage *b,        \
                meta m) {                                                      \
    compare<kind::K, D>(dst, a, b, m.n, m.groups, strided{&m});                \
  }
