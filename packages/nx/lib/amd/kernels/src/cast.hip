/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* cast: each element of an operand converted to another dtype.
   CAST(float32, int8) defines the kernels of the cast from float32 to
   int8. */

#include "common.h"

template <class S, class D, class I>
DEVICE void convert_all(typename D::storage *dst, const typename S::storage *src,
                        i64 n, i64 groups, I index) {
  for (i64 i = first(); i < n; i += groups * THREADS)
    dst[i] = cast<S, D>(src[index.at(0, i)]);
}

#define CAST(S, D)                                                             \
  KERNEL void NAME(c)(D::storage *dst, const S::storage *src, i64 n,           \
                      i64 groups) {                                            \
    convert_all<S, D>(dst, src, n, groups, contiguous());                      \
  }                                                                            \
  KERNEL void NAME(s)(D::storage *dst, const S::storage *src, meta m) {        \
    convert_all<S, D>(dst, src, m.n, m.groups, strided{&m});                   \
  }
