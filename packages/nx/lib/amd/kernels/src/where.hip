/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* where: the element of [a] where the condition's byte is not 0, of [b]
   elsewhere. It moves bytes, so a module serves every dtype of one width:
   WHERE(uint32_t) is the module of 4-byte elements. */

#include "common.h"

template <class T, class I>
DEVICE void select(T *dst, const uint8_t *cond, const T *a, const T *b, i64 n,
                   i64 groups, I index) {
  for (i64 i = first(); i < n; i += groups * THREADS)
    dst[i] = cond[index.at(0, i)] ? a[index.at(1, i)] : b[index.at(2, i)];
}

#define WHERE(T)                                                               \
  KERNEL void c(T *dst, const uint8_t *cond, const T *a, const T *b, i64 n,    \
                i64 groups) {                                                  \
    select(dst, cond, a, b, n, groups, contiguous());                          \
  }                                                                            \
  KERNEL void s(T *dst, const uint8_t *cond, const T *a, const T *b, meta m) { \
    select(dst, cond, a, b, m.n, m.groups, strided{&m});                       \
  }
