/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* contiguous: the elements of an operand of any layout, copied in C order.
   It moves bytes, so a module serves every dtype of one width:
   CONTIGUOUS(uint32_t) defines the kernels of 4-byte elements. */

#include "common.h"

template <class T, class I>
DEVICE void copy(T *dst, const T *src, i64 n, i64 groups, I index) {
  for (i64 i = first(); i < n; i += groups * THREADS) dst[i] = src[index.at(0, i)];
}

#define CONTIGUOUS(T)                                                          \
  KERNEL void NAME(c)(T *dst, const T *src, i64 n, i64 groups) {               \
    copy(dst, src, n, groups, contiguous());                                   \
  }                                                                            \
  KERNEL void NAME(s)(T *dst, const T *src, meta m) {                          \
    copy(dst, src, m.n, m.groups, strided{&m});                                \
  }
