/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* gather: at each index of the indices, the operand's element that the index
   there selects along the axis, or 0 where it lies outside [0, len). It moves
   bytes, so a module serves every dtype of one width: GATHER(uint32_t) is the
   module of 4-byte elements.

   The operand comes as a view of the indices' shape whose stride along the
   axis is 0, so that it coalesces with the indices: element i reads
   [x[at(0, i) + k * step]] for the index [k] at [idx[at(1, i)]], and reads
   nothing for an index out of range.
   - [c (dst, x, idx, n, groups, step, len)]
   - [s (dst, x, idx, meta, step, len)] */

#include "common.h"

template <class T, class I>
DEVICE void gather(T *dst, const T *x, const i64 *idx, i64 n, i64 groups,
                   i64 step, i64 len, I index) {
  for (i64 i = first(); i < n; i += groups * THREADS) {
    i64 k = idx[index.at(1, i)];
    dst[i] = k >= 0 && k < len ? x[index.at(0, i) + k * step] : (T)0;
  }
}

#define GATHER(T)                                                              \
  KERNEL void c(T *dst, const T *x, const i64 *idx, i64 n, i64 groups,         \
                i64 step, i64 len) {                                           \
    gather(dst, x, idx, n, groups, step, len, contiguous());                   \
  }                                                                            \
  KERNEL void s(T *dst, const T *x, const i64 *idx, meta m, i64 step,          \
                i64 len) {                                                     \
    gather(dst, x, idx, m.n, m.groups, step, len, strided{&m});                \
  }
