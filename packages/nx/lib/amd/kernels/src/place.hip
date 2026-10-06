/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* place: an operand's elements written into a window of the destination.
   It moves bytes, so a module serves every dtype of one width: PLACE(uint32_t)
   is the module of 4-byte elements.

   The window is a view of the destination, operand 0 of the index template,
   and the operand is operand 1. A corner of rank r moves the window by
   [sum_{d<r} starts[off + d * str] * dstr[d]]: the positions of a vector
   read when the kernel runs, along axes of strides [dstr]. A corner of rank 0
   leaves it at its view's offset.
   - [c (dst, x, starts, n, groups, corner)]
   - [s (dst, x, starts, meta, corner)] */

#include "common.h"

struct corner {
  i64 rank, off, str;
  i64 dstr[MAX_RANK];
};

template <class T, class I>
DEVICE void place(T *dst, const T *x, const i64 *starts, i64 n, i64 groups,
                  const corner &k, I index) {
  i64 base = 0;
  for (i64 d = 0; d < k.rank; d++) base += starts[k.off + d * k.str] * k.dstr[d];
  for (i64 i = first(); i < n; i += groups * THREADS)
    dst[base + index.at(0, i)] = x[index.at(1, i)];
}

#define PLACE(T)                                                               \
  KERNEL void c(T *dst, const T *x, const i64 *starts, i64 n, i64 groups,      \
                corner k) {                                                    \
    place(dst, x, starts, n, groups, k, contiguous());                         \
  }                                                                            \
  KERNEL void s(T *dst, const T *x, const i64 *starts, meta m, corner k) {     \
    place(dst, x, starts, m.n, m.groups, k, strided{&m});                      \
  }
