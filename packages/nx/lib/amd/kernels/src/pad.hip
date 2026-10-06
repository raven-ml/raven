/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* pad: the operand surrounded along each axis by elements of one value. It
   moves bytes, so a module serves every dtype of one width: PAD(uint32_t)
   defines the kernel of 4-byte elements, and the value comes as the low bytes
   of [fill].

   Element i of the result, at the index c of extents [ext], is the operand's
   at [off + sum_d (c_d - lo[d]) str[d]] where every [lo[d] <= c_d < hi[d]],
   and the value elsewhere.
   - [s (dst, x, pmeta)] */

#include "common.h"

struct pmeta {
  i64 n, rank, groups, off;
  i64 ext[MAX_RANK], lo[MAX_RANK], hi[MAX_RANK], str[MAX_RANK];
  uint64_t fill;
};

template <class T> DEVICE void pad(T *dst, const T *x, const pmeta &m) {
  for (i64 i = first(); i < m.n; i += m.groups * THREADS) {
    i64 r = i, o = m.off;
    bool inside = true;
    for (i64 d = m.rank - 1; d >= 0; d--) {
      i64 c = r % m.ext[d];
      r /= m.ext[d];
      inside = inside && c >= m.lo[d] && c < m.hi[d];
      o += (c - m.lo[d]) * m.str[d];
    }
    dst[i] = inside ? x[o] : (T)m.fill;
  }
}

#define PAD(T)                                                                 \
  KERNEL void NAME(s)(T *dst, const T *x, pmeta m) { pad(dst, x, m); }
