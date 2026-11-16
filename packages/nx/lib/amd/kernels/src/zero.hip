/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* zero: 32-bit words of memory set to 0 (kernels.h). */

#include "device.h"

KERNEL SIZED(NX_ZERO_THREADS) void zero(Words<NX_ZERO_WORDS>) {
  const int64_t *w = words();
  uint32_t *dst = address<uint32_t>(NX_ZERO_DST);
  const int64_t count = w[NX_ZERO_COUNT];
  const int64_t stride = w[NX_ZERO_GROUPS] * NX_ZERO_THREADS;
  for (int64_t i = (int64_t)group_x() * NX_ZERO_THREADS + item(); i < count;
       i += stride)
    dst[i] = 0;
}
