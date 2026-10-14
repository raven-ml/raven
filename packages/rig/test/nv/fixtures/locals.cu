/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Forty kernels, local0 to local39, each with local memory of its own size:
   local<k> keeps 16 (k + 1) words per thread in an array it indexes by the
   thread, which only local memory holds, filled by a loop kept rolled so
   that the cubin stays small. Thread i writes word i of [out]. */

#define LOCAL(k)                                                           \
  extern "C" __global__ void local##k(int *out, int n) {                   \
    volatile int a[16 * ((k) + 1)];                                        \
    int i = threadIdx.x;                                                   \
    _Pragma("unroll 1")                                                    \
    for (int j = 0; j < 16 * ((k) + 1); j++) a[j] = i + j;                 \
    if (i < n) out[i] = a[i % (16 * ((k) + 1))];                           \
  }

LOCAL(0)
LOCAL(1)
LOCAL(2)
LOCAL(3)
LOCAL(4)
LOCAL(5)
LOCAL(6)
LOCAL(7)
LOCAL(8)
LOCAL(9)
LOCAL(10)
LOCAL(11)
LOCAL(12)
LOCAL(13)
LOCAL(14)
LOCAL(15)
LOCAL(16)
LOCAL(17)
LOCAL(18)
LOCAL(19)
LOCAL(20)
LOCAL(21)
LOCAL(22)
LOCAL(23)
LOCAL(24)
LOCAL(25)
LOCAL(26)
LOCAL(27)
LOCAL(28)
LOCAL(29)
LOCAL(30)
LOCAL(31)
LOCAL(32)
LOCAL(33)
LOCAL(34)
LOCAL(35)
LOCAL(36)
LOCAL(37)
LOCAL(38)
LOCAL(39)
