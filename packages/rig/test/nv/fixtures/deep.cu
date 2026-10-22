/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Two kernels whose threads keep an array in local memory: deep keeps 480 KiB
   a thread, near the 512 KiB a thread may keep, and shallow 4 KiB. A GPU
   holds a thread's local memory for every thread its multiprocessors may run
   at once: deep needs some 100 GiB of it on an RTX 5000 Ada. Each indexes
   its array by the thread, which only local memory holds, and fills it by a
   loop kept rolled so that the cubin stays small. Thread i of the grid
   writes a[i % words], which is i + i % words, into word i of [out] if
   i < n. */

#define STACK(name, words)                                                 \
  extern "C" __global__ void name(int *out, int n) {                       \
    volatile int a[words];                                                 \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                         \
    _Pragma("unroll 1")                                                    \
    for (int j = 0; j < (words); j++) a[j] = i + j;                        \
    if (i < n) out[i] = a[i % (words)];                                    \
  }

STACK(deep, 120 * 1024)
STACK(shallow, 1024)
