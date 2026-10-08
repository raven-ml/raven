/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* 128 kernels, as one program compiles to: each keeps a row it indexes at run
   time, so its work-items take scratch. The loop stays a loop, which keeps the
   object near the size of a compiled program's. */

#define KERNEL(n)                                                              \
  kernel void k##n(global float *out, global const float *a, int j) {          \
    float row[64];                                                             \
    int i = __builtin_amdgcn_workitem_id_x();                                  \
    __attribute__((opencl_unroll_hint(1))) for (int r = 0; r < 64; r++)        \
      row[r] = a[(i + r) & 255] * (float)(n + r);                              \
    out[i] = row[j & 63];                                                      \
  }

#define K4(n) KERNEL(n##0) KERNEL(n##1) KERNEL(n##2) KERNEL(n##3)
#define K16(n) K4(n##0) K4(n##1) K4(n##2) K4(n##3)
#define K64(n) K16(n##0) K16(n##1) K16(n##2) K16(n##3)
K64(1) K64(2)
