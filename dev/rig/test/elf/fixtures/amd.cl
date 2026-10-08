/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* GPU kernels as the AMD loader takes them: a kernel, its descriptor, and a
   global in .bss. With KERNELS defined to 16 or 128, that many kernels. */

global int last;

#define KERNEL(n)                                                              \
  kernel void add##n(global int *out, global const int *a,                     \
                     global const int *b) {                                    \
    int i = __builtin_amdgcn_workitem_id_x();                                  \
    out[i] = a[i] + b[i];                                                      \
    last = i;                                                                  \
  }

#define K4(n) KERNEL(n##0) KERNEL(n##1) KERNEL(n##2) KERNEL(n##3)
#define K16(n) K4(n##0) K4(n##1) K4(n##2) K4(n##3)
#define K64(n) K16(n##0) K16(n##1) K16(n##2) K16(n##3)

#if KERNELS == 16
K16(0)
#elif KERNELS == 128
K64(0) K64(1)
#else
KERNEL()
#endif
