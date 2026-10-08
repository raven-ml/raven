/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* 128 kernels, k00 to k7f, as one program compiles to: each scales by a global
   of its own, whose address the cubin relocates. */

#define K(i)                                                                   \
  __device__ float scale_##i = 1;                                              \
  extern "C" __global__ void k##i(float *out, const float *a, int n) {         \
    int t = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (t < n) out[t] = a[t] * scale_##i;                                      \
  }

#define K16(h)                                                                 \
  K(h##0) K(h##1) K(h##2) K(h##3) K(h##4) K(h##5) K(h##6) K(h##7) K(h##8)      \
  K(h##9) K(h##a) K(h##b) K(h##c) K(h##d) K(h##e) K(h##f)

K16(0) K16(1) K16(2) K16(3) K16(4) K16(5) K16(6) K16(7)
