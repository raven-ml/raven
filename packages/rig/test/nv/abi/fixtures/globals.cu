/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A kernel that reads two globals: an uninitialised one, which the cubin
   holds in .nv.global without bytes, and an initialised one, in
   .nv.global.init. The cubin relocates the address of each. */

__device__ float scale;
__device__ int bias = 3;

extern "C" __global__ void affine(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * scale + bias;
}
