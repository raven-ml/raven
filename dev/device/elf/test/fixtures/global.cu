/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A kernel that uses a __device__ global it does not initialise: NVRTC puts it
   in .nv.global, a section with no bytes in the object (SHT_NOBITS). */

__device__ int scratch[256];

extern "C" __global__ void global_add(int* out, const int* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    scratch[i % 256] = a[i];
    out[i] = scratch[i % 256] + 1;
  }
}
