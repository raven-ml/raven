/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The 32-bit word i at [out] is 3i, for i < n, in blocks of 256 threads. Its
   twin twice.cu differs only in the factor. */
extern "C" __global__ void index(int *out, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) out[i] = 3 * i;
}
