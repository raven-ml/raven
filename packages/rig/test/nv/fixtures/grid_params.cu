/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

typedef unsigned int u32;

struct params {
  u32 *out;
  u32 values[4];
};

/* A generic address into the launch's parameters must reach their bytes. The
   volatile read in a separate function keeps the compiler from replacing it
   with a direct constant-bank load. */
__device__ __noinline__ u32 indexed(const volatile u32 *p, u32 at) {
  return p[at];
}

extern "C" __global__ void grid_params(const __grid_constant__ params p) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  p.out[i] = indexed(p.values, i % 4);
}
