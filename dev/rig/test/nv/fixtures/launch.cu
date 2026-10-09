/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The kernels of launches, whose parameters a launch's block holds: ids and
   twice for the conformance suite's launch laws, rotate for the NV suite. A
   thread's index k is g * T + t: g its group's index in the grid, x fastest,
   t its index in its group, x fastest, and T the threads of a group. */

typedef unsigned int u32;
typedef unsigned long long u64;

__device__ u32 group(void) {
  return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

__device__ u32 thread(void) {
  return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

__device__ u32 threads(void) { return blockDim.x * blockDim.y * blockDim.z; }

/* Stores (u32)a + b * k + (u32)f into the 32-bit word k at [out], f rounded
   toward zero. */
extern "C" __global__ void ids(u32 *out, u64 a, u32 b, float f) {
  u32 k = group() * threads() + thread();
  out[k] = (u32)a + b * k + (u32)f;
}

/* Stores 2x + c into the 32-bit word i at [dst], x the 32-bit word i at
   [src], for each thread i of a grid of one dimension. */
extern "C" __global__ void twice(u32 *dst, const u32 *src, u32 c) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  dst[i] = 2 * src[i] + c;
}

/* Stores k + 3i + 7g, i the thread's index in the grid, into the 32-bit word
   t of the group's dynamic shared memory, of at least 4T bytes, then the word
   t + 1 of it, the next thread's (thread 0's for the last), into the 32-bit
   word i at [out]. */
extern "C" __global__ void rotate(u32 *out, u32 k) {
  extern __shared__ u32 dyn[];
  u32 g = group(), t = thread(), n = threads(), i = g * n + t;
  dyn[t] = k + 3 * i + 7 * g;
  __syncthreads();
  out[i] = dyn[(t + 1) % n];
}
