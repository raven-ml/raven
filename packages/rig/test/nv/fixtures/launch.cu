/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The kernels of launches, whose parameters a launch's block holds: ids,
   twice and wide for the conformance suite's launch laws, rotate, turn and pick for
   the NV suite. A
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

/* As rotate, through 8 KiB of static shared memory. */
extern "C" __global__ void turn(u32 *out, u32 k) {
  __shared__ u32 kept[2048];
  u32 g = group(), t = thread(), n = threads(), i = g * n + t;
  kept[t] = k + 3 * i + 7 * g;
  __syncthreads();
  out[i] = kept[(t + 1) % n];
}

/* Element i of [p], of type [t]: 0 to 3 unsigned of 1, 2, 4 and 8 bytes, 4
   to 7 signed of those, 8 float, 9 double. A switch the compiler makes a
   jump table, out of line, which it keeps in constant bank 2, one of the
   kernel's own sections. */
__device__ __noinline__ u64 element(const void *p, u32 i, u32 t) {
  switch (t) {
  case 0: return ((const unsigned char *)p)[i];
  case 1: return ((const unsigned short *)p)[i];
  case 2: return ((const u32 *)p)[i];
  case 3: return ((const u64 *)p)[i];
  case 4: return (u64)(long long)((const signed char *)p)[i];
  case 5: return (u64)(long long)((const short *)p)[i];
  case 6: return (u64)(long long)((const int *)p)[i];
  case 7: return (u64)((const long long *)p)[i];
  case 8: return (u64)(long long)((const float *)p)[i];
  case 9: return (u64)(long long)((const double *)p)[i];
  }
  return 0;
}

/* Stores into the 32-bit word i at [out] the low 32 bits of element i of
   [src], of type [t] (element). */
extern "C" __global__ void pick(u32 *out, const void *src, u32 t) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = (u32)element(src, i, t);
}

/* The words of wide's parameters after [out]. */
struct wide_words {
  u32 w[2046];
};

/* Stores the 32-bit words at bytes 4,096 and 8,188 of its 8,192 bytes of
   parameters into words 0 and 1 at [out], its first 8. */
extern "C" __global__ void wide(u32 *out, struct wide_words p) {
  out[0] = p.w[1022];
  out[1] = p.w[2045];
}
