/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The harness's kernels, as harness.h states them. The host sizes their
   grids, and the blocks of all but the floors. */

#include "harness.h"
#include "nx_dtype.h"

typedef unsigned long long u64;

static __device__ u64 now(void) {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

static __device__ u64 thread_index(void) {
  return (u64)blockIdx.x * blockDim.x + threadIdx.x;
}

/* Timing */

extern "C" __global__ void empty(void) {}

extern "C" __global__ void stamp(const __grid_constant__ stamp_params p) {
  *p.at = now();
}

extern "C" __global__ void sm_clock(const __grid_constant__ sm_clock_params p) {
  u64 t0 = now();
  long long c0 = clock64();
  while (clock64() - c0 < (long long)p.cycles) {
  }
  *p.ns = now() - t0;
}

extern "C" __global__ void delay(const __grid_constant__ delay_params p) {
  u64 t0 = now();
  while (*p.flag < p.want)
    if (now() - t0 > p.ns) {
      *p.late = 1;
      return;
    }
}

/* Operands */

/* splitmix64's finaliser. */
static __device__ u64 mix(u64 x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static __device__ double draw_float(u64 h, uint32_t draw, int spread) {
  if (draw != NX_DRAW_WIDE)
    return (double)((int64_t)(h >> 40) - (1 << 23)) / (1 << 23);
  int e = (int)((h >> 32) % (u64)(2 * spread + 1)) - spread;
  double m = 1.0 + (double)(h & 0x7FFFFFFFull) / 2147483648.0;
  return (h >> 31) & 1 ? -ldexp(m, e) : ldexp(m, e);
}

extern "C" __global__ void generate(const __grid_constant__ generate_params p) {
  for (u64 i = thread_index(); i < p.n; i += (u64)gridDim.x * blockDim.x) {
    u64 h = mix(p.seed ^ mix(i));
    double f = draw_float(h, p.draw, p.spread);
    /* An integer's low bits; NX_DRAW_SMALL's from [lo, lo + 16). */
    u64 v = p.draw == NX_DRAW_SMALL ? (u64)((int64_t)(h & 15) + p.lo) : h;
    if (p.dtype == NX_FLOAT64) v = nx_double_bits(f);
    else if (p.dtype == NX_FLOAT32) v = nx_float_bits((float)f);
    else if (p.dtype == NX_BOOL) v = h & 1;
    else if (p.narrow) v = (u64)nx_double_to_bits(p.dtype, f);
    switch (p.bytes) {
    case 1: ((uint8_t *)p.out)[i] = (uint8_t)v; break;
    case 2: ((uint16_t *)p.out)[i] = (uint16_t)v; break;
    case 4: ((uint32_t *)p.out)[i] = (uint32_t)v; break;
    default: ((u64 *)p.out)[i] = v;
    }
  }
}

/* Floors */

extern "C" __global__ void __launch_bounds__(NX_COPY_THREADS)
    floor_copy(const __grid_constant__ copy_params p) {
  u64 i = (u64)blockIdx.x * NX_COPY_THREADS + threadIdx.x;
  if (i >= p.vecs) return;
  uint4 v = make_uint4(0, 0, 0, 0);
#pragma unroll
  for (int j = 0; j < 3; j++)
    if (i < p.lens[j]) {
      uint4 x = ((const uint4 *)p.in[j])[i];
      v.x ^= x.x, v.y ^= x.y, v.z ^= x.z, v.w ^= x.w;
    }
  ((uint4 *)p.out)[i] = v;
}

/* A thread's loads are issued before it reduces them. */
extern "C" __global__ void __launch_bounds__(NX_READ_THREADS)
    floor_read(const __grid_constant__ read_params p) {
  __shared__ uint32_t warps[NX_READ_THREADS / 32];
  u64 first = (u64)blockIdx.x * NX_READ_THREADS * NX_READ_VECS + threadIdx.x;
  uint32_t x = 0;
#pragma unroll
  for (int k = 0; k < NX_READ_VECS; k++) {
    u64 i = first + k * NX_READ_THREADS;
    if (i < p.vecs) {
      uint4 v = ((const uint4 *)p.in)[i];
      x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  for (int d = 16; d > 0; d /= 2) x ^= __shfl_xor_sync(0xFFFFFFFFu, x, d);
  if (threadIdx.x % 32 == 0) warps[threadIdx.x / 32] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < NX_READ_THREADS / 32; w++) x ^= warps[w];
    p.out[blockIdx.x] = x;
  }
}

/* Codecs */

extern "C" __global__ void codecs(const __grid_constant__ codecs_params p) {
  u64 i = thread_index();
  int wide = p.dtype == NX_FLOAT16 || p.dtype == NX_BFLOAT16;
  if (i < (wide ? 1u << 16 : 1u << 8))
    p.dec[i] = nx_bits_to_float(p.dtype, (uint32_t)i);
  if (i >= p.n) return;
  int64_t c = nx_double_to_bits(p.dtype, p.x[i]);
  if (wide) ((uint16_t *)p.enc)[i] = (uint16_t)c;
  else ((uint8_t *)p.enc)[i] = (uint8_t)c;
}
