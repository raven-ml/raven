/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The suite's kernels. Each runs in blocks of 256 threads along X and reads
   no block or grid size: thread i is blockIdx.x * 256 + threadIdx.x. */

typedef unsigned long long u64;

#define THREADS 256

/* The GPU's timer, in nanoseconds. */
__device__ u64 now(void) {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

/* Returns once the 64-bit word at [flag] is not 0, or after [ns]
   nanoseconds. */
__device__ void hold(volatile u64 *flag, u64 ns) {
  u64 t0 = now();
  while (*flag == 0 && now() - t0 < ns) {
  }
}

extern "C" __global__ void empty(void) {}

/* The 32-bit word i at [out] is 2i, for i < n. */
extern "C" __global__ void double_index(int *out, int n) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  if (i < n) out[i] = 2 * i;
}

/* Copies the [n] bytes at [src] to [dst], starting at least [ns] nanoseconds
   late. One block. */
extern "C" __global__ void copy_after(u64 ns, unsigned char *dst,
                                      const unsigned char *src, u64 n) {
  if (threadIdx.x == 0) {
    u64 t0 = now();
    while (now() - t0 < ns) {
    }
  }
  __syncthreads();
  for (u64 i = threadIdx.x; i < n; i += THREADS) dst[i] = src[i];
}

/* Runs until the 64-bit word at [flag] is not 0, or for [ns] nanoseconds. One
   thread. */
extern "C" __global__ void spin(volatile u64 *flag, u64 ns) { hold(flag, ns); }

/* The 32-bit word i at [out] is 512i + 130816, the sum of i + k over
   k < 512, each kept in 2 KiB of the thread's local memory, for i < n. */
extern "C" __global__ void stack(int *out, int n) {
  volatile int kept[512];
  int i = blockIdx.x * THREADS + threadIdx.x;
  for (int k = 0; k < 512; k++) kept[k] = i + k;
  int sum = 0;
  for (int k = 0; k < 512; k++) sum += kept[k];
  if (i < n) out[i] = sum;
}

/* [stack out n], holding the 2 KiB of each thread until the 64-bit word at
   [flag] is not 0, or for [ns] nanoseconds. */
extern "C" __global__ void stack_held(volatile u64 *flag, u64 ns, int *out,
                                      int n) {
  volatile int kept[512];
  int i = blockIdx.x * THREADS + threadIdx.x;
  for (int k = 0; k < 512; k++) kept[k] = i + k;
  hold(flag, ns);
  int sum = 0;
  for (int k = 0; k < 512; k++) sum += kept[k];
  if (i < n) out[i] = sum;
}
