/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* How partial sums combine: one output tile's sum cut into [splits] ranges,
   each summed by its own block, those blocks the same in every way but
   their range. Each block stores its partial; the last block to take the
   tile's ticket adds the partials in range order, s0 + s1 + ... + s(n-1),
   whichever order the blocks ran in. A family's kernels call it after
   their own sums, before their epilogue. */

#ifndef NX_CUDA_COMBINE_CUH
#define NX_CUDA_COMBINE_CUH

#include <stdint.h>

/* [v] is this thread's N values of the partial of its block: range
   blockIdx.y of the tile of blockIdx.x and blockIdx.z, whose [splits]
   partials go to [partials] (N * threads values each, the tiles one after
   another) and whose ticket is its word of [tickets], 0 before the first
   block arrives. The last block returns the ticket to 0, so tickets left
   by one launch are zero for the next. Returns true in the last block,
   its [v] then the tile's sum, and false in the others, which stop
   there. */
template <typename T, int N>
__device__ bool combine(T (&v)[N], T *partials, uint32_t *tickets,
                        int splits) {
  __shared__ int last;
  const int threads = blockDim.x * blockDim.y, tid =
      threadIdx.y * blockDim.x + threadIdx.x;
  const int tile = blockIdx.z * gridDim.x + blockIdx.x, z = blockIdx.y;
  T *part = partials + (size_t)tile * splits * N * threads;
  uint32_t *ticket = tickets + tile;
  T *mine = part + (size_t)z * N * threads;
#pragma unroll
  for (int f = 0; f < N; f++) __stcg(mine + (size_t)f * threads + tid, v[f]);
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(ticket, 1u) == (uint32_t)splits - 1;
  __syncthreads();
  if (!last) return false;
  /* Every other block of the tile has taken its ticket. */
  if (tid == 0) *ticket = 0;
  __threadfence();
#pragma unroll
  for (int f = 0; f < N; f++) v[f] = __ldcg(part + (size_t)f * threads + tid);
#pragma unroll 1
  for (int r = 1; r < splits; r++) {
    const T *q = part + (size_t)r * N * threads + tid;
#pragma unroll
    for (int f = 0; f < N; f++) v[f] += __ldcg(q + (size_t)f * threads);
  }
  return true;
}

#endif
