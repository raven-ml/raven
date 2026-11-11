/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* How partial sums combine: one output tile's sum cut into [splits] ranges
   of k (k_range), each summed by its own block, those blocks the same in
   every way but their range. Each block stores its partial; the last block
   to take the tile's ticket adds the partials in range order,
   s0 + s1 + ... + s(n-1), whichever order the blocks ran in. A family's
   kernels call it after their own sums, before their epilogue. Part of
   contract.cu's translation unit. */

#ifndef NX_CUDA_COMBINE_CUH
#define NX_CUDA_COMBINE_CUH

/* Range [s] of [S] of [k] terms: its runs of RUN terms shared out in
   order, range s taking runs s * runs / S to (s + 1) * runs / S, the last
   cut at k. The bounds depend on k, S and RUN alone. */
template <int RUN>
__device__ __forceinline__ void k_range(int64_t k, int s, int S, int64_t &k0,
                                        int64_t &k1) {
  const int64_t runs = (k + RUN - 1) / RUN;
  k0 = s * runs / S * RUN;
  k1 = min(k, (s + 1) * runs / S * RUN);
}

/* [v], this thread's N values of its block's range sum, the first [live]
   of them computed; the tile's [splits] ranges' sums go to [part]
   (N * THREADS values each, in range order) and its ticket is [ticket], 0
   before the first block arrives. The last block of the tile to take its
   ticket adds them in range order, s0 + s1 + ..., whichever order the
   blocks ran in, and returns true with the tile's sums in [v]; the others
   return false, and stop there. It leaves the ticket zero for the next
   launch. */
template <typename T, int N, int THREADS>
__device__ bool combine(T (&v)[N], T *part, uint32_t *ticket, int split,
                        int splits, int live = N) {
  __shared__ int last;
  const int tid = threadIdx.x;
  T *mine = part + (size_t)split * N * THREADS;
#pragma unroll
  for (int f = 0; f < N; f++)
    if (f < live) __stcg(mine + f * THREADS + tid, v[f]);
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(ticket, 1u) == (uint32_t)splits - 1;
  __syncthreads();
  if (!last) return false;
  /* Every other block of the tile has taken its ticket. */
  if (tid == 0) *ticket = 0;
  __threadfence();
#pragma unroll
  for (int f = 0; f < N; f++)
    if (f < live) v[f] = __ldcg(part + f * THREADS + tid);
  /* A thread of few values loads AHEAD ranges' before it adds them, so
     that their loads are in flight together. */
  constexpr int AHEAD = N <= 8 ? 4 : 1;
#pragma unroll 1
  for (int r = 1; r < splits; r += AHEAD) {
    T x[AHEAD][N];
#pragma unroll
    for (int a = 0; a < AHEAD; a++)
      if (r + a < splits)
#pragma unroll
        for (int f = 0; f < N; f++)
          if (f < live) x[a][f] = __ldcg(part + (size_t)(r + a) * N * THREADS + f * THREADS + tid);
#pragma unroll
    for (int a = 0; a < AHEAD; a++)
      if (r + a < splits)
#pragma unroll
        for (int f = 0; f < N; f++)
          if (f < live) v[f] += x[a][f];
  }
  return true;
}

/* A range's blocks. A float range of more than NX_CONTRACT_BLOCK terms
   sums its terms in blocks of that many from its start, each block from
   +0 in the thread's [v]; its running sum, the blocks' sums added in
   order, the first as it is, lies in slot [s] of [partials] (N * THREADS
   values a slot, as combine lays them out), so that it takes no
   register. fold() adds [v], the first [live] of them computed, into it
   and sets [v] to +0 for the next block; close() gives [v] the running
   sum plus the last block's. Each thread reads only what it wrote. */
template <typename T, int N, int THREADS>
__device__ void fold(T (&v)[N], void *partials, int64_t s, bool first,
                     int live = N) {
  const int tid = threadIdx.x;
  T *mine = (T *)partials + s * N * THREADS;
#pragma unroll
  for (int f = 0; f < N; f++) {
    if (f < live) {
      T *at = mine + f * THREADS + tid;
      __stcg(at, first ? v[f] : __ldcg(at) + v[f]);
    }
    v[f] = T(0);
  }
}

template <typename T, int N, int THREADS>
__device__ void close(T (&v)[N], const void *partials, int64_t s,
                      int live = N) {
  const int tid = threadIdx.x;
  const T *mine = (const T *)partials + s * N * THREADS;
#pragma unroll
  for (int f = 0; f < N; f++)
    if (f < live) v[f] = __ldcg(mine + f * THREADS + tid) + v[f];
}

#endif
