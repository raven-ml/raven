/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The skinny kernels (m <= 16): lane l of 32 sums, for each t, the run of
   four k0 + 128 t + 4 l + u, u < 4, in increasing k, and the 32 lane sums
   combine by the balanced tree (l, l + 16), then (l, l + 8), (l, l + 4),
   (l, l + 2), (l, l + 1). A block computes NX_SKINNY_ROWS rows of 32
   columns: few rows a thread keep its registers few, so that an SM holds
   enough warps to cover a load's latency. When b's k axis is contiguous,
   warp w takes the columns w + 8 q, q < 4, lanes across k, a run one
   vector, and loads each of a's runs once for its four columns; otherwise
   thread (x, y) sums the lane classes y, y + 8, y + 16, y + 24 of column
   x, reading rows of b across x.

   Part of contract.cu's translation unit, after elements.cuh. */

#ifndef NX_CUDA_SKINNY_CUH
#define NX_CUDA_SKINNY_CUH

/* The 32 lane sums [v] combined by the tree, the root in every lane. */
template <typename T> __device__ T tree32(T v) {
#pragma unroll
  for (int d = 16; d > 0; d /= 2) v += __shfl_xor_sync(0xFFFFFFFFu, v, d);
  return v;
}

/* The four elements of S at [src], aligned on their 4 sizeof(S) bytes,
   as one or two vector loads, each widened to T. */
template <typename T, typename S>
__device__ void load4(T (&v)[4], const char *src) {
  constexpr int N = 4 * sizeof(S);
  union {
    uint4 q[2];
    S e[4];
  } u;
  if (N == 32) u.q[0] = ((const uint4 *)src)[0], u.q[1] = ((const uint4 *)src)[1];
  if (N == 16) u.q[0] = *(const uint4 *)src;
  if (N == 8) *(uint2 *)u.q = *(const uint2 *)src;
  if (N == 4) *(uint32_t *)u.q = *(const uint32_t *)src;
#pragma unroll
  for (int e = 0; e < 4; e++) v[e] = widen<T>(u.e[e]);
}

/* The run of four k from [k] of the operand at [p], of elements S with
   strides [s_row] and [s_k], row [row]: its first [us] elements, the rest
   zero, widened to T, as one vector where [vectors] (its k contiguous, its
   rows aligned). */
template <typename T, typename S>
__device__ void run(T (&v)[4], const char *p, int64_t s_row, int64_t s_k,
                    int row, int k, int us, bool vectors) {
  if (vectors && us == 4) {
    load4<T, S>(v, p + (row * s_row + k) * sizeof(S));
    return;
  }
#pragma unroll
  for (int u = 0; u < 4; u++)
    v[u] = u < us ? widen<T>(((const S *)p)[row * s_row + (k + u) * s_k]) : T(0);
}

/* Adds the runs [bv] of k from [k] of Q columns, their first [us] inside
   the range, times a's [rows] rows from [a], to [c]: each of a's runs
   loaded once for the Q columns. a's runs are vectors where
   NX_CONTRACT_A_VECTORS says so. */
template <int Q, typename T, typename S>
__device__ void dot4(const contract_params &p, T (&c)[Q][NX_SKINNY_ROWS],
                     const char *a, int rows, const T (&bv)[Q][4], int k,
                     int us) {
#pragma unroll
  for (int i = 0; i < NX_SKINNY_ROWS; i++)
    if (i < rows) {
      T av[4];
      run<T, S>(av, a, p.sa[1], p.sa[2], i, k, us,
                p.aligned & NX_CONTRACT_A_VECTORS);
#pragma unroll
      for (int q = 0; q < Q; q++)
#pragma unroll
        for (int u = 0; u < 4; u++)
          if (u < us) c[q][i] = fma_(av[u], bv[q][u], c[q][i]);
    }
}

/* Adds the run of k from [k], its first [us] inside the range, of the
   columns j + 8 q, q < 4, times a's rows, to [c]; b's runs are vectors
   where NX_CONTRACT_B_VECTORS says so. A column past n reads column
   n - 1, whose sums the store drops: the loads take no branch. */
template <typename T, typename S>
__device__ void step4(const contract_params &p, T (&c)[4][NX_SKINNY_ROWS],
                      const char *a, int rows, const char *b, int j, int k,
                      int us) {
  T bv[4][4];
#pragma unroll
  for (int q = 0; q < 4; q++)
    run<T, S>(bv[q], b, p.sb[1], p.sb[2], min(j + 8 * q, p.n - 1), k, us,
              p.aligned & NX_CONTRACT_B_VECTORS);
  dot4<4, T, S>(p, c, a, rows, bv, k, us);
}

/* The block's sums of [rows] rows from [a] and the 32 columns from [j0]
   into [level8], row i of column c at element i * 32 + c, as the tree
   leaves them. */
template <typename T, typename S>
__device__ void skinny_sums(const contract_params &p,
                            T (*level8)[8][32], const char *a, int rows,
                            const char *b, int j0, int k0, int k1) {
  constexpr int R = NX_SKINNY_ROWS;
  T *sums = &level8[0][0][0];
  if ((p.aligned & NX_CONTRACT_B_ACROSS) == 0) {
    /* Warp w: columns j0 + w + 8 q, lane l holding class l. Whole runs,
       then the last one's elements inside the range. */
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    T acc[4][R];
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int i = 0; i < R; i++) acc[q][i] = T(0);
    int k = k0 + 4 * lane;
    for (; k + 3 < k1; k += 128)
      step4<T, S>(p, acc, a, rows, b, j0 + w, k, 4);
    if (k < k1) step4<T, S>(p, acc, a, rows, b, j0 + w, k, k1 - k);
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int i = 0; i < R; i++) acc[q][i] = tree32(acc[q][i]);
    if (lane == 0)
#pragma unroll
      for (int q = 0; q < 4; q++)
#pragma unroll
        for (int i = 0; i < R; i++) sums[i * 32 + w + 8 * q] = acc[q][i];
    return;
  }
  /* Thread (x, y): classes y + 8q of column x; its four sums reduce to the
     tree's level of 8, (y + y16) + (y8 + y24), then 8 threads' in shared
     memory to the root. */
  const int x = threadIdx.x % 32, y = threadIdx.x / 32, j = j0 + x;
  T c[4][1][R];
#pragma unroll
  for (int q = 0; q < 4; q++)
#pragma unroll
    for (int i = 0; i < R; i++) c[q][0][i] = T(0);
  if (j < p.n)
#pragma unroll
    for (int q = 0; q < 4; q++)
      for (int k = k0 + 4 * (y + 8 * q); k < k1; k += 128) {
        const int us = min(4, k1 - k);
        T bv[1][4];
        run<T, S>(bv[0], b, p.sb[1], p.sb[2], j, k, us, false);
        dot4<1, T, S>(p, c[q], a, rows, bv, k, us);
      }
#pragma unroll
  for (int i = 0; i < R; i++)
    level8[i][y][x] = (c[0][0][i] + c[2][0][i]) + (c[1][0][i] + c[3][0][i]);
  __syncthreads();
  T acc[R];
  if (y == 0)
#pragma unroll
    for (int i = 0; i < R; i++) {
      const T(&l)[8][32] = level8[i];
      acc[i] = ((l[0][x] + l[4][x]) + (l[2][x] + l[6][x])) +
               ((l[1][x] + l[5][x]) + (l[3][x] + l[7][x]));
    }
  __syncthreads();
  if (y == 0)
#pragma unroll
    for (int i = 0; i < R; i++) sums[i * 32 + x] = acc[i];
}

template <typename T, typename S>
__device__ void skinny_contract(const contract_params &p,
                                T (*level8)[8][32]) {
  constexpr int R = NX_SKINNY_ROWS;
  static_assert(R * 32 <= 256, "a thread holds one of the block's sums");
  /* Block x: the rows from R (x % groups) and the columns from 32
     (x / groups), so that the groups of a column run together and read b
     from L2. */
  const int groups = (p.m + R - 1) / R;
  const int i0 = blockIdx.x % groups * R, j0 = blockIdx.x / groups * 32;
  const int z = blockIdx.z, split = blockIdx.y, splits = p.splits;
  const char *a = (const char *)p.a + (z * p.sa[0] + i0 * p.sa[1]) * sizeof(S);
  const char *b = (const char *)p.b + z * p.sb[0] * sizeof(S);
  /* The range of k: whole 128-element runs, one t of the lanes. */
  const int runs = (p.k + 127) / 128;
  const int k0 = split * runs / splits * 128;
  const int k1 = min(p.k, (split + 1) * runs / splits * 128);
  skinny_sums<T, S>(p, level8, a, min(R, p.m - i0), b, j0, k0, k1);

  /* One of the block's sums per thread: the partials a split sum
     combines. */
  const T *sums = &level8[0][0][0];
  __syncthreads();
  T v[1] = {threadIdx.x < R * 32 ? sums[threadIdx.x] : T(0)};
  if (splits > 1 && !combine(v, (T *)p.partials, p.tickets, splits))
    return;
  const int i = threadIdx.x / 32;
  if (i < R && i0 + i < p.m) store(p, z, i0 + i, j0 + threadIdx.x % 32, v[0]);
}

/* The integer skinny loop of the sum T, for the integer dtype both
   operands are read in. */
template <typename T>
__device__ void skinny_ints(const contract_params &p, T (*level8)[8][32]) {
  switch (p.b_dtype) {
  case NX_INT8: return skinny_contract<T, int8_t>(p, level8);
  case NX_UINT8:
  case NX_BOOL: return skinny_contract<T, uint8_t>(p, level8);
  case NX_INT16: return skinny_contract<T, int16_t>(p, level8);
  case NX_UINT16: return skinny_contract<T, uint16_t>(p, level8);
  case NX_INT32: return skinny_contract<T, int32_t>(p, level8);
  case NX_UINT32: return skinny_contract<T, uint32_t>(p, level8);
  default: return skinny_contract<T, u64>(p, level8);
  }
}

/* The skinny kernel of the accumulator T: a float one reads T; the
   integer one reads the integer dtype both operands are in, its loop
   compiled for each, one shared buffer for all. A 32-bit accumulator
   sums in 32 bits, wrapping: the 64-bit sum's low half, which the store
   would keep. */
template <typename T> __device__ void skinny_kernel(const contract_params &p) {
  __shared__ T level8[NX_SKINNY_ROWS][8][32];
  if constexpr (T(-1) < T(0))
    skinny_contract<T, T>(p, level8);
  else if (p.acc_dtype == NX_INT32 || p.acc_dtype == NX_UINT32)
    skinny_ints<uint32_t>(p, (uint32_t(*)[8][32])level8);
  else
    skinny_ints<T>(p, level8);
}

#endif
