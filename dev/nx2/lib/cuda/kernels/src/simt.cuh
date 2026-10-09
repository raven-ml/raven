/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The SIMT kernels (m > 16): float32 or float64 sums by fma, integer sums
   wrapping; each output summed in increasing k. Part of contract.cu's translation unit,
   after elements.cuh. */

#ifndef NX_CUDA_SIMT_CUH
#define NX_CUDA_SIMT_CUH

/* A BM x BM tile of outputs per block of 256 threads (16 x 16), k-tiles of
   64 bytes of k (16 floats), each thread TM x TM outputs in two halves
   BM / 2 apart each way. Both operands are stored k-major in shared
   memory, the next k-tile loaded into registers while the current one
   computes. A thread loads E = BM * BK / 256 elements of each operand per
   k-tile: E consecutive k of a row, or E consecutive rows of a k, whichever
   axis is contiguous, as vectors where it can. */
template <typename T, int BM, int BK> struct Simt_operand {
  static constexpr int E = BM * BK / 256;
  const char *base;
  int64_t s_row, s_k;
  int rows, k, r0;
  bool rows_contiguous, vectors;

  __device__ void load(T (&v)[E], int k0, int tid) const {
    const int row = rows_contiguous ? r0 + tid % (BM / E) * E : r0 + tid / (BK / E);
    const int kk = rows_contiguous ? k0 + tid / (BM / E) : k0 + tid % (BK / E) * E;
    if (vectors &&
        (rows_contiguous ? kk < k && row + E - 1 < rows : row < rows && kk + E - 1 < k)) {
      const char *src = base + (row * s_row + kk * s_k) * sizeof(T);
      if (E * sizeof(T) == 8)
        *(uint2 *)v = *(const uint2 *)src;
      else
#pragma unroll
        for (int e = 0; e < E * (int)sizeof(T) / 16; e++)
          ((uint4 *)v)[e] = ((const uint4 *)src)[e];
      return;
    }
#pragma unroll
    for (int e = 0; e < E; e++) {
      const int r = rows_contiguous ? row + e : row, q = rows_contiguous ? kk : kk + e;
      v[e] = r < rows && q < k ? ((const T *)base)[r * s_row + q * s_k] : T(0);
    }
  }

  /* The loaded elements into the k-major tile [s]. */
  __device__ void stash(T (*s)[BM], const T (&v)[E], int tid) const {
#pragma unroll
    for (int e = 0; e < E; e++)
      if (rows_contiguous)
        s[tid / (BM / E)][tid % (BM / E) * E + e] = v[e];
      else
        s[tid % (BK / E) * E + e][tid / (BK / E)] = v[e];
  }
};

template <typename T, int BM>
__device__ void simt_contract(const contract_params &p) {
  constexpr int BK = 64 / sizeof(T), TM = BM / 16, H = TM / 2, E = BM * BK / 256;
  __shared__ __align__(16) T as[2][BK][BM], bs[2][BK][BM];
  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16, z = blockIdx.z;
  /* Grouped rasterisation, as the mma kernels'. */
  const int tiles_m = (p.m + BM - 1) / BM, tiles_n = (p.n + BM - 1) / BM;
  const int pid = blockIdx.x, group = 8 * tiles_n, first = (pid / group) * 8;
  const int gsz = min(tiles_m - first, 8);
  const int tm = first + (pid % group) % gsz, tn = (pid % group) / gsz;
  const int m0 = tm * BM, n0 = tn * BM;
  const int kts = (p.k + BK - 1) / BK, split = blockIdx.y;
  const int kt0 = split * kts / p.splits, kt1 = (split + 1) * kts / p.splits;

  const Simt_operand<T, BM, BK> a = {(const char *)p.a + z * p.sa[0] * sizeof(T),
                                     p.sa[1], p.sa[2], p.m, p.k, m0,
                                     p.sa[1] == 1 && p.sa[2] != 1,
                                     (p.aligned & NX_CONTRACT_A_VECTORS) != 0};
  const Simt_operand<T, BM, BK> b = {(const char *)p.b + z * p.sb[0] * sizeof(T),
                                     p.sb[1], p.sb[2], p.n, p.k, n0,
                                     p.sb[1] == 1 && p.sb[2] != 1,
                                     (p.aligned & NX_CONTRACT_B_VECTORS) != 0};

  T acc[TM][TM];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TM; j++) acc[i][j] = T(0);
  T ra[E], rb[E];
  if (kt0 < kt1) {
    a.load(ra, kt0 * BK, tid);
    b.load(rb, kt0 * BK, tid);
    a.stash(as[0], ra, tid);
    b.stash(bs[0], rb, tid);
  }
  __syncthreads();

  for (int kt = kt0, buf = 0; kt < kt1; kt++, buf ^= 1) {
    const bool more = kt + 1 < kt1;
    if (more) {
      a.load(ra, (kt + 1) * BK, tid);
      b.load(rb, (kt + 1) * BK, tid);
    }
#pragma unroll
    for (int k = 0; k < BK; k++) {
      T va[TM], vb[TM];
#pragma unroll
      for (int e = 0; e < H; e++) {
        va[e] = as[buf][k][ty * H + e], va[H + e] = as[buf][k][BM / 2 + ty * H + e];
        vb[e] = bs[buf][k][tx * H + e], vb[H + e] = bs[buf][k][BM / 2 + tx * H + e];
      }
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TM; j++) acc[i][j] = fma_(va[i], vb[j], acc[i][j]);
    }
    if (more) {
      a.stash(as[buf ^ 1], ra, tid);
      b.stash(bs[buf ^ 1], rb, tid);
    }
    __syncthreads();
  }

  if (p.splits > 1) {
    T(&flat)[TM * TM] = *reinterpret_cast<T(*)[TM * TM]>(acc);
    if (!combine(flat, (T *)p.partials, p.tickets, p.splits)) return;
  }
  /* A thread's columns in each half are consecutive: in pairs, from two
     on. */
#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int row = m0 + (i < H ? ty * H + i : BM / 2 + ty * H + i - H);
#pragma unroll
    for (int j = 0; j < TM; j += H > 1 ? 2 : 1) {
      const int col = n0 + (j < H ? tx * H + j : BM / 2 + tx * H + j - H);
      if (H > 1) store_pair(p, z, row, col, acc[i][j], acc[i][j + 1]);
      else store(p, z, row, col, acc[i][j]);
    }
  }
}

#endif
