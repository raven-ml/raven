/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contract: y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k]
   (kernels.h's contract_params), by three bodies, each its own
   association, fixed by the dtypes and the shape and never by a layout:
   mma (mma.cuh), SIMT and skinny (below), over the elements of
   elements.cuh. In each, [splits] ranges of k (whole k-tiles for mma and
   SIMT) combine in range order (combine.cuh); init then enters the sum,
   and the result rounds once to y's dtype. Each body reads its operands in
   the layout and dtype the plan packs them into (the pack, below): the
   same values in the same tiles, so the same bits. */

#include "combine.cuh"
#include "kernels.h"
#include "nx_dtype.h"

typedef unsigned long long u64;

/* No device function here or in the headers is static: nvcc names an
   out-of-line function of internal linkage after a hash of the directory
   it compiles in, so the cubin's bytes would change with the checkout. */

#include "elements.cuh"
#include "mma.cuh"

/* SIMT */

/* The SIMT kernels (m > 16): float32 or float64 sums by fma, integer sums
   wrapping; each output summed in increasing k. */

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

/* Skinny */

/* The skinny kernels (m <= 16): lane l of 32 sums, for each t, the run of
   four k0 + 128 t + 4 l + u, u < 4, in increasing k, and the 32 lane sums
   combine by the balanced tree (l, l + 16), then (l, l + 8), (l, l + 4),
   (l, l + 2), (l, l + 1). A block computes NX_SKINNY_ROWS rows of 32
   columns: few rows a thread keep its registers few, so that an SM holds
   enough warps to cover a load's latency. When b's k axis is contiguous,
   warp w takes the columns w + 8 q, q < 4, lanes across k, a run one
   vector, and loads each of a's runs once for its four columns; otherwise
   thread (x, y) sums the lane classes y, y + 8, y + 16, y + 24 of column
   x, reading rows of b across x. */

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

/* Packing */

/* The pack: an operand copied into the layout and dtype a body reads, each
   value converted exactly. */

/* An element of a packed operand: [dt]'s value as [out]'s, exactly, as
   [out]'s bits. */
__device__ __forceinline__ u64 packed(const void *src, int64_t x, int dt,
                                      int out) {
  switch (out) {
  case NX_BFLOAT16: return nx_float_to_bf16((float)float_at(src, x, dt));
  case NX_FLOAT16: return nx_float_to_f16((float)float_at(src, x, dt));
  case NX_FLOAT32: return nx_float_bits((float)float_at(src, x, dt));
  case NX_FLOAT64: return nx_double_bits(float_at(src, x, dt));
  }
  return int_at(src, x, dt);
}

/* The bytes of an element of the dtype [dt], at compile time. */
__host__ __device__ constexpr int dt_bytes(int dt) {
  return dt == NX_FLOAT64 || dt == NX_INT64 || dt == NX_UINT64   ? 8
         : dt == NX_FLOAT32 || dt == NX_INT32 || dt == NX_UINT32 ? 4
         : dt == NX_FLOAT16 || dt == NX_BFLOAT16 || dt == NX_INT16 ||
                 dt == NX_UINT16
             ? 2
             : 1;
}

/* N bytes as one load: 2, 4, 8 or 16. */
template <int N> struct Raw;
template <> struct Raw<2> { typedef uint16_t type; };
template <> struct Raw<4> { typedef uint32_t type; };
template <> struct Raw<8> { typedef uint2 type; };
template <> struct Raw<16> { typedef uint4 type; };

/* The unsigned type of N bytes. */
template <int N> struct Bits;
template <> struct Bits<1> { typedef uint8_t type; };
template <> struct Bits<2> { typedef uint16_t type; };
template <> struct Bits<4> { typedef uint32_t type; };
template <> struct Bits<8> { typedef u64 type; };

/* The packed operand's 16-byte vectors, one a thread, counted in [I], of
   [OB]-byte elements: from the dtype [DT] to [OUT], each compiled for its
   pair, or from [p.dtype] to [p.out] where they are -1. A vector is a run
   of one row: its indices are divided once for its elements, and a run of
   the source whose k is contiguous and aligned is one load. */
template <int DT, int OUT, int OB, typename I>
__device__ void pack_vectors(const pack_params &p) {
  typedef typename Bits<OB>::type Out;
  constexpr int SB = DT < 0 ? 0 : dt_bytes(DT);
  const int dt = DT < 0 ? p.dtype : DT, out = OUT < 0 ? p.out : OUT;
  constexpr int PER = 16 / OB;
  const char *src = (const char *)p.src;
  const I per_row = (I)(p.lead / PER), n = (I)p.batch * p.rows * per_row;
  /* A float8 source converts through a table of its 256 codes' values,
     made by the block from the same conversion. */
  constexpr bool TABLE = DT == NX_FLOAT8_E4M3FN || DT == NX_FLOAT8_E5M2;
  __shared__ Out table[TABLE ? 256 : 1];
  if (TABLE) {
    for (int c = threadIdx.x; c < 256; c += blockDim.x) {
      const uint8_t code = (uint8_t)c;
      table[c] = (Out)packed(&code, 0, dt, out);
    }
    __syncthreads();
  }
  auto conv = [&](const void *at, int64_t x) {
    return TABLE ? table[((const uint8_t *)at)[x]] : (Out)packed(at, x, dt, out);
  };
  for (I v = blockIdx.x * (I)blockDim.x + threadIdx.x; v < n;
       v += (I)gridDim.x * blockDim.x) {
    const I row = v / per_row, z = row / (I)p.rows;
    const int64_t q0 = (int64_t)(v - row * per_row) * PER;
    const int64_t at = (int64_t)z * p.s[0] +
                       (int64_t)(row - z * p.rows) * p.s[1] + q0 * p.s[2];
    union {
      Out e[PER];
      uint4 all;
    } o;
    if (SB > 0 && p.s[2] == 1 && q0 + PER <= p.k &&
        (uintptr_t)(src + at * SB) % (PER * SB) == 0) {
      typedef typename Raw<(SB > 0 ? PER * SB : 16)>::type R;
      union {
        R all;
        uint8_t b[sizeof(R)];
      } in = {*(const R *)(src + at * SB)};
#pragma unroll
      for (int e = 0; e < PER; e++) o.e[e] = conv(in.b, e);
    } else
#pragma unroll
      for (int e = 0; e < PER; e++)
        o.e[e] = q0 + e < p.k ? conv(src, at + e * p.s[2]) : Out(0);
    ((uint4 *)p.dst)[v] = o.all;
  }
}

/* X(dtype, out): the packs the plan makes, each compiled for its pair; any
   other runs the general loop. */
#define PACKS(X)                                                               \
  X(NX_INT8, NX_INT8)                                                          \
  X(NX_BFLOAT16, NX_BFLOAT16)                                                  \
  X(NX_FLOAT8_E4M3FN, NX_BFLOAT16)                                             \
  X(NX_FLOAT8_E5M2, NX_BFLOAT16)                                               \
  X(NX_FLOAT16, NX_FLOAT16)                                                    \
  X(NX_FLOAT16, NX_FLOAT32)                                                    \
  X(NX_BFLOAT16, NX_FLOAT32)                                                   \
  X(NX_FLOAT8_E4M3FN, NX_FLOAT32)                                              \
  X(NX_FLOAT8_E5M2, NX_FLOAT32)                                                \
  X(NX_FLOAT32, NX_FLOAT32)                                                    \
  X(NX_FLOAT16, NX_FLOAT64)                                                    \
  X(NX_BFLOAT16, NX_FLOAT64)                                                   \
  X(NX_FLOAT8_E4M3FN, NX_FLOAT64)                                              \
  X(NX_FLOAT8_E5M2, NX_FLOAT64)                                                \
  X(NX_FLOAT32, NX_FLOAT64)                                                    \
  X(NX_FLOAT64, NX_FLOAT64)                                                    \
  X(NX_BOOL, NX_INT64)                                                         \
  X(NX_INT8, NX_INT64)                                                         \
  X(NX_UINT8, NX_INT64)                                                        \
  X(NX_INT16, NX_INT64)                                                        \
  X(NX_UINT16, NX_INT64)                                                       \
  X(NX_INT32, NX_INT64)                                                        \
  X(NX_UINT32, NX_INT64)                                                       \
  X(NX_INT64, NX_INT64)                                                        \
  X(NX_UINT64, NX_INT64)

__device__ void pack_rows(const pack_params &p) {
  const u64 n = (u64)p.batch * p.rows * p.lead * p.bytes / 16;
  if (n <= UINT32_MAX) {
#define PACK_CASE(d, o)                                                        \
  if (p.dtype == d && p.out == o)                                              \
    return pack_vectors<d, o, dt_bytes(o), uint32_t>(p);
    PACKS(PACK_CASE)
#undef PACK_CASE
  }
  switch (p.bytes) {
  case 1: return pack_vectors<-1, -1, 1, u64>(p);
  case 2: return pack_vectors<-1, -1, 2, u64>(p);
  case 4: return pack_vectors<-1, -1, 4, u64>(p);
  default: return pack_vectors<-1, -1, 8, u64>(p);
  }
}

/* Instances */

extern "C" __global__ void zero_u32(const __grid_constant__ zero_params p) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < p.n;
       i += (u64)gridDim.x * blockDim.x)
    p.p[i] = 0;
}

#define TRANSPOSED_k false
#define TRANSPOSED_m true
#define TRANSPOSED_n true
#define ACC_f32 float
#define ACC_f64 double
#define ACC_i64 u64

#define TILE(name, bm, bn, bkb, wm, wn, s)                                     \
  struct name {                                                                \
    enum { BM = bm, BN = bn, BKB = bkb, WM = wm, WN = wn, STAGES = s };        \
    enum { THREADS = (bm / wm) * (bn / wn) * 32 };                             \
  };
NX_CUDA_TILES(TILE)
#undef TILE

/* The mma kernel of [KIND]; for KIND_any, each kind's by a's dtype, each
   loop compiled apart so that none keeps another's registers live. */
template <int KIND, bool A_T, bool B_T, typename Tile>
__device__ void mma_kernel(const contract_params &p) {
  if constexpr (KIND != KIND_any)
    mma_contract<KIND, A_T, B_T, Tile>(p);
  else if (p.a_dtype == NX_INT8)
    mma_contract<KIND_s8, A_T, B_T, Tile>(p);
  else if (p.a_dtype == NX_FLOAT16)
    mma_contract<KIND_f16, A_T, B_T, Tile>(p);
  else
    mma_contract<KIND_bf16, A_T, B_T, Tile>(p);
}

#define DEFINE(name, FAMILY, ...) FAMILY(name, __VA_ARGS__)
#define ZERO(name, ...)
#define PACK(name, ...)                                                        \
  extern "C" __global__ void name(const __grid_constant__ pack_params p) {     \
    pack_rows(p);                                                              \
  }
#define MMA(name, kind, a, b, tile)                                            \
  extern "C" __global__ void __launch_bounds__(tile::THREADS)                  \
      name(const __grid_constant__ contract_params p) {                        \
    mma_kernel<KIND_##kind, TRANSPOSED_##a, TRANSPOSED_##b, tile>(p);          \
  }
#define SIMT(name, acc, bm)                                                    \
  extern "C" __global__ void __launch_bounds__(256)                            \
      name(const __grid_constant__ contract_params p) {                        \
    simt_contract<ACC_##acc, bm>(p);                                           \
  }
#define SKINNY(name, acc)                                                      \
  extern "C" __global__ void __launch_bounds__(256)                            \
      name(const __grid_constant__ contract_params p) {                        \
    skinny_kernel<ACC_##acc>(p);                                               \
  }
NX_CUDA_KERNELS(DEFINE)
