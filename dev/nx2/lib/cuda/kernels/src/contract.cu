/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contract: y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k]
   (kernels.h's contract_params), three bodies, each its own association,
   fixed by the dtypes and the shape and never by a layout:

   - mma: bfloat16 or float16 operands summed in float32, int8 in int32,
     on mma.sync. Each output sums its k in steps of 16 (32 for int8) in
     increasing k, one mma.sync a step; the unit truncates its sum to
     float32 (23 bits kept: each addition errs by at most 2u).

       global --cp.async 16 B (zero past the extents)--> shared ring of
       STAGES k-tiles, 16-byte chunks swizzled --ldmatrix (.trans for an
       operand whose free axis is contiguous)--> fragments --mma.sync-->
       accumulators --> combine (splits) --> + init --> y

   - simt (m > 16): float32 or float64 sums by fma, integer sums wrapping
     in 64 bits (a 32-bit accumulator's wrap to 32 bits at the store: the
     same sum modulo 2^32); each output summed in increasing k.
   - skinny (m <= 16): lane l of 32 sums the runs k0 + 128 t + 4 l + u,
     u < 4, of its range in increasing k, and the 32 lane sums combine by
     the balanced tree (l, l + 16), then (l, l + 8), (l, l + 4), (l, l + 2),
     (l, l + 1).

   In each, [splits] ranges of k (whole k-tiles for mma and simt) combine
   in range order (combine.cuh); init then enters the sum, and the result
   rounds once to y's dtype. An mma operand whose rows are not 16-byte
   vectors, or in a layout its tile has no instance of, is first packed
   into rows that are, k contiguous (pack): the same values in the same
   tiles, so the same bits. A SIMT or float skinny operand of a dtype
   other than its accumulator's is packed into the accumulator's dtype,
   exactly; the integer skinny kernel reads one integer dtype both its
   operands hold, an operand of another packed into it. */

#include "combine.cuh"
#include "kernels.h"
#include "nx_dtype.h"

typedef unsigned long long u64;

/* No device function here is static: nvcc names an out-of-line function of
   internal linkage after a hash of the directory it compiles in, so the
   cubin's bytes would change with the checkout. */

/* Elements */

/* Element [i] of [p], of the dtype [dt], as a double or an int64. */
__device__ __forceinline__ double float_at(const void *p, int64_t i, int dt) {
  switch (dt) {
  case NX_FLOAT64: return ((const double *)p)[i];
  case NX_FLOAT32: return ((const float *)p)[i];
  case NX_FLOAT16: return nx_f16_to_float(((const uint16_t *)p)[i]);
  case NX_BFLOAT16: return nx_bf16_to_float(((const uint16_t *)p)[i]);
  case NX_FLOAT8_E4M3FN: return nx_e4m3fn_to_float(((const uint8_t *)p)[i]);
  case NX_FLOAT8_E5M2: return nx_e5m2_to_float(((const uint8_t *)p)[i]);
  }
  return 0;
}

__device__ __forceinline__ u64 int_at(const void *p, int64_t i, int dt) {
  switch (dt) {
  case NX_INT64:
  case NX_UINT64: return ((const u64 *)p)[i];
  case NX_INT32: return (u64)(int64_t)((const int32_t *)p)[i];
  case NX_UINT32: return ((const uint32_t *)p)[i];
  case NX_INT16: return (u64)(int64_t)((const int16_t *)p)[i];
  case NX_UINT16: return ((const uint16_t *)p)[i];
  case NX_INT8: return (u64)(int64_t)((const int8_t *)p)[i];
  case NX_UINT8:
  case NX_BOOL: return ((const uint8_t *)p)[i];
  }
  return 0;
}

/* The same out of line: the operands of the SIMT kernels, and init, read
   at every element of an unrolled tile. */
__device__ __noinline__ double read_float(const void *p, int64_t i, int dt) {
  return float_at(p, i, dt);
}

__device__ __noinline__ u64 read_int(const void *p, int64_t i, int dt) {
  return int_at(p, i, dt);
}

/* The accumulator types: integers wrap, so they sum unsigned. */
template <typename T> __device__ T read(const void *p, int64_t i, int dt);
template <> __device__ float read(const void *p, int64_t i, int dt) {
  return (float)read_float(p, i, dt);
}
template <> __device__ double read(const void *p, int64_t i, int dt) {
  return read_float(p, i, dt);
}
template <> __device__ u64 read(const void *p, int64_t i, int dt) {
  return read_int(p, i, dt);
}
template <> __device__ int read(const void *p, int64_t i, int dt) {
  return (int)read_int(p, i, dt);
}
template <> __device__ uint32_t read(const void *p, int64_t i, int dt) {
  return (uint32_t)read_int(p, i, dt);
}

/* Stores the bits [c] of an element of the integer, bool or narrow float
   dtype [dt] at element [i] of [p]: their low bits, as wide as [dt]. */
__device__ void put(void *p, int64_t i, int dt, int64_t c) {
  switch (dt) {
  case NX_INT64:
  case NX_UINT64: ((int64_t *)p)[i] = c; break;
  case NX_INT32:
  case NX_UINT32: ((uint32_t *)p)[i] = (uint32_t)c; break;
  case NX_INT16:
  case NX_UINT16:
  case NX_FLOAT16:
  case NX_BFLOAT16: ((uint16_t *)p)[i] = (uint16_t)c; break;
  default: ((uint8_t *)p)[i] = (uint8_t)c;
  }
}

/* The sum [v] at element [i] of y, as a cast from the accumulator writes
   it. A float sum rounds once to y's dtype, nx_dtype.h's conversions
   reaching the narrow floats, the integers and bool. */
__device__ void write(const contract_params &p, int64_t i, double v) {
  switch (p.y_dtype) {
  case NX_FLOAT64: ((double *)p.y)[i] = v; return;
  case NX_FLOAT32: ((float *)p.y)[i] = (float)v; return;
  }
  put(p.y, i, p.y_dtype, nx_double_to_bits(p.y_dtype, v));
}

__device__ void write(const contract_params &p, int64_t i, float v) {
  write(p, i, (double)v);
}

/* An integer sum [v], wrapped to the accumulator and widened by its sign:
   wrapped to an integer y, rounded once to a float y, nonzero for bool. A
   32-bit accumulator's sum, summed in 64 bits, wraps to 32 here: the same
   sum, modulo 2^32. */
__device__ void write(const contract_params &p, int64_t i, u64 v) {
  if (p.acc_dtype == NX_INT32) v = (u64)(int64_t)(int32_t)v;
  if (p.acc_dtype == NX_UINT32) v = (uint32_t)v;
  const bool s = p.acc_dtype == NX_INT32 || p.acc_dtype == NX_INT64;
  switch (p.y_dtype) {
  case NX_FLOAT64:
    ((double *)p.y)[i] = s ? (double)(int64_t)v : (double)v;
    return;
  case NX_FLOAT32:
    ((float *)p.y)[i] = s ? (float)(int64_t)v : (float)v;
    return;
  case NX_FLOAT16:
  case NX_BFLOAT16:
  case NX_FLOAT8_E4M3FN:
  case NX_FLOAT8_E5M2: {
    const double odd = s ? nx_i64_odd((int64_t)v) : nx_u64_odd(v);
    put(p.y, i, p.y_dtype, nx_double_to_bits(p.y_dtype, odd));
    return;
  }
  case NX_BOOL: put(p.y, i, NX_BOOL, v != 0); return;
  }
  put(p.y, i, p.y_dtype, (int64_t)v);
}

__device__ void write(const contract_params &p, int64_t i, int v) {
  write(p, i, (u64)(int64_t)v);
}

__device__ void write(const contract_params &p, int64_t i, uint32_t v) {
  write(p, i, (u64)v);
}

/* y's element (z, i, j), if inside: init added, then rounded once. Out of
   line: kernels call it at every element of an unrolled tile. */
template <typename T>
__device__ __noinline__ void store(const contract_params &p, int z,
                                          int i, int j, T v) {
  if (i >= p.m || j >= p.n) return;
  if (p.init)
    v += read<T>(p.init, z * p.si[0] + i * p.si[1] + j * p.si[2],
                 p.init_dtype);
  write(p, z * p.sy[0] + i * p.sy[1] + j * p.sy[2], v);
}

/* Two outputs (z, i, j) and (z, i, j + 1). With no init, y's pairs whole
   and aligned along a contiguous j, and y of the accumulator's dtype, or
   bfloat16 or float16 from float32 (NX_CONTRACT_Y_WHOLE), one
   store of both; through store otherwise. The conversions round to
   nearest even, as nx_dtype.h's: only a NaN's payload may differ. */
__device__ void store_pair(const contract_params &p, int z, int i,
                                  int j, float v0, float v1) {
  if ((p.aligned & NX_CONTRACT_Y_WHOLE) && i < p.m && j + 1 < p.n) {
    const int64_t o = z * p.sy[0] + i * p.sy[1] + j;
    uint32_t h;
    switch (p.y_dtype) {
    case NX_FLOAT32:
      *(float2 *)((float *)p.y + o) = make_float2(v0, v1);
      return;
    case NX_BFLOAT16:
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(v1), "f"(v0));
      *(uint32_t *)((uint16_t *)p.y + o) = h;
      return;
    case NX_FLOAT16:
      asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(v1), "f"(v0));
      *(uint32_t *)((uint16_t *)p.y + o) = h;
      return;
    }
  }
  store(p, z, i, j, v0);
  store(p, z, i, j + 1, v1);
}

template <typename T>
__device__ void store_pair(const contract_params &p, int z, int i,
                                  int j, T v0, T v1) {
  if ((p.aligned & NX_CONTRACT_Y_WHOLE) && i < p.m && j + 1 < p.n) {
    T *y = (T *)p.y + z * p.sy[0] + i * p.sy[1] + j;
    y[0] = v0, y[1] = v1;
    return;
  }
  store(p, z, i, j, v0);
  store(p, z, i, j + 1, v1);
}

/* Eight outputs (z, i, j + e), e < 8, of [v]. With no init, y's j
   contiguous, its rows aligned on 16 bytes, and y of the accumulator's
   dtype, or bfloat16 or float16 from float32 (NX_CONTRACT_Y_WHOLE): one
   16-byte store, rounding to nearest even as nx_dtype.h's encoders do
   (only a NaN's payload may differ). Through store otherwise. */
template <typename T>
__device__ void store8(const contract_params &p, int z, int i, int j,
                              const T *v) {
  if ((p.aligned & NX_CONTRACT_Y_WHOLE) && i < p.m && j + 8 <= p.n) {
    const int64_t o = z * p.sy[0] + i * p.sy[1] + j;
    if (sizeof(T) == 4 && (p.y_dtype == NX_BFLOAT16 || p.y_dtype == NX_FLOAT16)) {
      uint32_t h[4];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const float lo = (float)v[2 * e], hi = (float)v[2 * e + 1];
        if (p.y_dtype == NX_BFLOAT16)
          asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(h[e]) : "f"(hi), "f"(lo));
        else
          asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h[e]) : "f"(hi), "f"(lo));
      }
      *(uint4 *)((uint16_t *)p.y + o) = make_uint4(h[0], h[1], h[2], h[3]);
      return;
    }
#pragma unroll
    for (int e = 0; e < 8 * (int)sizeof(T) / 16; e++)
      ((uint4 *)((T *)p.y + o))[e] = ((const uint4 *)v)[e];
    return;
  }
  for (int e = 0; e < 8; e++) store(p, z, i, j + e, v[e]);
}

/* The block's BM x BN outputs, held in the mma accumulator layout,
   stored through shared memory ([CAP] bytes at [smem], the pipeline's) in
   passes of whole warp rows: a pass's warps write their accumulators,
   then every thread stores 8 outputs at a time. No accumulator is live
   across a call, and y's stores are whole rows. */
template <typename Acc, int BM, int BN, int WM, int WN, int THREADS, int CAP>
__device__ void epilogue(const contract_params &p, uint8_t *smem,
                                const Acc (&acc)[WM / 16][WN / 8][4], int z,
                                int m0, int n0, int wm0, int wn0) {
  constexpr int LD = BN + 8;
  constexpr int ROWS = CAP / (LD * (int)sizeof(Acc)) / WM * WM;
  static_assert(ROWS >= WM, "a pass holds a warp row");
  Acc *st = (Acc *)smem;
  const int tid = threadIdx.x, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int base = 0; base < BM; base += ROWS) {
    __syncthreads();
    if (wm0 >= base && wm0 < base + ROWS)
#pragma unroll
      for (int i = 0; i < WM / 16; i++)
#pragma unroll
        for (int h = 0; h < 2; h++)
#pragma unroll
          for (int j = 0; j < WN / 8; j++) {
            Acc *d = st + (wm0 - base + i * 16 + g + h * 8) * LD + wn0 + j * 8 + t4 * 2;
            d[0] = acc[i][j][2 * h], d[1] = acc[i][j][2 * h + 1];
          }
    __syncthreads();
    const int rows = min(ROWS, BM - base);
    for (int e = tid; e < rows * (BN / 8); e += THREADS) {
      const int r = e / (BN / 8), c = e % (BN / 8) * 8;
      store8(p, z, m0 + base + r, n0 + c, st + r * LD + c);
    }
  }
}

/* The mma kernels */

__device__ void mma(float *c, const uint32_t *a, const uint32_t *b,
                           int kind) {
  if (kind == 0)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ void mma(int *c, const uint32_t *a, const uint32_t *b,
                           int) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ void ldsm(uint32_t *r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ void ldsm_trans(uint32_t *r, uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

/* [bytes] of the 16 at [src] to shared [dst], the rest zero. */
__device__ void cp_async(uint32_t dst, const void *src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(bytes));
}

__device__ void cp_commit(void) {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ void cp_wait(void) {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/* The chunk a row's chunk [c] is stored at, in a tile whose rows hold [C]
   16-byte chunks: the 8 rows an ldmatrix phase reads then hit 8 distinct
   bank groups. */
__device__ int swz(int r, int c, int C) {
  return C >= 8 ? c ^ (r & 7) : c ^ ((r / (8 / C)) % C);
}

/* One operand of a block: its rows (m for a, n for b) from [r0], [R] of
   them, k-tiles of [BK] elements of [ES] bytes, loaded in 16-byte chunks
   along the contiguous axis: a row-major tile ([R] rows of k) if [T] is
   false, k rows of [R] elements if [T]. Its rows are whole vectors: the
   plan packs an operand whose are not.

   A thread loads the chunk at its column of every RS-th tile row, the
   same each k-tile: one address, moved along the rows and along k, and
   the bounds of its chunks, kept from make. */
template <int R, int BK, int ES, bool T, int THREADS> struct Operand {
  static constexpr int CHE = 16 / ES;               /* elements per chunk */
  static constexpr int C = T ? R / CHE : BK / CHE;  /* chunks per tile row */
  static constexpr int ROWS = T ? BK : R;           /* tile rows */
  static constexpr int BYTES = R * BK * ES;
  static constexpr int RS = THREADS / C;            /* rows a pass loads */
  static_assert(THREADS % C == 0, "a thread keeps its tile column");

  const char *first; /* this thread's first chunk at k-tile 0 */
  int64_t step;      /* bytes from a pass's chunk to the next pass's */
  int left;          /* rows from its first to the extent; if T, its
                        chunk's elements inside the rows */
  int kofs;          /* its k in a k-tile (its first, if T) */

  __device__ void make(const char *base, int64_t lead, int rows, int r0,
                       int tid) {
    const int tc = tid % C, tr = tid / C;
    step = RS * lead;
    if (T) {
      first = base + tr * lead + (r0 + tc * CHE) * ES;
      left = min(CHE, rows - (r0 + tc * CHE));
      kofs = tr;
    } else {
      first = base + (r0 + tr) * lead + tc * CHE * ES;
      left = rows - (r0 + tr);
      kofs = tc * CHE;
    }
  }

  /* The tile of k-tile [kt] of an operand of [k] k into the shared buffer
     at [dst], from thread [tid]: the chunks past the extents are zero. */
  __device__ void load(uint32_t dst, int kt, int k, int tid) const {
    static_assert(!T || BK % RS == 0, "a k-tile is whole passes");
    const int tc = tid % C, tr = tid / C;
    const char *src = first + (T ? kt * (BK / RS) * step : (int64_t)kt * BK * ES);
#pragma unroll
    for (int it = 0; it < (ROWS + RS - 1) / RS; it++) {
      const int row = tr + it * RS;
      if (ROWS % RS && row >= ROWS) break;
      const int kk = kt * BK + kofs + (T ? it * RS : 0);
      const int inside = T ? (kk < k ? left : 0)
                           : (it * RS < left ? min(CHE, k - kk) : 0);
      cp_async(dst + (row * C + swz(row, tc, C)) * 16,
               inside > 0 ? src + it * step : first, inside > 0 ? inside * ES : 0);
    }
  }
};

/* The accumulator of an mma kind: int32 for int8, float32 otherwise. */
template <int KIND> struct Mma_acc { typedef float type; };
template <> struct Mma_acc<2> { typedef int type; };

template <int KIND, bool AM, bool BN_, typename Tile>
__device__ void mma_contract(const contract_params &p) {
  typedef typename Mma_acc<KIND>::type Acc;
  constexpr int BM = Tile::BM, BN = Tile::BN, BKB = Tile::BKB, WM = Tile::WM,
                WN = Tile::WN, STAGES = Tile::STAGES;
  constexpr int ES = KIND == 2 ? 1 : 2, BK = BKB / ES;
  constexpr int WARPS_N = BN / WN, THREADS = (BM / WM) * WARPS_N * 32;
  constexpr int MI = WM / 16, NI = WN / 8, KS = BKB / 32;
  typedef Operand<BM, BK, ES, AM, THREADS> A;
  typedef Operand<BN, BK, ES, BN_, THREADS> B;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sa = __cvta_generic_to_shared(smem);
  const uint32_t sb = sa + STAGES * A::BYTES;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm0 = (warp / WARPS_N) * WM, wn0 = (warp % WARPS_N) * WN;

  /* Grouped rasterisation: 8 row-tiles sweep the column tiles together, so
     that a column tile of b is read from L2 by the 8 blocks in flight. */
  const int tiles_m = (p.m + BM - 1) / BM, tiles_n = (p.n + BN - 1) / BN;
  const int pid = blockIdx.x, group = 8 * tiles_n, first = (pid / group) * 8;
  const int gsz = min(tiles_m - first, 8);
  const int tm = first + (pid % group) % gsz, tn = (pid % group) / gsz;
  const int m0 = tm * BM, n0 = tn * BN, z = blockIdx.z;

  const int kts = (p.k + BK - 1) / BK, split = blockIdx.y;
  const int kt0 = split * kts / p.splits;
  const int KT = (split + 1) * kts / p.splits - kt0;

  A a;
  B b;
  a.make((const char *)p.a + z * p.sa[0] * ES, (AM ? p.sa[2] : p.sa[1]) * ES,
         p.m, m0, tid);
  b.make((const char *)p.b + z * p.sb[0] * ES, (BN_ ? p.sb[2] : p.sb[1]) * ES,
         p.n, n0, tid);

  auto load = [&](int slot, int kt) {
    a.load(sa + slot * A::BYTES, kt0 + kt, p.k, tid);
    b.load(sb + slot * B::BYTES, kt0 + kt, p.k, tid);
  };

  /* The fragments of k-step [ks] of a slot. A row-major tile gives them by
     ldmatrix; a transposed one by ldmatrix.trans of the 8 x 8 blocks the
     fragment's registers hold, in their order. */
  auto frag = [&](uint32_t (&fa)[MI][4], uint32_t (&fb)[NI][2], int slot,
                  int ks) {
    const uint32_t ta = sa + slot * A::BYTES, tb = sb + slot * B::BYTES;
    const int q = lane >> 3, r8 = lane & 7;
#pragma unroll
    for (int i = 0; i < MI; i++) {
      if (AM) {
        const int r = ks * 16 + r8 + 8 * (q >> 1), c = (wm0 + i * 16) / 8 + (q & 1);
        ldsm_trans(fa[i], ta + (r * A::C + swz(r, c, A::C)) * 16);
      } else {
        const int r = wm0 + i * 16 + (lane & 15), c = 2 * ks + (lane >> 4);
        ldsm(fa[i], ta + (r * A::C + swz(r, c, A::C)) * 16);
      }
    }
#pragma unroll
    for (int j = 0; j < NI / 2; j++) {
      uint32_t t[4];
      if (BN_) {
        const int r = ks * 16 + r8 + 8 * (q & 1), c = (wn0 + j * 16) / 8 + (q >> 1);
        ldsm_trans(t, tb + (r * B::C + swz(r, c, B::C)) * 16);
      } else {
        const int r = wn0 + j * 16 + r8 + ((lane >> 4) << 3);
        const int c = 2 * ks + ((lane >> 3) & 1);
        ldsm(t, tb + (r * B::C + swz(r, c, B::C)) * 16);
      }
      fb[2 * j][0] = t[0], fb[2 * j][1] = t[1];
      fb[2 * j + 1][0] = t[2], fb[2 * j + 1][1] = t[3];
    }
  };

  Acc acc[MI][NI][4] = {};
  uint32_t fa[2][MI][4], fb[2][NI][2];

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) load(s, s);
    cp_commit();
  }
  cp_wait<STAGES - 2>();
  __syncthreads();
  frag(fa[0], fb[0], 0, 0);

  /* Software pipeline: k-step ks multiplies the fragments loaded during
     ks - 1 while ldmatrix fetches ks + 1; the barrier for the next stage
     sits before the last k-step, so the next tile's first fragments load
     under its mma. The tile STAGES - 1 ahead refills the slot tile kt - 1
     freed: a's copies at the first k-step, b's at the last, which commits
     them as one group, so that the copies spread over the k-tile. */
  for (int kt = 0; kt < KT; kt++) {
    const int slot = kt % STAGES;
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
      const bool wrap = ks == KS - 1;
      if (wrap) {
        /* Tile kt + 1's group, which has STAGES - 3 committed after it:
           tile kt + STAGES - 1's commits after this wait. */
        cp_wait<STAGES - 3>();
        __syncthreads();
      }
      if (!wrap || kt + 1 < KT)
        frag(fa[(ks + 1) & 1], fb[(ks + 1) & 1],
             wrap ? (kt + 1) % STAGES : slot, (ks + 1) % KS);
      const int next = kt + STAGES - 1;
      if (ks == 0 && next < KT) a.load(sa + next % STAGES * A::BYTES, kt0 + next, p.k, tid);
      if (ks == KS - 1) {
        if (next < KT) b.load(sb + next % STAGES * B::BYTES, kt0 + next, p.k, tid);
        cp_commit();
      }
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NI; j++)
          mma(acc[i][j], fa[ks & 1][i], fb[ks & 1][j], KIND);
    }
  }
  cp_wait<0>();

  if (p.splits > 1) {
    Acc(&flat)[MI * NI * 4] = *reinterpret_cast<Acc(*)[MI * NI * 4]>(acc);
    const int tile = z * tiles_m * tiles_n + tm * tiles_n + tn;
    if (!combine(flat, (Acc *)p.partials + (size_t)tile * p.splits * MI * NI * 4 * THREADS,
                 p.tickets + tile, p.splits, split))
      return;
  }

  epilogue<Acc, BM, BN, WM, WN, THREADS, STAGES * (BM + BN) * BKB>(
      p, smem, acc, z, m0, n0, wm0, wn0);
}

/* The SIMT kernels */

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

__device__ float fma_(float a, float b, float c) { return fmaf(a, b, c); }
__device__ double fma_(double a, double b, double c) { return fma(a, b, c); }
__device__ uint32_t fma_(uint32_t a, uint32_t b, uint32_t c) { return a * b + c; }
__device__ u64 fma_(u64 a, u64 b, u64 c) { return a * b + c; }

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
    const int tile = z * gridDim.x + blockIdx.x;
    if (!combine(flat, (T *)p.partials + (size_t)tile * p.splits * TM * TM * 256,
                 p.tickets + tile, p.splits, split))
      return;
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
template <typename T> __device__ T tree32(T v) {
#pragma unroll
  for (int d = 16; d > 0; d /= 2) v += __shfl_xor_sync(0xFFFFFFFFu, v, d);
  return v;
}

/* [s] as the accumulator T: an integer widened by its own sign. */
template <typename T, typename S> __device__ T widen(S s) {
  if constexpr (T(-1) > T(0) && S(-1) < S(0))
    return (T)(int64_t)s;
  else
    return (T)s;
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
  if (splits > 1) {
    const int tile = z * gridDim.x + blockIdx.x;
    if (!combine(v, (T *)p.partials + (size_t)tile * splits * blockDim.x,
                 p.tickets + tile, splits, split))
      return;
  }
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

#define KIND_bf16 0
#define KIND_f16 1
#define KIND_s8 2
#define KIND_any 3
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
template <int KIND, bool AM, bool BN_, typename Tile>
__device__ void mma_kernel(const contract_params &p) {
  if constexpr (KIND != KIND_any)
    mma_contract<KIND, AM, BN_, Tile>(p);
  else if (p.a_dtype == NX_INT8)
    mma_contract<KIND_s8, AM, BN_, Tile>(p);
  else if (p.a_dtype == NX_FLOAT16)
    mma_contract<KIND_f16, AM, BN_, Tile>(p);
  else
    mma_contract<KIND_bf16, AM, BN_, Tile>(p);
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
