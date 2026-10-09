/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contract: y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k]
   (kernels.h's contract_params), three bodies, each its own association,
   fixed by the dtypes and the shape and never by a layout:

   - wmma: bfloat16 or float16 operands summed in float32, int8 in int32,
     on gfx12's WMMA 16 x 16 x 16. Each output sums its k in steps of 16
     in increasing k, one WMMA a step.

       global --16-byte loads (zero past the extents), two k-tiles ahead-->
       registers --> one of two local data share k-tiles --8 k a lane-->
       fragments --WMMA-->
       accumulators --> combine (splits) --> + init --> y

   - simt (m > 16): float32 or float64 sums by fma, integer sums wrapping
     in 64 bits (a 32-bit accumulator's wrap to 32 bits at the store: the
     same sum modulo 2^32); each output summed in increasing k.
   - skinny (m <= 16): lane l of 32 sums the runs k0 + 128 t + 4 l + u,
     u < 4, of its range in increasing k, and the 32 lane sums combine by
     the balanced tree (l, l + 16), then (l, l + 8), (l, l + 4), (l, l + 2),
     (l, l + 1).

   In each, [splits] ranges of k (whole k-tiles for wmma and simt) combine
   in range order (combine.h); init then enters the sum, and the result
   rounds once to y's dtype. A WMMA operand whose rows along k are not
   16-byte vectors is first packed into rows that are (pack): the same
   values in the same tiles, so the same bits. A SIMT or skinny operand of
   a dtype other than its accumulator's is packed into the accumulator's
   dtype, exactly, so these kernels read their own dtype alone. */

#include "combine.h"
#include "device.h"
#include "kernels.h"
#include "nx_dtype.h"

DEVICE int min_(int a, int b) { return a < b ? a : b; }

/* Elements */

/* Element [i] of [p], of the dtype [dt], as a double or an int64: the
   operands of the SIMT kernels, and init. */
DEVICE double read_float(const void *p, int64_t i, int dt) {
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

DEVICE u64 read_int(const void *p, int64_t i, int dt) {
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

/* The accumulator types: integers wrap, so they sum unsigned. */
template <typename T> DEVICE T read(const void *p, int64_t i, int dt);
template <> DEVICE float read(const void *p, int64_t i, int dt) {
  return (float)read_float(p, i, dt);
}
template <> DEVICE double read(const void *p, int64_t i, int dt) {
  return read_float(p, i, dt);
}
template <> DEVICE u64 read(const void *p, int64_t i, int dt) {
  return read_int(p, i, dt);
}
template <> DEVICE int read(const void *p, int64_t i, int dt) {
  return (int)read_int(p, i, dt);
}

/* Stores the bits [c] of an element of the integer, bool or narrow float
   dtype [dt] at element [i] of [p]: their low bits, as wide as [dt]. */
DEVICE void put(void *p, int64_t i, int dt, int64_t c) {
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
DEVICE void write(const contract_params &p, int64_t i, double v) {
  switch (p.y_dtype) {
  case NX_FLOAT64: ((double *)p.y)[i] = v; return;
  case NX_FLOAT32: ((float *)p.y)[i] = (float)v; return;
  }
  put(p.y, i, p.y_dtype, nx_double_to_bits(p.y_dtype, v));
}

DEVICE void write(const contract_params &p, int64_t i, float v) {
  write(p, i, (double)v);
}

/* An integer sum [v], wrapped to the accumulator and widened by its sign:
   wrapped to an integer y, rounded once to a float y, nonzero for bool. A
   32-bit accumulator's sum, summed in 64 bits, wraps to 32 here: the same
   sum, modulo 2^32. */
DEVICE void write(const contract_params &p, int64_t i, u64 v) {
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

DEVICE void write(const contract_params &p, int64_t i, int v) {
  write(p, i, (u64)(int64_t)v);
}

/* y's element (z, i, j), if inside: init added, then rounded once. */
template <typename T>
DEVICE void store(const contract_params &p, int z, int i, int j, T v) {
  if (i >= p.m || j >= p.n) return;
  if (p.init)
    v += read<T>(p.init, z * p.si[0] + i * p.si[1] + j * p.si[2],
                 p.init_dtype);
  write(p, z * p.sy[0] + i * p.sy[1] + j * p.sy[2], v);
}

/* Eight outputs (z, i, j + e), e < 8, of [v]. With no init, y's j
   contiguous, its rows aligned on 16 bytes, and y of the accumulator's
   dtype, or bfloat16 or float16 from float32 (NX_CONTRACT_Y_WHOLE): whole
   16-byte stores, rounding as nx_dtype.h's encoders do. Through store
   otherwise. */
template <typename T>
DEVICE void store8(const contract_params &p, int z, int i, int j,
                   const T *v) {
  if ((p.aligned & NX_CONTRACT_Y_WHOLE) && i < p.m && j + 8 <= p.n) {
    const int64_t o = z * p.sy[0] + i * p.sy[1] + j;
    if (sizeof(T) == 4 &&
        (p.y_dtype == NX_BFLOAT16 || p.y_dtype == NX_FLOAT16)) {
      u32x4 h;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float lo = (float)v[e], hi = (float)v[e + 1];
        h[e / 2] = p.y_dtype == NX_BFLOAT16
                       ? nx_float_to_bf16(lo) | (uint32_t)nx_float_to_bf16(hi) << 16
                       : nx_float_to_f16(lo) | (uint32_t)nx_float_to_f16(hi) << 16;
      }
      *(u32x4 *)((uint16_t *)p.y + o) = h;
      return;
    }
#pragma unroll
    for (int e = 0; e < 8 * (int)sizeof(T) / 16; e++)
      ((u32x4 *)((T *)p.y + o))[e] = ((const u32x4 *)v)[e];
    return;
  }
  for (int e = 0; e < 8; e++) store(p, z, i, j + e, v[e]);
}

/* [rows] x [cols] outputs from (z, i0, j0) staged in the local data share
   at [st], rows [ld] apart, stored 8 at a time along a row by the
   workgroup's [threads] work-items. */
template <typename T>
DEVICE void store_rows(const contract_params &p, const T *st, int ld,
                       int rows, int cols, int z, int i0, int j0, int threads) {
  for (int e = item(); e < rows * (cols / 8); e += threads) {
    const int r = e / (cols / 8), c = e % (cols / 8) * 8;
    store8(p, z, i0 + r, j0 + c, st + r * ld + c);
  }
}

/* The WMMA kernels */

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef int i32x2 __attribute__((ext_vector_type(2)));

#define KIND_bf16 0
#define KIND_f16 1
#define KIND_s8 2

/* A wave's 16 x 16 tile of sums: lane l holds rows 8 (l / 16) + e, e < 8,
   of column l % 16. Its operand fragments: lane l holds k 8 (l / 16) + e
   of row l % 16 (a's rows m, b's rows n), 8 elements of 16 or 8 bits. */
template <int KIND> struct Wmma;

template <> struct Wmma<KIND_bf16> {
  typedef float acc;
  typedef f32x8 tile;
  typedef u32x4 frag;
  static DEVICE tile step(const frag &a, const frag &b, tile c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
        __builtin_bit_cast(bf16x8, a), __builtin_bit_cast(bf16x8, b), c);
  }
};

template <> struct Wmma<KIND_f16> {
  typedef float acc;
  typedef f32x8 tile;
  typedef u32x4 frag;
  static DEVICE tile step(const frag &a, const frag &b, tile c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        __builtin_bit_cast(f16x8, a), __builtin_bit_cast(f16x8, b), c);
  }
};

template <> struct Wmma<KIND_s8> {
  typedef int acc;
  typedef i32x8 tile;
  typedef i32x2 frag;
  static DEVICE tile step(const frag &a, const frag &b, tile c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b,
                                                            c, false);
  }
};

/* One operand of a workgroup: [R] rows (m for a, n for b) from [r0], their
   k-tiles of [BKB] bytes loaded as 16-byte chunks, [CH] a work-item, into
   rows of [LDB] bytes of the local data share. Rows hold whole vectors, k's
   end included: the plan packs an operand whose do not, its rows padded
   with zeros. */
template <int R, int BKB, int ES, int LDB, int THREADS> struct Operand {
  static constexpr int C = BKB / 16;         /* chunks a row */
  static constexpr int CH = R * C / THREADS; /* chunks a work-item */
  static_assert(R * C % THREADS == 0, "whole chunks a work-item");

  const char *base; /* row 0, k 0 */
  int64_t lead;     /* bytes from a row to the next */
  int rows, k;      /* rows from r0 to the extent; k in elements */

  /* Whether chunk [e] of k-tile [kt] is inside the extents. */
  DEVICE bool inside(int e, int kt) const {
    return e / C < rows && kt * (BKB / ES) + e % C * (16 / ES) < k;
  }

  /* This work-item's chunks of k-tile [kt], as read. Every chunk is one
     16-byte load, with no branch, so that the compiler counts a k-tile's
     loads in flight: a chunk past the extents reads row 0's first. */
  DEVICE void load(u32x4 (&v)[CH], int kt, int tid) const {
#pragma unroll
    for (int it = 0; it < CH; it++) {
      const int e = tid + it * THREADS;
      const int64_t at = e / C * lead + (int64_t)kt * BKB + e % C * 16;
      v[it] = *(const u32x4 *)(inside(e, kt) ? base + at : base);
    }
  }

  /* The chunks [v] of k-tile [kt] into the tile [s], those past the
     extents zero. Cleared here, at the store, so that the loads stay in
     flight until their tile is stored. */
  DEVICE void stash(uint8_t *s, const u32x4 (&v)[CH], int kt,
                    int tid) const {
#pragma unroll
    for (int it = 0; it < CH; it++) {
      const int e = tid + it * THREADS;
      *(u32x4 *)(s + e / C * LDB + e % C * 16) =
          inside(e, kt) ? v[it] : (u32x4)0;
    }
  }
};

/* The fragment of row [r] at k [k] of a tile of rows of [LDB] bytes. */
template <typename F>
DEVICE F fragment(const uint8_t *s, int r, int k, int ldb, int es) {
  return *(const F *)(s + r * ldb + k * es);
}

template <int KIND, typename Tile>
DEVICE void wmma_contract(const contract_params &p) {
  constexpr int BM = Tile::BM, BN = Tile::BN, BKB = Tile::BKB, WM = Tile::WM,
                WN = Tile::WN;
  typedef Wmma<KIND> W;
  typedef typename W::acc Acc;
  constexpr int ES = KIND == KIND_s8 ? 1 : 2, BK = BKB / ES;
  constexpr int LDB = BKB + 16; /* a padded row: its 16 lanes' rows apart */
  constexpr int WAVES_N = BN / WN, THREADS = (BM / WM) * WAVES_N * 32;
  constexpr int MI = WM / 16, NI = WN / 16, KS = BK / 16;
  constexpr int STAGE = (BM + BN) * LDB, CAP = 2 * STAGE;
  typedef Operand<BM, BKB, ES, LDB, THREADS> A;
  typedef Operand<BN, BKB, ES, LDB, THREADS> B;
  static SHARED __attribute__((aligned(16))) uint8_t smem[CAP];

  const int tid = item(), lane = tid % 32, wave = tid / 32;
  const int wm0 = wave / WAVES_N * WM, wn0 = wave % WAVES_N * WN;

  /* Grouped rasterisation: 8 row-tiles sweep the column tiles together, so
     that a column tile of b is read from the L2 by the 8 in flight. */
  const int tiles_m = (p.m + BM - 1) / BM, tiles_n = (p.n + BN - 1) / BN;
  const int pid = group(), groups = 8 * tiles_n, first = pid / groups * 8;
  const int gsz = min_(tiles_m - first, 8);
  const int tm = first + pid % groups % gsz, tn = pid % groups / gsz;
  const int m0 = tm * BM, n0 = tn * BN, z = group_z();

  const int kts = (p.k + BK - 1) / BK, split = group_y();
  const int kt0 = split * kts / p.splits;
  const int KT = (split + 1) * kts / p.splits - kt0;

  const A a = {(const char *)p.a + (z * p.sa[0] + m0 * p.sa[1]) * ES,
               p.sa[1] * ES, p.m - m0, p.k};
  const B b = {(const char *)p.b + (z * p.sb[0] + n0 * p.sb[1]) * ES,
               p.sb[1] * ES, p.n - n0, p.k};

  typename W::tile acc[MI][NI] = {};

  /* Tile kt of this range's k into the register set [s]. Past the range
     it loads the last tile again, unused: under a branch, the loads would
     leave the compiler unsure which are the newest, and it would wait for
     every load in flight before each stash. */
  u32x4 ra[2][A::CH], rb[2][B::CH];
  auto fetch = [&](int s, int kt) __attribute__((always_inline)) {
    const int t = kt0 + min_(kt, KT - 1);
    a.load(ra[s], t, tid);
    b.load(rb[s], t, tid);
  };
  /* The register set [s], holding tile kt, into the local data share's
     stage [stage]. */
  auto stash = [&](int s, int stage, int kt) __attribute__((always_inline)) {
    uint8_t *st = smem + stage * STAGE;
    a.stash(st, ra[s], kt0 + kt, tid);
    b.stash(st + BM * LDB, rb[s], kt0 + kt, tid);
  };
  /* The WMMAs of the k-tile in the stage [stage]. */
  auto compute = [&](int stage) __attribute__((always_inline)) {
    const uint8_t *sa = smem + stage * STAGE, *sb = sa + BM * LDB;
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
      const int k = ks * 16 + lane / 16 * 8;
      typename W::frag fa[MI], fb[NI];
#pragma unroll
      for (int i = 0; i < MI; i++)
        fa[i] = fragment<typename W::frag>(sa, wm0 + i * 16 + lane % 16, k,
                                           LDB, ES);
#pragma unroll
      for (int j = 0; j < NI; j++)
        fb[j] = fragment<typename W::frag>(sb, wn0 + j * 16 + lane % 16, k,
                                           LDB, ES);
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NI; j++)
          acc[i][j] = W::step(fa[i], fb[j], acc[i][j]);
    }
  };

  /* Two k-tiles ahead: tile kt computes from one stage while tile kt + 1,
     loaded a tile earlier, goes to the other stage, which every wave
     finished reading at the last barrier, and tile kt + 2's loads are in
     flight in the register set tile kt left. Unrolled by two, so that
     each register set and stage is named by a constant. */
  if (KT > 0) {
    fetch(0, 0);
    fetch(1, 1);
    stash(0, 0, 0);
    barrier();
  }
  for (int kt = 0; kt < KT; kt += 2) {
    fetch(0, kt + 2);
    compute(0);
    if (kt + 1 < KT) stash(1, 1, kt + 1);
    barrier();
    if (kt + 1 >= KT) break;
    fetch(1, kt + 3);
    compute(1);
    if (kt + 2 < KT) stash(0, 0, kt + 2);
    barrier();
  }

  if (p.splits > 1) {
    Acc(&flat)[MI * NI * 8] = *reinterpret_cast<Acc(*)[MI * NI * 8]>(acc);
    const int tile = z * tiles_m * tiles_n + tm * tiles_n + tn;
    if (!combine(flat,
                 (Acc *)p.partials + (size_t)tile * p.splits * MI * NI * 8 * THREADS,
                 p.tickets + tile, p.splits, split, THREADS))
      return;
  }

  /* The outputs through the local data share, in passes of whole wave rows:
     a pass's waves write their sums, then every work-item stores 8 outputs
     at a time along a row. */
  constexpr int LD = BN + 8;
  constexpr int ROWS = CAP / (LD * (int)sizeof(Acc)) / WM * WM;
  static_assert(ROWS >= WM, "a pass holds a wave row");
  Acc *st = (Acc *)smem;
#pragma unroll
  for (int base = 0; base < BM; base += ROWS) {
    if (wm0 >= base && wm0 < base + ROWS)
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NI; j++)
#pragma unroll
          for (int e = 0; e < 8; e++)
            st[(wm0 - base + i * 16 + lane / 16 * 8 + e) * LD + wn0 + j * 16 +
               lane % 16] = acc[i][j][e];
    barrier();
    store_rows(p, st, LD, min_(ROWS, BM - base), BN, z, m0 + base, n0,
               THREADS);
    barrier();
  }
}

/* The SIMT kernels */

/* A BM x BM tile of outputs per workgroup of 256 work-items (16 x 16),
   k-tiles of 64 bytes of k (16 floats), each work-item TM x TM outputs in
   two halves BM / 2 apart each way. Both operands are stored k-major in the
   local data share, the next k-tile loaded into registers while the
   current one computes. A work-item loads E = BM * BK / 256 elements of
   each operand per k-tile: E consecutive k of a row, or E consecutive rows
   of a k, whichever axis is contiguous, as vectors where it can. */
template <typename T, int BM, int BK> struct Simt_operand {
  static constexpr int E = BM * BK / 256;
  const char *base;
  int64_t s_row, s_k;
  int rows, k, r0;
  bool rows_contiguous, vectors;

  DEVICE void load(T (&v)[E], int k0, int tid) const {
    const int row = rows_contiguous ? r0 + tid % (BM / E) * E : r0 + tid / (BK / E);
    const int kk = rows_contiguous ? k0 + tid / (BM / E) : k0 + tid % (BK / E) * E;
    if (vectors &&
        (rows_contiguous ? kk < k && row + E - 1 < rows : row < rows && kk + E - 1 < k)) {
      const char *src = base + (row * s_row + kk * s_k) * sizeof(T);
#pragma unroll
      for (int e = 0; e < E * (int)sizeof(T) / 8; e++)
        ((u64 *)v)[e] = ((const u64 *)src)[e];
      return;
    }
#pragma unroll
    for (int e = 0; e < E; e++) {
      const int r = rows_contiguous ? row + e : row, q = rows_contiguous ? kk : kk + e;
      v[e] = r < rows && q < k ? ((const T *)base)[r * s_row + q * s_k] : T(0);
    }
  }

  /* The loaded elements into the k-major tile [s]. */
  DEVICE void stash(T (*s)[BM], const T (&v)[E], int tid) const {
#pragma unroll
    for (int e = 0; e < E; e++)
      if (rows_contiguous)
        s[tid / (BM / E)][tid % (BM / E) * E + e] = v[e];
      else
        s[tid % (BK / E) * E + e][tid / (BK / E)] = v[e];
  }
};

DEVICE float fma_(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
DEVICE double fma_(double a, double b, double c) { return __builtin_fma(a, b, c); }
DEVICE u64 fma_(u64 a, u64 b, u64 c) { return a * b + c; }

template <typename T, int BM> DEVICE void simt_contract(const contract_params &p) {
  constexpr int BK = 64 / sizeof(T), TM = BM / 16, H = TM / 2;
  constexpr int THREADS = NX_CONTRACT_THREADS;
  /* Two k-tiles of each operand, then a half tile of outputs. */
  constexpr int TILES = 2 * 2 * BK * BM, HALF = BM / 2 * BM;
  static SHARED __attribute__((aligned(16))) T smem[TILES > HALF ? TILES : HALF];
  T(*as)[BK][BM] = (T(*)[BK][BM])smem, (*bs)[BK][BM] = as + 2;
  const int tid = item(), tx = tid % 16, ty = tid / 16, z = group_z();
  /* Grouped rasterisation, as the WMMA kernels'. */
  const int tiles_m = (p.m + BM - 1) / BM, tiles_n = (p.n + BM - 1) / BM;
  const int pid = group(), groups = 8 * tiles_n, first = pid / groups * 8;
  const int gsz = min_(tiles_m - first, 8);
  const int tm = first + pid % groups % gsz, tn = pid % groups / gsz;
  const int m0 = tm * BM, n0 = tn * BM;
  const int kts = (p.k + BK - 1) / BK, split = group_y();
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
  T ra[Simt_operand<T, BM, BK>::E], rb[Simt_operand<T, BM, BK>::E];
  if (kt0 < kt1) {
    a.load(ra, kt0 * BK, tid);
    b.load(rb, kt0 * BK, tid);
    a.stash(as[0], ra, tid);
    b.stash(bs[0], rb, tid);
  }
  barrier();

  for (int kt = kt0, buf = 0; kt < kt1; kt++, buf ^= 1) {
    const bool more = kt + 1 < kt1;
    if (more) {
      a.load(ra, (kt + 1) * BK, tid);
      b.load(rb, (kt + 1) * BK, tid);
    }
    /* Unrolled by two: the loads of every k at once would outgrow the
       registers. */
#pragma unroll 2
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
    barrier();
  }

  if (p.splits > 1) {
    T(&flat)[TM * TM] = *reinterpret_cast<T(*)[TM * TM]>(acc);
    const int tile = z * tiles_m * tiles_n + pid;
    if (!combine(flat, (T *)p.partials + (size_t)tile * p.splits * TM * TM * THREADS,
                 p.tickets + tile, p.splits, split, THREADS))
      return;
  }
  /* The outputs through the local data share, a half of the rows a pass:
     work-item (tx, ty)'s first H rows, then its last H. */
#pragma unroll
  for (int half = 0; half < 2; half++) {
#pragma unroll
    for (int i = 0; i < H; i++)
#pragma unroll
      for (int j = 0; j < TM; j++)
        smem[(ty * H + i) * BM + (j < H ? tx * H + j : BM / 2 + tx * H + j - H)] =
            acc[half * H + i][j];
    barrier();
    store_rows(p, smem, BM, BM / 2, BM, z, m0 + half * BM / 2, n0, THREADS);
    barrier();
  }
}

/* The skinny kernels (m <= 16): lane l of 32 sums, for each t, the run of
   four k0 + 128 t + 4 l + u, u < 4, in increasing k, and the 32 lane sums
   combine by the balanced tree (l, l + 16), then (l, l + 8), (l, l + 4),
   (l, l + 2), (l, l + 1). Two forms, each its own kernel so that each has
   the registers it needs: column, one wave per output column j when b's
   k axis is contiguous (8 columns a workgroup), lanes across k, a run one
   vector; across, a workgroup of 32 columns by 8 lane groups, work-item
   (x, y) summing the lane classes y, y + 8, y + 16, y + 24 of column x,
   reading rows of b across x. */
enum { FORM_column, FORM_across };

template <typename T> DEVICE T tree32(T v) {
#pragma unroll
  for (int d = 16; d > 0; d /= 2) v += xor_lane(v, d);
  return v;
}

/* The run of four k from [k] of the operand at [p] with strides [s_row]
   and [s_k], row [row]: its first [us] elements, the rest zero, as whole
   vectors where [vectors] (its k contiguous, its rows aligned). */
template <typename T>
DEVICE void run(T (&v)[4], const char *p, int64_t s_row, int64_t s_k,
                int row, int k, int us, bool vectors) {
  if (vectors && us == 4) {
    const T *src = (const T *)__builtin_assume_aligned(
        p + (row * s_row + k) * sizeof(T), 16);
#pragma unroll
    for (int u = 0; u < 4; u++) v[u] = src[u];
    return;
  }
#pragma unroll
  for (int u = 0; u < 4; u++)
    v[u] = u < us ? ((const T *)p)[row * s_row + (k + u) * s_k] : T(0);
}

/* Adds column j's runs [bv], run t of k from k + 128 t and the first
   [us] of each inside the range, times a's rows, to [c] in increasing k:
   each row's runs loaded together, then summed. a's runs are vectors
   where NX_CONTRACT_A_VECTORS says so. */
template <typename T, int F>
DEVICE void dot(const contract_params &p, T (&c)[16], const char *a,
                const T (&bv)[F][4], int k, int us) {
#pragma unroll
  for (int i = 0; i < 16; i++)
    if (i < p.m) {
      T av[F][4];
#pragma unroll
      for (int t = 0; t < F; t++)
        run(av[t], a, p.sa[1], p.sa[2], i, k + t * 128, us,
            p.aligned & NX_CONTRACT_A_VECTORS);
#pragma unroll
      for (int t = 0; t < F; t++)
#pragma unroll
        for (int u = 0; u < 4; u++)
          if (u < us) c[i] = fma_(av[t][u], bv[t][u], c[i]);
    }
}

/* Adds column j's runs from [k] to [k1], every 128th k, times a's rows,
   to [c]: F whole runs a step, their loads issued before their sums, so
   that a lane has F in flight; then the runs left, one by one. b's runs
   are vectors where [vectors] says so. */
template <typename T, int F>
DEVICE void add_runs(const contract_params &p, T (&c)[16], const char *a,
                     const char *b, int j, int k, int k1, bool vectors) {
  if constexpr (F > 1)
    for (; k + (F - 1) * 128 + 3 < k1; k += F * 128) {
      T bv[F][4];
#pragma unroll
      for (int t = 0; t < F; t++)
        run(bv[t], b, p.sb[1], p.sb[2], j, k + t * 128, 4, vectors);
      dot<T, F>(p, c, a, bv, k, 4);
    }
  for (; k < k1; k += 128) {
    T bv[1][4];
    const int us = min_(4, k1 - k);
    run(bv[0], b, p.sb[1], p.sb[2], j, k, us, vectors);
    dot<T, 1>(p, c, a, bv, k, us);
  }
}

/* The workgroup's sums into [level8], row i of column c at element
   i * 32 + c (its first 512), as the tree leaves them, by the form
   [FORM]. Rows from m on stay zero. */
template <typename T, int FORM>
DEVICE void skinny_sums(const contract_params &p, T (*level8)[8][32],
                        const char *a, const char *b, int k0, int k1) {
  T *sums = &level8[0][0][0];
  if constexpr (FORM == FORM_column) {
    /* Lane l holds class l, its runs vectors where b's rows are
       (NX_CONTRACT_B_VECTORS). */
    const int lane = item() % 32, j = group() * 8 + item() / 32;
    T acc[16];
#pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = T(0);
    if (j < p.n)
      add_runs<T, sizeof(T) == 8 ? 2 : 4>(p, acc, a, b, j, k0 + 4 * lane, k1,
                                          p.aligned & NX_CONTRACT_B_VECTORS);
#pragma unroll
    for (int i = 0; i < 16; i++)
      if (i < p.m) acc[i] = tree32(acc[i]);
    barrier();
    if (lane == 0)
#pragma unroll
      for (int i = 0; i < 16; i++) sums[i * 32 + item() / 32] = acc[i];
    return;
  }
  /* Work-item (x, y): classes y + 8q of column x; its four sums reduce to
     the tree's level of 8, (y + y16) + (y8 + y24), then 8 work-items' in
     the local data share to the root. The pairs q and q + 2 are summed one
     after the other, the first's sum waiting in the work-item's place of
     the local data share, so that two sets of sums are live at once. y is
     the wave's, as readfirstlane tells the compiler, so a's runs, the same
     for each lane of a wave, are scalar loads. */
  const int x = item() % 32, j = group() * 32 + x;
  const int y = __builtin_amdgcn_readfirstlane(item() / 32);
#pragma unroll
  for (int h = 0; h < 2; h++) {
    T c[2][16];
#pragma unroll
    for (int q = 0; q < 2; q++)
#pragma unroll
      for (int i = 0; i < 16; i++) c[q][i] = T(0);
    if (j < p.n)
#pragma unroll
      for (int q = 0; q < 2; q++)
        add_runs<T, 1>(p, c[q], a, b, j, k0 + 4 * (y + 8 * (h + 2 * q)), k1,
                       false);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      const T pair = c[0][i] + c[1][i];
      level8[i][y][x] = h == 0 ? pair : level8[i][y][x] + pair;
    }
  }
  barrier();
  T acc[16];
  if (y == 0)
#pragma unroll
    for (int i = 0; i < 16; i++) {
      const T(&l)[8][32] = level8[i];
      acc[i] = ((l[0][x] + l[4][x]) + (l[2][x] + l[6][x])) +
               ((l[1][x] + l[5][x]) + (l[3][x] + l[7][x]));
    }
  barrier();
  if (y == 0)
#pragma unroll
    for (int i = 0; i < 16; i++) sums[i * 32 + x] = acc[i];
}

template <typename T, int FORM>
DEVICE void skinny_contract(const contract_params &p) {
  constexpr int THREADS = NX_CONTRACT_THREADS;
  const int z = group_z(), split = group_y(), splits = p.splits;
  const char *a = (const char *)p.a + z * p.sa[0] * sizeof(T);
  const char *b = (const char *)p.b + z * p.sb[0] * sizeof(T);
  /* The range of k: whole 128-element runs, one t of the lanes. */
  const int runs = (p.k + 127) / 128;
  const int k0 = split * runs / splits * 128;
  const int k1 = min_(p.k, (split + 1) * runs / splits * 128);
  static SHARED T level8[16][8][32];
  skinny_sums<T, FORM>(p, level8, a, b, k0, k1);

  /* Two of the workgroup's sums per work-item: the partials a split sum
     combines. */
  constexpr int cols = FORM == FORM_across ? 32 : 8;
  const T *sums = &level8[0][0][0];
  barrier();
  T v[2] = {sums[item()], sums[256 + item()]};
  if (splits > 1) {
    const int tile = z * ((p.n + cols - 1) / cols) + group();
    if (!combine(v, (T *)p.partials + (size_t)tile * splits * 2 * THREADS,
                 p.tickets + tile, splits, split, THREADS))
      return;
  }
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int e = h * 256 + item(), i = e / 32, c = e % 32;
    if (i < p.m && c < cols) store(p, z, i, group() * cols + c, v[h]);
  }
}

/* Packing */

/* An element of a packed operand: [dt]'s value as [out]'s, exactly, as
   [out]'s bits. */
DEVICE u64 packed(const void *src, int64_t x, int dt, int out) {
  switch (out) {
  case NX_BFLOAT16: return nx_float_to_bf16((float)read_float(src, x, dt));
  case NX_FLOAT16: return nx_float_to_f16((float)read_float(src, x, dt));
  case NX_FLOAT32: return nx_float_bits((float)read_float(src, x, dt));
  case NX_FLOAT64: return nx_double_bits(read_float(src, x, dt));
  }
  return read_int(src, x, dt);
}

DEVICE void pack_rows(const pack_params &p) {
  const u64 i = index_of(NX_CONTRACT_THREADS);
  if (i >= (u64)p.batch * p.rows * p.lead) return;
  const int64_t q = i % p.lead, r = i / p.lead % p.rows, z = i / p.lead / p.rows;
  const int64_t x = z * p.s[0] + r * p.s[1] + q * p.s[2];
  const u64 v = q < p.k ? packed(p.src, x, p.dtype, p.out) : 0;
  switch (p.bytes) {
  case 1: ((uint8_t *)p.dst)[i] = (uint8_t)v; break;
  case 2: ((uint16_t *)p.dst)[i] = (uint16_t)v; break;
  case 4: ((uint32_t *)p.dst)[i] = (uint32_t)v; break;
  default: ((u64 *)p.dst)[i] = v;
  }
}

/* Instances */

KERNEL SIZED(NX_CONTRACT_THREADS) void zero_u32(zero_params p) {
  const u64 i = index_of(NX_CONTRACT_THREADS);
  if (i < p.n) p.p[i] = 0;
}

#define ACC_f32 float
#define ACC_f64 double
#define ACC_i64 u64

#define TILE(name, bm, bn, bkb, wm, wn)                                        \
  struct name {                                                                \
    enum { BM = bm, BN = bn, BKB = bkb, WM = wm, WN = wn };                    \
    enum { THREADS = (bm / wm) * (bn / wn) * 32 };                             \
  };
NX_AMD_TILES(TILE)
#undef TILE

#define DEFINE(name, FAMILY, ...) FAMILY(name, __VA_ARGS__)
#define ZERO(name, ...)
#define PACK(name, ...)                                                        \
  KERNEL SIZED(NX_CONTRACT_THREADS) void name(pack_params p) { pack_rows(p); }
#define WMMA(name, kind, tile)                                                \
  KERNEL SIZED(tile::THREADS) void name(contract_params p) {                   \
    wmma_contract<KIND_##kind, tile>(p);                                       \
  }
#define SIMT(name, acc, bm)                                                    \
  KERNEL SIZED(NX_CONTRACT_THREADS) void name(contract_params p) {             \
    simt_contract<ACC_##acc, bm>(p);                                           \
  }
#define SKINNY(name, acc, form)                                                \
  KERNEL SIZED(NX_CONTRACT_THREADS) void name(contract_params p) {             \
    skinny_contract<ACC_##acc, FORM_##form>(p);                                \
  }
NX_AMD_KERNELS(DEFINE)
