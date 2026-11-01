/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contract: y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k]
   (kernels.h's contract_params), four bodies, each its own association.
   A float sum's is fixed by the dtypes, k and the order given, never by a
   layout, m, n, the batch or the tile; an integer sum's skinny ranges also
   read n and the batch, which no association changes:

   - wmma: bfloat16 or float16 operands summed in float32, int8 in int32,
     on gfx12's WMMA 16 x 16 x 16. Each output sums a chunk's k in steps
     of 16 in increasing k, one WMMA a step.

       global --16-byte loads (zero past the extents; transposing 8 x 8
       blocks for an operand whose free axis is contiguous), two k-tiles
       ahead--> registers --> one of two local data share k-tiles, rows of
       k --8 k a lane--> fragments --WMMA-->
       accumulators --> combine (splits) --> + init --> y

   - simt (float sums a call gives chain order, and integer sums past 16
     rows): float32 or float64 sums by fma, integer sums wrapping in 64
     bits (a 32-bit accumulator's wrap to 32 bits at the store: the same
     sum modulo 2^32); each output sums a chunk's k in increasing k.
   - skinny (integer sums of 16 rows or fewer): lane l of 32 sums the runs
     k0 + 128 t + 4 l + u, u < 4, of its range in increasing k, and the 32
     lane sums combine by the balanced tree (l, l + 16), then (l, l + 8),
     (l, l + 4), (l, l + 2), (l, l + 1). Integers wrap, so any association
     gives their bits.
   - lane (float sums a call gives lane order, at any extents):
     Nx_kernel's lane order, the same bits as nx.cpu's; it never
     splits.

   wmma and simt cut k into chunks of [chunk] k, a whole number of
   k-tiles of every tile of the family, from 0: each chunk is summed from
   +0, and the sum is +0 plus the chunks' sums in increasing k. A split
   gives each workgroup a chunk, whose sum it puts for the last to add
   (combine.h); without one, a workgroup adds each chunk's sum as it
   ends. skinny's [splits] ranges add in range order. In each, init then
   enters the sum, and the result rounds once to y's dtype. A WMMA operand
   whose rows along its contiguous axis are not 16-byte vectors, or in a
   layout with no instance, is first packed into rows of k that are
   (pack): the same values in the same tiles, so the same bits. A SIMT,
   skinny or lane operand of a dtype other than its kernel's is packed
   into the accumulator's dtype, exactly, so these kernels read their own
   dtype alone. */

#include "combine.h"
#include "device.h"
#include "kernels.h"
#include "nx_dtype.h"

DEVICE int min_(int a, int b) { return a < b ? a : b; }
DEVICE int64_t min_(int64_t a, int64_t b) { return a < b ? a : b; }

/* Coordinates: an extent may be 2^31 - 1, so that a coordinate past one, a
   ceiling or a product of a range by a split count is computed in 64 bits,
   and narrowed only where it is below an extent. [ceil_div n d] is
   ceil(n / d), for n an extent. */
DEVICE int ceil_div(int64_t n, int d) { return (int)((n + d - 1) / d); }

/* The first of [units] units in range [r] of [ranges]. */
DEVICE int64_t range_start(int64_t units, int r, int ranges) {
  return (int64_t)r * units / ranges;
}
DEVICE int max_(int a, int b) { return a > b ? a : b; }

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
  if ((p.aligned & NX_CONTRACT_Y_WHOLE) && i < p.m && p.n - j >= 8) {
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

/* An operand's contiguous axis: k, or its free axis (m for a, n for b). */
#define AXIS_k 0
#define AXIS_m 1
#define AXIS_n 1

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
   rows of [LDB] bytes of the local data share, chunk q of row r at
   r * LDB + 16 q. Along [AXIS]'s contiguous axis the rows hold whole
   vectors, k's end included: the plan packs an operand whose do not, its
   rows padded with zeros.

   With k contiguous ([AXIS] is AXIS_k), a chunk is one row's k as they
   lie, a row's chunks on consecutive work-items. With the rows contiguous
   (16-bit elements), gfx12's transposing load reads a chunk: each group of
   8 lanes reads 8 runs of 8 rows, one run a k, and lane j of the group
   receives row j's 8 k, so a group holds 8 consecutive rows' chunk q.
   Either way the local data share holds the same values. */
template <int R, int BKB, int ES, int LDB, int THREADS, int AXIS>
struct Operand {
  static constexpr int C = BKB / 16;         /* chunks a row */
  static constexpr int CH = R * C / THREADS; /* chunks a work-item */
  static_assert(R * C % THREADS == 0, "whole chunks a work-item");
  static_assert(AXIS == AXIS_k || (ES == 2 && R % 8 == 0),
                "a transposing load reads 8 rows of 16-bit elements");

  const char *base; /* row 0, k 0 */
  int64_t lead;     /* bytes from a row, or a k, to the next */
  int rows, k;      /* rows from r0 to the extent; k in elements */

  /* The row and chunk of a k-tile of chunk [e]. */
  DEVICE static int row(int e) { return AXIS == AXIS_k ? e / C : e % R; }
  DEVICE static int chunk(int e) { return AXIS == AXIS_k ? e % C : e / R; }

  /* Whether chunk [e] of k-tile [kt], a tile k reaches, is inside the
     extents: kt * BK < k, so neither side overflows. With the rows
     contiguous, the plan admits rows and k of whole chunks, so a group's
     chunks are all inside or all outside. */
  DEVICE bool inside(int e, int kt) const {
    return row(e) < rows && chunk(e) * (16 / ES) < k - kt * (BKB / ES);
  }

  /* This work-item's chunks of k-tile [kt], as read. Every chunk is one
     16-byte load, with no branch, so that the compiler counts a k-tile's
     loads in flight: a chunk past the extents reads row 0's first. */
  DEVICE void load(u32x4 (&v)[CH], int kt, int tid) const {
#pragma unroll
    for (int it = 0; it < CH; it++) {
      const int e = tid + it * THREADS, r = row(e), q = chunk(e);
      if constexpr (AXIS == AXIS_k) {
        const int64_t at = r * lead + (int64_t)kt * BKB + q * 16;
        v[it] = *(const u32x4 *)(inside(e, kt) ? base + at : base);
      } else {
        /* Lane j of a group reads the run from row r - j at its own k. */
        const int j = e % 8;
        const int64_t at =
            ((int64_t)kt * (BKB / ES) + q * 8 + j) * lead + (r - j) * ES;
        typedef short short8 __attribute__((ext_vector_type(8)));
        typedef __attribute__((address_space(1))) short8 *global_short8;
        v[it] = __builtin_bit_cast(
            u32x4, __builtin_amdgcn_global_load_tr_b128_v8i16(
                       (global_short8)(inside(e, kt) ? base + at : base)));
      }
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
      *(u32x4 *)(s + row(e) * LDB + chunk(e) * 16) =
          inside(e, kt) ? v[it] : (u32x4)0;
    }
  }
};

/* The fragment of row [r] at k [k] of a tile of rows of [LDB] bytes. */
template <typename F>
DEVICE F fragment(const uint8_t *s, int r, int k, int ldb, int es) {
  return *(const F *)(s + r * ldb + k * es);
}

/* A constant flag as a type, which selects a lambda's instance. */
template <bool B> struct Flag {
  static constexpr bool value = B;
};

template <int KIND, int AXIS_A, int AXIS_B, typename Tile>
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
  typedef Operand<BM, BKB, ES, LDB, THREADS, AXIS_A> A;
  typedef Operand<BN, BKB, ES, LDB, THREADS, AXIS_B> B;
  static SHARED __attribute__((aligned(16))) uint8_t smem[CAP];

  const int tid = item(), lane = tid % 32, wave = tid / 32;
  const int wm0 = wave / WAVES_N * WM, wn0 = wave % WAVES_N * WN;

  /* Grouped rasterisation: 8 row-tiles sweep the column tiles together, so
     that a column tile of b is read from the L2 by the 8 in flight. */
  const int tiles_m = ceil_div(p.m, BM), tiles_n = ceil_div(p.n, BN);
  const int pid = group(), groups = 8 * tiles_n, first = pid / groups * 8;
  const int gsz = min_(tiles_m - first, 8);
  const int tm = first + pid % groups % gsz, tn = pid % groups / gsz;
  const int m0 = tm * BM, n0 = tn * BN, z = group_z();

  /* k in chunks of [ck] k-tiles: a split's workgroup takes chunk [split],
     the only one takes them all. Its KT k-tiles start at kt0. */
  const int kts = ceil_div(p.k, BK), split = group_y();
  const int ck = p.chunk / BK;
  const int kt0 = p.splits > 1 ? split * ck : 0;
  const int KT = p.splits > 1 ? min_(kts - kt0, ck) : kts;

  const A a = {(const char *)p.a + (z * p.sa[0] + m0 * p.sa[1]) * ES,
               p.sa[AXIS_A == AXIS_k ? 1 : 2] * ES, p.m - m0, p.k};
  const B b = {(const char *)p.b + (z * p.sb[0] + n0 * p.sb[1]) * ES,
               p.sb[AXIS_B == AXIS_k ? 1 : 2] * ES, p.n - n0, p.k};

  /* [acc] sums the chunk under way, from +0; at the chunk's end it is
     added to [sum], or put as its partial in a split. */
  typename W::tile acc[MI][NI] = {}, sum[MI][NI] = {};
  constexpr int N = MI * NI * 8;
  const int64_t tile = ((int64_t)z * tiles_m + tm) * tiles_n + tn;
  Acc *const part = (Acc *)p.partials + (size_t)tile * p.splits * N * THREADS;

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
  /* The end of tile kt: at a chunk's end before the last, before tile
     [end], the chunk's sum leaves [acc] for [sum]. */
  int end = min_(KT, ck);
  auto close = [&](int kt) __attribute__((always_inline)) {
    if (kt + 1 != end || end == KT) return;
    end = min_(KT, end + ck);
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
      for (int j = 0; j < NI; j++) {
        sum[i][j] += acc[i][j];
        acc[i][j] = typename W::tile{};
      }
  };
  /* The k loop, closing chunks where [closes]. On the 16-row tile, a
     workgroup whose k-tiles are one chunk, a split's or the only one's,
     runs it without [close]: on the R9700 the float16 kernel of 16 x 4096
     x 4096 takes 5% longer with it. The other tiles run faster with it
     (2048 cubed 5%, 512 x 201088 x 2880 7%), and keep it. */
  auto run = [&](auto closes) __attribute__((always_inline)) {
    for (int kt = 0; kt < KT; kt += 2) {
      fetch(0, kt + 2);
      compute(0);
      if constexpr (decltype(closes)::value) close(kt);
      if (kt + 1 < KT) stash(1, 1, kt + 1);
      barrier();
      if (kt + 1 >= KT) break;
      fetch(1, kt + 3);
      compute(1);
      if constexpr (decltype(closes)::value) close(kt + 1);
      if (kt + 2 < KT) stash(0, 0, kt + 2);
      barrier();
    }
  };
  if (BM == 16 && KT <= ck) run(Flag<false>{});
  else run(Flag<true>{});

  /* The last chunk's sum: added, or put as the split's partial. */
  if (p.splits > 1) {
    put(*reinterpret_cast<Acc(*)[N]>(acc), part + (size_t)split * N * THREADS,
        THREADS);
    if (!last(p.tickets + tile, p.splits)) return;
    chain(*reinterpret_cast<Acc(*)[N]>(sum), part, p.splits, THREADS);
  } else
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
      for (int j = 0; j < NI; j++) sum[i][j] += acc[i][j];

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
               lane % 16] = sum[i][j][e];
    barrier();
    store_rows(p, st, LD, min_(ROWS, BM - base), BN, z, m0 + base, n0,
               THREADS);
    barrier();
  }
}

/* The SIMT kernels */

/* A BM x BN tile of outputs per workgroup of 256 work-items, TY rows of
   them by TX, each TM x TN outputs in two halves BM / 2 and BN / 2 apart:
   16 x 16 on a square tile, 4 x 64 on a short one (16 or 8 rows), which
   keeps a sum of few rows from computing rows past m. k-tiles of 64 bytes
   of k (16 floats). Both operands are stored k-major in the local data
   share, the next k-tile loaded into registers while the current one
   computes. Per k-tile, the R * BK elements of an operand of R rows are
   loaded E at a time, E consecutive k of a row or E consecutive rows of a
   k, whichever axis is contiguous, as vectors where it can: E = R * BK /
   256, or 1 by the first R * BK work-items where a tile holds fewer
   elements than a workgroup has work-items. */
template <typename T, int R, int BK> struct Simt_operand {
  static constexpr int E = R * BK >= 256 ? R * BK / 256 : 1;
  static constexpr int LOADERS = R * BK / E;
  static_assert(E <= BK && E <= R, "a work-item loads a run of a row or a k");
  const char *base;
  int64_t s_row, s_k;
  int rows, k, r0;
  bool rows_contiguous, vectors;

  DEVICE void load(T (&v)[E], int64_t k0, int tid) const {
    if (tid >= LOADERS) return;
    const int row = rows_contiguous ? r0 + tid % (R / E) * E : r0 + tid / (BK / E);
    const int64_t kk =
        rows_contiguous ? k0 + tid / (R / E) : k0 + tid % (BK / E) * E;
    if (E * sizeof(T) % 8 == 0 && vectors &&
        (rows_contiguous ? kk < k && row + E - 1 < rows : row < rows && kk + E - 1 < k)) {
      const char *src = base + (row * s_row + kk * s_k) * sizeof(T);
#pragma unroll
      for (int e = 0; e < E * (int)sizeof(T) / 8; e++)
        ((u64 *)v)[e] = ((const u64 *)src)[e];
      return;
    }
#pragma unroll
    for (int e = 0; e < E; e++) {
      const int r = rows_contiguous ? row + e : row;
      const int64_t q = rows_contiguous ? kk : kk + e;
      v[e] = r < rows && q < k ? ((const T *)base)[r * s_row + q * s_k] : T(0);
    }
  }

  /* The loaded elements into the k-major tile [s]. */
  DEVICE void stash(T (*s)[R], const T (&v)[E], int tid) const {
    if (tid >= LOADERS) return;
#pragma unroll
    for (int e = 0; e < E; e++)
      if (rows_contiguous)
        s[tid / (R / E)][tid % (R / E) * E + e] = v[e];
      else
        s[tid % (BK / E) * E + e][tid / (BK / E)] = v[e];
  }
};

DEVICE float fma_(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
DEVICE double fma_(double a, double b, double c) { return __builtin_fma(a, b, c); }
DEVICE u64 fma_(u64 a, u64 b, u64 c) { return a * b + c; }

template <typename T, int BM, int BN> DEVICE void simt_contract(const contract_params &p) {
  constexpr int THREADS = NX_CONTRACT_THREADS;
  constexpr int TY = BM >= 32 ? 16 : 4, TX = THREADS / TY;
  constexpr int TM = BM / TY, TN = BN / TX, HM = TM / 2, HN = TN / 2;
  constexpr int BK = 64 / sizeof(T);
  static_assert(TM % 2 == 0 && TN % 2 == 0, "a work-item's outputs in two halves each way");
  /* Two k-tiles of each operand, then a half tile of outputs. */
  constexpr int TILES = 2 * BK * (BM + BN), HALF = BM / 2 * BN;
  static SHARED __attribute__((aligned(16))) T smem[TILES > HALF ? TILES : HALF];
  T(*as)[BK][BM] = (T(*)[BK][BM])smem;
  T(*bs)[BK][BN] = (T(*)[BK][BN])(smem + 2 * BK * BM);
  const int tid = item(), tx = tid % TX, ty = tid / TX, z = group_z();
  /* Grouped rasterisation, as the WMMA kernels'. */
  const int tiles_m = ceil_div(p.m, BM), tiles_n = ceil_div(p.n, BN);
  const int pid = group(), groups = 8 * tiles_n, first = pid / groups * 8;
  const int gsz = min_(tiles_m - first, 8);
  const int tm = first + pid % groups % gsz, tn = pid % groups / gsz;
  const int m0 = tm * BM, n0 = tn * BN;
  /* k in chunks of [ck] k-tiles: a split's workgroup takes chunk [split],
     the only one takes them all, its k-tiles [kt0, kt1). */
  const int kts = ceil_div(p.k, BK), split = group_y();
  const int ck = p.chunk / BK;
  const int kt0 = p.splits > 1 ? split * ck : 0;
  const int kt1 = p.splits > 1 ? min_(kts, kt0 + ck) : kts;

  const Simt_operand<T, BM, BK> a = {(const char *)p.a + z * p.sa[0] * sizeof(T),
                                     p.sa[1], p.sa[2], p.m, p.k, m0,
                                     p.sa[1] == 1 && p.sa[2] != 1,
                                     (p.aligned & NX_CONTRACT_A_VECTORS) != 0};
  const Simt_operand<T, BN, BK> b = {(const char *)p.b + z * p.sb[0] * sizeof(T),
                                     p.sb[1], p.sb[2], p.n, p.k, n0,
                                     p.sb[1] == 1 && p.sb[2] != 1,
                                     (p.aligned & NX_CONTRACT_B_VECTORS) != 0};

  /* [acc] sums the chunk under way, from +0; at the chunk's end it is
     added to [sum], or put as its partial in a split. */
  constexpr int N = TM * TN;
  T acc[TM][TN], sum[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = sum[i][j] = T(0);
  const int64_t tile = (int64_t)z * tiles_m * tiles_n + pid;
  T *const part = (T *)p.partials + (size_t)tile * p.splits * N * THREADS;
  T ra[Simt_operand<T, BM, BK>::E], rb[Simt_operand<T, BN, BK>::E];
  if (kt0 < kt1) {
    a.load(ra, (int64_t)kt0 * BK, tid);
    b.load(rb, (int64_t)kt0 * BK, tid);
    a.stash(as[0], ra, tid);
    b.stash(bs[0], rb, tid);
  }
  barrier();

  /* A chunk before the last ends before tile [end]. */
  int end = min_(kt1, kt0 + ck);
  for (int kt = kt0, buf = 0; kt < kt1; kt++, buf ^= 1) {
    const bool more = kt + 1 < kt1;
    if (more) {
      a.load(ra, (int64_t)(kt + 1) * BK, tid);
      b.load(rb, (int64_t)(kt + 1) * BK, tid);
    }
    /* Unrolled by two: the loads of every k at once would outgrow the
       registers. */
#pragma unroll 2
    for (int k = 0; k < BK; k++) {
      T va[TM], vb[TN];
#pragma unroll
      for (int e = 0; e < HM; e++)
        va[e] = as[buf][k][ty * HM + e],
        va[HM + e] = as[buf][k][BM / 2 + ty * HM + e];
#pragma unroll
      for (int e = 0; e < HN; e++)
        vb[e] = bs[buf][k][tx * HN + e],
        vb[HN + e] = bs[buf][k][BN / 2 + tx * HN + e];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = fma_(va[i], vb[j], acc[i][j]);
    }
    /* At a chunk's end before the last, its sum leaves [acc] for [sum]. */
    if (kt + 1 == end && more) {
      end = min_(kt1, end + ck);
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) {
          sum[i][j] += acc[i][j];
          acc[i][j] = T(0);
        }
    }
    if (more) {
      a.stash(as[buf ^ 1], ra, tid);
      b.stash(bs[buf ^ 1], rb, tid);
    }
    barrier();
  }

  /* The last chunk's sum: added, or put as the split's partial. */
  if (p.splits > 1) {
    put(*reinterpret_cast<T(*)[N]>(acc), part + (size_t)split * N * THREADS,
        THREADS);
    if (!last(p.tickets + tile, p.splits)) return;
    chain(*reinterpret_cast<T(*)[N]>(sum), part, p.splits, THREADS);
  } else
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++) sum[i][j] += acc[i][j];
  /* The outputs through the local data share, a half of the rows a pass:
     work-item (tx, ty)'s first HM rows, then its last HM. */
#pragma unroll
  for (int half = 0; half < 2; half++) {
#pragma unroll
    for (int i = 0; i < HM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++)
        smem[(ty * HM + i) * BN +
             (j < HN ? tx * HN + j : BN / 2 + tx * HN + j - HN)] =
            sum[half * HM + i][j];
    barrier();
    store_rows(p, smem, BN, BM / 2, BN, z, m0 + half * BM / 2, n0, THREADS);
    barrier();
  }
}

/* The skinny kernels (integers, m <= 16): lane l of 32 sums, for each t,
   the run of four k0 + 128 t + 4 l + u, u < 4, in increasing k, and the
   32 lane sums combine by the balanced tree (l, l + 16), then (l, l + 8),
   (l, l + 4), (l, l + 2), (l, l + 1). Two forms, each its own kernel so
   that each has the registers it needs: column, one wave per output
   column j when b's k axis is contiguous (8 columns a workgroup), lanes
   across k, a run one vector; across, a workgroup of 32 columns by 8 lane
   groups, work-item (x, y) summing the lane classes y, y + 8, y + 16,
   y + 24 of column x, reading rows of b across x. */
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
   [us] of each inside the range, times a's rows below [ROWS] and m, to
   [c] in increasing k: each row's runs loaded together, then summed. a's
   runs are vectors where NX_CONTRACT_A_VECTORS says so. */
template <typename T, int F, int ROWS>
DEVICE void dot(const contract_params &p, T (&c)[16], const char *a,
                const T (&bv)[F][4], int k, int us) {
#pragma unroll
  for (int i = 0; i < ROWS; i++)
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
   that a lane has F in flight, two of 8-byte elements to keep the
   registers they take; then the runs left, one by one. a's rows are
   those below [ROWS] and m; b's runs are vectors where [vectors] says so.
   [k] steps up to a step past [k1], so it counts in 32 bits unsigned; a
   run it reads starts below [k1], so a signed int holds it. */
template <typename T, int ROWS>
DEVICE void add_runs(const contract_params &p, T (&c)[16], const char *a,
                     const char *b, int j, uint32_t k, uint32_t k1,
                     bool vectors) {
  constexpr int F = sizeof(T) == 8 ? 2 : 4;
  for (; k + (F - 1) * 128 + 3 < k1; k += F * 128) {
    T bv[F][4];
#pragma unroll
    for (int t = 0; t < F; t++)
      run(bv[t], b, p.sb[1], p.sb[2], j, k + t * 128, 4, vectors);
    dot<T, F, ROWS>(p, c, a, bv, k, 4);
  }
  for (; k < k1; k += 128) {
    T bv[1][4];
    const int us = k1 - k < 4 ? (int)(k1 - k) : 4;
    run(bv[0], b, p.sb[1], p.sb[2], j, k, us, vectors);
    dot<T, 1, ROWS>(p, c, a, bv, k, us);
  }
}

/* [add_runs] with a loop for one row of a, and one for up to 16. The
   loop for 16 unrolls to some 29 KiB, which a product of one row walked
   by branching over 15 rows a step; on the R9700 it then ran at one of
   two speeds by where its code landed, mod 32 KiB. The sums are the
   same. */
template <typename T>
DEVICE void add_rows(const contract_params &p, T (&c)[16], const char *a,
                     const char *b, int j, uint32_t k, uint32_t k1,
                     bool vectors) {
  if (p.m == 1)
    add_runs<T, 1>(p, c, a, b, j, k, k1, vectors);
  else
    add_runs<T, 16>(p, c, a, b, j, k, k1, vectors);
}

/* The workgroup's sums of its range's [k1] k into [level8], row i of
   column c at element i * 32 + c (its first 512), as the tree leaves them,
   by the form [FORM]. Rows from m on stay zero. */
template <typename T, int FORM>
DEVICE void skinny_sums(const contract_params &p, T (*level8)[8][32],
                        const char *a, const char *b, uint32_t k1) {
  T *sums = &level8[0][0][0];
  if constexpr (FORM == FORM_column) {
    /* Lane l holds class l, its runs vectors where b's rows are
       (NX_CONTRACT_B_VECTORS). */
    const int lane = item() % 32, j = group() * 8 + item() / 32;
    T acc[16];
#pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = T(0);
    if (j < p.n)
      add_rows(p, acc, a, b, j, 4 * lane, k1,
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
        add_rows(p, c[q], a, b, j, 4 * (y + 8 * (h + 2 * q)), k1,
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
  /* The range of k: whole 128-element runs, one t of the lanes, from k0,
     at which a and b start. Its k are counted from k0 in 32 bits unsigned,
     past its end by at most a step. */
  const int runs = ceil_div(p.k, 128);
  const int64_t k0 = range_start(runs, split, splits) * 128;
  const int64_t k1 =
      min_((int64_t)p.k, range_start(runs, split + 1, splits) * 128);
  const char *a =
      (const char *)p.a + (z * p.sa[0] + k0 * p.sa[2]) * sizeof(T);
  const char *b =
      (const char *)p.b + (z * p.sb[0] + k0 * p.sb[2]) * sizeof(T);
  static SHARED T level8[16][8][32];
  skinny_sums<T, FORM>(p, level8, a, b, (uint32_t)(k1 - k0));

  /* Two of the workgroup's sums per work-item: the partials a split sum
     combines. */
  constexpr int cols = FORM == FORM_across ? 32 : 8;
  const T *sums = &level8[0][0][0];
  barrier();
  T v[2] = {sums[item()], sums[256 + item()]};
  if (splits > 1) {
    const int64_t tile = (int64_t)z * ceil_div(p.n, cols) + group();
    T *const part = (T *)p.partials + (size_t)tile * splits * 2 * THREADS;
    put(v, part + (size_t)split * 2 * THREADS, THREADS);
    if (!last(p.tickets + tile, splits)) return;
    chain(v, part, splits, THREADS);
  }
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int e = h * 256 + item(), i = e / 32, c = e % 32;
    if (i < p.m && c < cols) store(p, z, i, group() * cols + c, v[h]);
  }
}

/* The lane kernels: rows of a (the plan makes a the operand of the short
   extent) against the columns of b, each output in Nx_kernel's lane
   order. Its k falls into blocks of 1024 and a block into 16 lanes by k
   modulo 16, each lane a fused chain from +0 in increasing k. Lane i takes
   lane i + 8, then i + 4, i + 2 and i + 1: the block's sum. The blocks sum
   by a tree whose left part holds the largest power of two of blocks below
   their count; then init is added. With no block, y is init's bits, or +0.

   A work-item sums the 16 lanes of one block of one column, for each of
   its workgroup's NX_CONTRACT_LANE_ROWS rows of a, and reduces them to the
   block's sum. G blocks of a column run at once, in chunks: G is k's
   blocks, at most 8, so that past 8 it is a power of two. A wave holds one
   block of 32 columns, so that its work-items share a's loads, and a
   workgroup G blocks of 32 columns; across, b's n axis contiguous, it
   holds 8 / G times as many columns, which measured faster there.
   Through the local data share, each chunk's sums reduce to its tree, and
   the chunks combine by a binary counter: a full chunk's tree merges with
   the counter's equal ones as two subtrees of the blocks' tree do, and the
   last chunk's tree ends the right-nested fold of what the counter holds.
   A column's work-items are one workgroup, so a long k over few columns
   runs on few work-group processors. */
enum {
  LANE_BLOCK = 1024,
  LANE_ROWS = NX_CONTRACT_LANE_ROWS,
  LANE_AT_ONCE = 8,
  LANE_DEPTH = 24
};

/* An element as the accumulator reads it: a float, a double, or a bfloat16's
   bits. */
template <typename T> DEVICE T value(float x) { return x; }
template <typename T> DEVICE T value(double x) { return x; }
template <typename T> DEVICE T value(uint16_t x) {
  return __builtin_bit_cast(float, (uint32_t)x << 16);
}

/* 128 bytes of elements, a cache line: the column form's step. */
template <typename E> struct line {
  E e[128 / sizeof(E)];
};

/* 16 elements, [s] apart: a run, one element of each lane. */
template <typename E> struct run16 {
  E e[16];
};

/* The run of [p] from its first element. */
template <typename E> DEVICE run16<E> load_run(const E *p, int64_t s) {
  run16<E> r;
#pragma unroll
  for (int u = 0; u < 16; u++) r.e[u] = p[u * s];
  return r;
}

/* The line of [p] from element [at], 16-byte aligned, as eight loads. */
template <typename E> DEVICE line<E> load_line(const E *p, int64_t at) {
  const u32x4 *src = (const u32x4 *)__builtin_assume_aligned(p + at, 16);
  u32x4 r[8];
#pragma unroll
  for (int v = 0; v < 8; v++) r[v] = src[v];
  line<E> l;
  __builtin_memcpy(&l, r, sizeof l);
  return l;
}

/* The sum of a block of [n] elements of a row of a at [a] and a column of b
   at [b], [sa] and [sb] elements apart along k: its 16 lanes' chains, then
   their tree. Where both are contiguous along k and aligned ([lines]), a line
   of each at a time while whole lines remain; then a run of each at a time,
   then the last, partial run. */
template <typename T, typename E>
DEVICE T lane_sum(const E *a, int64_t sa, const E *b, int64_t sb, int n,
                  bool lines) {
  constexpr int LINE = 128 / sizeof(E);
  T l[16];
#pragma unroll
  for (int u = 0; u < 16; u++) l[u] = T(0);
  int k = 0;
  if (lines && LINE <= n) {
    /* The next line's loads go out before this one's sums: the barrier keeps
       the compiler from sinking them to their uses, which would leave one
       line in flight. */
    line<E> x = load_line(b, 0), w = load_line(a, 0);
    for (k = LINE; k + LINE <= n; k += LINE) {
      const line<E> x1 = load_line(b, k), w1 = load_line(a, k);
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int e = 0; e < LINE; e++)
        l[e % 16] = fma_(value<T>(w.e[e]), value<T>(x.e[e]), l[e % 16]);
      x = x1;
      w = w1;
    }
#pragma unroll
    for (int e = 0; e < LINE; e++)
      l[e % 16] = fma_(value<T>(w.e[e]), value<T>(x.e[e]), l[e % 16]);
  }
  if (k + 16 <= n) {
    /* Whole runs of 16, the next one's loads out before this one's sums. */
    run16<E> x = load_run(b + k * sb, sb), w = load_run(a + k * sa, sa);
    for (k += 16; k + 16 <= n; k += 16) {
      const run16<E> x1 = load_run(b + k * sb, sb);
      const run16<E> w1 = load_run(a + k * sa, sa);
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int u = 0; u < 16; u++)
        l[u] = fma_(value<T>(w.e[u]), value<T>(x.e[u]), l[u]);
      x = x1;
      w = w1;
    }
#pragma unroll
    for (int u = 0; u < 16; u++)
      l[u] = fma_(value<T>(w.e[u]), value<T>(x.e[u]), l[u]);
  }
  /* The block's last, partial run. */
#pragma unroll
  for (int u = 0; u < 16; u++)
    if (k + u < n)
      l[u] = fma_(value<T>(a[(k + u) * sa]), value<T>(b[(k + u) * sb]), l[u]);
#pragma unroll
  for (int w = 8; w > 0; w /= 2)
#pragma unroll
    for (int x = 0; x < w; x++) l[x] = l[x] + l[x + w];
  return l[0];
}

/* The tree of the [n] block sums at [s], which it overwrites: n's binary
   digits from the lowest, each the perfect tree of the blocks it ends with,
   folded from the right. */
template <typename T> DEVICE T block_tree(T *s, int n) {
  T acc = T(0);
  for (int d = 0, end = n; (1 << d) <= n; d++) {
    if (!(n >> d & 1)) continue;
    const int size = 1 << d, lo = end - size;
    for (int h = 1; h < size; h *= 2)
      for (int x = lo; x + h < lo + size; x += 2 * h) s[x] = s[x] + s[x + h];
    acc = end == n ? s[lo] : s[lo] + acc;
    end = lo;
  }
  return acc;
}

/* y's element (z, i, j) of a sum of no product: init as it is, or +0. */
template <typename T>
DEVICE void no_product(const contract_params &p, int z, int i, int j) {
  const int64_t yi = z * p.sy[0] + i * p.sy[1] + j * p.sy[2];
  if (!p.init) {
    write(p, yi, T(0));
    return;
  }
  const int64_t ii = z * p.si[0] + i * p.si[1] + j * p.si[2];
  if (p.init_dtype != p.y_dtype) {
    write(p, yi, read<T>(p.init, ii, p.init_dtype));
    return;
  }
  switch (p.y_dtype) {
  case NX_FLOAT64:
    ((u64 *)p.y)[yi] = ((const u64 *)p.init)[ii];
    return;
  case NX_FLOAT32:
    ((uint32_t *)p.y)[yi] = ((const uint32_t *)p.init)[ii];
    return;
  case NX_FLOAT16:
  case NX_BFLOAT16:
    ((uint16_t *)p.y)[yi] = ((const uint16_t *)p.init)[ii];
    return;
  }
  ((uint8_t *)p.y)[yi] = ((const uint8_t *)p.init)[ii];
}

template <typename T, typename E, int FORM>
DEVICE void lane_contract(const contract_params &p) {
  const int blocks = ceil_div(p.k, LANE_BLOCK);
  const int G = max_(1, min_(blocks, LANE_AT_ONCE));
  const int C = FORM == FORM_across ? 32 * (LANE_AT_ONCE / G) : 32;
  const int threads = G * C, tid = item(), z = group_z();
  /* The workgroup's rows from r0 and columns from j0. */
  const int column_groups = ceil_div(p.n, C);
  const int r0 = group() / column_groups * LANE_ROWS;
  const int j0 = group() % column_groups * C;
  const int rows = min_(p.m - r0, (int)LANE_ROWS);
  /* Wave w holds block w % G of columns 32 (w / G) on. */
  const int w = tid / 32;
  const int c = w / G * 32 + tid % 32, g = w % G;
  const int j = j0 + c;
  const E *a = (const E *)p.a + z * p.sa[0] + r0 * p.sa[1];
  const E *b = (const E *)p.b + z * p.sb[0];
  static SHARED T sums[LANE_ROWS][NX_CONTRACT_THREADS];
  static SHARED T stack[LANE_ROWS * 32][LANE_DEPTH];

  if (blocks == 0) {
    for (int e = tid; e < LANE_ROWS * C; e += threads)
      if (e / C < rows && j0 + e % C < p.n)
        no_product<T>(p, z, r0 + e / C, j0 + e % C);
    return;
  }
  for (int base = 0; base < blocks; base += G) {
    const int blk = base + g;
    if (j < p.n && blk < blocks) {
      const int k0 = blk * LANE_BLOCK, n = min_(p.k - k0, (int)LANE_BLOCK);
      const bool lines = FORM == FORM_column &&
                         (p.aligned & NX_CONTRACT_A_VECTORS) &&
                         (p.aligned & NX_CONTRACT_B_VECTORS);
      const E *bj = b + j * p.sb[1] + (int64_t)k0 * p.sb[2];
      const E *ak = a + (int64_t)k0 * p.sa[2];
      for (int i = 0; i < rows; i++)
        sums[i][c * G + g] =
            lane_sum<T>(ak + i * p.sa[1], p.sa[2], bj, p.sb[2], n, lines);
    }
    barrier();

    /* Row i, column cc's chunk: its tree, then into the counter, which holds
       the [base / G] chunks before it, or, the last, the fold. */
    const int present = min_(G, blocks - base), full = base / G;
    for (int e = tid; e < LANE_ROWS * C; e += threads) {
      const int i = e / C, cc = e % C;
      if (i >= rows || j0 + cc >= p.n) continue;
      T carry = block_tree(&sums[i][cc * G], present);
      if (base + G < blocks) {
        int d = 0;
        for (; full >> d & 1; d++) carry = stack[e][d] + carry;
        stack[e][d] = carry;
      } else {
        for (int d = 0; d < LANE_DEPTH; d++)
          if (full >> d & 1) carry = stack[e][d] + carry;
        store(p, z, r0 + i, j0 + cc, carry);
      }
    }
    barrier();
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
#define WMMA(name, kind, la, lb, tile)                                        \
  KERNEL SIZED(tile::THREADS) void name(contract_params p) {                   \
    wmma_contract<KIND_##kind, AXIS_##la, AXIS_##lb, tile>(p);                 \
  }
#define SIMT(name, acc, bm, bn)                                                \
  KERNEL SIZED(NX_CONTRACT_THREADS) void name(contract_params p) {             \
    simt_contract<ACC_##acc, bm, bn>(p);                                       \
  }
#define SKINNY(name, acc, form)                                                \
  KERNEL SIZED(NX_CONTRACT_THREADS) void name(contract_params p) {             \
    skinny_contract<ACC_##acc, FORM_##form>(p);                                \
  }
#define ELEM_f32 float
#define ELEM_f64 double
#define ELEM_bf16 uint16_t
#define LANE(name, acc, elem, form)                                            \
  KERNEL UP_TO(NX_CONTRACT_THREADS) void name(contract_params p) {             \
    lane_contract<ACC_##acc, ELEM_##elem, FORM_##form>(p);                     \
  }
NX_AMD_KERNELS(DEFINE)
