/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The matrix-unit kernels: float16, bfloat16, float8 or 8-bit integer
   operands summed in float32 or int32 on RDNA4's WMMA instructions, each
   16 x 16 outputs of 16 terms on a wave of 32. An output's accumulator
   starts from +0 in each of the plan's ranges of k, and each range sums
   its k in increasing k, 16 at a time on an instruction. The instruction adds its terms in steps, each
   step's products and the running sum aligned to one grid, truncated and
   rounded once (nx_amd.mli states it): four steps of 4 terms of 16-bit
   floats, two of 8 of float8. A lane's fragment of 16-bit floats holds a
   row's k 0-3 and 8-11 (lanes 16 to 31: 4-7 and 12-15), so that the
   unit's steps, over its layout's k 0-3, 8-11, 4-7 then 12-15, take the
   terms in increasing k. The ranges' sums add in order, s0 + s1 + ...,
   then init, each addition of two NaNs keeping the first, and the result
   rounds once to y's dtype.

     global --16-byte loads, the next k-tile's held in registers while
     this one sums--> LDS k-tile, a row of k per output row or column,
     two stages --> fragments --> WMMA --> accumulators (from +0)
     --> combine (ranges) --> + init --> y

   Part of contract.hip's translation unit, in its namespace. */

#ifndef NX_AMD_WMMA_H
#define NX_AMD_WMMA_H

/* The pairs of operands an instruction multiplies, by the operands'
   dtypes: both float16, both bfloat16, the four float8 pairs, the four
   8-bit integer pairs (signed or not, each). */
enum {
  W_F16,
  W_BF16,
  W_FP8_FP8,
  W_FP8_BF8,
  W_BF8_FP8,
  W_BF8_BF8,
  W_S8_S8,
  W_S8_U8,
  W_U8_S8,
  W_U8_U8,
};

DEVICE int wmma_kind(int a, int b) {
  if (a == NX_FLOAT16) return W_F16;
  if (a == NX_BFLOAT16) return W_BF16;
  const bool fa = a == NX_FLOAT8_E4M3FN, fb = b == NX_FLOAT8_E4M3FN;
  if (a == NX_FLOAT8_E4M3FN || a == NX_FLOAT8_E5M2)
    return fa ? (fb ? W_FP8_FP8 : W_FP8_BF8) : (fb ? W_BF8_FP8 : W_BF8_BF8);
  const bool sa = a == NX_INT8, sb = b == NX_INT8;
  return sa ? (sb ? W_S8_S8 : W_S8_U8) : (sb ? W_U8_S8 : W_U8_U8);
}

typedef _Float16 w_f16x8 __attribute__((ext_vector_type(8)));
typedef __bf16 w_bf16x8 __attribute__((ext_vector_type(8)));
typedef float w_f32x8 __attribute__((ext_vector_type(8)));
typedef int w_i32x8 __attribute__((ext_vector_type(8)));
typedef int w_i32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t w_u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t w_u32x2 __attribute__((ext_vector_type(2)));

/* A kind's element bytes, accumulator and fragment: a lane's 8 elements
   of a row along k. */
template <int KIND> constexpr int w_bytes = KIND <= W_BF16 ? 2 : 1;
template <int KIND> constexpr bool w_int = KIND >= W_S8_S8;
template <int KIND>
using w_acc = typename conditional<w_int<KIND>, w_i32x8, w_f32x8>::type;
template <int KIND>
using w_frag = typename conditional<w_bytes<KIND> == 2, w_u32x4, w_u32x2>::type;

/* One instruction of [KIND] into [c]. */
template <int KIND>
DEVICE void wmma(w_acc<KIND> &c, w_frag<KIND> a, w_frag<KIND> b) {
  if constexpr (KIND == W_F16)
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        __builtin_bit_cast(w_f16x8, a), __builtin_bit_cast(w_f16x8, b), c);
  else if constexpr (KIND == W_BF16)
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
        __builtin_bit_cast(w_bf16x8, a), __builtin_bit_cast(w_bf16x8, b), c);
  else {
    const w_i32x2 ia = __builtin_bit_cast(w_i32x2, a);
    const w_i32x2 ib = __builtin_bit_cast(w_i32x2, b);
    if constexpr (KIND == W_FP8_FP8)
      c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(ia, ib, c);
    else if constexpr (KIND == W_FP8_BF8)
      c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(ia, ib, c);
    else if constexpr (KIND == W_BF8_FP8)
      c = __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(ia, ib, c);
    else if constexpr (KIND == W_BF8_BF8)
      c = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(ia, ib, c);
    else
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
          KIND == W_S8_S8 || KIND == W_S8_U8, ia,
          KIND == W_S8_S8 || KIND == W_U8_S8, ib, c, false);
  }
}

/* A group of WM x WN waves computes BM x BN outputs over k-tiles of BKB
   bytes a row; a wave's outputs are MI x NI instructions' 16 x 16. An LDS
   row holds a k-tile's row and 16 bytes of padding, so that a wave's
   fragment reads spread over the banks. */
template <int BM_, int BN_, int BKB_, int WM_, int WN_> struct Wtile {
  static constexpr int BM = BM_, BN = BN_, BKB = BKB_, WM = WM_, WN = WN_;
  static constexpr int THREADS = WM * WN * 32;
  static constexpr int MI = BM / WM / 16, NI = BN / WN / 16;
  static constexpr int LD = BKB + 16;
  static constexpr int STAGE = (BM + BN) * LD;
  static constexpr int LDS = 2 * STAGE;
  static_assert(MI * WM * 16 == BM && NI * WN * 16 == BN, "tile shape");
  static_assert(NX_WMMA_RUN % (BKB / 2) == 0 && NX_WMMA_RUN % BKB == 0,
                "a run is whole k-tiles");
};

/* An operand of R rows (a's, or b's columns) of ES-byte elements whose k
   (or, with NX_OP_FREE, its rows) is contiguous, in 16-byte vectors: its
   k-tile into an LDS tile of rows along k. A thread holds its chunks in
   registers from their load to their store, so that a k-tile's loads are
   in flight while the group sums the one before. A k-tile not whole along
   k, or by rows along the tile's rows, a condition the group shares,
   loads at its store, element by element straight into LDS: rows past the
   operand's read its last row (their outputs are never stored), and k
   past the range reads +0, a zero product, which takes no part in a step's
   grids and leaves its sum: a zero total is +0 either way. */
template <class C, int R, int ES> struct Wop {
  static constexpr int BKB = C::BKB, BK = BKB / ES, V = 16 / ES;
  static constexpr int THREADS = C::THREADS, LD = C::LD;
  static constexpr int CH = R * BKB / 16; /* chunks a k-tile */
  static constexpr int NC = (CH + THREADS - 1) / THREADS;
  static constexpr int RV = R / V > 0 ? R / V : 1, KV = BK / V;
  static_assert(R % V == 0, "whole vectors");
  typedef typename conditional<ES == 2, uint16_t, uint8_t>::type E;

  w_u32x4 v[NC];
  const E *ptr[NC]; /* each chunk's first element, the k-tile to load */
  int64_t step;     /* elements a k-tile */
  int r0, rows;
  int edge;         /* the k inside a k-tile loaded at its store, or 0 */
  bool rmode;

  /* Chunk [i]'s k and row in the tile. */
  DEVICE_MEMBER void at(int i, int &kk, int &rr) const {
    const int c = item() + i * THREADS;
    if (rmode) kk = c / RV, rr = c % RV * V;
    else rr = c / KV, kk = c % KV * V;
  }
  DEVICE_MEMBER bool live(int i) const {
    return CH % THREADS == 0 || item() + i * THREADS < CH;
  }

  DEVICE_MEMBER void make(const contract_operand &o, int z, int r0_,
                          int64_t k0) {
    r0 = r0_, rows = o.rows, rmode = (o.flags & NX_OP_FREE) != 0, edge = 0;
    step = BK * o.s[2];
    const E *base = (const E *)o.p + z * o.s[0];
#pragma unroll
    for (int i = 0; i < NC; i++) {
      int kk, rr;
      at(i, kk, rr);
      /* Along k, a stride of 1, or 0 where the operand repeats an element
         along k, which its rows hold in BK copies (contract.ml). */
      ptr[i] = rmode ? base + r0 + rr + (k0 + kk) * o.s[2]
                     : base + min(r0 + rr, rows - 1) * o.s[1] + k0 * o.s[2] + kk;
    }
  }

  /* The next k-tile, [kl] of its k inside. */
  DEVICE_MEMBER void load(int kl) {
    if (kl < BK || (rmode && r0 + R > rows)) {
      edge = kl;
      return;
    }
#pragma unroll
    for (int i = 0; i < NC; i++) {
      if (!live(i)) continue;
      v[i] = *(const w_u32x4 *)ptr[i];
      ptr[i] += step;
    }
  }

  /* The k-tile into the LDS tile at [s]: a chunk along k as one vector, a
     chunk along the rows element by element down its column of k. */
  DEVICE_MEMBER void store(uint8_t *s) {
#pragma unroll
    for (int i = 0; i < NC; i++) {
      if (!live(i)) continue;
      int kk, rr;
      at(i, kk, rr);
      if (edge) {
        const int left = rmode ? rows - (r0 + rr) : V; /* rows inside */
#pragma unroll 1
        for (int u = 0; u < V; u++) {
          const bool in = rmode ? kk < edge : kk + u < edge;
          const E e = !in ? E(0) : rmode ? ptr[i][min(u, left - 1)] : ptr[i][u];
          *(E *)(s + (rmode ? (rr + u) * LD + kk * ES : rr * LD + (kk + u) * ES)) = e;
        }
        ptr[i] += step;
      } else if (!rmode)
        *(w_u32x4 *)(s + rr * LD + kk * ES) = v[i];
      else {
        const E *e = (const E *)&v[i];
#pragma unroll
        for (int u = 0; u < V; u++) *(E *)(s + (rr + u) * LD + kk * ES) = e[u];
      }
    }
    edge = 0;
  }
};

/* A lane's fragment of row [r] of the LDS tile at [s], at its k [kb]:
   for 16-bit elements k kb + 4h to kb + 4h + 3 and kb + 8 + 4h to
   kb + 11 + 4h, h = lane / 16, each an 8-byte read, so that the unit's
   steps take the terms in increasing k; for 8-bit ones kb + 8h to
   kb + 8h + 7. */
template <int KIND, int LD>
DEVICE w_frag<KIND> wfrag(const uint8_t *s, int r, int kb, int lane) {
  const uint8_t *row = s + r * LD;
  const int h = lane / 16;
  if constexpr (w_bytes<KIND> == 2) {
    const w_u32x2 lo = *(const w_u32x2 *)(row + 2 * (kb + 4 * h));
    const w_u32x2 hi = *(const w_u32x2 *)(row + 2 * (kb + 8 + 4 * h));
    return w_u32x4{lo[0], lo[1], hi[0], hi[1]};
  } else
    return *(const w_u32x2 *)(row + kb + 8 * h);
}

/* init's element (z, i, j) as the accumulator holds it: float32,
   bfloat16 or float16 decoded into float32, a 32-bit integer's bits (the
   plan casts another init into float32 or int32). */
template <int KIND>
DEVICE auto winit(const contract_params &p, int z, int i, int j) {
  const int64_t at = z * p.si[0] + (int64_t)i * p.si[1] + (int64_t)j * p.si[2];
  if constexpr (w_int<KIND>)
    return ((const int32_t *)p.init)[at];
  else {
    if (p.init_dtype == NX_BFLOAT16)
      return nx_bf16_to_float(((const uint16_t *)p.init)[at]);
    if (p.init_dtype == NX_FLOAT16)
      return nx_f16_to_float(((const uint16_t *)p.init)[at]);
    return ((const float *)p.init)[at];
  }
}

/* Output (z, i, j) of [v] into y: float32, or rounded once into bfloat16
   or float16 by nx_dtype.h's encoders, or a 32-bit integer's bits (the
   plan casts into another y). */
template <int KIND, typename T>
DEVICE void wstore(const contract_params &p, int z, int i, int j, T v) {
  const int64_t at = z * p.sy[0] + (int64_t)i * p.sy[1] + (int64_t)j * p.sy[2];
  if constexpr (w_int<KIND>)
    ((int32_t *)p.y)[at] = v;
  else if (p.y_dtype == NX_BFLOAT16)
    ((uint16_t *)p.y)[at] = nx_float_to_bf16(v);
  else if (p.y_dtype == NX_FLOAT16)
    ((uint16_t *)p.y)[at] = nx_float_to_f16(v);
  else
    ((float *)p.y)[at] = v;
}

/* The matrix-unit contraction of [KIND] on tiles [C]: group (x, y, z) is
   tile x of batch element z's [tiles_m] x [tiles_n], in groups of 8 row
   tiles that sweep the column tiles together so that a column tile's b
   stays in L2, and range y of its [s1] ranges of k, whole runs of
   NX_WMMA_RUN terms shared out in order (k_range). */
template <int KIND, class C> DEVICE void wmma_contract(const contract_params &p) {
  typedef w_acc<KIND> Acc;
  /* An integer sum's ranges add as unsigned words, wrapping. */
  typedef typename conditional<w_int<KIND>, uint32_t, float>::type T;
  constexpr int ES = w_bytes<KIND>, BK = C::BKB / ES;
  constexpr int BM = C::BM, BN = C::BN, MI = C::MI, NI = C::NI;
  constexpr int THREADS = C::THREADS, LD = C::LD;
  extern SHARED __attribute__((aligned(16))) unsigned char smem_raw[];
  uint8_t *smem = (uint8_t *)smem_raw;

  const int z = (int)group_z(), split = (int)group_y(), splits = p.s1;
  const int pid = (int)group_x(), group = 8 * p.tiles_n;
  const int first = pid / group * 8, gsz = min(p.tiles_m - first, 8);
  const int tm = first + pid % group % gsz, tn = pid % group / gsz;
  const int m0 = tm * BM, n0 = tn * BN;
  int64_t k0, k1;
  k_range<NX_WMMA_RUN>(p.k, split, splits, k0, k1);

  /* A wave's outputs: WCOLS columns from wn0, MI * 16 rows from wm0, an
     instruction's output (i, j) e (e < 8) on lane l at row 16 i + 8 (l /
     16) + e, column 16 j + l mod 16. */
  constexpr int WCOLS = NI * 16;
  static_assert(C::WM * C::WN * 16 * WCOLS * 4 <= C::LDS, "LDS pieces");
  const int tid = item(), lane = tid % 32, wave = tid / 32;
  const int wm0 = wave / C::WN * (BM / C::WM), wn0 = wave % C::WN * (BN / C::WN);

  Wop<C, BM, ES> la;
  Wop<C, BN, ES> lb;
  la.make(p.a, z, m0, k0);
  lb.make(p.b, z, n0, k0);
  auto inside = [&](int64_t kb) { return (int)min((int64_t)BK, k1 - kb); };

  Acc acc[MI][NI];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NI; j++) acc[i][j] = Acc{};

  /* Two stages: k-tile kt + 1's loads are in flight while kt sums, then
     stored into the stage kt - 1 left; one barrier a k-tile. */
  const int64_t KT = (k1 - k0 + BK - 1) / BK;
  if (KT > 0) {
    la.load(inside(k0)), lb.load(inside(k0));
    la.store(smem), lb.store(smem + BM * LD);
  }
  barrier();
#pragma unroll 1
  for (int64_t kt = 0; kt < KT; kt++) {
    const int s = (int)(kt & 1);
    const int64_t kn = k0 + (kt + 1) * BK;
    if (kt + 1 < KT) la.load(inside(kn)), lb.load(inside(kn));
    const uint8_t *as = smem + s * C::STAGE, *bs = as + BM * LD;
#pragma unroll
    for (int kb = 0; kb < BK; kb += 16) {
      w_frag<KIND> fa[MI], fb[NI];
#pragma unroll
      for (int i = 0; i < MI; i++)
        fa[i] = wfrag<KIND, LD>(as, wm0 + 16 * i + lane % 16, kb, lane);
#pragma unroll
      for (int j = 0; j < NI; j++)
        fb[j] = wfrag<KIND, LD>(bs, wn0 + 16 * j + lane % 16, kb, lane);
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NI; j++) wmma<KIND>(acc[i][j], fa[i], fb[j]);
    }
    if (kt + 1 < KT) {
      uint8_t *next = smem + (s ^ 1) * C::STAGE;
      la.store(next), lb.store(next + BM * LD);
    }
    barrier();
  }

  if (splits > 1) {
    constexpr int N = MI * NI * 8;
    T(&flat)[N] = *reinterpret_cast<T(*)[N]>(acc);
    const int64_t tile = (int64_t)z * p.tiles_m * p.tiles_n + pid;
    if (!combine<T, N, THREADS>(flat, (T *)p.partials + tile * splits * N * THREADS,
                                p.tickets + tile, split, splits))
      return;
  }

  /* A wave's outputs pass through its own piece of LDS a row of
     instructions at a time, so that one loop, not unrolled, adds init and
     stores them, a row's columns from consecutive lanes, with the
     decoders in one place. An integer sum wraps, so init's place in it
     changes no bit. */
  T *own = (T *)smem + wave * 16 * WCOLS;
#pragma unroll
  for (int i = 0; i < MI; i++) {
#pragma unroll
    for (int j = 0; j < NI; j++)
#pragma unroll
      for (int e = 0; e < 8; e++)
        own[(8 * (lane / 16) + e) * WCOLS + 16 * j + lane % 16] = (T)acc[i][j][e];
    wave_barrier();
#pragma unroll 1
    for (int e = lane; e < 16 * WCOLS; e += 32) {
      const int r = m0 + wm0 + 16 * i + e / WCOLS, c = n0 + wn0 + e % WCOLS;
      if (r >= p.m || c >= p.n) continue;
      T v = own[e];
      if (p.init) {
        const T x = winit<KIND>(p, z, r, c);
        v = kept(v, x, v + x);
      }
      wstore<KIND>(p, z, r, c, v);
    }
    wave_barrier();
  }
}

/* The kernel of [KIND], or W_ANY's: each kind's by the operands' dtypes,
   each compiled whole and apart so that none keeps another's registers
   live; W_8BIT's: the float8 and 8-bit integer kinds'. The float8 kinds
   differ only in their instruction, yet one body for them, its loop over
   the k-tiles chosen by kind, takes 256 VGPRs and spills; apart, the
   8-bit tile takes 216. */
enum { W_ANY = 100, W_8BIT = 101 };
#define W_KIND_f16 W_F16
#define W_KIND_bf16 W_BF16
#define W_KIND_any W_ANY
#define W_KIND_8bit W_8BIT

template <int KIND, class C> DEVICE void wmma_kernel(const contract_params &p) {
  if constexpr (KIND < W_ANY)
    wmma_contract<KIND, C>(p);
  else {
    const int k = wmma_kind(p.a.dtype, p.b.dtype);
    if constexpr (KIND == W_ANY) {
      if (k == W_F16) return wmma_contract<W_F16, C>(p);
      if (k == W_BF16) return wmma_contract<W_BF16, C>(p);
    }
    if (k == W_FP8_FP8) wmma_contract<W_FP8_FP8, C>(p);
    else if (k == W_FP8_BF8) wmma_contract<W_FP8_BF8, C>(p);
    else if (k == W_BF8_FP8) wmma_contract<W_BF8_FP8, C>(p);
    else if (k == W_BF8_BF8) wmma_contract<W_BF8_BF8, C>(p);
    else if (k == W_S8_S8) wmma_contract<W_S8_S8, C>(p);
    else if (k == W_S8_U8) wmma_contract<W_S8_U8, C>(p);
    else if (k == W_U8_S8) wmma_contract<W_U8_S8, C>(p);
    else if (k == W_U8_U8) wmma_contract<W_U8_U8, C>(p);
  }
}

#endif
