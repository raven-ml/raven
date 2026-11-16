/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contract: y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k]
   (kernels.h's contract_ words), by fused multiply-adds on SIMT tiles
   (GEMM, GEMV) for every accumulator and dtype pair. A pack copies an
   operand the kernels do not read where it lies into the workspace.

     global --(load: native, or convert: decode, expand)--> registers
       --> LDS k-major tile, two stages --> fragments --> FMA
       --> combine (split tiles) --> + init --> cast --> y

   Association of the SIMT kernels (what fixes a float output's bits): the
   plan gives each tile S ranges of k (S = s1 for the call's first [main]
   tiles, s2 for the others), whole runs of NX_CONTRACT_RUN terms shared
   out in order, as k_range computes. A range sums its terms in increasing
   k from +0, one fused multiply-add each; the ranges' sums add in
   increasing k; then init, each addition of two NaNs keeping the first
   ({!kept}); then one rounding into y. An output without
   products is init itself. With NX_CONTRACT_INIT_FIRST the first range
   sums from init instead, and init is not added after. Loads, layouts and
   packs change no bit beyond the values the terms take: the plan chooses
   the tiles and S from the dtypes and the extents alone.

   The inner loop is the same for every dtype: only the load differs. A
   native load (operands of the compute type, in 16-byte vectors) copies;
   a converting load decodes each element once per tile, with one switch
   on the dtype per operand per k-tile, outside the FMA loop. A complex
   call is the real contraction of its parts: a's (re, im) at k' = 2p,
   2p + 1, b's as the 2 x 2 block [[re, -im], [im, re]] of columns 2j,
   2j + 1, so the FMA loop sums the contract's numbered terms in order.

   A kernel reads its words once, into a struct the compiler keeps in
   registers; a pack stages its axes' words in LDS, which its loop over
   the axes reads. */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "device.h"
#include "nx_dtype.h"

typedef unsigned long long u64;

namespace nx_contract {

/* What CUDA gives and HIP without its headers does not. */

DEVICE float as_float(uint32_t x) { return __builtin_bit_cast(float, x); }
DEVICE uint32_t float_word(float x) { return __builtin_bit_cast(uint32_t, x); }
DEVICE double as_double(u64 x) { return __builtin_bit_cast(double, x); }
DEVICE int64_t double_word(double x) { return __builtin_bit_cast(int64_t, x); }

struct alignas(16) uint4 {
  uint32_t x, y, z, w;
};

template <bool C, class T, class F> struct conditional { typedef T type; };
template <class T, class F> struct conditional<false, T, F> { typedef F type; };
template <typename T, T v> struct integral_constant {
  static constexpr T value = v;
};

/* Narrow floats: float16 by the hardware's conversion, exact but for a
   NaN, which it may quiet; float8 e4m3 by nx_dtype.h's decoder. For
   operands, whose NaN sums have no stated bits. */
DEVICE float f16_float(uint32_t h) {
  return (float)__builtin_bit_cast(_Float16, (uint16_t)h);
}
DEVICE float e4m3_float(uint32_t c) {
  return nx_bits_to_float(NX_FLOAT8_E4M3FN, c & 0xFF);
}

/* Words */

/* An operand as contract_ words state it (NX_OPERAND_). */
struct contract_operand {
  const void *p;
  int64_t s[3];
  int32_t rows, dtype, flags;
};

/* A contraction's words (NX_CONTRACT_), addresses 0 for none. */
struct contract_params {
  contract_operand a, b;
  const void *init;
  void *y;
  int64_t si[3], sy[3];
  int32_t init_dtype, y_dtype, acc_dtype, flags;
  int32_t batch, m, n, kgroups;
  int64_t k;
  int32_t tiles_m, tiles_n, main, s1, s2, groups;
  void *partials;
  uint32_t *tickets;
  const int32_t *perm, *desc, *ntiles;
  const void *kperm;
  const int64_t *koff;
  int64_t sbg, syg, sig;
};

template <typename T> DEVICE T *pointer(int64_t w) {
  return (T *)(uintptr_t)w;
}

DEVICE contract_operand operand_words(const int64_t *w) {
  contract_operand o;
  o.p = pointer<const void>(uniform(w[NX_OPERAND_P]));
  for (int i = 0; i < 3; i++) o.s[i] = uniform(w[NX_OPERAND_S + i]);
  o.rows = (int32_t)uniform(w[NX_OPERAND_ROWS]);
  o.dtype = (int32_t)uniform(w[NX_OPERAND_DTYPE]);
  o.flags = (int32_t)uniform(w[NX_OPERAND_FLAGS]);
  return o;
}

/* The words a contraction kernel reads, from [w], their copy in LDS
   ({!stage}), in scalar registers. */
DEVICE contract_params contract_words(const int64_t *w) {
  contract_params p;
  p.a = operand_words(w + NX_CONTRACT_A);
  p.b = operand_words(w + NX_CONTRACT_B);
  p.init = pointer<const void>(uniform(w[NX_CONTRACT_INIT]));
  p.y = pointer<void>(uniform(w[NX_CONTRACT_Y]));
  for (int i = 0; i < 3; i++)
    p.si[i] = uniform(w[NX_CONTRACT_SI + i]), p.sy[i] = uniform(w[NX_CONTRACT_SY + i]);
  p.init_dtype = (int32_t)uniform(w[NX_CONTRACT_INIT_DTYPE]);
  p.y_dtype = (int32_t)uniform(w[NX_CONTRACT_Y_DTYPE]);
  p.acc_dtype = (int32_t)uniform(w[NX_CONTRACT_ACC_DTYPE]);
  p.flags = (int32_t)uniform(w[NX_CONTRACT_FLAGS]);
  p.batch = (int32_t)uniform(w[NX_CONTRACT_BATCH]);
  p.m = (int32_t)uniform(w[NX_CONTRACT_M]);
  p.n = (int32_t)uniform(w[NX_CONTRACT_N]);
  p.kgroups = (int32_t)uniform(w[NX_CONTRACT_KGROUPS]);
  p.k = uniform(w[NX_CONTRACT_K]);
  p.tiles_m = (int32_t)uniform(w[NX_CONTRACT_TILES_M]);
  p.tiles_n = (int32_t)uniform(w[NX_CONTRACT_TILES_N]);
  p.main = (int32_t)uniform(w[NX_CONTRACT_MAIN]);
  p.s1 = (int32_t)uniform(w[NX_CONTRACT_S1]);
  p.s2 = (int32_t)uniform(w[NX_CONTRACT_S2]);
  p.groups = (int32_t)uniform(w[NX_CONTRACT_GROUPS]);
  p.partials = pointer<void>(uniform(w[NX_CONTRACT_PARTIALS]));
  p.tickets = pointer<uint32_t>(uniform(w[NX_CONTRACT_TICKETS]));
  p.perm = pointer<const int32_t>(uniform(w[NX_CONTRACT_PERM]));
  p.desc = pointer<const int32_t>(uniform(w[NX_CONTRACT_DESC]));
  p.ntiles = pointer<const int32_t>(uniform(w[NX_CONTRACT_NTILES]));
  p.kperm = pointer<const void>(uniform(w[NX_CONTRACT_KPERM]));
  p.koff = pointer<const int64_t>(uniform(w[NX_CONTRACT_KOFF]));
  p.sbg = uniform(w[NX_CONTRACT_SBG]);
  p.syg = uniform(w[NX_CONTRACT_SYG]);
  p.sig = uniform(w[NX_CONTRACT_SIG]);
  return p;
}

/* A pack's words (NX_PACK_), staged in LDS. */
struct pack_params {
  const void *src;
  void *dst;
  const void *scale;
  int64_t src_first, dst_first, scale_first, numel, block, groups;
  int32_t rank, src_dtype, dst_dtype, scale_dtype, mode, acc_dtype;
  bool small;
  const int64_t *ax;
};

DEVICE pack_params pack_words(const int64_t *w) {
  pack_params p;
  p.src = pointer<const void>(uniform(w[NX_PACK_SRC]));
  p.dst = pointer<void>(uniform(w[NX_PACK_DST]));
  p.scale = pointer<const void>(uniform(w[NX_PACK_SCALE]));
  p.src_first = uniform(w[NX_PACK_SRC_FIRST]);
  p.dst_first = uniform(w[NX_PACK_DST_FIRST]);
  p.scale_first = uniform(w[NX_PACK_SCALE_FIRST]);
  p.numel = uniform(w[NX_PACK_NUMEL]);
  p.block = uniform(w[NX_PACK_BLOCK_WORD]);
  p.groups = uniform(w[NX_PACK_GROUPS]);
  p.rank = (int32_t)uniform(w[NX_PACK_RANK]);
  p.src_dtype = (int32_t)uniform(w[NX_PACK_SRC_DTYPE]);
  p.dst_dtype = (int32_t)uniform(w[NX_PACK_DST_DTYPE]);
  p.scale_dtype = (int32_t)uniform(w[NX_PACK_SCALE_DTYPE]);
  p.mode = (int32_t)uniform(w[NX_PACK_MODE]);
  p.acc_dtype = (int32_t)uniform(w[NX_PACK_ACC_DTYPE]);
  p.small = uniform(w[NX_PACK_SMALL]);
  p.ax = w + NX_PACK_AX;
  return p;
}

/* Ranges */

/* Range [s] of [S] of [k] terms: its runs of RUN terms shared out in
   order, range s taking runs s * runs / S to (s + 1) * runs / S, the last
   cut at k. The bounds depend on k, S and RUN alone. */
template <int RUN>
DEVICE void k_range(int64_t k, int s, int S, int64_t &k0, int64_t &k1) {
  const int64_t runs = (k + RUN - 1) / RUN;
  k0 = s * runs / S * RUN;
  k1 = min(k, (s + 1) * runs / S * RUN);
}

/* [v], this thread's N values of its group's range sum, the first [live]
   of them computed; the tile's [splits] ranges' sums go to [part] (N *
   THREADS values each, in range order) and its ticket is [ticket], 0
   before the first group arrives. The last group of the tile to take its
   ticket (last_of) adds them in range order, s0 + s1 + ..., whichever
   order the groups ran in, a NaN sum keeping the first NaN ({!kept}), and
   returns true with the tile's sums in [v];
   the others return false, and stop there. */
template <typename T, int N, int THREADS>
DEVICE bool combine(T (&v)[N], T *part, uint32_t *ticket, int split,
                    int splits, int live = N) {
  const int tid = item();
  T *mine = part + (size_t)split * N * THREADS;
#pragma unroll
  for (int f = 0; f < N; f++)
    if (f < live) mine[f * THREADS + tid] = v[f];
  if (!last_of(ticket, (uint32_t)splits)) return false;
#pragma unroll
  for (int f = 0; f < N; f++)
    if (f < live) v[f] = part[f * THREADS + tid];
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
          if (f < live) x[a][f] = part[(size_t)(r + a) * N * THREADS + f * THREADS + tid];
#pragma unroll
    for (int a = 0; a < AHEAD; a++)
      if (r + a < splits)
#pragma unroll
        for (int f = 0; f < N; f++)
          if (f < live) v[f] = kept(v[f], x[a][f], v[f] + x[a][f]);
  }
  return true;
}

#include "wmma.h"

/* Elements */

/* a * b + c in T: one rounding for floats, wrapping for integers. */
DEVICE float fma_(float a, float b, float c) {
  return __builtin_fmaf(a, b, c);
}
DEVICE double fma_(double a, double b, double c) {
  return __builtin_fma(a, b, c);
}
DEVICE uint32_t fma_(uint32_t a, uint32_t b, uint32_t c) {
  return a * b + c;
}
DEVICE u64 fma_(u64 a, u64 b, u64 c) { return a * b + c; }

template <typename T> constexpr bool is_float = T(0.5) != T(0);

/* The bytes an element of [fmt] takes in the kernels' operands: a
   sub-byte dtype's, packed one a byte, one. */
DEVICE constexpr int width_of(int fmt) {
  return fmt == NX_COMPLEX128 ? 16
         : fmt == NX_FLOAT64 || fmt == NX_INT64 || fmt == NX_UINT64 ||
                 fmt == NX_COMPLEX64
             ? 8
         : fmt == NX_FLOAT32 || fmt == NX_INT32 || fmt == NX_UINT32 ? 4
         : fmt == NX_FLOAT16 || fmt == NX_BFLOAT16 || fmt == NX_INT16 ||
                 fmt == NX_UINT16
             ? 2
             : 1;
}

/* The bits an element of [fmt] takes in memory. */
DEVICE int bits_of(int fmt) {
  return fmt == NX_BIT ? 1
         : fmt == NX_INT4 || fmt == NX_UINT4 || fmt == NX_FLOAT4_E2M1FN
             ? 4
             : 8 * width_of(fmt);
}

/* An integer accumulator's range, as a cast saturates a float into it. */
DEVICE void range_of(int acc, int64_t &lo, int64_t &hi) {
  switch (acc) {
  case NX_INT32: lo = INT32_MIN, hi = INT32_MAX; return;
  case NX_INT16: lo = INT16_MIN, hi = INT16_MAX; return;
  case NX_UINT16: lo = 0, hi = UINT16_MAX; return;
  case NX_INT8: lo = INT8_MIN, hi = INT8_MAX; return;
  case NX_UINT8: lo = 0, hi = UINT8_MAX; return;
  case NX_INT4: lo = -8, hi = 7; return;
  case NX_UINT4: lo = 0, hi = 15; return;
  default: lo = 0, hi = UINT32_MAX;
  }
}

/* An element of the dtype [fmt], its bits [lo] and [hi], as the compute
   type T holds it, as a cast into the accumulator [acc] stores it: the
   real part of a complex, rounded to nearest into a float, toward zero and
   saturated into an integer accumulator, an integer wrapped into it; [im]
   gets a complex element's imaginary part, else +0. A sub-byte element's
   code is in the low bits of [lo]. */
template <typename T>
DEVICE T decode(int fmt, int acc, u64 lo, u64 hi, T &im) {
  im = T(0);
  if constexpr (is_float<T>) {
    switch (fmt) {
    case NX_FLOAT64: return (T)as_double(lo);
    case NX_FLOAT32: return (T)as_float((uint32_t)lo);
    /* nx_dtype.h's decoders, which keep a NaN's sign and payload where
       the hardware's conversions make it canonical. */
    case NX_FLOAT16: case NX_BFLOAT16: case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: case NX_FLOAT4_E2M1FN:
      return (T)nx_bits_to_float(fmt, (uint32_t)lo);
    case NX_INT64: return (T)(int64_t)lo;
    case NX_UINT64: return (T)lo;
    case NX_INT32: return (T)(int32_t)lo;
    case NX_UINT32: return (T)(uint32_t)lo;
    case NX_INT16: return (T)(int16_t)lo;
    case NX_UINT16: return (T)(uint16_t)lo;
    case NX_INT8: return (T)(int8_t)lo;
    case NX_UINT8: return (T)(uint8_t)lo;
    case NX_INT4: return (T)((int32_t)((uint32_t)lo << 28) >> 28);
    case NX_UINT4: return (T)(uint32_t)(lo & 15);
    case NX_BOOL: return (T)((uint8_t)lo != 0);
    case NX_BIT: return (T)((lo & 1) != 0);
    case NX_COMPLEX64:
      im = (T)as_float((uint32_t)(lo >> 32));
      return (T)as_float((uint32_t)lo);
    case NX_COMPLEX128:
      im = (T)as_double(hi);
      return (T)as_double(lo);
    }
    return T(0);
  } else {
    double f;
    switch (fmt) {
    case NX_INT64: case NX_UINT64: return (T)lo;
    case NX_INT32: return (T)(int64_t)(int32_t)lo;
    case NX_UINT32: return (T)(uint32_t)lo;
    case NX_INT16: return (T)(int64_t)(int16_t)lo;
    case NX_UINT16: return (T)(uint16_t)lo;
    case NX_INT8: return (T)(int64_t)(int8_t)lo;
    case NX_UINT8: return (T)(uint8_t)lo;
    case NX_INT4: return (T)(int64_t)((int32_t)((uint32_t)lo << 28) >> 28);
    case NX_UINT4: return (T)(lo & 15);
    case NX_BOOL: return (T)((uint8_t)lo != 0);
    case NX_BIT: return (T)((lo & 1) != 0);
    case NX_FLOAT64: case NX_COMPLEX128: f = as_double(lo); break;
    case NX_COMPLEX64: case NX_FLOAT32: f = as_float((uint32_t)lo); break;
    default: f = nx_bits_to_float(fmt, (uint32_t)lo);
    }
    switch (acc) {
    case NX_INT64: return (T)nx_double_to_i64(f);
    case NX_UINT64: return (T)nx_double_to_u64(f);
    default: {
      int64_t l, h;
      range_of(acc, l, h);
      return (T)nx_double_to_int(f, l, h);
    }
    }
  }
}

/* The same out of line, for the paths that take any dtype. */
template <typename T>
DEVICE T decode_any(int fmt, int acc, u64 lo, u64 hi, T &im) {
  return decode<T>(fmt, acc, lo, hi, im);
}

/* The bits of element [i] of [fmt] at [p], an operand of the kernels: one
   a byte for a sub-byte dtype. An element of w bytes lies on w bytes, but
   one of 8 on 4 (complex64's promise) and one of 16 on 8: one load of its
   width where its address allows, else two (a runtime memcpy would load a
   byte at a time). */
DEVICE void bits_at(const void *p, int64_t i, int fmt,
                                        u64 &lo, u64 &hi) {
  const int w = width_of(fmt);
  const char *q = (const char *)p + i * w;
  lo = hi = 0;
  switch (w) {
  case 1: lo = *(const uint8_t *)q; return;
  case 2: lo = *(const uint16_t *)q; return;
  case 4: lo = *(const uint32_t *)q; return;
  case 8:
    if ((uintptr_t)q % 8 == 0) lo = *(const u64 *)q;
    else lo = *(const uint32_t *)q | (u64)((const uint32_t *)q)[1] << 32;
    return;
  default: lo = ((const u64 *)q)[0], hi = ((const u64 *)q)[1];
  }
}

/* Output */

/* The bits of a float sum [re] (and [im], a complex sum's, else +0) of
   the dtype [fmt], as a cast from the accumulator writes it (nx_dtype.h's
   encoders): the real part alone into a real dtype, nonzero where either
   part is for a boolean; [hi] a complex128's imaginary part. */
DEVICE u64 float_bits(int fmt, double re, double im,
                                          u64 &hi) {
  hi = 0;
  switch (fmt) {
  case NX_FLOAT64: return (u64)double_word(re);
  case NX_FLOAT32: return float_word((float)re);
  case NX_COMPLEX128: hi = (u64)double_word(im); return (u64)double_word(re);
  case NX_COMPLEX64:
    return float_word((float)re) | (u64)float_word((float)im) << 32;
  case NX_BOOL: case NX_BIT: return re != 0 || im != 0;
  }
  return (u64)nx_double_to_bits(fmt, re);
}

/* The bits of an integer sum [v] of the accumulator [acc], of the dtype
   [fmt]: wrapped to the accumulator's width and widened by its sign, then
   wrapped to an integer dtype, rounded once to a float, nonzero for a
   boolean. */
DEVICE u64 int_bits(int fmt, int acc, u64 v, u64 &hi) {
  const int b = bits_of(acc);
  const bool s = acc == NX_INT64 || acc == NX_INT32 || acc == NX_INT16 ||
                 acc == NX_INT8 || acc == NX_INT4;
  if (b < 64) {
    v &= (1ull << b) - 1;
    if (s && (v >> (b - 1)) & 1) v |= ~0ull << b;
  }
  hi = 0;
  switch (fmt) {
  case NX_FLOAT64: case NX_COMPLEX128:
    return (u64)double_word(s ? (double)(int64_t)v : (double)v);
  case NX_FLOAT32: case NX_COMPLEX64:
    return float_word(s ? (float)(int64_t)v : (float)v);
  case NX_FLOAT16: case NX_BFLOAT16: case NX_FLOAT8_E4M3FN: case NX_FLOAT8_E5M2:
  case NX_FLOAT4_E2M1FN:
    return (u64)nx_double_to_bits(fmt, s ? nx_i64_odd((int64_t)v) : nx_u64_odd(v));
  case NX_BOOL: case NX_BIT: return v != 0;
  }
  return v;
}

/* init's element (z, i, j), of the compute type's own bits (the plan
   casts another init into them). */
template <typename T>
DEVICE T init_at(const contract_params &p, int z, int i,
                                     int j, int64_t is) {
  return ((const T *)p.init)[is + z * p.si[0] + (int64_t)i * p.si[1] + (int64_t)j * p.si[2]];
}

/* Output (z, i, j) of the sum [re]: init added where
   it has products (an output of none, k = 0 or [none], is init itself),
   then stored into y of the accumulator's own bits, or an integer sum into
   an integer y no wider, which keeps its low bits (the plan casts it into
   another y). */
template <typename T>
DEVICE void store(const contract_params &p, int z, int i,
                                   int j, T re, bool none, int64_t ys,
                                   int64_t is) {
  if (i >= p.m || j >= p.n) return;
  /* With NX_CONTRACT_INIT_FIRST the sum began from init. */
  if (p.init && !(p.flags & NX_CONTRACT_INIT_FIRST)) {
    const T x = init_at<T>(p, z, i, j, is);
    re = p.k > 0 && !none ? kept(re, x, re + x) : x;
  }
  const int64_t at = ys + z * p.sy[0] + (int64_t)i * p.sy[1] + (int64_t)j * p.sy[2];
  if constexpr (is_float<T>) ((T *)p.y)[at] = re;
  else
    switch (width_of(p.y_dtype)) {
    case 1: ((uint8_t *)p.y)[at] = (uint8_t)re; return;
    case 2: ((uint16_t *)p.y)[at] = (uint16_t)re; return;
    case 4: ((uint32_t *)p.y)[at] = (uint32_t)re; return;
    default: ((u64 *)p.y)[at] = (u64)re;
    }
}

/* Tiles */

/* A block of WM x WN warps computes BM x BN outputs over k-tiles of BK; a
   warp's lanes are LM x LN, a lane's outputs TM x TN, its rows in runs of
   RM (a 16-byte fragment load) RM * LM apart, its columns in runs of CN
   CN * LN apart. */
template <typename T_, int BM_, int BN_, int BK_, int TM_, int TN_, int LM_,
          int WM_, int WN_>
struct Tile {
  typedef T_ T;
  static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_;
  static constexpr int LM = LM_, LN = 32 / LM_, WM = WM_, WN = WN_;
  static constexpr int THREADS = WM * WN * 32;
  static constexpr int V = 16 / sizeof(T); /* elements of a vector */
  static constexpr int RM = TM < V ? TM : V, CN = TN < V ? TN : V;
  static constexpr int LDA = BM + V, LDB = BN + V; /* k-major rows, padded */
  static constexpr int STAGE = BK * (LDA + LDB);   /* elements */
  static constexpr int LDS = 2 * STAGE * (int)sizeof(T); /* bytes */
  static_assert(BM == WM * LM * TM && BN == WN * LN * TN, "tile shape");
  static_assert(WM * WN * 32 * TN <= 2 * BK * (BM + V + BN + V), "epilogue slots");
  static_assert(NX_CONTRACT_RUN % BK == 0, "a run is whole k-tiles");
};

/* Schedule */

/* The work of block [b]: its tile, its range of k, the ranges its tile
   sums in, and where the tile's ranges' sums go. */
struct Work {
  int64_t tile, k0, k1, part; /* part: the tile's first partial, or -1 */
  int split, splits;
};

DEVICE Work schedule(const contract_params &p) {
  Work w;
  const int64_t b = group_x(), head = (int64_t)p.main * p.s1;
  if (b < head)
    w.tile = b / p.s1, w.split = (int)(b % p.s1), w.splits = p.s1;
  else
    w.tile = p.main + (b - head) / p.s2, w.split = (int)((b - head) % p.s2),
    w.splits = p.s2;
  k_range<NX_CONTRACT_RUN>(p.k, w.split, w.splits, w.k0, w.k1);
  /* Split tiles' partials, in tile order: the main tiles' if s1 > 1, then
     the others' if s2 > 1. */
  if (w.splits == 1)
    w.part = -1;
  else if (w.tile < p.main)
    w.part = w.tile * p.s1;
  else
    w.part = (p.s1 > 1 ? head : 0) + (w.tile - p.main) * p.s2;
  return w;
}

/* Tile [t] as batch element, row tile and column tile: tiles of 8 row
   tiles run together, so that a column tile's b stays in L2. */
DEVICE void tile_of(const contract_params &p, int64_t t,
                                        int &z, int &tm, int &tn) {
  const int64_t per = (int64_t)p.tiles_m * p.tiles_n;
  z = (int)(t / per);
  const int pid = (int)(t - z * per), group = 8 * p.tiles_n;
  const int first = pid / group * 8, gsz = min(p.tiles_m - first, 8);
  tm = first + pid % group % gsz, tn = pid % group / gsz;
}

/* Native loads */

/* A pad: k past a range reads -0 in a and +0 in b, whose product -0 adds
   nothing to any sum, -0 included. */
template <typename T> DEVICE T pad(bool a) {
  if constexpr (is_float<T>) return a ? -T(0) : T(0);
  else return T(0);
}

/* An operand of the compute type T whose free axis (rows mode) or k is
   contiguous, in 16-byte vectors: its R x BK k-tile into a k-major shared
   tile. A thread holds its chunks in registers from their load to their
   store, so that a k-tile's loads are in flight while the block sums the
   one before. Rows past the operand's read its last row (k mode) or are
   loaded one by one (rows mode): their outputs are never stored. */
template <class C, int R, bool A> struct Native {
  typedef typename C::T T;
  static constexpr int BK = C::BK, V = C::V, THREADS = C::THREADS, LD = R + V;
  static constexpr int CH = R * BK / V; /* chunks a k-tile */
  static constexpr int NC = (CH + THREADS - 1) / THREADS;
  static constexpr int RV = R / V > 0 ? R / V : 1, KV = BK / V;
  static_assert(R * BK % V == 0 && BK % V == 0, "whole vectors");
  struct alignas(16) Vec { T e[V]; };

  Vec v[NC];
  const T *ptr[NC]; /* each chunk's first element, this k-tile */
  int64_t step;     /* elements a k-tile */
  int r0, rows;
  bool rmode;
  const int32_t *map; /* a segmented tile's rows, [count] of them, or NULL */
  int count;

  /* Tile row [rr] as the operand's row: rows past the operand's, or past
     a segmented tile's, read its last. */
  DEVICE_MEMBER int64_t row_of(int rr) const {
    return map ? map[min(rr, count - 1)] : min(r0 + rr, rows - 1);
  }

  /* Chunk [i]'s k and row in the tile. */
  DEVICE_MEMBER void at(int i, int &kk, int &rr) const {
    const int c = item() + i * THREADS;
    if (rmode) kk = c / RV, rr = c % RV * V;
    else rr = c / KV, kk = c % KV * V;
  }
  DEVICE_MEMBER bool live(int i) const {
    return CH % THREADS == 0 || item() + i * THREADS < CH;
  }

  DEVICE_MEMBER void make(const contract_operand &o, int z, int r0_, int64_t k0,
                       const int32_t *map_ = nullptr, int count_ = 0,
                       int64_t shift = 0) {
    r0 = r0_, rows = o.rows, rmode = (o.flags & NX_OP_FREE) != 0;
    map = map_, count = count_;
    step = BK * o.s[2];
    const T *base = (const T *)o.p + z * o.s[0] + shift;
#pragma unroll
    for (int i = 0; i < NC; i++) {
      int kk, rr;
      at(i, kk, rr);
      /* Along k, a stride of 1, or 0 where the operand repeats an element
         along k, which its rows hold in BK copies (contract.ml). */
      ptr[i] = rmode ? base + r0 + rr + (k0 + kk) * o.s[2]
                     : base + row_of(rr) * o.s[1] + k0 * o.s[2] + kk;
    }
  }

  /* The next k-tile, [kl] of its k inside. A k-tile whole along k and,
     in rows mode, along the tile's rows loads whole vectors: a condition
     the group shares, so that the edge's loads branch around the vectors'
     rather than mask them. */
  DEVICE_MEMBER void load(int kl) {
    const bool whole = kl == BK && (!rmode || r0 + R <= rows);
#pragma unroll
    for (int i = 0; i < NC; i++) {
      if (!live(i)) continue;
      if (whole) {
        v[i] = *(const Vec *)ptr[i];
        ptr[i] += step;
        continue;
      }
      int kk, rr;
      at(i, kk, rr);
      const int edge = rmode ? rows - (r0 + rr) : V; /* rows inside */
#pragma unroll
        for (int u = 0; u < V; u++) {
          const bool in = rmode ? kk < kl : kk + u < kl;
          v[i].e[u] = !in ? pad<T>(A) : rmode ? ptr[i][min(u, edge - 1)] : ptr[i][u];
        }
      ptr[i] += step;
    }
  }

  DEVICE_MEMBER void store(T *s) const {
#pragma unroll
    for (int i = 0; i < NC; i++) {
      if (!live(i)) continue;
      int kk, rr;
      at(i, kk, rr);
      if (rmode)
        *(Vec *)&s[kk * LD + rr] = v[i];
      else
#pragma unroll
        for (int u = 0; u < V; u++) s[(kk + u) * LD + rr] = v[i].e[u];
    }
  }
};

/* Converting loads */

/* An operand's k-tile of any dtype into the compute type's k-major tile,
   decoded as a cast into the accumulator stores each element. Where its
   contiguous axis allows, a thread loads 16 bytes at once, C = 16 / width
   elements, and decodes them with one switch on the dtype; otherwise an
   element at a time. The loads are not held across the sums: the SM's
   other warps cover their latency. */
template <class C, int R, bool A> struct Convert {
  typedef typename C::T T;
  static constexpr int BK = C::BK, THREADS = C::THREADS, V = C::V;
  static constexpr int RR = R, KK = BK, LD = R + V;

  contract_operand o;
  const char *base;
  int r0, rows, fmt, acc, width;
  bool kfast, chunks, pairs;
  const int32_t *map; /* a segmented tile's rows, [count] of them, or NULL */
  int count;
  const void *kmap; /* a segmented k's indices, or NULL */
  bool wide;        /* kmap's words int64, else int32 */

  DEVICE_MEMBER int64_t row_of(int rr) const {
    return map ? map[min(rr, count - 1)] : min(r0 + rr, rows - 1);
  }
  /* A segmented k's index; with [pairs], a complex call's real parts, two
     a complex index. */
  DEVICE_MEMBER int64_t k_of(int64_t k) const {
    if (!kmap) return k;
    const int64_t i = pairs ? k >> 1 : k;
    const int64_t q = wide ? ((const int64_t *)kmap)[i] : ((const int32_t *)kmap)[i];
    return pairs ? 2 * q + (k & 1) : q;
  }

  DEVICE_MEMBER void make(const contract_operand &o_, int z, int r0_, int acc_,
                       const int32_t *map_ = nullptr, int count_ = 0,
                       int64_t shift = 0, const void *kmap_ = nullptr,
                       bool wide_ = false, bool pairs_ = false) {
    o = o_, fmt = o_.dtype, acc = acc_, width = width_of(fmt);
    map = map_, count = count_, kmap = kmap_, wide = wide_, pairs = pairs_;
    base = (const char *)o_.p + (z * o_.s[0] + shift) * width;
    rows = o_.rows, r0 = r0_;
    kfast = !(o_.flags & NX_OP_FREE);
    /* 16-byte chunks: the contiguous axis's tile extent whole chunks, the
       operand aligned for them. */
    const int c = 16 / width;
    const int64_t lead = kfast ? o_.s[1] : o_.s[2];
    chunks = fast(fmt) && (kfast ? !kmap : !map) &&
             (kfast ? KK % c == 0 : RR % c == 0) &&
             (kfast ? o_.s[2] == 1 : o_.s[1] == 1) && lead % c == 0 &&
             o_.s[0] % c == 0 && shift % c == 0 && (uintptr_t)o_.p % 16 == 0;
  }

  /* Element (rr, kk) of the tile, decoded value [v] (pads past [kl]),
     into the shared tile [s]. */
  DEVICE_MEMBER void put(T *s, int rr, int kk, int kl, T v) const {
    s[kk * LD + rr] = kk >= kl ? pad<T>(A) : v;
  }

  /* A chunk's [ce] elements of W bytes from [at], those past the tile's
     edge left zero. */
  template <int W>
  DEVICE_MEMBER void partial(uint4 &raw, const char *at, int rr, int kk,
                             int kl) const {
    constexpr int CE = 16 / W;
#pragma unroll
    for (int u = 0; u < CE; u++) {
      const bool in = kfast ? kk + u < kl : kk < kl;
      const int64_t d = kfast ? u : min(u, rows - 1 - (r0 + rr));
      if (in) memcpy((char *)&raw + u * W, at + d * W, W);
    }
  }

  /* A chunk's elements decoded: the common dtypes inline (fast() names
     them). */
  template <int W> DEVICE_MEMBER void decode(const uint4 &raw, T *val) const {
    constexpr int CE = 16 / W;
    auto bits = [&](int u) -> u64 {
      u64 x = 0;
      memcpy(&x, (const char *)&raw + u * W, W < 8 ? W : 8);
      return x;
    };
#define FAST(F, EXPR)                                                          \
  if (fmt == F) {                                                              \
    _Pragma("unroll") for (int u = 0; u < CE; u++) {                           \
      const u64 x = bits(u);                                                   \
      val[u] = (EXPR);                                                         \
    }                                                                          \
    return;                                                                    \
  }
    if constexpr (W == 1) {
      FAST(NX_FLOAT8_E4M3FN, (T)e4m3_float((uint32_t)x))
      FAST(NX_FLOAT8_E5M2, (T)f16_float((uint32_t)x << 8))
      FAST(NX_INT8, (T)(int8_t)x)
      FAST(NX_UINT8, (T)(uint8_t)x)
    }
    if constexpr (W == 2) {
      FAST(NX_FLOAT16, (T)f16_float((uint32_t)x))
      FAST(NX_BFLOAT16, (T)as_float((uint32_t)x << 16))
      FAST(NX_INT16, (T)(int16_t)x)
    }
    if constexpr (W == 4 && is_float<T>) FAST(NX_FLOAT32, (T)as_float((uint32_t)x))
    if constexpr (W == 4) FAST(NX_INT32, (T)(int32_t)x)
#undef FAST
  }

  /* 16-byte chunks along the contiguous axis, [ce] elements a chunk. One
     loop for every width: a loop a width would hold each one's addresses
     across the k loop, and the three together spill. */
  DEVICE_MEMBER void chunked(T *s, int64_t kb, int kl) const {
    const int ce = 16 / width;
    /* Chunks along a tile's k and rows; store() takes this path only
       where they are whole. */
    const int n = RR * KK / ce, kc = max(KK / ce, 1), rc = max(RR / ce, 1);
#pragma unroll 1
    for (int c = item(); c < n; c += THREADS) {
      int rr, kk;
      if (kfast) rr = c / kc, kk = c % kc * ce;
      else kk = c / rc, rr = c % rc * ce;
      const int64_t row = row_of(rr);
      const int64_t kq = kfast || kk >= kl ? kb + kk : k_of(kb + kk);
      const char *at = base + (kfast ? row * o.s[1] + kq : r0 + rr + kq * o.s[2]) * width;
      uint4 raw = uint4{0, 0, 0, 0};
      const bool whole = kfast ? kk + ce <= kl : (kk < kl && r0 + rr + ce <= rows);
      if (whole) raw = *(const uint4 *)at;
      else if (width == 1) partial<1>(raw, at, rr, kk, kl);
      else if (width == 2) partial<2>(raw, at, rr, kk, kl);
      else partial<4>(raw, at, rr, kk, kl);
      T val[16];
#pragma unroll
      for (int u = 0; u < 16; u++) val[u] = T(0);
      if (width == 1) decode<1>(raw, val);
      else if (width == 2) decode<2>(raw, val);
      else decode<4>(raw, val);
#pragma unroll
      for (int u = 0; u < 16; u++)
        if (u < ce) put(s, kfast ? rr : rr + u, kfast ? kk + u : kk, kl, val[u]);
    }
  }

  /* The dtypes chunked() decodes inline. */
  static DEVICE_MEMBER bool fast(int fmt) {
    return fmt == NX_FLOAT8_E4M3FN || fmt == NX_FLOAT8_E5M2 || fmt == NX_INT8 ||
           fmt == NX_UINT8 || fmt == NX_FLOAT16 || fmt == NX_BFLOAT16 ||
           fmt == NX_INT16 || (is_float<T> && fmt == NX_FLOAT32) || fmt == NX_INT32;
  }

  /* An element at a time: any layout. */
  DEVICE_MEMBER void elements(T *s, int64_t kb, int kl) const {
#pragma unroll 1
    for (int e = item(); e < RR * KK; e += THREADS) {
      const int rr = kfast ? e / KK : e % RR, kk = kfast ? e % KK : e / RR;
      u64 lo = 0, hi = 0;
      if (kk < kl)
        bits_at(base, row_of(rr) * o.s[1] + k_of(kb + kk) * o.s[2], fmt, lo, hi);
      T im, re = decode_any<T>(fmt, acc, lo, hi, im);
      put(s, rr, kk, kl, re);
    }
  }

  /* The k-tile from k [kb], [kl] of its k inside. */
  DEVICE_MEMBER void store(T *s, int64_t kb, int kl) const {
    if (!chunks) return elements(s, kb, kl);
    chunked(s, kb, kl);
  }
};

/* The GEMM kernels */

/* LOAD: 0 native, 1 converting. */
template <class C, int LOAD> DEVICE void gemm(const contract_params &p) {
  typedef typename C::T T;
  constexpr int BM = C::BM, BN = C::BN, BK = C::BK, TM = C::TM, TN = C::TN;
  constexpr int RM = C::RM, CN = C::CN, LM = C::LM, LN = C::LN;
  constexpr int THREADS = C::THREADS;
  extern SHARED __attribute__((aligned(16))) unsigned char smem_raw[];
  T *smem = (T *)smem_raw;

  Work w = schedule(p);
  int z, tm, tn;
  /* A Rows-segmented call's tile: a row tile of one group's rows, the
     same ranges of k for every tile, whatever the ids. */
  const int32_t *map = nullptr;
  int count = 0;
  bool none = false;
  int64_t shift = 0;
  if (p.desc) {
    tn = (int)(w.tile % p.tiles_n), tm = (int)(w.tile / p.tiles_n);
    if (tm >= *p.ntiles) return;
    const int32_t *d = p.desc + 4 * (int64_t)tm;
    z = d[0], map = p.perm + d[2], count = d[3];
    none = d[1] >= p.groups, shift = none ? 0 : d[1] * p.sbg;
    if (none) w.k1 = w.k0;
  } else
    tile_of(p, w.tile, z, tm, tn);
  /* A Contracting segment's slice e of batch element z: its group's k,
     in increasing k, cut into the plan's ranges by its count; a complex
     call's two real parts a complex index (NX_CONTRACT_K_PAIRS). */
  const void *kmap = nullptr;
  int64_t yshift = 0, ishift = 0;
  const bool pairs = p.flags & NX_CONTRACT_K_PAIRS;
  const bool wide = p.flags & NX_CONTRACT_K_WIDE;
  if (p.kperm) {
    const int e = z / p.batch;
    z %= p.batch;
    const int64_t *off = p.koff + (int64_t)z * (p.kgroups + 1) + e;
    kmap = wide ? (const void *)((const int64_t *)p.kperm + off[0])
                : (const void *)((const int32_t *)p.kperm + off[0]);
    const int64_t n = (off[1] - off[0]) * (pairs ? 2 : 1);
    k_range<NX_CONTRACT_RUN>(n, w.split, w.splits, w.k0, w.k1);
    none = n == 0, yshift = e * p.syg, ishift = e * p.sig;
  }
  const int m0 = map ? 0 : tm * BM, n0 = tn * BN;
  /* A tile wholly outside the triangle a masked call computes: its
     columns complex ones on a complex accumulator. */
  if (p.flags & (NX_CONTRACT_LOWER | NX_CONTRACT_UPPER)) {
    const int c = p.acc_dtype == NX_COMPLEX64 || p.acc_dtype == NX_COMPLEX128 ? 2 : 1;
    const int j0 = n0 / c, j1 = (n0 + BN) / c;
    if ((p.flags & NX_CONTRACT_LOWER) && m0 + BM <= j0) return;
    if ((p.flags & NX_CONTRACT_UPPER) && j1 <= m0) return;
  }
  const int tid = item(), lane = tid % 32, warp = tid / 32;
  const int lm = warp / C::WN * LM * TM + lane / LN * RM;
  const int ln = warp % C::WN * LN * TN + lane % LN * CN;

  /* The loads, by kind. */
  typedef Native<C, BM, true> NA;
  typedef Native<C, BN, false> NB;
  typedef Convert<C, BM, true> XA;
  typedef Convert<C, BN, false> XB;
  typename conditional<LOAD == 0, NA, XA>::type la;
  typename conditional<LOAD == 0, NB, XB>::type lb;
  if constexpr (LOAD == 0)
    la.make(p.a, z, m0, w.k0, map, count), lb.make(p.b, z, n0, w.k0, nullptr, 0, shift);
  else
    la.make(p.a, z, m0, p.acc_dtype, map, count, 0, kmap, wide, pairs),
        lb.make(p.b, z, n0, p.acc_dtype, nullptr, 0, shift, kmap, wide, pairs);
  /* k-tile [kb]'s k inside the range. */
  auto inside = [&](int64_t kb) { return (int)min((int64_t)BK, w.k1 - kb); };
  /* A native k-tile's loads, issued before the sums of the one before. */
  auto load = [&](int64_t kb) {
    if constexpr (LOAD == 0) la.load(inside(kb)), lb.load(inside(kb));
  };
  auto put = [&](int s, int64_t kb) {
    T *as = smem + s * C::STAGE, *bs = as + BK * C::LDA;
    if constexpr (LOAD == 0)
      la.store(as), lb.store(bs);
    else
      la.store(as, kb, inside(kb)), lb.store(bs, kb, inside(kb));
  };

  T acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = T(0);
  /* NX_CONTRACT_INIT_FIRST: the first range starts from init. */
  if ((p.flags & NX_CONTRACT_INIT_FIRST) && p.init && w.split == 0)
#pragma unroll
    for (int i = 0; i < TM; i++) {
      const int r = m0 + lm + i / RM * LM * RM + i % RM;
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const int c = n0 + ln + j / CN * LN * CN + j % CN;
        if (r < p.m && c < p.n) acc[i][j] = init_at<T>(p, z, r, c, ishift);
      }
    }

  T fa[2][TM], fb[2][TN];
  auto frag = [&](int f, const T *as, const T *bs, int k) {
#pragma unroll
    for (int i = 0; i < TM; i += RM) {
      struct alignas(sizeof(T) * RM) Run_ { T e[RM]; };
      *(Run_ *)&fa[f][i] = *(const Run_ *)&as[k * C::LDA + lm + i / RM * LM * RM];
    }
#pragma unroll
    for (int j = 0; j < TN; j += CN) {
      struct alignas(sizeof(T) * CN) Run_ { T e[CN]; };
      *(Run_ *)&fb[f][j] = *(const Run_ *)&bs[k * C::LDB + ln + j / CN * LN * CN];
    }
  };

  /* Two stages: k-tile kt + 1's loads are in flight while kt sums, then
     stored into the stage kt - 1 left; one barrier a k-tile. */
  const int64_t KT = (w.k1 - w.k0 + BK - 1) / BK;
  if (KT > 0) {
    load(w.k0);
    put(0, w.k0);
  }
  barrier();
#pragma unroll 1
  for (int64_t kt = 0; kt < KT; kt++) {
    const int s = (int)(kt & 1);
    const int64_t kn = w.k0 + (kt + 1) * BK;
    if (kt + 1 < KT) load(kn);
    const T *as = smem + s * C::STAGE, *bs = as + BK * C::LDA;
    frag(0, as, bs, 0);
    /* Two k a step: fully unrolled, the compiler loads every k's
       fragments ahead of the sums, which spills; a step of two keeps the
       fragments' double buffer at fixed registers. */
#pragma unroll 2
    for (int k = 0; k < BK; k++) {
      if (k + 1 < BK) frag((k + 1) & 1, as, bs, k + 1);
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] = fma_(fa[k & 1][i], fb[k & 1][j], acc[i][j]);
    }
    if (kt + 1 < KT) put(s ^ 1, kn);
    barrier();
  }

  if (w.splits > 1) {
    T(&flat)[TM * TN] = *reinterpret_cast<T(*)[TM * TN]>(acc);
    if (!combine<T, TM * TN, THREADS>(flat, (T *)p.partials +
                                                  w.part * TM * TN * THREADS,
                                      p.tickets + w.tile, w.split, w.splits))
      return;
  }

  /* Rows in runs of RM, columns in runs of CN. A row of a thread's
     outputs passes through its own slots of shared memory, so that one
     loop, not unrolled, stores them: a run of columns as one vector
     where y is laid out for it, else each through store(), called from
     one place. */
  const bool vec = (p.flags & NX_CONTRACT_Y_VECTORS) && !p.init;
  T *slots = smem + tid * TN;
#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int rr = lm + i / RM * LM * RM + i % RM;
    const int r = !map ? m0 + rr : rr < count ? map[rr] : p.m;
#pragma unroll
    for (int j = 0; j < TN; j++) slots[j] = acc[i][j];
#pragma unroll 1
    for (int j = 0; j < TN; j += CN) {
      const int c = n0 + ln + j / CN * LN * CN;
      if (vec && r < p.m && c + CN <= p.n) {
        struct alignas(sizeof(T) * CN) Run_ { T e[CN]; };
        T *at = (T *)p.y + yshift + z * p.sy[0] + (int64_t)r * p.sy[1] + c;
        *(Run_ *)at = *(const Run_ *)&slots[j];
      } else
#pragma unroll 1
        for (int u = 0; u < CN; u++)
          store<T>(p, z, r, c + u, slots[j + u], none, yshift, ishift);
    }
  }
}

/* gemv */

/* Few rows by a b of any real dtype of at most 8 bytes: a thread a
   column, never a tile in shared memory but a's. The association is
   gemm()'s: the plan's ranges of k, each summed in increasing k from +0 by
   fused multiply-adds, the ranges added in order by combine(); so a call
   gives the bits a tile of NX_GEMV_ROWS x NX_GEMV_THREADS outputs gives.
   a's rows over a stretch of the range sit in shared memory, read by
   every thread at once (a broadcast).

   b comes 16 k a step. Where its columns are contiguous, a lane loads its
   own column's 16 elements, the warp's lanes neighbours. Where its k is
   contiguous, a column's step is a run of 16 W bytes, W an element's
   bytes: the warp loads its 32 columns' runs together, its lanes in turn
   along each run in 16-byte vectors, so that a load instruction reads
   whole runs; each lane decodes what it loaded into the warp's rows of
   shared memory (NX_GEMV_STAGE), a row a column, and takes its column's
   16 values from there. A dtype decode16 does not decode, or any other
   layout, takes an element at a time. */

/* Whether decode16 decodes [fmt] into T: the integers of 8 to 64 bits,
   and into a float the floats of 8 to 64 bits. */
template <typename T> DEVICE bool decodes(int fmt) {
  switch (fmt) {
  case NX_INT8: case NX_UINT8: case NX_INT16: case NX_UINT16: case NX_INT32:
  case NX_UINT32: case NX_INT64: case NX_UINT64: return true;
  case NX_FLOAT8_E4M3FN: case NX_FLOAT8_E5M2: case NX_BFLOAT16:
  case NX_FLOAT16: case NX_FLOAT32: case NX_FLOAT64: return is_float<T>;
  }
  return false;
}

/* 16 elements of [fmt], W bytes each, in [q], as T as a cast into the
   accumulator stores them, for a [fmt] decodes() names. */
template <typename T, int W>
DEVICE void decode16(T (&v)[16], const uint4 (&q)[W],
                                         int fmt) {
  const unsigned char *b = (const unsigned char *)q;
  auto word = [&](int u) {
    u64 x = 0;
    memcpy(&x, b + u * W, W);
    return x;
  };
#define ALL(F, EXPR)                                                           \
  if (fmt == F) {                                                              \
    _Pragma("unroll") for (int u = 0; u < 16; u++) {                           \
      const u64 x = word(u);                                                   \
      v[u] = (EXPR);                                                           \
    }                                                                          \
    return;                                                                    \
  }
  if constexpr (W == 1 && is_float<T>) {
    ALL(NX_FLOAT8_E4M3FN, (T)e4m3_float((uint32_t)x))
    ALL(NX_FLOAT8_E5M2, (T)f16_float((uint32_t)x << 8))
  }
  if constexpr (W == 1) ALL(NX_INT8, (T)(int8_t)x)
  if constexpr (W == 2 && is_float<T>) {
    ALL(NX_BFLOAT16, (T)as_float((uint32_t)x << 16))
    ALL(NX_FLOAT16, (T)f16_float((uint32_t)x))
  }
  if constexpr (W == 2) ALL(NX_INT16, (T)(int16_t)x)
  if constexpr (W == 4 && is_float<T>) ALL(NX_FLOAT32, (T)as_float((uint32_t)x))
  if constexpr (W == 4) ALL(NX_INT32, (T)(int32_t)x)
  if constexpr (W == 8 && is_float<T>) {
    ALL(NX_FLOAT64, (T)as_double(x))
    ALL(NX_INT64, (T)(int64_t)x)
  }
#undef ALL
  /* The unsigned integers of W bytes, and a 64-bit one into an integer. */
#pragma unroll
  for (int u = 0; u < 16; u++) v[u] = (T)word(u);
}

/* The bits of element [e] of [w] bytes at [p], which lies on [w]
   bytes. */
DEVICE u64 element(const char *p, int64_t e, int w) {
  const char *at = p + e * w;
  switch (w) {
  case 1: return *(const uint8_t *)at;
  case 2: return *(const uint16_t *)at;
  case 4: return *(const uint32_t *)at;
  default: return *(const u64 *)at;
  }
}

/* f(W) for b's element bytes [wb], W a constant. */
template <typename F> DEVICE void by_width(int wb, F f) {
  switch (wb) {
  case 1: return f(integral_constant<int, 1>());
  case 2: return f(integral_constant<int, 2>());
  case 4: return f(integral_constant<int, 4>());
  default: return f(integral_constant<int, 8>());
  }
}

/* Shared memory: a's [m] rows of the stretch, then each warp's rows of
   b's values (NX_GEMV_STAGE). */
template <typename T> DEVICE void gemv(const contract_params &p) {
  static_assert(NX_GEMV_STRETCH >= NX_GEMV_THREADS, "a's stretch holds the sums");
  constexpr int ROW = 16 * sizeof(T) + 16;
  extern SHARED __attribute__((aligned(16))) unsigned char smem_raw[];
  T *as = (T *)smem_raw;
  const Work w = schedule(p);
  const int z = (int)(w.tile / p.tiles_n), tn = (int)(w.tile % p.tiles_n);
  const int lane = item() % 32, j0 = tn * NX_GEMV_THREADS + item() - lane;
  const int j = j0 + lane, m = p.m;
  const int fmt = p.b.dtype, accf = p.acc_dtype, wb = width_of(fmt);
  const int64_t sk = p.b.s[2], sr = p.b.s[1];
  const char *b = (const char *)p.b.p + z * p.b.s[0] * wb;
  const char *col = b + (int64_t)min(j, p.n - 1) * sr * wb;
  /* A step loads whole vectors along k; a dtype decode16 does not decode
     takes an element at a time. */
  const bool fast = decodes<T>(fmt) && (uintptr_t)p.b.p % wb == 0;
  const bool runs = fast && sk == 1 && (sr * wb) % 16 == 0 &&
                    (p.b.s[0] * wb) % 16 == 0 && (uintptr_t)p.b.p % 16 == 0;
  unsigned char *st = smem_raw + (m * NX_GEMV_STRETCH * sizeof(T) + 15) / 16 * 16 +
                      item() / 32 * (sizeof(T) == 4 ? NX_GEMV_STAGE_4 : NX_GEMV_STAGE_8);
  T acc[NX_GEMV_ROWS];
#pragma unroll
  for (int i = 0; i < NX_GEMV_ROWS; i++) acc[i] = T(0);

  for (int64_t k0 = w.k0; k0 < w.k1; k0 += NX_GEMV_STRETCH) {
    const int len = (int)min((int64_t)NX_GEMV_STRETCH, w.k1 - k0);
    barrier();
    /* a's rows over the stretch, in T as a cast stores them. */
#pragma unroll 1
    for (int e = item(); e < m * NX_GEMV_STRETCH; e += NX_GEMV_THREADS) {
      const int i = e / NX_GEMV_STRETCH, q = e % NX_GEMV_STRETCH;
      T x = T(0), im;
      if (q < len) {
        u64 lo, hi;
        bits_at(p.a.p, z * p.a.s[0] + (int64_t)i * p.a.s[1] + (k0 + q) * p.a.s[2],
                p.a.dtype, lo, hi);
        x = decode_any<T>(p.a.dtype, accf, lo, hi, im);
      }
      as[i * NX_GEMV_STRETCH + q] = x;
    }
    barrier();
    const char *at = col + k0 * sk * wb;
    /* A step's bytes of b: the lane's W vectors where [runs], else its
       column's 16 elements. */
    auto fetch = [&](auto c, uint4 *raw, int q) {
      constexpr int W = decltype(c)::value;
      if (runs) {
        /* Chunk 32 t + lane of the warp's step: bytes 16 (lane % W) of the
           run of column t (32 / W) + lane / W. */
#pragma unroll
        for (int t = 0; t < W; t++) {
          const int cc = t * (32 / W) + lane / W;
          raw[t] = *(const uint4 *)(b + ((int64_t)min(j0 + cc, p.n - 1) * sr + k0 + q) * W +
                                    lane % W * 16);
        }
        return;
      }
      unsigned char *r = (unsigned char *)raw;
#pragma unroll
      for (int u = 0; u < 16; u++) {
        const u64 x = element(at, (int64_t)(q + u) * sk, W);
        memcpy(r + u * W, &x, W);
      }
    };
    /* The next step's bytes, in flight while this one sums, where they
       take no more registers than its values: W <= sizeof(T). */
    constexpr int NV = sizeof(T);
    uint4 next[NV];
    int q = 0;
    if (fast && len >= 16)
      by_width(wb, [&](auto c) {
        if constexpr (decltype(c)::value <= NV) fetch(c, next, 0);
      });
#pragma unroll 1
    for (; fast && q + 16 <= len; q += 16) {
      T v[16];
      by_width(wb, [&](auto c) {
        constexpr int W = decltype(c)::value, PER = 16 / W;
        uint4 raw[W];
        if constexpr (W <= NV) {
#pragma unroll
          for (int t = 0; t < W; t++) raw[t] = next[t];
          if (q + 32 <= len) fetch(c, next, q + 16);
        } else
          fetch(c, raw, q);
        decode16<T, W>(v, raw, fmt);
        if (runs) {
          wave_barrier();
#pragma unroll
          for (int u = 0; u < 16; u++) {
            const int cc = u / PER * (32 / W) + lane / W;
            ((T *)(st + cc * ROW))[lane % W * PER + u % PER] = v[u];
          }
          wave_barrier();
#pragma unroll
          for (int t = 0; t < (int)sizeof(T); t++)
            ((uint4 *)v)[t] = ((const uint4 *)(st + lane * ROW))[t];
        }
      });
      /* Each row's sums in increasing k, k outermost: rows innermost,
         the integer sums, free to reassociate, otherwise hold every
         row's values of a at once and spill. Rows past m skip their sums:
         a predicated multiply-add would still take its issue slot. */
#pragma unroll
      for (int u = 0; u < 16; u++)
#pragma unroll
        for (int i = 0; i < NX_GEMV_ROWS; i++) {
          if (i >= m) break;
          acc[i] = fma_(as[i * NX_GEMV_STRETCH + q + u], v[u], acc[i]);
        }
    }
    /* Other dtypes and layouts, and the range's last k: an element at a
       time, in order. */
#pragma unroll 1
    for (; q < len; q++) {
      u64 x, hi;
      bits_at(at, (int64_t)q * sk, fmt, x, hi);
      T im, v = decode_any<T>(fmt, accf, x, hi, im);
#pragma unroll
      for (int i = 0; i < NX_GEMV_ROWS; i++) {
        if (i >= m) break;
        acc[i] = fma_(as[i * NX_GEMV_STRETCH + q], v, acc[i]);
      }
    }
  }

  if (w.splits > 1 &&
      !combine<T, NX_GEMV_ROWS, NX_GEMV_THREADS>(
          acc, (T *)p.partials + w.part * NX_GEMV_ROWS * NX_GEMV_THREADS,
          p.tickets + w.tile, w.split, w.splits, m))
    return;
  /* Each thread's m sums through a's stretch, NX_GEMV_THREADS of them. */
  barrier();
  T *slots = as + item() * m;
#pragma unroll
  for (int i = 0; i < NX_GEMV_ROWS; i++)
    if (i < m) slots[i] = acc[i];
#pragma unroll 1
  for (int i = 0; i < m; i++)
    store<T>(p, z, i, j, slots[i], false, 0, 0);
}


/* Segments */

/* A prepass's words (kernels.h, NX_SEG_), read once. */
struct segment_params {
  const int64_t *ids;
  void *counts, *cursor, *perm, *desc, *ntiles;
  int64_t sz, sx, inner, count, extent, first;
  int batch, groups, bm, phase, grid;
  bool wide;
};

DEVICE segment_params segment_words(void) {
  const int64_t *w = words();
  segment_params p;
  p.ids = pointer<const int64_t>(w[NX_SEG_IDS]);
  p.counts = pointer<void>(w[NX_SEG_COUNTS]);
  p.cursor = pointer<void>(w[NX_SEG_CURSOR]);
  p.perm = pointer<void>(w[NX_SEG_PERM]);
  p.desc = pointer<void>(w[NX_SEG_DESC]);
  p.ntiles = pointer<void>(w[NX_SEG_NTILES]);
  p.sz = w[NX_SEG_SZ], p.sx = w[NX_SEG_SX], p.inner = w[NX_SEG_INNER];
  p.count = w[NX_SEG_COUNT], p.extent = w[NX_SEG_EXTENT];
  p.first = w[NX_SEG_FIRST];
  p.batch = (int)w[NX_SEG_BATCH], p.groups = (int)w[NX_SEG_GROUPS];
  p.bm = (int)w[NX_SEG_BM], p.phase = (int)w[NX_SEG_PHASE];
  p.wide = w[NX_SEG_WIDE] != 0, p.grid = (int)w[NX_SEG_GRID];
  return p;
}

/* The group of index [q] of batch element [z], [first] past the call's
   first, or [groups] for none. */
DEVICE int group_of(const segment_params &p, int z, int64_t q) {
  const int64_t x = (p.first + q) / p.inner % p.extent;
  const int64_t e = p.ids[z * p.sz + x * p.sx];
  return e >= 0 && e < p.groups ? (int)e : p.groups;
}

/* The active lanes of the wave whose [e] is this lane's: one ballot for
   each distinct [e], the lowest lane not yet matched broadcasting its
   own. */
DEVICE uint32_t same_lanes(int64_t e) {
  uint32_t todo = __builtin_amdgcn_ballot_w32(true), same = 0;
  while (todo) {
    const int l = __builtin_ctz(todo);
    const uint32_t lo = __builtin_amdgcn_readlane((int)(uint32_t)e, l);
    const uint32_t hi = __builtin_amdgcn_readlane((int)(e >> 32), l);
    const bool mine = e == (int64_t)(((u64)hi << 32) | lo);
    const uint32_t m = __builtin_amdgcn_ballot_w32(mine);
    if (mine) same = m;
    todo &= ~m;
  }
  return same;
}

/* [x] of lane [l], which is active. */
DEVICE int64_t from_lane(int64_t x, int l) {
  const uint32_t lo = __builtin_amdgcn_ds_bpermute(4 * l, (int)(uint32_t)x);
  const uint32_t hi = __builtin_amdgcn_ds_bpermute(4 * l, (int)(x >> 32));
  return (int64_t)(((u64)hi << 32) | lo);
}

/* x[e] += 1 for each active lane's [e], one atomic a set of the wave's
   lanes of one [e]: the old value plus the lane's rank among them. The
   sets' order follows the atomics', as each lane's does. W is int32_t or
   int64_t. */
template <typename W> DEVICE W add_one(W *x, int64_t e) {
  const uint32_t same = same_lanes(e);
  const int me = (int)lane(), leader = __builtin_ctz(same);
  int64_t base = 0;
  if (me == leader)
    base = (int64_t)__hip_atomic_fetch_add(&x[e], (W)__builtin_popcount(same),
                                           __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
  base = from_lane(base, leader);
  return (W)(base + __builtin_popcount(same & ((1u << me) - 1)));
}

/* A loop of [i] over [0, n), strided over the launch's groups. */
#define GRID_LOOP(i, n, p)                                                     \
  for (int64_t i = group_x() * (int64_t)NX_SEG_THREADS + item(); i < (n);     \
       i += (int64_t)(p).grid * NX_SEG_THREADS)

/* Rows segments: the rows of a grouped by their ids, so that a tile of
   the contraction holds one group's rows and multiplies one slice of b.
   Four launches, by [phase], over the batch's G = groups + 1 groups,
   group [groups] the rows whose id is outside [0, groups), whose outputs
   have no products:

     0  zeroes [counts], G words a batch element
     1  counts each group's rows (integer atomics, a wave's lanes of one
        group at once, whose order changes no count)
     2  one work-item: lays the groups out one after another in each batch
        element's run of [count] slots, and cuts each into row tiles of
        [bm] rows: [cursor] the group's first slot, [counts] its first
        tile, [counts] past the batch's G words its first slot again, and
        [ntiles] the tiles
     3  places each row in its group's run, and writes each tile's [desc]

   A row's place within its group follows the atomics' order, which
   changes no output's bits: a tile's rows are summed alike wherever they
   sit. */
DEVICE void segment_rows(const segment_params &p) {
  const int G = p.groups + 1, m = (int)p.count;
  const int64_t slots = (int64_t)p.batch * G, rows = (int64_t)p.batch * m;
  int32_t *counts = (int32_t *)p.counts, *cursor = (int32_t *)p.cursor;
  int32_t *perm = (int32_t *)p.perm, *ntiles = (int32_t *)p.ntiles;
  int32_t *first = counts + slots; /* each group's first slot, then the end */
  switch (p.phase) {
  case 0:
    GRID_LOOP(i, slots, p) counts[i] = 0;
    return;
  case 1:
    GRID_LOOP(i, rows, p) {
      const int z = (int)(i / m), r = (int)(i % m);
      add_one(counts, (int64_t)z * G + group_of(p, z, r));
    }
    return;
  case 2:
    if (group_x() == 0 && item() == 0) {
      int32_t t = 0, at = 0;
      for (int64_t i = 0; i < slots; i++) {
        const int32_t n = counts[i];
        cursor[i] = first[i] = at, counts[i] = t;
        t += (n + p.bm - 1) / p.bm, at += n;
      }
      first[slots] = at;
      *ntiles = t;
    }
    return;
  default:
    GRID_LOOP(i, rows, p) {
      const int z = (int)(i / m), r = (int)(i % m);
      perm[add_one(cursor, (int64_t)z * G + group_of(p, z, r))] = r;
    }
    /* Tile t's group: the last whose first tile is at most t. */
    GRID_LOOP(t, (int64_t)*ntiles, p) {
      int64_t lo = 0, hi = slots - 1;
      while (lo < hi) {
        const int64_t mid = (lo + hi + 1) / 2;
        if (counts[mid] <= t) lo = mid;
        else hi = mid - 1;
      }
      const int32_t f = first[lo] + (int32_t)(t - counts[lo]) * p.bm;
      int32_t *d = (int32_t *)p.desc + 4 * t;
      d[0] = (int32_t)(lo / G), d[1] = (int32_t)(lo % G), d[2] = f;
      d[3] = min(p.bm, first[lo + 1] - f);
    }
  }
}

/* Contracting segments: per batch element z, the k of each group in
   increasing k, the groups one after another in z's run of [count] slots
   of [perm]; [counts] holds each group's first slot and, after the last
   group, the end of those in groups. [perm] holds int64 indices where
   [wide], int32 ones otherwise. z's k are cut into [bm] parts of
   [part] k, whose groups' counts [desc] holds, G = groups words a part.
   Five launches, by [phase]:

     0  zeroes [desc]
     1  a group a part: counts each group's k in it (integer atomics)
     2  each group of each z: its parts' first slots within the group,
        exclusive sums in part order, into [desc], and its count into
        [cursor]
     3  a work-item a z: each group's first slot into [counts]
     4  a group a part: one wave places its k, 32 at a time, in order, so
        that a group's k keep their order: the order its products are
        numbered in */
DEVICE void segment_k(const segment_params &p) {
  const int G = p.groups, P = p.bm;
  const int64_t k = p.count, part = (k + P - 1) / P;
  int64_t *counts = (int64_t *)p.counts, *cursor = (int64_t *)p.cursor;
  int64_t *desc = (int64_t *)p.desc;
  switch (p.phase) {
  case 0:
    GRID_LOOP(i, (int64_t)p.batch * P * G, p) desc[i] = 0;
    return;
  case 1: {
    const int z = (int)group_x() / P, c = (int)group_x() % P;
    int64_t *cnt = desc + ((int64_t)z * P + c) * G;
    const int64_t end = min(k, (c + 1) * part);
    for (int64_t q = c * part + item(); q < end; q += NX_SEG_THREADS) {
      const int e = group_of(p, z, q);
      if (e < G) add_one(cnt, e);
    }
    return;
  }
  case 2:
    GRID_LOOP(i, (int64_t)p.batch * G, p) {
      const int64_t z = i / G, e = i % G;
      int64_t run = 0;
      for (int c = 0; c < P; c++) {
        int64_t *x = desc + (z * P + c) * G + e;
        const int64_t n = *x;
        *x = run, run += n;
      }
      cursor[i] = run;
    }
    return;
  case 3:
    GRID_LOOP(z, (int64_t)p.batch, p) {
      int64_t *off = counts + z * (G + 1);
      int64_t at = z * k;
      for (int e = 0; e < G; e++) {
        off[e] = at;
        at += cursor[z * G + e];
      }
      off[G] = at;
    }
    return;
  default: {
    if (item() >= 32) return;
    const int z = (int)group_x() / P, c = (int)group_x() % P;
    int64_t *at = desc + ((int64_t)z * P + c) * G;
    const int64_t *off = counts + (int64_t)z * (G + 1);
    const int64_t end = min(k, (c + 1) * part);
    for (int64_t q0 = c * part; q0 < end; q0 += 32) {
      const int64_t q = q0 + lane();
      const int e = q < end ? group_of(p, z, q) : G;
      const uint32_t same = same_lanes(e);
      if (e < G) {
        const int64_t slot = off[e] + at[e] + __builtin_popcount(same & ((1u << lane()) - 1));
        if (p.wide) ((int64_t *)p.perm)[slot] = q;
        else ((int32_t *)p.perm)[slot] = (int32_t)q;
      }
      /* Every lane reads at[e] before its set's lowest lane moves it. */
      wave_barrier();
      if (e < G && (int)lane() == __builtin_ctz(same)) at[e] += __builtin_popcount(same);
      wave_barrier();
    }
  }
  }
}

#undef GRID_LOOP

/* Packs */

/* The bits of element [e] of [fmt] at [b], in the low bits of [lo] (and
   [hi] for a complex128); [spread] reads a sub-byte dtype one a byte. */
DEVICE void read_bits(const char *b, int64_t e, int fmt,
                                          bool spread, u64 &lo, u64 &hi) {
  const int n = bits_of(fmt);
  lo = hi = 0;
  if (n < 8 && !spread) {
    const u64 bit = (u64)e * n;
    lo = (((const uint8_t *)b)[bit >> 3] >> (bit & 7)) & ((1u << n) - 1);
  } else
    bits_at(b, e, fmt, lo, hi);
}

/* Stores [lo] (and [hi]) as element [e] of [fmt] at [b], a sub-byte one
   into the word that holds it (device.h's store_bits); [spread] writes a sub-byte
   dtype one a byte. */
DEVICE void write_bits(char *b, int64_t e, int fmt,
                                           bool spread, u64 lo, u64 hi) {
  const int n = bits_of(fmt), w = width_of(fmt);
  if (n < 8 && !spread) {
    store_bits(b, e, n, (uint32_t)lo);
    return;
  }
  char *q = b + e * w;
  if (n < 8) lo &= (1u << n) - 1;
  /* As bits_at loads. */
  switch (w) {
  case 1: *(uint8_t *)q = (uint8_t)lo; return;
  case 2: *(uint16_t *)q = (uint16_t)lo; return;
  case 4: *(uint32_t *)q = (uint32_t)lo; return;
  case 8:
    if ((uintptr_t)q % 8 == 0) *(u64 *)q = lo;
    else ((uint32_t *)q)[0] = (uint32_t)lo, ((uint32_t *)q)[1] = (uint32_t)(lo >> 32);
    return;
  default: ((u64 *)q)[0] = lo, ((u64 *)q)[1] = hi;
  }
}

/* A scale's element, as the accumulator T holds it: 2^(e - 127) of an
   Exponent code e, NaN of 255; a Value as a cast stores it. */
template <typename T>
DEVICE T scale_at(const pack_params &p, int64_t at) {
  if (p.mode & NX_PACK_EXPONENT) {
    const uint8_t e = ((const uint8_t *)p.scale)[at];
    if (e == 255) return T(NAN);
    if constexpr (sizeof(T) == 4) return __builtin_ldexpf(1.0f, e - 127);
    else return __builtin_ldexp(1.0, e - 127);
  }
  u64 lo, hi;
  read_bits((const char *)p.scale, at, p.scale_dtype, false, lo, hi);
  T im;
  return decode_any<T>(p.scale_dtype, p.acc_dtype, lo, hi, im);
}

/* An element's value into [dst_dtype], the accumulator's dtype, its
   compute type's, bfloat16 or float16, or an integer one that holds it:
   decoded into the accumulator, times its scale, rounded once. */
template <typename T>
DEVICE void put_value(const pack_params &p, char *d, int64_t e, u64 lo,
                          u64 hi, bool scaled, int64_t at) {
  T im, v = decode<T>(p.src_dtype, p.acc_dtype, lo, hi, im);
  if constexpr (is_float<T>) {
    if (scaled) v = v * scale_at<T>(p, at);
    if (p.mode & NX_PACK_NEG) v = -v, im = -im;
    if (p.mode & NX_PACK_BLOCK) {
      /* A complex b's element as the 2 x 2 real block of its rows 2j,
         2j + 1 and k 2p, 2p + 1, [[re, -im], [im, re]]. */
      T *x = (T *)d + e, *y = x + p.block;
      if (p.mode & NX_PACK_CONJ) im = -im;
      x[0] = v, x[1] = -im, y[0] = im, y[1] = v;
      if (p.mode & NX_PACK_FLIP) x[1] = -x[1], y[1] = -y[1];
      return;
    }
    switch (p.dst_dtype) {
    case NX_FLOAT64: ((double *)d)[e] = (double)v; return;
    case NX_FLOAT32: ((float *)d)[e] = (float)v; return;
    case NX_COMPLEX64: ((float *)d)[2 * e] = (float)v, ((float *)d)[2 * e + 1] = (float)im; return;
    case NX_COMPLEX128: ((double *)d)[2 * e] = (double)v, ((double *)d)[2 * e + 1] = (double)im; return;
    case NX_FLOAT16: ((uint16_t *)d)[e] = nx_float_to_f16((float)v); return;
    default: ((uint16_t *)d)[e] = nx_float_to_bf16((float)v);
    }
  } else
    switch (width_of(p.dst_dtype)) {
    case 1: ((uint8_t *)d)[e] = (uint8_t)v; return;
    case 2: ((uint16_t *)d)[e] = (uint16_t)v; return;
    case 4: ((uint32_t *)d)[e] = (uint32_t)v; return;
    default: ((u64 *)d)[e] = (u64)v;
    }
}

/* A sum of the workspace's dtype [src_dtype], the bits [lo] and [hi], into
   [dst_dtype] as a cast from the accumulator writes it. */
DEVICE u64 cast_bits(const pack_params &p, u64 lo, u64 &hi) {
  switch (p.src_dtype) {
  case NX_FLOAT32: return float_bits(p.dst_dtype, as_float((uint32_t)lo), 0, hi);
  case NX_FLOAT64: return float_bits(p.dst_dtype, as_double(lo), 0, hi);
  case NX_COMPLEX64:
    return float_bits(p.dst_dtype, as_float((uint32_t)lo),
                      as_float((uint32_t)(lo >> 32)), hi);
  case NX_COMPLEX128:
    return float_bits(p.dst_dtype, as_double(lo), as_double(hi), hi);
  }
  return int_bits(p.dst_dtype, p.acc_dtype, lo, hi);
}

/* The pack. */
DEVICE void pack(const pack_params &p) {
  const bool value = p.mode & NX_PACK_VALUE, cast = p.mode & NX_PACK_CAST;
  const bool spread_src = p.mode & NX_PACK_SPREAD_SRC;
  const bool spread_dst = p.mode & NX_PACK_SPREAD_DST;
  const int64_t sf = p.src_first, df = p.dst_first;
  const bool scaled = p.scale != nullptr;
  const int acc = p.acc_dtype;
  const int t = acc == NX_FLOAT32 || acc == NX_COMPLEX64     ? 0
                : acc == NX_FLOAT64 || acc == NX_COMPLEX128 ? 1
                : acc == NX_INT64 || acc == NX_UINT64       ? 2
                                                            : 3;
  for (int64_t x = (int64_t)group_x() * NX_PACK_THREADS + item();
       x < p.numel; x += p.groups * NX_PACK_THREADS) {
    int64_t rest = x;
    int64_t s = sf, d = df, c = p.scale_first;
    for (int a = p.rank - 1; a >= 0; a--) {
      const int64_t *r = p.ax + a * NX_PACK_AXIS;
      const int64_t i = split(rest, r, p.small);
      s += i * r[NX_PACK_AXIS_SS], d += i * r[NX_PACK_AXIS_DS];
      if (scaled) {
        int64_t q = i;
        split(q, r + NX_PACK_AXIS_RATIO, p.small);
        c += q * r[NX_PACK_AXIS_CS];
      }
    }
    u64 lo, hi;
    read_bits((const char *)p.src, s, p.src_dtype, spread_src, lo, hi);
    char *dst = (char *)p.dst;
    if (cast) {
      lo = cast_bits(p, lo, hi);
      write_bits(dst, d, p.dst_dtype, false, lo, hi);
    } else if (!value)
      write_bits(dst, d, p.dst_dtype, spread_dst, lo, hi);
    else if (t == 0)
      put_value<float>(p, dst, d, lo, hi, scaled, c);
    else if (t == 1)
      put_value<double>(p, dst, d, lo, hi, scaled, c);
    else if (t == 2)
      put_value<u64>(p, dst, d, lo, hi, scaled, c);
    else
      put_value<uint32_t>(p, dst, d, lo, hi, scaled, c);
  }
}
} /* namespace nx_contract */

/* Kernels */

using namespace nx_contract;

#define T_f32 float
#define T_f64 double
#define T_u32 uint32_t
#define T_u64 u64
#define LOAD_native 0
#define LOAD_convert 1

/* A pack stages its words up to its [rank] axes' records. */
KERNEL SIZED(NX_PACK_THREADS) void contract_pack(Words<NX_PACK_WORDS>) {
  static SHARED int64_t w[NX_PACK_WORDS];
  const int64_t rank = words()[NX_PACK_RANK];
  stage(w, NX_PACK_AX + rank * NX_PACK_AXIS, NX_PACK_THREADS);
  pack(pack_words(w));
}

KERNEL SIZED(NX_SEG_THREADS) void contract_rows(Words<NX_SEG_WORDS>) {
  segment_rows(segment_words());
}

KERNEL SIZED(NX_SEG_THREADS) void contract_ks(Words<NX_SEG_WORDS>) {
  segment_k(segment_words());
}

/* A contraction kernel's words, copied into LDS once a group so that they
   cross from the argument segment in one round trip. */
#define CONTRACT_WORDS(threads)                                               \
  static SHARED int64_t w[NX_CONTRACT_WORDS];                                 \
  stage(w, NX_CONTRACT_WORDS, threads)

#define GEMM_KERNEL(X, name, t, bm, bn, bk, tm, tn, lm, wm, wn, load)         \
  KERNEL SIZED((wm) * (wn) * 32) void name(Words<NX_CONTRACT_WORDS>) {        \
    typedef Tile<T_##t, bm, bn, bk, tm, tn, lm, wm, wn> C;                    \
    static_assert(C::LDS == NX_GEMM_LDS(t, bm, bn, bk), "kernels.h's LDS");   \
    CONTRACT_WORDS((wm) * (wn) * 32);                                         \
    gemm<C, LOAD_##load>(contract_words(w));                                  \
  }
NX_AMD_GEMM_TILES(GEMM_KERNEL, _)
#undef GEMM_KERNEL

#define WMMA_KERNEL(X, name, kind, bm, bn, bkb, wm, wn)                       \
  KERNEL SIZED((wm) * (wn) * 32) void name(Words<NX_CONTRACT_WORDS>) {        \
    typedef Wtile<bm, bn, bkb, wm, wn> C;                                     \
    static_assert(C::LDS == NX_WMMA_LDS(bm, bn, bkb), "kernels.h's LDS");     \
    CONTRACT_WORDS((wm) * (wn) * 32);                                         \
    wmma_kernel<W_KIND_##kind, C>(contract_words(w));                         \
  }
NX_AMD_WMMA_TILES(WMMA_KERNEL, _)
#undef WMMA_KERNEL

#define GEMV_KERNEL(t)                                                        \
  KERNEL SIZED(NX_GEMV_THREADS) void contract_gemv_##t(                       \
      Words<NX_CONTRACT_WORDS>) {                                             \
    CONTRACT_WORDS(NX_GEMV_THREADS);                                          \
    gemv<T_##t>(contract_words(w));                                           \
  }
GEMV_KERNEL(f32)
GEMV_KERNEL(f64)
GEMV_KERNEL(u32)
GEMV_KERNEL(u64)
#undef GEMV_KERNEL
