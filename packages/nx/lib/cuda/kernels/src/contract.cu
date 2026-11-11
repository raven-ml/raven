/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contract: y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k]
   (kernels.h's contract_params), by fused multiply-adds (GEMM, GEMV) for
   every accumulator and dtype pair, or on the matrix units (mma.cuh) for
   bfloat16, float16 and int8 operands. Packs copy an operand the kernels
   do not read where it lies into the workspace; ROWS and KS lay out a
   segmented call's groups.

     global --(load: native, or convert: decode, expand)--> registers
       --> shared k-major tile, two stages --> fragments --> FMA
       --> combine (split tiles) --> + init --> cast --> y

   Association of the FMA kernels (what fixes a float output's bits): the
   plan gives each tile S ranges of k (S = s1 for the call's first [main]
   tiles, s2 for the others), whole runs of NX_CONTRACT_RUN terms shared
   out in order, as k_range computes. A range sums its terms in increasing
   k, in blocks of NX_CONTRACT_BLOCK from its start, each from +0, one
   fused multiply-add a term, the blocks' sums added in order
   (combine.cuh's fold); the ranges' sums add in increasing k; then init;
   then one rounding into y. An output without products is init itself.
   With NX_CONTRACT_INIT_FIRST (the matrix routines' products) each range
   is one chain, the first from init, and init is not added after: the
   unblocked algorithms' chains, whose bounds the routines state. Loads,
   layouts and packs change no bit beyond
   the values the terms take: the plan chooses the tiles and S from the
   dtypes and the extents alone.

   The inner loop is the same for every dtype: only the load differs. A
   native load (operands of the compute type, in 16-byte vectors) copies;
   a converting load decodes each element once per tile, with one switch
   on the dtype per operand per k-tile, outside the FMA loop. A complex
   call is the real contraction of its parts: a's (re, im) at k' = 2p,
   2p + 1, b's as the 2 x 2 block [[re, -im], [im, re]] of columns 2j,
   2j + 1, so the FMA loop sums the contract's numbered terms in order. */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <type_traits>

#include "kernels.h"
#include "nx_dtype.h"
#include "words.cuh"

typedef unsigned long long u64;

/* No device function here or in the headers is static: nvcc names an
   out-of-line function of internal linkage after a hash of the directory
   it compiles in, so the cubin's bytes would change with the checkout. */

namespace nx_contract {

#include "combine.cuh"

/* Elements */

/* a * b + c in T: one rounding for floats, wrapping for integers. */
__device__ __forceinline__ float fma_(float a, float b, float c) {
  return __fmaf_rn(a, b, c);
}
__device__ __forceinline__ double fma_(double a, double b, double c) {
  return __fma_rn(a, b, c);
}
__device__ __forceinline__ uint32_t fma_(uint32_t a, uint32_t b, uint32_t c) {
  return a * b + c;
}
__device__ __forceinline__ u64 fma_(u64 a, u64 b, u64 c) { return a * b + c; }

template <typename T> constexpr bool is_float = T(0.5) != T(0);

/* Narrow floats by the hardware's conversions (sm_89), exact but for a
   NaN, which they make canonical: for operands, whose NaN sums have no
   stated bits. */
__device__ __forceinline__ float f16_float(uint32_t h) {
  float f;
  asm("{.reg .b16 t; cvt.u16.u32 t, %1; cvt.f32.f16 %0, t;}" : "=f"(f) : "r"(h));
  return f;
}
__device__ __forceinline__ float e4m3_float(uint32_t c) {
  uint32_t h;
  asm("{.reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}"
      : "=r"(h) : "r"(c & 0xFF));
  return f16_float(h & 0xFFFF);
}

/* The bytes an element of [fmt] takes in the kernels' operands: a
   sub-byte dtype's, packed one a byte, one. */
__host__ __device__ constexpr int width_of(int fmt) {
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
__device__ __forceinline__ int bits_of(int fmt) {
  return fmt == NX_BIT ? 1
         : fmt == NX_INT4 || fmt == NX_UINT4 || fmt == NX_FLOAT4_E2M1FN
             ? 4
             : 8 * width_of(fmt);
}

/* An integer accumulator's range, as a cast saturates a float into it. */
__device__ __forceinline__ void range_of(int acc, int64_t &lo, int64_t &hi) {
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
__device__ __forceinline__ T decode(int fmt, int acc, u64 lo, u64 hi, T &im) {
  im = T(0);
  if constexpr (is_float<T>) {
    switch (fmt) {
    case NX_FLOAT64: return (T)__longlong_as_double(lo);
    case NX_FLOAT32: return (T)__uint_as_float((uint32_t)lo);
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
      im = (T)__uint_as_float((uint32_t)(lo >> 32));
      return (T)__uint_as_float((uint32_t)lo);
    case NX_COMPLEX128:
      im = (T)__longlong_as_double(hi);
      return (T)__longlong_as_double(lo);
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
    case NX_FLOAT64: case NX_COMPLEX128: f = __longlong_as_double(lo); break;
    case NX_COMPLEX64: case NX_FLOAT32: f = __uint_as_float((uint32_t)lo); break;
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
__device__ __noinline__ T decode_any(int fmt, int acc, u64 lo, u64 hi, T &im) {
  return decode<T>(fmt, acc, lo, hi, im);
}

/* The bits of element [i] of [fmt] at [p], an operand of the kernels: one
   a byte for a sub-byte dtype. An element of w bytes lies on w bytes, but
   one of 8 on 4 (complex64's promise) and one of 16 on 8: one load of its
   width where its address allows, else two (a runtime memcpy would load a
   byte at a time). */
__device__ __forceinline__ void bits_at(const void *p, int64_t i, int fmt,
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
__device__ __forceinline__ u64 float_bits(int fmt, double re, double im,
                                          u64 &hi) {
  hi = 0;
  switch (fmt) {
  case NX_FLOAT64: return (u64)__double_as_longlong(re);
  case NX_FLOAT32: return __float_as_uint((float)re);
  case NX_COMPLEX128: hi = (u64)__double_as_longlong(im); return (u64)__double_as_longlong(re);
  case NX_COMPLEX64:
    return __float_as_uint((float)re) | (u64)__float_as_uint((float)im) << 32;
  case NX_BOOL: case NX_BIT: return re != 0 || im != 0;
  }
  return (u64)nx_double_to_bits(fmt, re);
}

/* The bits of an integer sum [v] of the accumulator [acc], of the dtype
   [fmt]: wrapped to the accumulator's width and widened by its sign, then
   wrapped to an integer dtype, rounded once to a float, nonzero for a
   boolean. */
__device__ __forceinline__ u64 int_bits(int fmt, int acc, u64 v, u64 &hi) {
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
    return (u64)__double_as_longlong(s ? (double)(int64_t)v : (double)v);
  case NX_FLOAT32: case NX_COMPLEX64:
    return __float_as_uint(s ? (float)(int64_t)v : (float)v);
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
__device__ __forceinline__ T init_at(const contract_params &p, int z, int i,
                                     int j, int64_t is) {
  return ((const T *)p.init)[is + z * p.si[0] + (int64_t)i * p.si[1] + (int64_t)j * p.si[2]];
}

/* Output (z, i, j) of the sum [re]: init added where
   it has products (an output of none, k = 0 or [none], is init itself),
   then stored into y of the accumulator's own bits, or an integer sum into
   an integer y no wider, which keeps its low bits (the plan casts it into
   another y). */
template <typename T>
__device__ __noinline__ void store(const contract_params &p, int z, int i,
                                   int j, T re, bool none, int64_t ys,
                                   int64_t is) {
  if (i >= p.m || j >= p.n) return;
  /* With NX_CONTRACT_INIT_FIRST the sum began from init. */
  if (p.init && !(p.flags & NX_CONTRACT_INIT_FIRST)) {
    const T x = init_at<T>(p, z, i, j, is);
    re = p.k > 0 && !none ? re + x : x;
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

__device__ Work schedule(const contract_params &p) {
  Work w;
  const int64_t b = blockIdx.x, head = (int64_t)p.main * p.s1;
  if (b < head)
    w.tile = b / p.s1, w.split = (int)(b % p.s1), w.splits = p.s1;
  else
    w.tile = p.main + (b - head) / p.s2, w.split = (int)((b - head) % p.s2),
    w.splits = p.s2;
  k_range<NX_CONTRACT_RUN>(p.k, w.split, w.splits, w.k0, w.k1);
  /* A split tile's partials: one slot a block, in block order. */
  if (w.splits == 1)
    w.part = -1;
  else if (w.tile < p.main)
    w.part = w.tile * p.s1;
  else
    w.part = head + (w.tile - p.main) * p.s2;
  return w;
}

/* Tile [t] as batch element, row tile and column tile: tiles of 8 row
   tiles run together, so that a column tile's b stays in L2. */
__device__ __forceinline__ void tile_of(const contract_params &p, int64_t t,
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
template <typename T> __device__ __forceinline__ T pad(bool a) {
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
  __device__ __forceinline__ int64_t row_of(int rr) const {
    return map ? map[min(rr, count - 1)] : min(r0 + rr, rows - 1);
  }

  /* Chunk [i]'s k and row in the tile. */
  __device__ __forceinline__ void at(int i, int &kk, int &rr) const {
    const int c = threadIdx.x + i * THREADS;
    if (rmode) kk = c / RV, rr = c % RV * V;
    else rr = c / KV, kk = c % KV * V;
  }
  __device__ __forceinline__ bool live(int i) const {
    return CH % THREADS == 0 || threadIdx.x + i * THREADS < CH;
  }

  __device__ void make(const contract_operand &o, int z, int r0_, int64_t k0,
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

  /* The next k-tile, [kl] of its k inside. */
  __device__ void load(int kl) {
#pragma unroll
    for (int i = 0; i < NC; i++) {
      if (!live(i)) continue;
      int kk, rr;
      at(i, kk, rr);
      const int edge = rmode ? rows - (r0 + rr) : V; /* rows inside */
      if (kl == BK && edge >= V)
        v[i] = *(const Vec *)ptr[i];
      else
#pragma unroll
        for (int u = 0; u < V; u++) {
          const bool in = rmode ? kk < kl : kk + u < kl;
          v[i].e[u] = !in ? pad<T>(A) : rmode ? ptr[i][min(u, edge - 1)] : ptr[i][u];
        }
      ptr[i] += step;
    }
  }

  __device__ void store(T *s) const {
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

  const contract_operand *o;
  const char *base;
  int r0, rows, fmt, acc, width;
  bool kfast, chunks, pairs;
  const int32_t *map; /* a segmented tile's rows, [count] of them, or NULL */
  int count;
  const void *kmap; /* a segmented k's indices, or NULL */
  bool wide;        /* kmap's words int64, else int32 */

  __device__ __forceinline__ int64_t row_of(int rr) const {
    return map ? map[min(rr, count - 1)] : min(r0 + rr, rows - 1);
  }
  /* A segmented k's index; with [pairs], a complex call's real parts, two
     a complex index. */
  __device__ __forceinline__ int64_t k_of(int64_t k) const {
    if (!kmap) return k;
    const int64_t i = pairs ? k >> 1 : k;
    const int64_t q = wide ? ((const int64_t *)kmap)[i] : ((const int32_t *)kmap)[i];
    return pairs ? 2 * q + (k & 1) : q;
  }

  __device__ void make(const contract_operand &o_, int z, int r0_, int acc_,
                       const int32_t *map_ = nullptr, int count_ = 0,
                       int64_t shift = 0, const void *kmap_ = nullptr,
                       bool wide_ = false, bool pairs_ = false) {
    o = &o_, fmt = o_.dtype, acc = acc_, width = width_of(fmt);
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
  __device__ __forceinline__ void put(T *s, int rr, int kk, int kl, T v) const {
    s[kk * LD + rr] = kk >= kl ? pad<T>(A) : v;
  }

  /* Chunks of W-byte elements: C along the contiguous axis. */
  template <int W> __device__ void chunked(T *s, int64_t kb, int kl) const {
    constexpr int CE = 16 / W, N = RR * KK / CE;
    /* Chunks along a tile's k and rows; store() takes this path only
       where they are whole. */
    constexpr int KC = KK / CE > 0 ? KK / CE : 1, RC = RR / CE > 0 ? RR / CE : 1;
    for (int c = threadIdx.x; c < N; c += THREADS) {
      int rr, kk;
      if (kfast) rr = c / KC, kk = c % KC * CE;
      else kk = c / RC, rr = c % RC * CE;
      const int64_t row = row_of(rr);
      const int64_t kq = kfast || kk >= kl ? kb + kk : k_of(kb + kk);
      const char *at = base + (kfast ? row * o->s[1] + kq : r0 + rr + kq * o->s[2]) * W;
      uint4 raw = make_uint4(0, 0, 0, 0);
      const bool whole = kfast ? kk + CE <= kl : (kk < kl && r0 + rr + CE <= rows);
      if (whole)
        raw = *(const uint4 *)at;
      else
        for (int u = 0; u < CE; u++) {
          const bool in = kfast ? kk + u < kl : kk < kl;
          const int64_t d = kfast ? u : min(u, rows - 1 - (r0 + rr));
          if (in) memcpy((char *)&raw + u * W, at + d * W, W);
        }
      auto bits = [&](int u) -> u64 {
        u64 x = 0;
        memcpy(&x, (const char *)&raw + u * W, W < 8 ? W : 8);
        return x;
      };
      /* The common dtypes decoded inline (fast() names them), then
         stored by one loop. */
      T val[CE];
#pragma unroll
      for (int u = 0; u < CE; u++) val[u] = T(0);
      bool done = false;
#define FAST(F, EXPR)                                                          \
  if (!done && fmt == F) {                                                     \
    _Pragma("unroll") for (int u = 0; u < CE; u++) {                           \
      const u64 x = bits(u);                                                   \
      val[u] = (EXPR);                                                         \
    }                                                                          \
    done = true;                                                               \
  }
      if constexpr (W == 1) {
        FAST(NX_FLOAT8_E4M3FN, (T)e4m3_float((uint32_t)x))
        FAST(NX_FLOAT8_E5M2, (T)f16_float((uint32_t)x << 8))
        FAST(NX_INT8, (T)(int8_t)x)
        FAST(NX_UINT8, (T)(uint8_t)x)
      }
      if constexpr (W == 2) {
        FAST(NX_FLOAT16, (T)f16_float((uint32_t)x))
        FAST(NX_BFLOAT16, (T)__uint_as_float((uint32_t)x << 16))
        FAST(NX_INT16, (T)(int16_t)x)
      }
      if constexpr (W == 4 && is_float<T>) FAST(NX_FLOAT32, (T)__uint_as_float((uint32_t)x))
      if constexpr (W == 4) FAST(NX_INT32, (T)(int32_t)x)
#undef FAST
#pragma unroll
      for (int u = 0; u < CE; u++)
        put(s, kfast ? rr : rr + u, kfast ? kk + u : kk, kl, val[u]);
    }
  }

  /* The dtypes chunked() decodes inline. */
  static __device__ bool fast(int fmt) {
    return fmt == NX_FLOAT8_E4M3FN || fmt == NX_FLOAT8_E5M2 || fmt == NX_INT8 ||
           fmt == NX_UINT8 || fmt == NX_FLOAT16 || fmt == NX_BFLOAT16 ||
           fmt == NX_INT16 || (is_float<T> && fmt == NX_FLOAT32) || fmt == NX_INT32;
  }

  /* An element at a time: any layout. */
  __device__ void elements(T *s, int64_t kb, int kl) const {
#pragma unroll 1
    for (int e = threadIdx.x; e < RR * KK; e += THREADS) {
      const int rr = kfast ? e / KK : e % RR, kk = kfast ? e % KK : e / RR;
      u64 lo = 0, hi = 0;
      if (kk < kl)
        bits_at(base, row_of(rr) * o->s[1] + k_of(kb + kk) * o->s[2], fmt, lo, hi);
      T im, re = decode_any<T>(fmt, acc, lo, hi, im);
      put(s, rr, kk, kl, re);
    }
  }

  /* The k-tile from k [kb], [kl] of its k inside. */
  __device__ void store(T *s, int64_t kb, int kl) const {
    if (!chunks) return elements(s, kb, kl);
    switch (width) {
    case 1: return chunked<1>(s, kb, kl);
    case 2: return chunked<2>(s, kb, kl);
    default: return chunked<4>(s, kb, kl);
    }
  }
};

/* The GEMM kernels */

/* LOAD: 0 native, 1 converting. */
template <class C, int LOAD> __device__ void gemm(const contract_params &p) {
  typedef typename C::T T;
  constexpr int BM = C::BM, BN = C::BN, BK = C::BK, TM = C::TM, TN = C::TN;
  constexpr int RM = C::RM, CN = C::CN, LM = C::LM, LN = C::LN;
  constexpr int THREADS = C::THREADS;
  extern __shared__ __align__(16) unsigned char smem_raw[];
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
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int lm = warp / C::WN * LM * TM + lane / LN * RM;
  const int ln = warp % C::WN * LN * TN + lane % LN * CN;

  /* The loads, by kind. */
  typedef Native<C, BM, true> NA;
  typedef Native<C, BN, false> NB;
  typedef Convert<C, BM, true> XA;
  typedef Convert<C, BN, false> XB;
  typename std::conditional<LOAD == 0, NA, XA>::type la;
  typename std::conditional<LOAD == 0, NB, XB>::type lb;
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
     stored into the stage kt - 1 left; one barrier a k-tile. A float
     range of more than one block folds each block's sums into its slot. */
  const int64_t KT = (w.k1 - w.k0 + BK - 1) / BK;
  static_assert(NX_CONTRACT_BLOCK % BK == 0, "a block is whole k-tiles");
  constexpr int64_t PER = NX_CONTRACT_BLOCK / BK;
  T(&flat)[TM * TN] = *reinterpret_cast<T(*)[TM * TN]>(acc);
  if (KT > 0) {
    load(w.k0);
    put(0, w.k0);
  }
  __syncthreads();
  auto tile = [&](int64_t kt) {
    const int s = (int)(kt & 1);
    const int64_t kn = w.k0 + (kt + 1) * BK;
    if (kt + 1 < KT) load(kn);
    const T *as = smem + s * C::STAGE, *bs = as + BK * C::LDA;
    frag(0, as, bs, 0);
#pragma unroll
    for (int k = 0; k < BK; k++) {
      if (k + 1 < BK) frag((k + 1) & 1, as, bs, k + 1);
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] = fma_(fa[k & 1][i], fb[k & 1][j], acc[i][j]);
    }
    if (kt + 1 < KT) put(s ^ 1, kn);
    __syncthreads();
  };
  if constexpr (!is_float<T>) {
#pragma unroll 1
    for (int64_t kt = 0; kt < KT; kt++) tile(kt);
  } else {
    /* The k-tiles block by block, each fold between two blocks, outside
       the loop of the sums. */
    const int64_t per =
        p.flags & NX_CONTRACT_INIT_FIRST ? max(KT, (int64_t)1) : PER;
#pragma unroll 1
    for (int64_t kb = 0; kb < KT; kb += per) {
      if (kb > 0)
        fold<T, TM * TN, THREADS>(flat, p.partials, blockIdx.x, kb == per);
      const int64_t ke = min(KT, kb + per);
#pragma unroll 1
      for (int64_t kt = kb; kt < ke; kt++) tile(kt);
    }
    if (KT > per) close<T, TM * TN, THREADS>(flat, p.partials, blockIdx.x);
  }

  if (w.splits > 1) {
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

template <class C> constexpr int shared_bytes() {
  return 2 * C::STAGE * (int)sizeof(typename C::T);
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
template <typename T> __device__ __forceinline__ bool decodes(int fmt) {
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
__device__ __forceinline__ void decode16(T (&v)[16], const uint4 (&q)[W],
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
    if (fmt == NX_FLOAT8_E4M3FN) {
#pragma unroll
      for (int u = 0; u < 16; u += 2) {
        uint32_t h;
        const uint32_t pair = b[u] | (uint32_t)b[u + 1] << 8;
        asm("{.reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}"
            : "=r"(h) : "r"(pair));
        v[u] = (T)f16_float(h & 0xFFFF), v[u + 1] = (T)f16_float(h >> 16);
      }
      return;
    }
    ALL(NX_FLOAT8_E5M2, (T)f16_float((uint32_t)x << 8))
  }
  if constexpr (W == 1) ALL(NX_INT8, (T)(int8_t)x)
  if constexpr (W == 2 && is_float<T>) {
    ALL(NX_BFLOAT16, (T)__uint_as_float((uint32_t)x << 16))
    ALL(NX_FLOAT16, (T)f16_float((uint32_t)x))
  }
  if constexpr (W == 2) ALL(NX_INT16, (T)(int16_t)x)
  if constexpr (W == 4 && is_float<T>) ALL(NX_FLOAT32, (T)__uint_as_float((uint32_t)x))
  if constexpr (W == 4) ALL(NX_INT32, (T)(int32_t)x)
  if constexpr (W == 8 && is_float<T>) {
    ALL(NX_FLOAT64, (T)__longlong_as_double(x))
    ALL(NX_INT64, (T)(int64_t)x)
  }
#undef ALL
  /* The unsigned integers of W bytes, and a 64-bit one into an integer. */
#pragma unroll
  for (int u = 0; u < 16; u++) v[u] = (T)word(u);
}

/* The bits of element [e] of [w] bytes at [p], which lies on [w]
   bytes. */
__device__ __forceinline__ u64 element(const char *p, int64_t e, int w) {
  const char *at = p + e * w;
  switch (w) {
  case 1: return *(const uint8_t *)at;
  case 2: return *(const uint16_t *)at;
  case 4: return *(const uint32_t *)at;
  default: return *(const u64 *)at;
  }
}

/* f(W) for b's element bytes [wb], W a constant. */
template <typename F> __device__ __forceinline__ void by_width(int wb, F f) {
  switch (wb) {
  case 1: return f(std::integral_constant<int, 1>());
  case 2: return f(std::integral_constant<int, 2>());
  case 4: return f(std::integral_constant<int, 4>());
  default: return f(std::integral_constant<int, 8>());
  }
}

/* Shared memory: a's [m] rows of the stretch, then each warp's rows of
   b's values (NX_GEMV_STAGE). */
template <typename T> __device__ void gemv(const contract_params &p) {
  static_assert(NX_GEMV_STRETCH >= NX_GEMV_THREADS, "a's stretch holds the sums");
  constexpr int ROW = 16 * sizeof(T) + 16;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T *as = (T *)smem_raw;
  const Work w = schedule(p);
  const int z = (int)(w.tile / p.tiles_n), tn = (int)(w.tile % p.tiles_n);
  const int lane = threadIdx.x % 32, j0 = tn * NX_GEMV_THREADS + threadIdx.x - lane;
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
                      threadIdx.x / 32 * NX_GEMV_STAGE(sizeof(T));
  T acc[NX_GEMV_ROWS];
#pragma unroll
  for (int i = 0; i < NX_GEMV_ROWS; i++) acc[i] = T(0);
  /* A stretch is a block: a float range of more than one folds each
     stretch's sums into the block's slot. */
  static_assert(NX_GEMV_STRETCH == NX_CONTRACT_BLOCK, "a stretch is a block");
  constexpr bool FLOAT = is_float<T>;

  for (int64_t k0 = w.k0; k0 < w.k1; k0 += NX_GEMV_STRETCH) {
    if (FLOAT && k0 > w.k0 && !(p.flags & NX_CONTRACT_INIT_FIRST))
      fold<T, NX_GEMV_ROWS, NX_GEMV_THREADS>(acc, p.partials, blockIdx.x,
                                             k0 == w.k0 + NX_GEMV_STRETCH, m);
    const int len = (int)min((int64_t)NX_GEMV_STRETCH, w.k1 - k0);
    __syncthreads();
    /* a's rows over the stretch, in T as a cast stores them. */
    for (int e = threadIdx.x; e < m * NX_GEMV_STRETCH; e += NX_GEMV_THREADS) {
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
    __syncthreads();
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
          __syncwarp();
#pragma unroll
          for (int u = 0; u < 16; u++) {
            const int cc = u / PER * (32 / W) + lane / W;
            ((T *)(st + cc * ROW))[lane % W * PER + u % PER] = v[u];
          }
          __syncwarp();
#pragma unroll
          for (int t = 0; t < (int)sizeof(T); t++)
            ((uint4 *)v)[t] = ((const uint4 *)(st + lane * ROW))[t];
        }
      });
#pragma unroll
      for (int i = 0; i < NX_GEMV_ROWS; i++) {
        /* Rows past m skip their sums: a predicated multiply-add would
           still take its issue slot. */
        if (i >= m) break;
#pragma unroll
        for (int u = 0; u < 16; u++)
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

  if (FLOAT && w.k1 - w.k0 > NX_GEMV_STRETCH &&
      !(p.flags & NX_CONTRACT_INIT_FIRST))
    close<T, NX_GEMV_ROWS, NX_GEMV_THREADS>(acc, p.partials, blockIdx.x, m);
  if (w.splits > 1 &&
      !combine<T, NX_GEMV_ROWS, NX_GEMV_THREADS>(
          acc, (T *)p.partials + w.part * NX_GEMV_ROWS * NX_GEMV_THREADS,
          p.tickets + w.tile, w.split, w.splits, m))
    return;
  /* Each thread's m sums through a's stretch, NX_GEMV_THREADS of them. */
  __syncthreads();
  T *slots = as + threadIdx.x * m;
#pragma unroll
  for (int i = 0; i < NX_GEMV_ROWS; i++)
    if (i < m) slots[i] = acc[i];
#pragma unroll 1
  for (int i = 0; i < m; i++)
    store<T>(p, z, i, j, slots[i], false, 0, 0);
}

/* Segments */

/* The group of index [q] of batch element [z], [first] past the call's
   first, or [groups] for none. */
__device__ __forceinline__ int group_of(const segment_params &p, int z,
                                        int64_t q) {
  const int64_t x = (p.first + q) / p.inner % p.extent;
  const int64_t e = p.ids[z * p.sz + x * p.sx];
  return e >= 0 && e < p.groups ? (int)e : p.groups;
}

/* atomicAdd(&x[e], 1) for each active lane's [e], one atomic a group of
   the warp's lanes of one [e]: the old value plus the lane's rank among
   them. The groups' order follows the atomics', as each lane's does. W is
   int32_t or int64_t. */
template <typename W>
__device__ __forceinline__ W add_one(W *x, int64_t e) {
  const unsigned live = __activemask();
  const unsigned same = __match_any_sync(live, e);
  const int lane = threadIdx.x % 32, leader = __ffs(same) - 1;
  W base = 0;
  if (lane == leader) {
    if constexpr (sizeof(W) == 8)
      base = (W)atomicAdd((unsigned long long *)&x[e], (unsigned long long)__popc(same));
    else
      base = atomicAdd(&x[e], __popc(same));
  }
  base = __shfl_sync(same, base, leader);
  return base + __popc(same & ((1u << lane) - 1));
}

/* A grid-stride loop of [i] over [0, n). */
#define GRID_LOOP(i, n)                                                        \
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < (n);   \
       i += (int64_t)gridDim.x * blockDim.x)

/* Rows segments: the rows of a grouped by their ids, so that a tile of
   the contraction holds one group's rows and multiplies one slice of b.
   Four launches, by [phase], over the batch's G = groups + 1 groups,
   group [groups] the rows whose id is outside [0, groups), whose outputs
   have no products:

     0  zeroes [counts], G words a batch element
     1  counts each group's rows (integer atomics, a warp's lanes of one
        group at once, whose order changes no count)
     2  one thread: lays the groups out one after another in each batch
        element's run of [count] slots, and cuts each into row tiles of
        [bm] rows: [cursor] the group's first slot, [counts] its first
        tile, [counts] past the batch's G words its first slot again, and
        [ntiles] the tiles
     3  places each row in its group's run, and writes each tile's [desc]

   A row's place within its group follows the atomics' order, which
   changes no output's bits: a tile's rows are summed alike wherever they
   sit. */
__device__ void segment_rows(const segment_params &p) {
  const int G = p.groups + 1, m = (int)p.count;
  const int64_t slots = (int64_t)p.batch * G, rows = (int64_t)p.batch * m;
  int32_t *counts = (int32_t *)p.counts, *cursor = (int32_t *)p.cursor;
  int32_t *perm = (int32_t *)p.perm, *ntiles = (int32_t *)p.ntiles;
  int32_t *first = counts + slots; /* each group's first slot, then the end */
  switch (p.phase) {
  case 0:
    GRID_LOOP(i, slots) counts[i] = 0;
    return;
  case 1:
    GRID_LOOP(i, rows) {
      const int z = (int)(i / m), r = (int)(i % m);
      add_one(counts, (int64_t)z * G + group_of(p, z, r));
    }
    return;
  case 2:
    if (blockIdx.x == 0 && threadIdx.x == 0) {
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
    GRID_LOOP(i, rows) {
      const int z = (int)(i / m), r = (int)(i % m);
      perm[add_one(cursor, (int64_t)z * G + group_of(p, z, r))] = r;
    }
    /* Tile t's group: the last whose first tile is at most t. */
    GRID_LOOP(t, (int64_t)*ntiles) {
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
     1  a block a part: counts each group's k in it (integer atomics)
     2  each group of each z: its parts' first slots within the group,
        exclusive sums in part order, into [desc], and its count into
        [cursor]
     3  a thread a z: each group's first slot into [counts]
     4  a block a part: one warp places its k, 32 at a time, in order, so
        that a group's k keep their order: the order its products are
        numbered in */
__device__ void segment_k(const segment_params &p) {
  const int G = p.groups, P = p.bm;
  const int64_t k = p.count, part = (k + P - 1) / P;
  int64_t *counts = (int64_t *)p.counts, *cursor = (int64_t *)p.cursor;
  int64_t *desc = (int64_t *)p.desc;
  switch (p.phase) {
  case 0:
    GRID_LOOP(i, (int64_t)p.batch * P * G) desc[i] = 0;
    return;
  case 1: {
    const int z = blockIdx.x / P, c = blockIdx.x % P;
    int64_t *cnt = desc + ((int64_t)z * P + c) * G;
    const int64_t end = min(k, (c + 1) * part);
    for (int64_t q = c * part + threadIdx.x; q < end; q += blockDim.x) {
      const int e = group_of(p, z, q);
      if (e < G) add_one(cnt, e);
    }
    return;
  }
  case 2:
    GRID_LOOP(i, (int64_t)p.batch * G) {
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
    GRID_LOOP(z, (int64_t)p.batch) {
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
    if (threadIdx.x >= 32) return;
    const int z = blockIdx.x / P, c = blockIdx.x % P, lane = threadIdx.x;
    int64_t *at = desc + ((int64_t)z * P + c) * G;
    const int64_t *off = counts + (int64_t)z * (G + 1);
    const int64_t end = min(k, (c + 1) * part);
    for (int64_t q0 = c * part; q0 < end; q0 += 32) {
      const int64_t q = q0 + lane;
      const int e = q < end ? group_of(p, z, q) : G;
      const unsigned same = __match_any_sync(0xFFFFFFFFu, e);
      if (e < G) {
        const int64_t slot = off[e] + at[e] + __popc(same & ((1u << lane) - 1));
        if (p.wide) ((int64_t *)p.perm)[slot] = q;
        else ((int32_t *)p.perm)[slot] = (int32_t)q;
      }
      __syncwarp();
      if (e < G && lane == __ffs(same) - 1) at[e] += __popc(same);
      __syncwarp();
    }
  }
  }
}

/* Packs */

/* The bits of element [e] of [fmt] at [b], in the low bits of [lo] (and
   [hi] for a complex128); [spread] reads a sub-byte dtype one a byte. */
__device__ __forceinline__ void read_bits(const char *b, int64_t e, int fmt,
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
   into the word that holds it (nx_store_bits); [spread] writes a sub-byte
   dtype one a byte. */
__device__ __forceinline__ void write_bits(char *b, int64_t e, int fmt,
                                           bool spread, u64 lo, u64 hi) {
  const int n = bits_of(fmt), w = width_of(fmt);
  if (n < 8 && !spread) {
    nx_store_bits(b, e, n, (uint32_t)lo);
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
__device__ T scale_at(const pack_params &p, int64_t at) {
  if (p.mode & NX_PACK_EXPONENT) {
    const uint8_t e = ((const uint8_t *)p.scale)[at];
    if (e == 255) return T(NAN);
    if constexpr (sizeof(T) == 4) return ldexpf(1.0f, e - 127);
    else return ldexp(1.0, e - 127);
  }
  u64 lo, hi;
  read_bits((const char *)p.scale, at, p.scale_dtype, false, lo, hi);
  T im;
  return decode_any<T>(p.scale_dtype, p.acc_dtype, lo, hi, im);
}

/* An element's value into [dst_dtype], the accumulator's dtype, its
   compute type's, or bfloat16: decoded into the accumulator, times its
   scale, rounded once. */
template <typename T>
__device__ void put_value(const pack_params &p, char *d, int64_t e, u64 lo,
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
    default: ((uint16_t *)d)[e] = nx_float_to_bf16((float)v);
    }
  } else
    ((T *)d)[e] = v;
}

/* A sum of the workspace's dtype [src_dtype], the bits [lo] and [hi], into
   [dst_dtype] as a cast from the accumulator writes it. */
__device__ __forceinline__ u64 cast_bits(const pack_params &p, u64 lo, u64 &hi) {
  switch (p.src_dtype) {
  case NX_FLOAT32: return float_bits(p.dst_dtype, __uint_as_float((uint32_t)lo), 0, hi);
  case NX_FLOAT64: return float_bits(p.dst_dtype, __longlong_as_double(lo), 0, hi);
  case NX_COMPLEX64:
    return float_bits(p.dst_dtype, __uint_as_float((uint32_t)lo),
                      __uint_as_float((uint32_t)(lo >> 32)), hi);
  case NX_COMPLEX128:
    return float_bits(p.dst_dtype, __longlong_as_double(lo), __longlong_as_double(hi), hi);
  }
  return int_bits(p.dst_dtype, p.acc_dtype, lo, hi);
}

/* The pack, its index [I] wide enough for numel. */
template <typename I> __device__ void pack(const pack_params &p) {
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
  for (I x = blockIdx.x * (I)blockDim.x + threadIdx.x; x < (I)p.numel;
       x += (I)gridDim.x * blockDim.x) {
    I rest = x;
    int64_t s = sf, d = df, c = p.scale_first;
    for (int a = p.rank - 1; a >= 0; a--) {
      const I n = (I)p.n[a], i = rest % n;
      rest /= n;
      s += (int64_t)i * p.ss[a], d += (int64_t)i * p.ds[a];
      if (scaled) c += (int64_t)(i / (I)p.ratio[a]) * p.cs[a];
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

#include "mma.cuh"

} /* namespace nx_contract */

/* Instances */

#define T_f32 float
#define T_f64 double
#define T_u32 uint32_t
#define T_u64 u64
#define LOAD_native 0
#define LOAD_convert 1
#define TRANSPOSED_k false
#define TRANSPOSED_m true
#define TRANSPOSED_n true

#define TILE(name, bm, bn, bkb, wm, wn, s)                                     \
  struct name {                                                                \
    enum { BM = bm, BN = bn, BKB = bkb, WM = wm, WN = wn, STAGES = s };        \
    enum { THREADS = (bm / wm) * (bn / wn) * 32 };                             \
  };
NX_CUDA_TILES(TILE)
#undef TILE

/* Each family's definition of its rows of NX_CUDA_KERNELS, which
   kernels.cu defines. */
#define PACK(name, ...)                                                        \
  extern "C" __global__ void name(const __grid_constant__ pack_params p) {     \
    if (p.numel < INT32_MAX) nx_contract::pack<uint32_t>(p);                   \
    else nx_contract::pack<u64>(p);                                            \
  }
#define ROWS(name, ...)                                                        \
  extern "C" __global__ void __launch_bounds__(1024)                           \
      name(const __grid_constant__ segment_params p) {                         \
    nx_contract::segment_rows(p);                                              \
  }
#define KS(name, ...)                                                          \
  extern "C" __global__ void __launch_bounds__(1024)                           \
      name(const __grid_constant__ segment_params p) {                         \
    nx_contract::segment_k(p);                                                 \
  }
#define GEMM(name, t, bm, bn, bk, tm, tn, lm, wm, wn, minb, occ, load)         \
  extern "C" __global__ void __launch_bounds__(wm * wn * 32, minb)             \
      name(const __grid_constant__ contract_params p) {                        \
    nx_contract::gemm<                                                         \
        nx_contract::Tile<T_##t, bm, bn, bk, tm, tn, lm, wm, wn>,              \
        LOAD_##load>(p);                                                       \
  }
#define GEMV(name, t)                                                          \
  extern "C" __global__ void __launch_bounds__(NX_GEMV_THREADS)                \
      name(const __grid_constant__ contract_params p) {                        \
    nx_contract::gemv<T_##t>(p);                                               \
  }
#define MMA(name, kind, a, b, tile)                                            \
  extern "C" __global__ void __launch_bounds__(tile::THREADS)                  \
      name(const __grid_constant__ contract_params p) {                        \
    nx_contract::mma_kernel<KIND_##kind, TRANSPOSED_##a, TRANSPOSED_##b,       \
                            tile>(p);                                          \
  }
