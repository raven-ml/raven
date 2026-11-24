/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contractions: on the matrix units, tiles of float32 sums of float32,
   float16 or bfloat16 products, a Rows call's grouped tiles of them after
   its prepass, and integer tiles of bytes, staged as half, whose
   fragments sum blocks of 256 products exactly and add them into 64-bit
   sums; complex tiles, which stage a complex64 b as the real blocks
   that make a complex call a float32 one; the finish that adds a call's
   ranges and its init and rounds into its destination; the SIMT sums of
   the other integer calls and of Contracting segments, a thread an
   output, and their integer finish; and the prepass that lists a
   Contracting call's groups' terms.

   A tile's group of four SIMD-groups computes BM x BN outputs of one
   batch element over one range of k, or one chunk of it. Each step it
   stages BK terms of a's tile rows and of b's tile columns in threadgroup
   memory, each thread loading one run of a's and one of b's, then each
   SIMD-group's fragments, 8 x 8 each, multiply-accumulate. A lane reads
   its fragment elements itself, as float, so every product is a float32
   multiply of exact values. A multiply-accumulate is a chain of eight
   fused multiply-adds in increasing k, so an output's sum over its range
   is one chain of fused multiply-adds from +0 in increasing k:

     tile order --> per step: loads --> threadgroup memory --> MMAs
     --> sums: dst, or the range's float32 plane in the workspace

   b is staged as BK rows of BN (b k x n, its columns contiguous) or BN
   rows of BK (b n x k, its terms contiguous). Rows of a past the tile's
   last read its last row, and rows of an n x k b past n read its last,
   so every step takes the vector loads. The edge functions (ALIGNED
   false) also take a partial last step, each element of which past the
   range reads a's -0 and b's +0, whose product, -0, adds nothing to any
   sum. No output past the tile's rows and columns is stored. A b n x k
   tile has its edge function alone: it differs from the aligned one
   only by the last step, and runs whole steps as fast.

   A b k x n tile has two edge functions. The wide one, for n of at
   least BN, starts its last tile column at n - BN, so every column it
   reads lies inside, and stores only the columns past the tile column
   before, so each output is stored once: no thread reads past n, and
   each run is one vector load. The narrow one (NARROW), for n under BN,
   reads b's columns past n as its last, element by element. They are
   apart because a second path through the steps, even one no thread
   takes, slows every step: 20-35% at n of 250-300 on the M1 Max. The
   converting tiles read element by element at any n.

   A chunk past a range's first resumes its outputs' sums from the plane
   it stores into, so chunks in increasing k make the range's chain.

   A grouped tile (ROWS) is one of a Rows call: its rows, of one bucket,
   come from contract_buckets' list, each read and stored at its own
   batch element and row, and its b is its bucket's group's. A row's
   chain is the same in any tile, so no bit depends on which rows share
   one. The bucket of no group computes no product and stores +0.

   A complex tile (CPLX) is a float32 tile over a complex call's real
   form. It reads a complex64 b in place, as float pairs, and stages
   each element br + i bi as the real 2 x 2 block of terms 2r, 2r + 1
   and columns 2j, 2j + 1: [br, bi] above [-bi, br]. Column 2j then sums
   ar br then ai (-bi), and column 2j + 1 ar bi then ai br, the
   contract's numbering of a complex product's real terms. A routine's
   sign flips, [codes]' NX_CODES_FLIPS, conjugate b (its bit 0: [br, -bi]
   above [bi, br]), conjugate a (bit 1: the lower row negated, since
   (-ai) x = ai (-x) bit for bit) or negate (bit 2: the whole block).
   A thread's run lies in one real row of the blocks (b k x n) or one
   real column (b n x k), so it stages each of its elements from its
   pair's same or other part, as the row or column's parity gives, and
   flips its sign bit by one of two masks, for even and odd elements.
   A sign-bit flip is negation's, a NaN's included. */

#include "device.h"
#include "nx_dtype.h"
#include "nx_kinds.h"
#include "values.h"

#define NX_UNROLL _Pragma("clang loop unroll(full)")
#define NX_INLINE_TILE static inline __attribute__((always_inline))

struct contract_words {
  device const uchar *a;
  device const uchar *b;
  device float *y;
  long lda;
  long ldb;
  long sa;
  long sb;
  long ldy;
  long cy;
  long sy;
  long plane;
  ulong m;
  ulong n;
  ulong k;
  ulong tiles;
  ulong bkb;
  ulong padded;
  ulong first;
  ulong ranges;
  ulong spans;
  ulong length;
  ulong chunk;
  ulong codes;
  device const uint *list;
  device const uint *table;
  ulong groups;
  long sg;
  device const uchar *bs;
  ulong bsz;
  ulong bsn;
  ulong bsk;
  ulong drn;
  ulong drk;
  ulong dbl;
};

NX_FIELD(contract_words, a, NX_CONTRACT_A);
NX_FIELD(contract_words, b, NX_CONTRACT_B);
NX_FIELD(contract_words, y, NX_CONTRACT_Y);
NX_FIELD(contract_words, lda, NX_CONTRACT_LDA);
NX_FIELD(contract_words, ldb, NX_CONTRACT_LDB);
NX_FIELD(contract_words, sa, NX_CONTRACT_SA);
NX_FIELD(contract_words, sb, NX_CONTRACT_SB);
NX_FIELD(contract_words, ldy, NX_CONTRACT_LDY);
NX_FIELD(contract_words, cy, NX_CONTRACT_CY);
NX_FIELD(contract_words, sy, NX_CONTRACT_SY);
NX_FIELD(contract_words, plane, NX_CONTRACT_PLANE);
NX_FIELD(contract_words, m, NX_CONTRACT_M);
NX_FIELD(contract_words, n, NX_CONTRACT_N);
NX_FIELD(contract_words, k, NX_CONTRACT_K);
NX_FIELD(contract_words, tiles, NX_CONTRACT_TILES);
NX_FIELD(contract_words, bkb, NX_CONTRACT_BKB);
NX_FIELD(contract_words, drn, NX_CONTRACT_DRN);
NX_FIELD(contract_words, drk, NX_CONTRACT_DRK);
NX_FIELD(contract_words, dbl, NX_CONTRACT_DBL);
NX_FIELD(contract_words, padded, NX_CONTRACT_PADDED);
NX_FIELD(contract_words, first, NX_CONTRACT_FIRST);
NX_FIELD(contract_words, ranges, NX_CONTRACT_RANGES);
NX_FIELD(contract_words, spans, NX_CONTRACT_SPANS);
NX_FIELD(contract_words, length, NX_CONTRACT_LENGTH);
NX_FIELD(contract_words, chunk, NX_CONTRACT_CHUNK);
NX_FIELD(contract_words, codes, NX_CONTRACT_CODES);
NX_FIELD(contract_words, list, NX_CONTRACT_LIST);
NX_FIELD(contract_words, table, NX_CONTRACT_TABLE);
NX_FIELD(contract_words, groups, NX_CONTRACT_GROUPS);
NX_FIELD(contract_words, sg, NX_CONTRACT_SG);
NX_FIELD(contract_words, bs, NX_CONTRACT_BS);
NX_FIELD(contract_words, bsz, NX_CONTRACT_BSZ);
NX_FIELD(contract_words, bsn, NX_CONTRACT_BSN);
NX_FIELD(contract_words, bsk, NX_CONTRACT_BSK);
NX_SIZE(contract_words, NX_CONTRACT_WORDS);

/* A tile row of a grouped tile's table that holds no rows. */
#define NX_ROWS_UNUSED 0xFFFFFFFFu

/* The offset of a grouped tile's row at entry [j] of the list, of batch
   element z and row r: z [sz] + r [sr]. */
static inline long contract_grouped_row(constant contract_words &w, uint j,
                                        long sz, long sr) {
  const uint y = w.list[j], m = uint(w.m);
  const uint z = y / m;
  return long(z) * sz + long(y - z * m) * sr;
}

/* A run of [N] elements, loaded and stored whole at [T]'s alignment. */
template <typename T, int N> struct contract_run { T v[N]; };

/* The elements a tile staging [T] reads: [T]'s, or an integer tile's
   bytes. */
template <typename T, bool BYTES> struct contract_source { typedef T type; };
template <typename T> struct contract_source<T, true> { typedef uchar type; };

/* An element as staged: a float tile's as read; an integer tile's byte
   as the integer it holds by [kind] (NX_BYTES_*), exact in half: a
   boolean's 1 where it is not zero. */
template <typename T> static inline T contract_elem(T x, ushort) { return x; }
static inline half contract_elem(uchar x, ushort kind) {
  if (kind == NX_BYTES_BOOLEAN) return half(x != 0);
  return half(kind == NX_BYTES_SIGNED ? short(char(x)) : short(x));
}

/* A loader's element of [dt] at [p], by its cast into the staged [T],
   float32, the stage of every loader. */
static inline float contract_as(float, u64 c) { return as_type<float>(uint(c)); }
static inline int contract_code(float) { return NX_FLOAT32; }

template <typename T>
static inline T contract_load(device const uchar *p, int dt) {
  return contract_as(T(), cast(dt, contract_code(T()), read(p, 0, 0, dt)).lo);
}

/* [v] times 2^(e - 127), by the arithmetic of its bits, never through a
   float scale, which code 0 would make subnormal and the flush zero:
   NaN of code 255; NaN and the infinities kept; a zero or a subnormal
   [v] a zero of its sign; an exponent past 254 an infinity, below 1 a
   zero of the sign, the exact product otherwise. */
static inline float contract_exponent(float v, uint e) {
  if (e == 255) return as_type<float>(0x7FC00000u);
  const uint u = as_type<uint>(v), sign = u & 0x80000000u;
  const int f = int((u >> 23) & 255);
  if (f == 255) return v;
  if (f == 0) return as_type<float>(sign);
  const int g = f + int(e) - 127;
  if (g > 254) return as_type<float>(sign | 0x7F800000u);
  if (g < 1) return as_type<float>(sign);
  return as_type<float>((u & 0x807FFFFFu) | (uint(g) << 23));
}

/* A loader's scales of b: none, a code's power of two (Exponent) or a
   value of its dtype (Value). */
#define NX_LOADER_UNSCALED 0
#define NX_LOADER_EXPONENT 1
#define NX_LOADER_VALUE 2

/* [v] times its scale at [s], of kind [SK] and, for a Value, dtype
   [SD], as contract_scale computes it: the code's power of two by
   contract_exponent, or one rounded product with the Value's float32
   cast. */
template <int SK, int SD>
static inline float contract_scaled(float v, device const uchar *s) {
  if (SK == NX_LOADER_EXPONENT) return contract_exponent(v, uint(*s));
  return v * as_type<float>(uint(cast(SD, NX_FLOAT32, read(s, 0, 0, SD)).lo));
}

/* The low and high halves of a word of two signed 32-bit fields. */
static inline long contract_low(ulong w) { return long(int(uint(w))); }
static inline long contract_high(ulong w) { return long(int(uint(w >> 32))); }

/* [u] / d by d's divisor word [dw] (kernels.h): Apple GPUs divide in
   software, and a loader divides each element's column and term. */
static inline uint contract_div(uint u, ulong dw) {
  const uint t = metal::mulhi(u, uint(dw));
  return (t + ((u - t) >> uint((dw >> 32) & 255))) >> uint((dw >> 40) & 255);
}

/* Stores an integer tile's run [v] of bytes at [s], element by
   element, read by [kind]. */
template <typename T, typename S, int N>
NX_INLINE_TILE void contract_widen_run(threadgroup T *s,
                                       thread const contract_run<S, N> &v,
                                       ushort kind) {
  NX_UNROLL for (int e = 0; e < N; e++) s[e] = contract_elem(v.v[e], kind);
}

/* How a tile stages its operands' elements, by its kind: [get] reads
   element [e] of a run at [p], and [put] stores a run [v] read whole.
   A float tile's are of the staged dtype (0); an integer tile's bytes
   read by [kind] (1); a converting tile's (2) of [dt], [w] bytes an
   element, which it reads one by one. */
template <typename T, typename S, int KIND> struct contract_reader;
template <typename T> struct contract_reader<T, T, 0> {
  static T get(device const T *p, long e, ushort, int, long) { return p[e]; }
  template <int N>
  static void put(threadgroup T *s, thread const contract_run<T, N> &v,
                  ushort) {
    *(threadgroup contract_run<T, N> *)s = v;
  }
};
template <typename T> struct contract_reader<T, uchar, 1> {
  static T get(device const uchar *p, long e, ushort kind, int, long) {
    return contract_elem(p[e], kind);
  }
  template <int N>
  static void put(threadgroup T *s, thread const contract_run<uchar, N> &v,
                  ushort kind) {
    contract_widen_run(s, v, kind);
  }
};
template <typename T> struct contract_reader<T, uchar, 2> {
  static T get(device const uchar *p, long e, ushort, int dt, long w) {
    return contract_load<T>(p + e * w, dt);
  }
  template <int N>
  static void put(threadgroup T *, thread const contract_run<uchar, N> &,
                  ushort) {}
};

/* The terms an integer tile's fragments sum before its 64-bit sums take
   them: products of bytes are below 2^16 in magnitude, so 256 of them
   sum below 2^24, within which float holds every integer. */
#define NX_CONTRACT_BLOCK 256

/* Adds each fragment element, an integer below 2^24 in magnitude, to
   its 64-bit sum, and clears the fragments. */
template <int TM, int TN>
NX_INLINE_TILE void contract_flush(
    thread metal::simdgroup_float8x8 (&acc)[TM][TN],
    thread ulong (&sums)[TM][TN][2]) {
  NX_UNROLL for (int fi = 0; fi < TM; fi++) {
    NX_UNROLL for (int fj = 0; fj < TN; fj++) {
      thread auto &e = acc[fi][fj].thread_elements();
      sums[fi][fj][0] += ulong(long(int(e[0])));
      sums[fi][fj][1] += ulong(long(int(e[1])));
      acc[fi][fj] = metal::simdgroup_float8x8(0);
    }
  }
}

/* [v] rounded into the 16-bit staged dtype, as the finish rounds it. */
static inline ushort contract_round(half, float v) {
  return nx_float_to_f16(v);
}
static inline ushort contract_round(bfloat, float v) {
  return nx_float_to_bf16(v);
}

/* Stores the sum [v] at element [at]: of [y], as float32, or of [yt],
   rounded into [T], where [narrow]. A float32 tile's outputs are
   float32. */
template <typename T>
NX_INLINE_TILE void contract_put(device float *y, device ushort *yt, long at,
                                 float v, bool narrow) {
  if (narrow)
    yt[at] = contract_round(T(), v);
  else
    y[at] = v;
}

template <>
inline __attribute__((always_inline)) void
contract_put<float>(device float *y, device ushort *, long at, float v,
                    bool) {
  y[at] = v;
}

/* One step's multiply-accumulates of a SIMD-group: for each 8 terms of
   the step, its fragments of a's rows and b's columns, read lane by lane
   as float, then each fragment pair's multiply-accumulate, rows in turn,
   columns back and forth. SIMD-group barriers between the reads where
   [BAR]. Fragment [i] of the SIMD-group's rows is at tile row
   [8 (WM i) + wr], fragment [j] of its columns at [8 (WN j) + wc]. */
template <typename T, int BM, int BN, int BK, int WM, int WN, bool NK,
          bool BAR>
NX_INLINE_TILE void contract_step(
    threadgroup const T *As, threadgroup const T *Bs, short a_lane,
    short b_lane,
    thread metal::simdgroup_float8x8 (&acc)[BM / (8 * WM)][BN / (8 * WN)]) {
  constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);
  constexpr int LDA = BK + 16 / sizeof(T);
  constexpr int LDB = (NK ? BK : BN) + 16 / sizeof(T);
  NX_UNROLL for (int kk = 0; kk < BK; kk += 8) {
    metal::simdgroup_float8x8 af[TM], bf[TN];
    if (BAR) metal::simdgroup_barrier(metal::mem_flags::mem_none);
    NX_UNROLL for (int i = 0; i < TM; i++) {
      threadgroup const T *src = As + a_lane + 8 * WM * i * LDA + kk;
      thread auto &d = af[i].thread_elements();
      d[0] = float(src[0]);
      d[1] = float(src[1]);
    }
    if (BAR) metal::simdgroup_barrier(metal::mem_flags::mem_none);
    NX_UNROLL for (int j = 0; j < TN; j++) {
      const int nj = 8 * WN * j;
      threadgroup const T *src =
          Bs + b_lane + (NK ? nj * LDB + kk : kk * LDB + nj);
      thread auto &d = bf[j].thread_elements();
      d[0] = float(src[0]);
      d[1] = float(src[NK ? LDB : 1]);
    }
    if (BAR) metal::simdgroup_barrier(metal::mem_flags::mem_none);
    NX_UNROLL for (int i = 0; i < TM; i++) {
      NX_UNROLL for (int j0 = 0; j0 < TN; j0++) {
        const int j = (i % 2) ? TN - 1 - j0 : j0;
        metal::simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j],
                                             acc[i][j]);
      }
    }
  }
}

template <typename T, int BM, int BN, int BK, int WM, int WN, bool NK,
          bool BAR, bool ALIGNED, bool NARROW, bool INT, bool CONV, bool ROWS,
          int DA, int DB, int BB, int SK, int SD, bool NEG = false,
          bool CPLX = false, bool RT = false, bool BAND = false>
NX_INLINE_TILE void contract_tile(constant contract_words &w,
                                  threadgroup T *As, threadgroup T *Bs,
                                  uint gx, uint s, ushort t, ushort sg,
                                  ushort lane) {
  constexpr int THREADS = WM * WN * 32;
  constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);
  constexpr int LDA = BK + 16 / sizeof(T);
  constexpr int LDB = (NK ? BK : BN) + 16 / sizeof(T);
  /* Each thread's runs: a's of VA terms, RA a row; b's of VB elements,
     RB a staged row. */
  constexpr int VA = BM * BK / THREADS, RA = BK / VA;
  constexpr int BC = NK ? BK : BN, BR = NK ? BN : BK;
  constexpr int VB = BK * BN / THREADS, RB = BC / VB;
  static_assert(THREADS == NX_CONTRACT_THREADS, "a tile's threads");
  static_assert(RA * VA == BK && THREADS / RA == BM, "one run of a");
  static_assert(RB * VB == BC && THREADS / RB == BR, "one run of b");
  static_assert(!INT || NX_CONTRACT_BLOCK % BK == 0, "whole blocks");
  typedef typename contract_source<T, INT || CONV>::type S;
  typedef contract_run<S, VA> run_a;
  typedef contract_run<S, VB> run_b;
  typedef contract_reader<T, S, INT ? 1 : CONV ? 2 : 0> reader;
  static_assert(!NEG || RT, "a negated tile is a routine's");
  static_assert(!RT || (metal::is_same_v<T, float> && !INT && !CONV && !ROWS),
                "a routine's tiles");
  static_assert(!CPLX || (!INT && !CONV && !NEG && metal::is_same_v<T, float>),
                "a complex tile stages float32");
  static_assert(!BAND || (!INT && !CPLX && !RT && !ALIGNED),
                "a banded tile is an edge tile of a cast operand");
  static_assert(SK == NX_LOADER_UNSCALED || (CONV && DB >= 0 &&
                                             metal::is_same_v<T, float>),
                "a loader of a scaled b stages float32");
  const ushort kind_a = ushort((w.codes >> NX_CODES_KIND_A) & 3);
  const ushort kind_b = ushort((w.codes >> NX_CODES_KIND_B) & 3);
  /* A loader's operands' dtypes, its constants, and the bytes of their
     elements, by which it scales every offset in elements. */
  const int dt_a = DA >= 0 ? DA : int((w.codes >> NX_CODES_DTYPE_A) & 255);
  const int dt_b = DB >= 0 ? DB : int((w.codes >> NX_CODES_DTYPE_B) & 255);
  const long wa = CONV ? long(bits_of(dt_a) / 8) : 1;
  /* A loader of b narrower than a byte (BB bits) counts b's offsets in
     elements: pb is w.b plus b's element offset, which it reads at the
     bit phase of codes' NX_CODES_PHASE. */
  constexpr bool SUB = BB > 0 && BB < 8;
  const long wb = SUB ? 1 : CONV ? long(bits_of(dt_b) / 8) : 1;
  const uint b_phase = uint((w.codes >> NX_CODES_PHASE) & 7);

  /* The tile: batch element [z], then blocks of 2^swizzle tile rows, in
     each block the tile rows of a column in turn. Tiles past the last
     tile row compute nothing. */
  const uint i = uint(w.first) + gx;
  const uint per = uint(w.padded);
  const uint swz = uint((w.codes >> NX_CODES_SWIZZLE) & 15);
  const uint z = i / per, q = i - z * per;
  const uint tiles_m = uint(w.tiles), tiles_n = uint(w.tiles >> 32);
  const uint block = tiles_n << swz;
  const uint blk = q / block, within = q - blk * block;
  const uint tm = (blk << swz) | (within & ((1u << swz) - 1));
  const uint tn = within >> swz;
  if (tm >= tiles_m) return;
  const uint m = uint(w.m), n = uint(w.n);
  /* A wide k x n edge tile (WIDE) stores its columns from lo; the last
     starts at n - BN, so that it reads BN columns inside n. */
  constexpr bool WIDE = !NK && !ALIGNED && !NARROW && !CONV;
  const uint lo = tn * BN;
  const uint m0 = tm * BM, n0 = WIDE ? metal::min(lo, n - uint(BN)) : lo;
  /* A routine's triangle (codes' NX_CODES_TRI, lower and upper bits;
     NX_CODES_PAIR where a column is half an element's): a tile of no
     element of it computes nothing, and a tile stores only the outputs
     inside it. Only the routines' tiles [RT] take a routine's options,
     so that a plain call's code is as without them. */
  constexpr bool ROUTINE = RT;
  const uint tri = ROUTINE ? uint(w.codes >> NX_CODES_TRI) & 3 : 0;
  const uint pair = ROUTINE ? uint(w.codes >> NX_CODES_PAIR) & 1 : 0;
  if (tri != 0) {
    const uint c_lo = n0 >> pair, c_hi = (n0 + uint(BN) - 1) >> pair;
    if (tri == 1 && m0 + uint(BM) - 1 < c_lo) return;
    if (tri == 2 && m0 > c_hi) return;
  }

  /* A grouped tile's rows: [count] from entry [start] of the list, of
     [bucket], whose group's b it multiplies, or none for the bucket of
     no group. */
  uint start = 0, count = 0, bucket = 0;
  if (ROWS) {
    device const uint *e = w.table + 3 * ulong(tm);
    bucket = e[0];
    if (bucket == NX_ROWS_UNUSED) return;
    start = e[1];
    count = e[2];
  }
  const uint groups = uint(w.groups);
  const bool none = ROWS && bucket == uint(w.groups >> 32);
  if (none && w.chunk != 0) return;
  const uint rows = ROWS ? count : metal::min(uint(BM), m - m0);

  /* Range [s]'s terms, spans from s r / S to (s + 1) r / S, and the
     launch's chunk of them. */
  const ulong r = w.spans, ranges = w.ranges;
  const ulong k0 = NX_CONTRACT_SPAN * (s * r / ranges);
  const ulong k1 =
      metal::min(w.k, ulong(NX_CONTRACT_SPAN) * ((s + 1) * r / ranges));
  ulong c0 = k0, c1 = k1;
  if (w.length > 0) {
    c0 = k0 + w.chunk * w.length;
    if (c0 >= k1) return;
    c1 = metal::min(k1, c0 + w.length);
  }
  /* A banded tile's [BAND] banded operand (codes' NX_CODES_BAND_A and
     NX_CODES_BAND_B) is read from a band of [length] terms a range, the
     chunk's terms of range s from term s length: [ka] and [kb] are where
     each operand's chunk starts. Steps and bounds still count from c0.
     Other tiles read from c0, without the select. */
  const ulong ka = BAND && ((w.codes >> NX_CODES_BAND_A) & 1) != 0
                       ? s * w.length
                       : c0;
  const ulong kb = BAND && ((w.codes >> NX_CODES_BAND_B) & 1) != 0
                       ? s * w.length
                       : c0;
  const uint steps = none ? 0 : uint((c1 - c0) / BK);
  const uint tail = none ? 0 : uint((c1 - c0) % BK);

  /* Thread t's runs: a's at tile row ar, term ac; b's at staged row br,
     element bc. */
  const short ar = t / RA, ac = VA * (t % RA);
  const short br = t / RB, bc = VB * (t % RB);
  device const S *pa =
      (device const S *)w.a +
      wa * (ROWS ? contract_grouped_row(
                       w, start + metal::min(uint(ar), rows - 1), w.sa,
                       w.lda) +
                       long(ka) + ac
                 : long(z) * w.sa +
                       long(metal::min(m0 + ar, m - 1)) * w.lda + long(ka) +
                       ac);
  /* b's batch element, or a grouped tile's group of its bucket. */
  long b_base = long(z) * w.sb;
  device const S *b0 = (device const S *)w.b + wb * b_base;
  /* A loader's scale of b: [ps] that of b's batch element or group, at
     [sw] bytes an element. */
  const long sw = SK == NX_LOADER_VALUE ? long(bits_of(SD) / 8) : 1;
  device const uchar *ps = w.bs + sw * long(z) * contract_low(w.bsz);
  if (ROWS && !none) {
    const uint zb = bucket / groups, j = bucket - zb * groups;
    b_base = long(zb) * w.sb + long(j) * w.sg;
    b0 = (device const S *)w.b + wb * b_base;
    ps = w.bs + sw * (long(zb) * contract_low(w.bsz) +
                      long(j) * contract_high(w.bsz));
  }
  /* The scale's strides, signed; its blocks' elements divide a column
     and a term by their divisor words. */
  const long sn = contract_low(w.bsn), sk = contract_low(w.bsk);
  const uint bl = uint(w.bkb);
  const long bs = contract_high(w.bkb);
  device const S *pb;
  long b_step;
  bool b_inside = true;
  /* The real row (k x n) or column (n x k) of the thread's run. A
     complex tile reads it in b's row of complex elements r / 2, [ldb]
     floats apart, at its pair's float. */
  const long b_line = NK ? long(metal::min(n0 + br, n - 1)) : long(kb + br);
  const long b_row = CPLX ? b_line >> 1 : b_line;
  if (NK) {
    pb = b0 + wb * (b_row * w.ldb + long(kb) + bc);
    b_step = wb * BK;
  } else {
    pb = b0 + wb * (b_row * w.ldb + n0 + bc);
    b_step = wb * ((CPLX ? BK / 2 : BK) * w.ldb);
    b_inside = ALIGNED || WIDE || n0 + bc + VB <= n;
  }
  /* A complex tile's run: its line's parity, the block's lower row
     (k x n) or odd column (n x k), takes each element from its pair's
     other part ([swap]), and the sign table of the block's (row, column)
     with the routine's flips gives the even and odd elements' masks. */
  const uint b_mode = uint(w.codes >> NX_CODES_FLIPS) & 7;
  const uint cb = b_mode & 1, ca = (b_mode >> 1) & 1, ng = b_mode >> 2;
  const uint b_odd = uint(b_line) & 1;
  const uint flips[2][2] = {{ng, cb ^ ng}, {1 ^ cb ^ ca ^ ng, ca ^ ng}};
  const bool swap = b_odd != 0;
  const uint m_even = (NK ? flips[0][b_odd] : flips[b_odd][0]) << 31;
  const uint m_odd = (NK ? flips[1][b_odd] : flips[b_odd][1]) << 31;
  /* Element [at] of the run, at an even float from its pair's first. */
  auto c_part = [&](float v, long at) -> float {
    return as_type<float>(as_type<uint>(v) ^ ((at & 1) != 0 ? m_odd : m_even));
  };
  auto c_get = [&](long at) -> float {
    return c_part(float(pb[swap ? at ^ 1 : at]), at);
  };
  /* A complex tile's whole steps stage a pair of lines, its block's
     rows (k x n) or columns (n x k), from one read of b's line: thread t
     reads VH floats of pair lp's line at lc and stages both lines, the
     odd one from each element's other part. The masks are then the
     same for every thread, and no element is selected. */
  constexpr int VH = VB / 2, TP = 2 * RB;
  static_assert(!CPLX || (VH % 2 == 0 && BR % 2 == 0),
                "a complex tile's pairs of lines and of parts");
  const short lp = t / TP, lc = VH * (t % TP);
  device const S *qb = pb;
  long q_step = b_step;
  bool q_inside = true;
  if (CPLX && NK) {
    const long row = metal::min(long(n0 / 2 + lp), long(n / 2) - 1);
    qb = b0 + wb * (row * w.ldb + long(kb) + lc);
    q_step = wb * BK;
  } else if (CPLX) {
    qb = b0 + wb * ((long(kb) / 2 + lp) * w.ldb + n0 + lc);
    q_step = wb * (BK / 2 * w.ldb);
    q_inside = ALIGNED || WIDE || n0 + lc + VH <= n;
  }
  const uint q_even[2] = {flips[0][0] << 31,
                          (NK ? flips[1][0] : flips[0][1]) << 31};
  const uint q_odd[2] = {(NK ? flips[0][1] : flips[1][0]) << 31,
                         flips[1][1] << 31};
  auto q_flip = [](float v, uint m) -> float {
    return as_type<float>(as_type<uint>(v) ^ m);
  };
  threadgroup T *sa = As + ar * LDA + ac;
  threadgroup T *sb = Bs + br * LDB + bc;
  threadgroup T *sq = Bs + 2 * lp * LDB + lc;
  /* A loader's scale of b's element [at] of the thread's run, at term
     [t] of the run's first, as its scaled value: an n x k run is along
     k in one column, a k x n run along n at one term. */
  const long b_col = NK ? long(metal::min(n0 + br, n - 1)) : long(n0 + bc);
  auto b_get = [&](long at, ulong t) -> T {
    if (SK == NX_LOADER_UNSCALED) return reader::get(pb, at, kind_b, dt_b, wb);
    const uint c = uint(NK ? b_col : b_col + at);
    const uint u = uint(NK ? long(t) + at : long(t));
    const long so = long(contract_div(c, w.drn)) * sn + long(contract_div(u, w.drk)) * sk;
    /* Term u's offset: blocks of bl terms, bs elements apart, where bl
       is not 0, as Q8_0's quants lie between their scales. */
    const uint q = contract_div(u, w.dbl);
    const long ku = bl == 0 ? long(u) : long(q) * bs + long(u - q * bl);
    const long e = b_base + (NK ? long(c) * w.ldb + ku : ku * w.ldb + long(c));
    const float v =
        SUB ? contract_as(float(),
                          cast(dt_b, NX_FLOAT32, read(w.b, u64(e), b_phase, dt_b))
                              .lo)
            : contract_load<float>((device const uchar *)w.b + e * wb, dt_b);
    return T(contract_scaled<SK, SD>(v, ps + sw * so));
  };
  ulong b_term = c0 + (NK ? bc : br);
  /* A scaled n x k loader's whole steps as runs: where blocks and scale
     blocks are multiples of VB terms and at least BK, no block splits a
     run, whose VB terms then share their block and scale. The tile takes
     the run's block and scale block once, by division, and steps them by
     a compare a step, at most one wrap since BK <= bl, rk. The condition
     is uniform: every thread takes the same path. */
  constexpr bool RUNS = SK != NX_LOADER_UNSCALED && NK;
  constexpr bool RUNS_KN = SK != NX_LOADER_UNSCALED && !NK && SUB && VB * BB % 32 == 0;
  static_assert(!SUB || 8 % BB == 0, "a run's element within a byte");
  const uint rk = uint(contract_high(w.bsk));
  const uint u0 = uint(b_term);
  const bool runs = RUNS && rk >= uint(BK) && rk % uint(VB) == 0 &&
                    (bl == 0 || (bl >= uint(BK) && bl % uint(VB) == 0)) &&
                    u0 % uint(VB) == 0;
  /* The run's block [bq] and term in it [bo], its scale block [kq] and
     term in it [ko]; the column's first element and scale. */
  uint bq = 0, bo = 0, kq = 0, ko = 0;
  long col_at = 0, col_scale = 0;
  if (runs) {
    const uint c = uint(b_col);
    col_at = b_base + long(c) * w.ldb;
    col_scale = long(contract_div(c, w.drn)) * sn;
    kq = contract_div(u0, w.drk);
    ko = u0 - kq * rk;
    bq = bl == 0 ? 0 : contract_div(u0, w.dbl);
    bo = bl == 0 ? u0 : u0 - bq * bl;
  }
  /* An e2m1 element's exponent field is 0 or 126-129, so for a code e in
     [2, 252] its product with 2^(e - 127) has a field in [1, 254]:
     contract_exponent's shift, which one multiply by the normal 2^(e -
     127) gives exactly. Other codes take contract_exponent. */
  constexpr bool POW2 = SK == NX_LOADER_EXPONENT && DB == NX_FLOAT4_E2M1FN;
  /* Whether a run of b narrower than a byte reads its VB elements as
     32-bit words: codes' bit 36, where the host found every run's start
     on 4 bytes of the buffer, and VB elements whole words. A run lies in
     a whole step inside k, so its words are the bytes it reads by bytes:
     none past its last. At 16 rows over MXFP4's 5760 x 2880, 400 us a
     call against 492 by bytes. */
  constexpr int NW = VB * BB / 32;
  const bool words = SUB && VB * BB % 32 == 0 && ((w.codes >> NX_CODES_WORDS) & 1) != 0;

  /* Lane l holds, of each fragment, the elements at row
     fr = 4 l4 + 2 l2 + l1 and columns fc = 4 l3 + 2 l0 and fc + 1, li
     bit i of l. SIMD-group sg's first row and column in the tile are
     [wr] and [wc]. */
  const short fr = ((lane >> 2) & 4) | ((lane >> 1) & 3);
  const short fc = ((lane >> 1) & 4) | ((lane & 1) << 1);
  const short wr = 8 * (sg / WN), wc = 8 * (sg % WN);
  const short a_lane = (wr + fr) * LDA + fc;
  const short b_lane = NK ? (wc + fc) * LDB + fr : fr * LDB + wc + fc;

  metal::simdgroup_float8x8 acc[TM][TN];
  NX_UNROLL for (int fi = 0; fi < TM; fi++)
    NX_UNROLL for (int fj = 0; fj < TN; fj++)
      acc[fi][fj] = metal::simdgroup_float8x8(0);
  /* An integer tile's 64-bit sums, of the blocks its fragments flush. */
  ulong sums[TM][TN][2];
  /* Whether the tile reads and stores column [c]'s outputs: those inside
     n, and of a wide tile those from lo, which the tile before does not
     store, so no output is stored or resumed by two tiles. A fragment's
     pair of columns may straddle lo, since n - BN may be odd. */
  auto owns = [&](uint c) { return c < n && (!WIDE || c >= lo); };
  /* Whether output (r, c) of the matrix lies in the routine's triangle:
     every output without one. One compare an output, outside the steps,
     as the finish's at S > 1, so that no output outside the triangle is
     stored, whatever the tile. */
  auto inside = [&](uint r, uint c) {
    return tri == 0 || (tri == 1 ? (c >> pair) <= r : (c >> pair) >= r);
  };
  if (INT) {
    NX_UNROLL for (int fi = 0; fi < TM; fi++)
      NX_UNROLL for (int fj = 0; fj < TN; fj++)
        sums[fi][fj][0] = sums[fi][fj][1] = 0;
  }
  /* A chunk past its range's first resumes the sums from the plane: one
     branch a tile, outside the fragments' loops, where the steps of a
     tile that never resumes run as fast as without it (inside, they ran
     7% slower at 4096^3). */
  if (INT && w.chunk != 0) {
    device const ulong *y =
        (device const ulong *)w.y + long(z) * w.sy + long(s) * w.plane;
    NX_UNROLL for (int fi = 0; fi < TM; fi++) {
      NX_UNROLL for (int fj = 0; fj < TN; fj++) {
        const uint rr = wr + 8 * WM * fi + fr;
        const uint cc = n0 + wc + 8 * WN * fj + fc;
        if (rr >= rows) continue;
        device const ulong *o = y + long(m0 + rr) * w.ldy + long(cc) * w.cy;
        if (owns(cc)) sums[fi][fj][0] = o[0];
        if (owns(cc + 1)) sums[fi][fj][1] = o[w.cy];
      }
    }
  } else if (w.chunk != 0 ||
             (ROUTINE && s == 0 && ((w.codes >> NX_CODES_FIRST) & 1) != 0)) {
    /* A routine's init_first (codes' NX_CODES_FIRST) starts range 0 from
       y. */
    device const float *y = w.y + long(z) * w.sy + long(s) * w.plane;
    NX_UNROLL for (int fi = 0; fi < TM; fi++) {
      NX_UNROLL for (int fj = 0; fj < TN; fj++) {
        const uint rr = wr + 8 * WM * fi + fr;
        const uint cc = n0 + wc + 8 * WN * fj + fc;
        if (rr >= rows) continue;
        const long row =
            ROWS ? contract_grouped_row(w, start + rr, w.sy, w.ldy)
                 : long(m0 + rr) * w.ldy;
        device const float *o = y + row + long(cc) * w.cy;
        thread auto &e = acc[fi][fj].thread_elements();
        if (owns(cc) && inside(m0 + rr, cc)) e[0] = o[0];
        if (owns(cc + 1) && inside(m0 + rr, cc + 1)) e[1] = o[w.cy];
      }
    }
  }

  for (uint st = 0; st < steps; st++) {
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (CONV) {
      /* Element by element, each cast: a converting tile's elements are
         of its operands' widths. */
      NX_UNROLL for (int e = 0; e < VA; e++)
        sa[e] = reader::get(pa, e, kind_a, dt_a, wa);
      if (RUNS && runs) {
        /* The run's elements read as read() and contract_load read them,
           a byte at a time, or as words where [words]: Q8_0's runs lie 2
           bytes off 4 in every other block, and an e2m1 run at bit phase
           4 spans 3 bytes. */
        const long e0 = col_at + long(bq) * bs + long(bo);
        device const uchar *sc = ps + sw * (col_scale + long(kq) * sk);
        const uint code = POW2 ? uint(*sc) : 0;
        const bool pow2 = POW2 && code - 2 <= 250;
        const float p2 = as_type<float>(code << 23);
        const u64 bit0 = u64(b_phase) + u64(BB) * u64(e0);
        device const uchar *b8 = (device const uchar *)w.b + (bit0 >> 3);
        uint wd[NW > 0 ? NW : 1];
        if (words)
          NX_UNROLL for (int j = 0; j < NW; j++) wd[j] = ((device const uint *)b8)[j];
        NX_UNROLL for (int e = 0; e < VB; e++) {
          float v;
          if (SUB) {
            const uint at = uint(bit0 & 7) + uint(BB) * uint(e);
            const uint raw =
                (words ? wd[(BB * e) >> 5] >> ((BB * e) & 31) : uint(b8[at >> 3]) >> (at & 7)) &
                ((1u << BB) - 1);
            v = contract_as(
                float(),
                cast(dt_b, NX_FLOAT32, held_of(of_raw(dt_b, raw), 0)).lo);
          } else {
            v = contract_load<float>((device const uchar *)w.b + (e0 + e) * wb,
                                     dt_b);
          }
          sb[e] = T(pow2 ? v * p2 : contract_scaled<SK, SD>(v, sc));
        }
        bo += uint(BK);
        if (bl != 0 && bo >= bl) {
          bo -= bl;
          bq += 1;
        }
        ko += uint(BK);
        if (ko >= rk) {
          ko -= rk;
          kq += 1;
        }
      } else if (!NK && RUNS_KN && words && b_inside) {
        /* A k x n run of b narrower than a byte: VB adjacent columns at
           one term, read as words, the term's scale block found once. */
        const uint u = uint(b_term), c = uint(b_col);
        const long e0 = b_base + long(u) * w.ldb + long(c);
        device const uint *bw =
            (device const uint *)((device const uchar *)w.b + ((u64(b_phase) + u64(BB) * u64(e0)) >> 3));
        uint wd[NW > 0 ? NW : 1];
        NX_UNROLL for (int j = 0; j < NW; j++) wd[j] = bw[j];
        const long sterm = long(contract_div(u, w.drk)) * sk;
        NX_UNROLL for (int e = 0; e < VB; e++) {
          const uint raw = (wd[(BB * e) >> 5] >> ((BB * e) & 31)) & ((1u << BB) - 1);
          const float v =
              contract_as(float(), cast(dt_b, NX_FLOAT32, held_of(of_raw(dt_b, raw), 0)).lo);
          device const uchar *sc = ps + sw * (long(contract_div(c + e, w.drn)) * sn + sterm);
          const uint code = POW2 ? uint(*sc) : 0;
          const bool pow2 = POW2 && code - 2 <= 250;
          sb[e] = T(pow2 ? v * as_type<float>(code << 23) : contract_scaled<SK, SD>(v, sc));
        }
      } else {
        NX_UNROLL for (int e = 0; e < VB; e++) {
          const long at =
              b_inside ? long(e)
                       : long(metal::min(n0 + bc + e, n - 1)) - long(n0 + bc);
          sb[e] = b_get(at, b_term);
        }
      }
    } else {
      const run_a va = *(device const run_a *)pa;
      run_b vb;
      if (CPLX) {
        /* Line 0 takes element j of the read, line 1 its pair's other
           part, j ^ 1; a narrow tile's elements past n repeat the
           last's. */
        contract_run<S, VH> l0, l1;
        if (q_inside) {
          const contract_run<S, VH> x = *(device const contract_run<S, VH> *)qb;
          NX_UNROLL for (int j = 0; j < VH; j++) {
            l0.v[j] = S(q_flip(float(x.v[j]), q_even[j & 1]));
            l1.v[j] = S(q_flip(float(x.v[j ^ 1]), q_odd[j & 1]));
          }
        } else {
          NX_UNROLL for (int j = 0; j < VH; j++) {
            const long a = long(metal::min(n0 + lc + j, n - 1)) - long(n0 + lc);
            l0.v[j] = S(q_flip(float(qb[a]), q_even[a & 1]));
            l1.v[j] = S(q_flip(float(qb[a ^ 1]), q_odd[a & 1]));
          }
        }
        reader::put(sq, l0, kind_b);
        reader::put(sq + LDB, l1, kind_b);
      } else if (b_inside) {
        vb = *(device const run_b *)pb;
      } else {
        NX_UNROLL for (int e = 0; e < VB; e++) {
          const long at = long(metal::min(n0 + bc + e, n - 1)) - long(n0 + bc);
          vb.v[e] = pb[at];
        }
      }
      if (NEG) {
        /* A whole step has no padding: every element of va is a's. */
        run_a na;
        NX_UNROLL for (int e = 0; e < VA; e++) na.v[e] = -va.v[e];
        reader::put(sa, na, kind_a);
      } else {
        reader::put(sa, va, kind_a);
      }
      if (!CPLX) reader::put(sb, vb, kind_b);
    }
    pa += wa * BK;
    pb += b_step;
    qb += q_step;
    b_term += BK;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    contract_step<T, BM, BN, BK, WM, WN, NK, BAR>(As, Bs, a_lane, b_lane,
                                                  acc);
    if (INT && st % (NX_CONTRACT_BLOCK / BK) == NX_CONTRACT_BLOCK / BK - 1)
      contract_flush(acc, sums);
  }

  if (!ALIGNED && tail > 0) {
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    /* Past the tail a's padding is -0, whose products with b's +0 leave
       a -0 sum alone; a negated tile flips a's elements alone. */
    NX_UNROLL for (int e = 0; e < VA; e++)
      sa[e] = uint(ac + e) >= tail ? T(-0.0f)
              : NEG ? T(-reader::get(pa, e, kind_a, dt_a, wa))
                    : reader::get(pa, e, kind_a, dt_a, wa);
    NX_UNROLL for (int e = 0; e < VB; e++) {
      const bool in = NK ? uint(bc + e) < tail : uint(br) < tail;
      const long at =
          NK || WIDE ? long(e)
                     : long(metal::min(n0 + bc + e, n - 1)) - long(n0 + bc);
      /* c_get stays out of b_get, which loaders call out of line, so
         their code holds none of the complex run. */
      sb[e] = !in ? T(0.0f) : CPLX ? T(c_get(at)) : b_get(at, b_term);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    contract_step<T, BM, BN, BK, WM, WN, NK, BAR>(As, Bs, a_lane, b_lane,
                                                  acc);
  }

  const bool narrow = ((w.codes >> NX_CODES_NARROW) & 1) != 0;
  const long base = long(z) * w.sy + long(s) * w.plane;
  if (INT) {
    contract_flush(acc, sums);
    device ulong *yl = (device ulong *)w.y + base;
    device uint *yu = (device uint *)w.y + base;
    NX_UNROLL for (int fi = 0; fi < TM; fi++) {
      const uint rr = wr + 8 * WM * fi + fr;
      if (rr >= rows) continue;
      NX_UNROLL for (int fj = 0; fj < TN; fj++) {
        const uint cc = n0 + wc + 8 * WN * fj + fc;
        const long at = long(m0 + rr) * w.ldy + long(cc) * w.cy;
        NX_UNROLL for (int h = 0; h < 2; h++) {
          if (!owns(cc + h)) continue;
          const long o = at + h * w.cy;
          if (narrow)
            yu[o] = uint(sums[fi][fj][h]);
          else
            yl[o] = sums[fi][fj][h];
        }
      }
    }
    return;
  }
  device float *y = w.y + base;
  device ushort *yt = (device ushort *)w.y + base;
  NX_UNROLL for (int fi = 0; fi < TM; fi++) {
    const uint rr = wr + 8 * WM * fi + fr;
    if (rr >= rows) continue;
    NX_UNROLL for (int fj = 0; fj < TN; fj++) {
      const uint cc = n0 + wc + 8 * WN * fj + fc;
      const long at = (ROWS ? contract_grouped_row(w, start + rr, w.sy, w.ldy)
                            : long(m0 + rr) * w.ldy) +
                      long(cc) * w.cy;
      thread auto &e = acc[fi][fj].thread_elements();
      if (owns(cc) && inside(m0 + rr, cc))
        contract_put<T>(y, yt, at, e[0], narrow);
      if (owns(cc + 1) && inside(m0 + rr, cc + 1))
        contract_put<T>(y, yt, at + w.cy, e[1], narrow);
    }
  }
}

/* A tile kernel: [T] staged, BM x BN outputs, BK terms a step, WM x WN
   SIMD-groups, b n x k where [NK], SIMD-group barriers where [BAR],
   every step whole and every column inside where [ALIGNED], fewer
   columns than BN where [NARROW], a's values negated where [NEG], b
   complex64 where [CPLX], a routine's options read where [RT], and an
   operand read through its band where [BAND]. Its threadgroup memory
   holds a's staged tile, then b's. */
#define CONTRACT_TILE_OF(name, T, BM, BN, BK, WM, WN, NK, BAR, ALIGNED,       \
                         NARROW, INT, CONV, ROWS, NEG, CPLX, RT, BAND)        \
  NX_KERNEL(NX_CONTRACT_THREADS)                                              \
  void name(constant contract_words &w [[buffer(0)]],                         \
            threadgroup uchar *shared [[threadgroup(0)]],                     \
            uint3 g [[threadgroup_position_in_grid]],                         \
            ushort t [[thread_index_in_threadgroup]],                         \
            ushort sg [[simdgroup_index_in_threadgroup]],                     \
            ushort lane [[thread_index_in_simdgroup]]) {                      \
    threadgroup T *As = (threadgroup T *)shared;                              \
    threadgroup T *Bs = As + BM * (BK + 16 / sizeof(T));                      \
    contract_tile<T, BM, BN, BK, WM, WN, NK, BAR, ALIGNED, NARROW, INT, CONV, \
                  ROWS, -1, -1, 0, NX_LOADER_UNSCALED, 0, NEG, CPLX, RT,      \
                  BAND>(                                                      \
        w, As, Bs, g.x, g.z, t, sg, lane);                                    \
  }
#define CONTRACT_TILE(name, T, BM, BN, BK, WM, WN, NK, BAR, ALIGNED, NARROW,  \
                      INT, CONV, ROWS)                                        \
  CONTRACT_TILE_OF(name, T, BM, BN, BK, WM, WN, NK, BAR, ALIGNED, NARROW,     \
                   INT, CONV, ROWS, false, false, false, false)

/* A b k x n tile in its three functions, aligned (_a), wide edge (_e)
   and narrow edge (_n); a b n x k tile in its edge function alone. */
#define CONTRACT_TILES_kn(name, T, BM, BN, BK, WM, WN, BAR, NEG, CPLX, RT)    \
  CONTRACT_TILE_OF(name##_a, T, BM, BN, BK, WM, WN, false, BAR, true, false,  \
                   false, false, false, NEG, CPLX, RT, false)                 \
  CONTRACT_TILE_OF(name##_e, T, BM, BN, BK, WM, WN, false, BAR, false, false, \
                   false, false, false, NEG, CPLX, RT, false)                 \
  CONTRACT_TILE_OF(name##_n, T, BM, BN, BK, WM, WN, false, BAR, false, true,  \
                   false, false, false, NEG, CPLX, RT, false)
#define CONTRACT_TILES_nk(name, T, BM, BN, BK, WM, WN, BAR, NEG, CPLX, RT)    \
  CONTRACT_TILE_OF(name, T, BM, BN, BK, WM, WN, true, BAR, false, false,      \
                   false, false, false, NEG, CPLX, RT, false)

/* The tiles of a staged dtype and b's layout: 64 x 64 and 32 x 32 of 2 x 2
   SIMD-groups, 16 x 64 of 1 x 4, of the depths and barriers contract.ml
   states, negated where [NEG], of a complex64 b where [CPLX], a routine's
   where [RT]. */
#define CONTRACT_SET(stage, T, BK64, BK32, BAR, layout, NEG, CPLX, RT)        \
  CONTRACT_TILES_##layout(contract_##stage##_64_##layout, T, 64, 64, BK64, 2, \
                          2, BAR, NEG, CPLX, RT)                              \
  CONTRACT_TILES_##layout(contract_##stage##_32_##layout, T, 32, 32, BK32, 2, \
                          2, BAR, NEG, CPLX, RT)                              \
  CONTRACT_TILES_##layout(contract_##stage##_16_##layout, T, 16, 64, 32, 1,   \
                          4, true, NEG, CPLX, RT)

CONTRACT_SET(f32, float, 16, 16, false, kn, false, false, false)
CONTRACT_SET(f32, float, 16, 16, false, nk, false, false, false)
CONTRACT_SET(f16, half, 32, 32, true, kn, false, false, false)
CONTRACT_SET(f16, half, 32, 32, true, nk, false, false, false)
CONTRACT_SET(bf16, bfloat, 16, 16, true, kn, false, false, false)
CONTRACT_SET(bf16, bfloat, 16, 16, true, nk, false, false, false)
/* The routines' float32 tiles, plain and negated (contract.mli's [neg]). */
CONTRACT_SET(f32_rt, float, 16, 16, false, kn, false, false, true)
CONTRACT_SET(f32_rt, float, 16, 16, false, nk, false, false, true)
CONTRACT_SET(f32_neg, float, 16, 16, false, kn, true, false, true)
CONTRACT_SET(f32_neg, float, 16, 16, false, nk, true, false, true)
/* The complex tiles: the float32 tiles' shapes, depths and barriers. */
CONTRACT_SET(c64, float, 16, 16, false, kn, false, true, true)
CONTRACT_SET(c64, float, 16, 16, false, nk, false, true, true)

/* The banded tiles of a staged dtype: a tile's edge functions, which a
   call that reads an operand through its band takes in place of the
   tile's, at its shape, depth and barriers. An aligned call takes the
   wide edge function. */
#define CONTRACT_BANDED(name, T, BM, BN, BK, WM, WN, BAR)                     \
  CONTRACT_TILE_OF(name##_kn_e_band, T, BM, BN, BK, WM, WN, false, BAR,       \
                   false, false, false, false, false, false, false, false,    \
                   true)                                                      \
  CONTRACT_TILE_OF(name##_kn_n_band, T, BM, BN, BK, WM, WN, false, BAR,       \
                   false, true, false, false, false, false, false, false,     \
                   true)                                                      \
  CONTRACT_TILE_OF(name##_nk_band, T, BM, BN, BK, WM, WN, true, BAR, false,   \
                   false, false, false, false, false, false, false, true)
#define CONTRACT_BANDED_SET(stage, T, BK64, BK32, BAR)                        \
  CONTRACT_BANDED(contract_##stage##_64, T, 64, 64, BK64, 2, 2, BAR)          \
  CONTRACT_BANDED(contract_##stage##_32, T, 32, 32, BK32, 2, 2, BAR)          \
  CONTRACT_BANDED(contract_##stage##_16, T, 16, 64, 32, 1, 4, true)

CONTRACT_BANDED_SET(f32, float, 16, 16, false)
CONTRACT_BANDED_SET(f16, half, 32, 32, true)
CONTRACT_BANDED_SET(bf16, bfloat, 16, 16, true)

/* The integer tiles: bytes staged as half, at the float16 tiles' shapes
   and depths, each of its edge function alone, which with b k x n is
   the wide one; a b k x n tile's narrow function is _n. */
#define CONTRACT_INT_TILE(name, BM, BN, WM, WN, NK, NARROW)                   \
  CONTRACT_TILE(name, half, BM, BN, 32, WM, WN, NK, true, false, NARROW,      \
                true, false, false)
#define CONTRACT_INT_SET(layout, NK, suffix, NARROW)                          \
  CONTRACT_INT_TILE(contract_i8_64_##layout##suffix, 64, 64, 2, 2, NK,        \
                    NARROW)                                                   \
  CONTRACT_INT_TILE(contract_i8_32_##layout##suffix, 32, 32, 2, 2, NK,        \
                    NARROW)                                                   \
  CONTRACT_INT_TILE(contract_i8_16_##layout##suffix, 16, 64, 1, 4, NK, NARROW)

CONTRACT_INT_SET(kn, false, , false)
CONTRACT_INT_SET(kn, false, _n, true)
CONTRACT_INT_SET(nk, true, , false)

/* The grouped tiles of a staged dtype and b's layout: 32 x 64 of 2 x 2
   SIMD-groups at the 32 x 32 tile's depth and barriers, 16 x 64 of
   1 x 4 as the 16 x 64 tile, each of its edge function alone, which
   with b k x n is the wide one; a b k x n tile's narrow function is
   _n; of a complex64 b where [CPLX], of a banded operand where [BAND]. */
#define CONTRACT_GROUPED_SET(stage, T, BK32, BAR, layout, NK, suffix, NARROW, \
                             CPLX, BAND)                                      \
  CONTRACT_TILE_OF(contract_rows_##stage##_32_##layout##suffix, T, 32, 64,    \
                   BK32, 2, 2, NK, BAR, false, NARROW, false, false, true,    \
                   false, CPLX, false, BAND)                                  \
  CONTRACT_TILE_OF(contract_rows_##stage##_16_##layout##suffix, T, 16, 64,    \
                   32, 1, 4, NK, true, false, NARROW, false, false, true,     \
                   false, CPLX, false, BAND)

CONTRACT_GROUPED_SET(f32, float, 16, false, kn, false, , false, false, false)
CONTRACT_GROUPED_SET(f32, float, 16, false, kn, false, _n, true, false, false)
CONTRACT_GROUPED_SET(f32, float, 16, false, nk, true, , false, false, false)
CONTRACT_GROUPED_SET(f16, half, 32, true, kn, false, , false, false, false)
CONTRACT_GROUPED_SET(f16, half, 32, true, kn, false, _n, true, false, false)
CONTRACT_GROUPED_SET(f16, half, 32, true, nk, true, , false, false, false)
CONTRACT_GROUPED_SET(bf16, bfloat, 16, true, kn, false, , false, false, false)
CONTRACT_GROUPED_SET(bf16, bfloat, 16, true, kn, false, _n, true, false, false)
CONTRACT_GROUPED_SET(bf16, bfloat, 16, true, nk, true, , false, false, false)
CONTRACT_GROUPED_SET(c64, float, 16, false, kn, false, , false, true, false)
CONTRACT_GROUPED_SET(c64, float, 16, false, kn, false, _n, true, true, false)
CONTRACT_GROUPED_SET(c64, float, 16, false, nk, true, , false, true, false)
/* The banded grouped tiles. */
CONTRACT_GROUPED_SET(f32, float, 16, false, kn, false, _band, false, false,
                     true)
CONTRACT_GROUPED_SET(f32, float, 16, false, kn, false, _n_band, true, false,
                     true)
CONTRACT_GROUPED_SET(f32, float, 16, false, nk, true, _band, false, false,
                     true)
CONTRACT_GROUPED_SET(f16, half, 32, true, kn, false, _band, false, false, true)
CONTRACT_GROUPED_SET(f16, half, 32, true, kn, false, _n_band, true, false,
                     true)
CONTRACT_GROUPED_SET(f16, half, 32, true, nk, true, _band, false, false, true)
CONTRACT_GROUPED_SET(bf16, bfloat, 16, true, kn, false, _band, false, false,
                     true)
CONTRACT_GROUPED_SET(bf16, bfloat, 16, true, kn, false, _n_band, true, false,
                     true)
CONTRACT_GROUPED_SET(bf16, bfloat, 16, true, nk, true, _band, false, false,
                     true)

/* Loaders: tiles that read a weight of another dtype than their stage,
   made for their operands' dtypes, the dtype codes and b's scale as
   constants, so each element's read and cast compile to the code of its
   dtypes. A key's tiles are the float32 32 x 32, 16 x 64 and grouped
   tiles' shapes, depths and barriers.
   [akey] expands to a's dtype code, [bkey] to b's dtype code, its bits,
   its scale's kind and a Value scale's dtype code. A banded loader
   ([BAND]) reads a through its band. */
#define CONTRACT_LOADER(name, T, BM, BN, BK, WM, WN, NK, BAR, ROWS, akey,    \
                        bkey, BAND)                                           \
  NX_KERNEL(NX_CONTRACT_THREADS)                                              \
  void name(constant contract_words &w [[buffer(0)]],                         \
            threadgroup uchar *shared [[threadgroup(0)]],                     \
            uint3 g [[threadgroup_position_in_grid]],                         \
            ushort t [[thread_index_in_threadgroup]],                         \
            ushort sg [[simdgroup_index_in_threadgroup]],                     \
            ushort lane [[thread_index_in_simdgroup]]) {                      \
    threadgroup T *As = (threadgroup T *)shared;                              \
    threadgroup T *Bs = As + BM * (BK + 16 / sizeof(T));                      \
    contract_tile<T, BM, BN, BK, WM, WN, NK, BAR, false, false, false, true,  \
                  ROWS, akey, bkey, false, false, false, BAND>(               \
        w, As, Bs, g.x, g.z, t, sg, lane);                                    \
  }

/* Each key's dtypes. */
#define NX_LOADER_A_f32 NX_FLOAT32
#define NX_LOADER_A_f16 NX_FLOAT16
#define NX_LOADER_A_bf16 NX_BFLOAT16
#define NX_LOADER_B_e2m1_by_exp NX_FLOAT4_E2M1FN, 4, NX_LOADER_EXPONENT, 0
#define NX_LOADER_B_i8_by_f16 NX_INT8, 8, NX_LOADER_VALUE, NX_FLOAT16
#define NX_LOADER_B_i8_by_f32 NX_INT8, 8, NX_LOADER_VALUE, NX_FLOAT32

/* The 32 x 32 tile's depth and barriers at each stage. */
#define NX_LOADER_BK32_float 16
#define NX_LOADER_BK32_half 32
#define NX_LOADER_BK32_bfloat 16
#define NX_LOADER_BAR_float false
#define NX_LOADER_BAR_half true
#define NX_LOADER_BAR_bfloat true

#define CONTRACT_LOADER_SET(b, a, T, layout, NK, BAND)                        \
  CONTRACT_LOADER(contract_ld_##b##_##a##_32_##layout, T, 32, 32,             \
                  NX_LOADER_BK32_##T, 2, 2, NK, NX_LOADER_BAR_##T, false,     \
                  NX_LOADER_A_##a, NX_LOADER_B_##b, BAND)                     \
  CONTRACT_LOADER(contract_ld_##b##_##a##_16_##layout, T, 16, 64, 32, 1, 4,   \
                  NK, true, false, NX_LOADER_A_##a, NX_LOADER_B_##b, BAND)    \
  CONTRACT_LOADER(contract_rows_ld_##b##_##a##_32_##layout, T, 32, 64,        \
                  NX_LOADER_BK32_##T, 2, 2, NK, NX_LOADER_BAR_##T, true,      \
                  NX_LOADER_A_##a, NX_LOADER_B_##b, BAND)                     \
  CONTRACT_LOADER(contract_rows_ld_##b##_##a##_16_##layout, T, 16, 64, 32, 1, \
                  4, NK, true, true, NX_LOADER_A_##a, NX_LOADER_B_##b, BAND)

#define CONTRACT_LOADERS(X, b, B, a, A, T)                                    \
  CONTRACT_LOADER_SET(b, a, T, kn, false, false)                              \
  CONTRACT_LOADER_SET(b, a, T, nk, true, false)

NX_CONTRACT_LOADER_KEYS(CONTRACT_LOADERS, _)

/* The banded loaders: a key's tiles reading a, cast into float32,
   through its band. */
#define CONTRACT_LOADER_BANDS(X, b, B)                                        \
  CONTRACT_LOADER_SET(b, f32, float, kn_band, false, true)                    \
  CONTRACT_LOADER_SET(b, f32, float, nk_band, true, true)

NX_CONTRACT_LOADER_BANDS(CONTRACT_LOADER_BANDS, _)

/* Buckets

   contract_buckets, a Rows call's prepass in one group: each row's
   bucket, then over chunks of NX_BUCKETS_CHUNK buckets, the chunk's
   rows counted with threadgroup atomics, the counts and tile counts
   scanned across the group by SIMD-group prefix sums, each bucket's
   first entry and first tile row written, and its rows listed, in an
   order the atomics leave free, which no bit depends on. Then each tile
   row finds its bucket by binary search over the first tile rows. Each
   thread reads the buckets of the rows it wrote, so only the search
   waits on device memory. */

struct buckets_words {
  device const long *ids;
  device uint *of;
  device uint *list;
  device uint *table;
  device uint *firsts;
  long ids_z;
  long ids_r;
  ulong rows;
  ulong m;
  ulong inner;
  ulong extent;
  ulong groups;
  ulong buckets;
  ulong tile;
  ulong bound;
  ulong sort;
};

NX_FIELD(buckets_words, ids, NX_BUCKETS_IDS);
NX_FIELD(buckets_words, of, NX_BUCKETS_OF);
NX_FIELD(buckets_words, list, NX_BUCKETS_LIST);
NX_FIELD(buckets_words, table, NX_BUCKETS_TABLE);
NX_FIELD(buckets_words, firsts, NX_BUCKETS_FIRSTS);
NX_FIELD(buckets_words, ids_z, NX_BUCKETS_IDS_Z);
NX_FIELD(buckets_words, ids_r, NX_BUCKETS_IDS_R);
NX_FIELD(buckets_words, rows, NX_BUCKETS_ROWS);
NX_FIELD(buckets_words, m, NX_BUCKETS_M);
NX_FIELD(buckets_words, inner, NX_BUCKETS_INNER);
NX_FIELD(buckets_words, extent, NX_BUCKETS_EXTENT);
NX_FIELD(buckets_words, groups, NX_BUCKETS_GROUPS);
NX_FIELD(buckets_words, buckets, NX_BUCKETS_BUCKETS);
NX_FIELD(buckets_words, tile, NX_BUCKETS_TILE);
NX_FIELD(buckets_words, bound, NX_BUCKETS_BOUND);
NX_FIELD(buckets_words, sort, NX_BUCKETS_SORT);
NX_SIZE(buckets_words, NX_BUCKETS_WORDS);

NX_KERNEL(NX_BUCKETS_THREADS)
void contract_buckets(constant buckets_words &w [[buffer(0)]],
                      threadgroup uchar *shared [[threadgroup(0)]],
                      uint t [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NT = NX_BUCKETS_THREADS, C = NX_BUCKETS_CHUNK;
  constexpr uint PER = C / NT, NSG = NT / 32;
  constexpr auto relaxed = metal::memory_order_relaxed;
  constexpr auto group = metal::mem_flags::mem_threadgroup;
  threadgroup metal::atomic_uint *count =
      (threadgroup metal::atomic_uint *)shared;
  threadgroup uint *first = (threadgroup uint *)(shared + 4 * C);
  threadgroup uint2 *part = (threadgroup uint2 *)(shared + 8 * C);
  /* The rows and tile rows of the chunks before, and the chunk's. */
  threadgroup uint2 *carry = part + NSG, *total = part + NSG + 1;
  const uint rows = uint(w.rows), m = uint(w.m), g = uint(w.groups);
  const uint buckets = uint(w.buckets), none = buckets - 1;
  const uint bm = uint(w.tile);
  device uint *rs = w.firsts, *ts = w.firsts + buckets + 1;

  /* A row's id is tested signed and whole: an id past 2^32 or below 0
     never reaches a group by its low bits. */
  for (uint i = t; i < rows; i += NT) {
    const uint z = i / m, r = i - z * m;
    const ulong ra = (ulong(r) / w.inner) % w.extent;
    const long id = w.ids[long(z) * w.ids_z + long(ra) * w.ids_r];
    w.of[i] = id >= 0 && id < long(g) ? z * g + uint(id) : none;
  }
  if (w.sort == 0) return;

  if (t == 0) *carry = uint2(0);
  for (uint c0 = 0; c0 < buckets; c0 += C) {
    const uint nc = metal::min(C, buckets - c0);
    for (uint e = t; e < C; e += NT)
      metal::atomic_store_explicit(&count[e], 0u, relaxed);
    metal::threadgroup_barrier(group);
    /* A bucket below the chunk's wraps past it. */
    for (uint i = t; i < rows; i += NT) {
      const uint e = w.of[i] - c0;
      if (e < nc) metal::atomic_fetch_add_explicit(&count[e], 1u, relaxed);
    }
    metal::threadgroup_barrier(group);
    /* Thread t owns the chunk's buckets from t PER: its rows and tile
       rows, then their exclusive scan in the group. */
    uint2 own = uint2(0);
    for (uint u = 0; u < PER; u++) {
      const uint e = t * PER + u;
      if (e >= nc) break;
      const uint c = metal::atomic_load_explicit(&count[e], relaxed);
      own += uint2(c, (c + bm - 1) / bm);
    }
    const uint2 ex = uint2(metal::simd_prefix_exclusive_sum(own.x),
                           metal::simd_prefix_exclusive_sum(own.y));
    if (lane == 31) part[sg] = ex + own;
    metal::threadgroup_barrier(group);
    if (sg == 0) {
      const uint2 v = part[lane];
      const uint2 e2 = uint2(metal::simd_prefix_exclusive_sum(v.x),
                             metal::simd_prefix_exclusive_sum(v.y));
      part[lane] = e2;
      if (lane == NSG - 1) *total = e2 + v;
    }
    metal::threadgroup_barrier(group);
    uint2 at = *carry + part[sg] + ex;
    for (uint u = 0; u < PER; u++) {
      const uint e = t * PER + u;
      if (e >= nc) break;
      const uint c = metal::atomic_load_explicit(&count[e], relaxed);
      rs[c0 + e] = at.x;
      ts[c0 + e] = at.y;
      first[e] = at.x;
      at += uint2(c, (c + bm - 1) / bm);
    }
    metal::threadgroup_barrier(group);
    if (t == 0) *carry += *total;
    for (uint e = t; e < C; e += NT)
      metal::atomic_store_explicit(&count[e], 0u, relaxed);
    metal::threadgroup_barrier(group);
    for (uint i = t; i < rows; i += NT) {
      const uint e = w.of[i] - c0;
      if (e < nc)
        w.list[first[e] + metal::atomic_fetch_add_explicit(&count[e], 1u,
                                                           relaxed)] = i;
    }
    metal::threadgroup_barrier(group);
  }
  if (t == 0) {
    rs[buckets] = carry->x;
    ts[buckets] = carry->y;
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_device |
                             metal::mem_flags::mem_threadgroup);

  /* Tile row i's bucket: the last whose first tile row is at most i,
     which holds rows. */
  const uint used = carry->y;
  for (uint i = t; i < uint(w.bound); i += NT) {
    device uint *o = w.table + 3 * ulong(i);
    if (i >= used) {
      o[0] = NX_ROWS_UNUSED;
      continue;
    }
    uint lo = 0, hi = buckets;
    while (lo + 1 < hi) {
      const uint mid = lo + (hi - lo) / 2;
      if (ts[mid] <= i)
        lo = mid;
      else
        hi = mid;
    }
    const uint r0 = (i - ts[lo]) * bm;
    o[0] = lo;
    o[1] = rs[lo] + r0;
    o[2] = metal::min(bm, rs[lo + 1] - rs[lo] - r0);
  }
}

/* Finish */

struct finish_words {
  device const float *planes;
  device const uchar *init;
  device uchar *dst;
  ulong rows;
  ulong cols;
  ulong plane;
  ulong ranges;
  ulong first;
  long init_dtype;
  long init_z;
  long init_r;
  long init_c;
  long dst_dtype;
  long dst_z;
  long dst_r;
  long dst_c;
  long acc;
  ulong init_at;
  ulong init_phase;
  ulong dst_at;
  ulong dst_phase;
  device const uint *buckets;
  long none;
  device const ulong *koff;
  ulong kgroups;
  ulong batch;
  ulong cap;
  ulong tri;
};

NX_FIELD(finish_words, planes, NX_FINISH_PLANES);
NX_FIELD(finish_words, init, NX_FINISH_INIT);
NX_FIELD(finish_words, dst, NX_FINISH_DST);
NX_FIELD(finish_words, rows, NX_FINISH_ROWS);
NX_FIELD(finish_words, cols, NX_FINISH_COLS);
NX_FIELD(finish_words, plane, NX_FINISH_PLANE);
NX_FIELD(finish_words, ranges, NX_FINISH_RANGES);
NX_FIELD(finish_words, first, NX_FINISH_FIRST);
NX_FIELD(finish_words, init_dtype, NX_FINISH_INIT_DTYPE);
NX_FIELD(finish_words, init_z, NX_FINISH_INIT_Z);
NX_FIELD(finish_words, init_r, NX_FINISH_INIT_R);
NX_FIELD(finish_words, init_c, NX_FINISH_INIT_C);
NX_FIELD(finish_words, dst_dtype, NX_FINISH_DST_DTYPE);
NX_FIELD(finish_words, dst_z, NX_FINISH_DST_Z);
NX_FIELD(finish_words, dst_r, NX_FINISH_DST_R);
NX_FIELD(finish_words, dst_c, NX_FINISH_DST_C);
NX_FIELD(finish_words, acc, NX_FINISH_ACC);
NX_FIELD(finish_words, init_at, NX_FINISH_INIT_AT);
NX_FIELD(finish_words, init_phase, NX_FINISH_INIT_PHASE);
NX_FIELD(finish_words, dst_at, NX_FINISH_DST_AT);
NX_FIELD(finish_words, dst_phase, NX_FINISH_DST_PHASE);
NX_FIELD(finish_words, buckets, NX_FINISH_BUCKETS);
NX_FIELD(finish_words, none, NX_FINISH_NONE);
NX_FIELD(finish_words, koff, NX_FINISH_KOFF);
NX_FIELD(finish_words, kgroups, NX_FINISH_KGROUPS);
NX_FIELD(finish_words, batch, NX_FINISH_BATCH);
NX_FIELD(finish_words, cap, NX_FINISH_CAP);
NX_FIELD(finish_words, tri, NX_FINISH_TRI);
NX_SIZE(finish_words, NX_FINISH_WORDS);

/* A Contracting call's group of [count] terms, under the call's cap of
   ranges [cap]: its spans of NX_CONTRACT_SPAN terms of the real form,
   a complex64 term two of them, and its ranges, at least 8 spans each,
   as a tile's. */
static inline ulong contract_span(long acc) {
  return acc == NX_COMPLEX64 ? NX_CONTRACT_SPAN / 2 : NX_CONTRACT_SPAN;
}

static inline ulong contract_ranges(ulong spans, ulong cap) {
  return metal::max(1ul, metal::min(cap, spans / 8));
}

/* Batch element z of a Contracting call, group e = z / [batch] of
   operand batch element z mod [batch]: the first and the end of its
   entries in the prepass's list. */
static inline ulong2 contract_group(device const ulong *koff, uint z,
                                    uint batch, uint groups) {
  const uint e = z / batch, zb = z - e * batch;
  device const ulong *o = koff + ulong(zb) * (groups + 1) + e;
  return ulong2(o[0], o[1]);
}

/* Element [at] of [p], of the float dtype [dt], as float32; of a
   complex64 [p], part [part] of it. */
static inline float finish_read(device const uchar *p, long dt, long at,
                                ulong part) {
  switch (dt) {
  case NX_FLOAT16: return nx_f16_to_float(((device const ushort *)p)[at]);
  case NX_BFLOAT16: return nx_bf16_to_float(((device const ushort *)p)[at]);
  case NX_COMPLEX64: return ((device const float *)p)[2 * at + long(part)];
  default: return ((device const float *)p)[at];
  }
}

static inline void finish_write(device uchar *p, long dt, long at, ulong part,
                                float v) {
  switch (dt) {
  case NX_FLOAT16: ((device ushort *)p)[at] = nx_float_to_f16(v); break;
  case NX_BFLOAT16: ((device ushort *)p)[at] = nx_float_to_bf16(v); break;
  case NX_COMPLEX64: ((device float *)p)[2 * at + long(part)] = v; break;
  default: ((device float *)p)[at] = v; break;
  }
}

/* Column [c] of a site of [dt]: its element's column and part. */
static inline ulong finish_col(long dt, ulong c) {
  return dt == NX_COMPLEX64 ? c >> 1 : c;
}
static inline ulong finish_part(long dt, ulong c) {
  return dt == NX_COMPLEX64 ? c & 1 : 0;
}

/* Whether a finish reads init and writes dst as float32, float16 or
   bfloat16, or a complex64 call's parts, at their elements' offsets;
   else by their casts, at their positions. */
static inline bool finish_typed(long acc, long dt) {
  return acc == NX_COMPLEX64 || dt == NX_FLOAT32 || dt == NX_FLOAT16 ||
         dt == NX_BFLOAT16;
}

/* contract_finish: each output of row y = first + group y, (z, r) with
   y = z rows + r, and column c: its ranges' planes added in order from
   the first, then init as float32 holds it, rounded into dst by its
   cast. Without ranges, an output of no product, of a Rows call's row of
   no group or of a Contracting call's group without terms, it is init
   alone, or +0 without init. A Contracting call's output adds its
   group's ranges alone. The planes
   hold batch element z's sums at ((s B + z) rows + r) cols + c,
   [plane] apart from range to range. A complex64 call's columns are its
   real form's: column c is part c mod 2 of the element at column
   c / 2. */
NX_KERNEL(NX_FINISH_THREADS)
void contract_finish(constant finish_words &w [[buffer(0)]],
                     uint2 g [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]]) {
  const ulong c = ulong(g.x) * NX_FINISH_THREADS + t;
  if (c >= w.cols) return;
  const uint y = uint(w.first) + g.y;
  const uint rows = uint(w.rows);
  const uint z = y / rows, r = y - z * rows;
  /* A routine's triangle (bits 0 and 1, lower and upper; bit 2 where a
     column is half an element's): an output outside it is left. */
  if (w.tri != 0) {
    const ulong col = c >> ((w.tri >> 2) & 1);
    if (((w.tri & 1) != 0 && col > r) || ((w.tri & 2) != 0 && col < r))
      return;
  }
  bool sums = w.ranges > 0 && (w.none < 0 || w.buckets[y] != uint(w.none));
  ulong ranges = w.ranges;
  if (sums && w.kgroups != 0) {
    const ulong2 at =
        contract_group(w.koff, z, uint(w.batch), uint(w.kgroups));
    const ulong span = contract_span(w.acc);
    sums = at.y > at.x;
    ranges = contract_ranges((at.y - at.x + span - 1) / span, w.cap);
  }
  float v = 0.0f;
  if (sums) {
    device const float *p = w.planes + ulong(y) * w.cols + c;
    v = p[0];
    for (ulong s = 1; s < ranges; s++) v = v + p[s * w.plane];
  }
  if (!finish_typed(w.acc, w.dst_dtype)) {
    if (w.init_dtype >= 0) {
      const int dt = int(w.init_dtype);
      const long at = long(w.init_at) + long(z) * w.init_z +
                      long(r) * w.init_r + long(c) * w.init_c;
      const held h = cast(dt, NX_FLOAT32,
                          read(w.init, u64(at), uint(w.init_phase), dt));
      const float i = as_type<float>(uint(h.lo));
      v = sums ? v + i : i;
    }
    const int dt = int(w.dst_dtype);
    const long at = long(w.dst_at) + long(z) * w.dst_z + long(r) * w.dst_r +
                    long(c) * w.dst_c;
    write(w.dst, u64(at), uint(w.dst_phase), dt,
          cast(NX_FLOAT32, dt, held_of(as_type<uint>(v), 0)));
    return;
  }
  if (w.init_dtype >= 0) {
    const long ic = long(finish_col(w.init_dtype, c));
    const float i = finish_read(
        w.init, w.init_dtype,
        long(z) * w.init_z + long(r) * w.init_r + ic * w.init_c,
        finish_part(w.init_dtype, c));
    v = sums ? v + i : i;
  }
  const long dc = long(finish_col(w.dst_dtype, c));
  finish_write(w.dst, w.dst_dtype,
               long(z) * w.dst_z + long(r) * w.dst_r + dc * w.dst_c,
               finish_part(w.dst_dtype, c), v);
}

/* Routines' copies */

struct copy_words {
  device const float *src;
  device float *dst;
  long sz;
  long sr;
  long sc;
  long dz;
  long dr;
  long dc;
  ulong rows;
  ulong cols;
  ulong parts;
  ulong first;
  ulong start;
};

NX_FIELD(copy_words, src, NX_CCOPY_SRC);
NX_FIELD(copy_words, dst, NX_CCOPY_DST);
NX_FIELD(copy_words, sz, NX_CCOPY_SZ);
NX_FIELD(copy_words, sr, NX_CCOPY_SR);
NX_FIELD(copy_words, sc, NX_CCOPY_SC);
NX_FIELD(copy_words, dz, NX_CCOPY_DZ);
NX_FIELD(copy_words, dr, NX_CCOPY_DR);
NX_FIELD(copy_words, dc, NX_CCOPY_DC);
NX_FIELD(copy_words, rows, NX_CCOPY_ROWS);
NX_FIELD(copy_words, cols, NX_CCOPY_COLS);
NX_FIELD(copy_words, parts, NX_CCOPY_PARTS);
NX_FIELD(copy_words, first, NX_CCOPY_FIRST);
NX_FIELD(copy_words, start, NX_CCOPY_START);
NX_SIZE(copy_words, NX_CCOPY_WORDS);

/* contract_copy: element (z, r, j) of src into dst, row y = first + g.y
   = z rows + r, each of [parts] float32 parts read flushed: a subnormal
   part is a zero of its sign. Strides count elements. */
NX_KERNEL(NX_CCOPY_THREADS)
void contract_copy(constant copy_words &w [[buffer(0)]],
                   uint2 g [[threadgroup_position_in_grid]],
                   uint t [[thread_index_in_threadgroup]]) {
  const ulong j = w.start + ulong(g.x) * NX_CCOPY_THREADS + t;
  if (j >= w.cols) return;
  const uint y = uint(w.first) + g.y;
  const uint rows = uint(w.rows);
  const uint z = y / rows, r = y - z * rows;
  const long parts = long(w.parts);
  device const float *s =
      w.src + parts * (long(z) * w.sz + long(r) * w.sr + long(j) * w.sc);
  device float *d =
      w.dst + parts * (long(z) * w.dz + long(r) * w.dr + long(j) * w.dc);
  for (long p = 0; p < parts; p++) d[p] = nx_flush_f32(s[p]);
}

/* Scales */

struct scale_words {
  device float *dst;
  device const uchar *src;
  device const uchar *scale;
  constant long *axes;
  ulong first;
  ulong count;
  ulong rank;
  ulong codes;
  ulong src_bit;
  ulong scale_bit;
  long held[NX_SCALE_AXIS_WORDS * NX_SCALE_HOLDS];
};

NX_FIELD(scale_words, dst, NX_SCALE_DST);
NX_FIELD(scale_words, src, NX_SCALE_SRC);
NX_FIELD(scale_words, scale, NX_SCALE_SCALE);
NX_FIELD(scale_words, axes, NX_SCALE_AXES);
NX_FIELD(scale_words, first, NX_SCALE_FIRST);
NX_FIELD(scale_words, count, NX_SCALE_COUNT);
NX_FIELD(scale_words, rank, NX_SCALE_RANK);
NX_FIELD(scale_words, codes, NX_SCALE_CODES);
NX_FIELD(scale_words, src_bit, NX_SCALE_SRC_BIT);
NX_FIELD(scale_words, scale_bit, NX_SCALE_SCALE_BIT);
NX_FIELD(scale_words, held, NX_SCALE_HELD);
NX_SIZE(scale_words, NX_SCALE_WORDS);

/* The element at bit [base] + [bits] [at] of [p], of dtype [dt]. */
static inline held scale_read(device const uchar *p, ulong base, long at,
                              int dt) {
  const long b = long(bits_of(dt));
  const u64 bit = u64(long(base) + b * at);
  return read(p, bit / u64(b), uint(bit % u64(b)), dt);
}

NX_KERNEL(NX_SCALE_THREADS)
void contract_scale(constant scale_words &w [[buffer(0)]],
                    uint g [[threadgroup_position_in_grid]],
                    uint t [[thread_index_in_threadgroup]]) {
  const ulong k = w.first + nx_thread(g, t, NX_SCALE_THREADS);
  if (k >= w.count) return;

  /* The element's and its scale's positions of index [k], as copy finds
     a source's. */
  constant long *axes = w.rank <= NX_SCALE_HOLDS ? w.held : w.axes;
  long at = 0, sat = 0;
  ulong rest = k;
  for (int j = int(w.rank) - 1; j >= 1; j--) {
    constant long *axis = axes + NX_SCALE_AXIS_WORDS * j;
    const ulong e = ulong(axis[NX_SCALE_AXIS_EXTENT]);
    const long i = long(rest % e);
    at += i * axis[NX_SCALE_AXIS_SRC];
    sat += i * axis[NX_SCALE_AXIS_SCALE];
    rest /= e;
  }
  at += long(rest) * axes[NX_SCALE_AXIS_SRC];
  sat += long(rest) * axes[NX_SCALE_AXIS_SCALE];

  const int from = int(w.codes & 255), by = int((w.codes >> 8) & 255);
  const held h = cast(from, NX_FLOAT32, scale_read(w.src, w.src_bit, at, from));
  const float v = as_type<float>(uint(h.lo));
  const held s = scale_read(w.scale, w.scale_bit, sat, by);
  if (w.codes & NX_SCALE_EXPONENT) {
    w.dst[k] = contract_exponent(v, uint(s.lo) & 255);
    return;
  }
  w.dst[k] = v * as_type<float>(uint(cast(by, NX_FLOAT32, s).lo));
}

/* Bands */

struct band_words {
  device uchar *dst;
  device const uchar *src;
  ulong src_bit;
  long stride;
  long group;
  ulong codes;
  ulong k;
  ulong quot;
  ulong rem;
  ulong ds;
  ulong length;
  ulong chunk;
};

NX_FIELD(band_words, dst, NX_BAND_DST);
NX_FIELD(band_words, src, NX_BAND_SRC);
NX_FIELD(band_words, src_bit, NX_BAND_SRC_BIT);
NX_FIELD(band_words, stride, NX_BAND_STRIDE);
NX_FIELD(band_words, group, NX_BAND_GROUP);
NX_FIELD(band_words, codes, NX_BAND_CODES);
NX_FIELD(band_words, k, NX_BAND_K);
NX_FIELD(band_words, quot, NX_BAND_QUOT);
NX_FIELD(band_words, rem, NX_BAND_REM);
NX_FIELD(band_words, ds, NX_BAND_DS);
NX_FIELD(band_words, length, NX_BAND_LENGTH);
NX_FIELD(band_words, chunk, NX_BAND_CHUNK);
NX_SIZE(band_words, NX_BAND_WORDS);

/* Range s's first span, ⌊s r / S⌋ = s q + ⌊s e / S⌋ for r = q S + e:
   a thread divides only s e, under 64², by S's divisor word. */
static inline ulong band_span(constant band_words &w, uint s) {
  return ulong(s) * w.quot + contract_div(s * uint(w.rem), w.ds);
}

NX_KERNEL(NX_BAND_THREADS)
void contract_band(constant band_words &w [[buffer(0)]],
                   uint3 g [[threadgroup_position_in_grid]],
                   uint3 grid [[threadgroups_per_grid]],
                   uint t [[thread_index_in_threadgroup]]) {
  const ulong j = ulong(g.x) * NX_BAND_THREADS + t;
  if (j >= w.length) return;
  const uint s = g.y;
  const ulong k1 =
      metal::min(w.k, ulong(NX_CONTRACT_SPAN) * band_span(w, s + 1));
  const ulong term =
      ulong(NX_CONTRACT_SPAN) * band_span(w, s) + w.chunk * w.length + j;
  if (term >= k1) return;
  const int from = int(w.codes & 255), to = int((w.codes >> 8) & 255);
  const long at = long(g.z) * w.group + long(term) * w.stride;
  const held h = cast(from, to, scale_read(w.src, w.src_bit, at, from));
  write(w.dst, (u64(g.z) * grid.y + s) * w.length + j, 0, to, h);
}

/* Integers */

/* The held value [h] of [dt] as the accumulator [acc] holds it, as 64
   bits: a float's or a complex64's by its cast into [acc], an integer's
   value or a boolean's 0 or 1. Products and sums of these wrap modulo
   2^64, whose low bits [acc]'s cast then keeps, so an integer need not
   enter [acc] before its product: the cast's remainder is the same. */
static inline u64 contract_value(int dt, int acc, held h) {
  if (float_dt(dt) || dt == NX_COMPLEX64) {
    h = cast(dt, acc, h);
    dt = acc;
  }
  if (bool_dt(dt)) return h.lo != 0;
  if (signed_dt(dt) && bits_of(dt) <= 32) return u64(long(int(uint(h.lo))));
  return h.lo;
}

struct simt_words {
  device const uchar *a;
  device const uchar *b;
  device ulong *y;
  ulong a_at;
  ulong a_phase;
  long a_dtype;
  long a_z;
  long a_r;
  long a_k;
  ulong b_at;
  ulong b_phase;
  long b_dtype;
  long b_z;
  long b_k;
  long b_c;
  long acc;
  ulong m;
  ulong n;
  ulong k;
  ulong plane;
  ulong first;
  ulong last;
  ulong ranges;
  ulong spans;
  ulong length;
  ulong chunk;
  device const uint *buckets;
  ulong groups;
  long b_g;
  ulong segmented;
  device const ulong *koff;
  device const uint *klist;
  ulong kgroups;
  ulong batch;
  ulong cap;
};

NX_FIELD(simt_words, a, NX_SIMT_A);
NX_FIELD(simt_words, b, NX_SIMT_B);
NX_FIELD(simt_words, y, NX_SIMT_Y);
NX_FIELD(simt_words, a_at, NX_SIMT_A_AT);
NX_FIELD(simt_words, a_phase, NX_SIMT_A_PHASE);
NX_FIELD(simt_words, a_dtype, NX_SIMT_A_DTYPE);
NX_FIELD(simt_words, a_z, NX_SIMT_A_Z);
NX_FIELD(simt_words, a_r, NX_SIMT_A_R);
NX_FIELD(simt_words, a_k, NX_SIMT_A_K);
NX_FIELD(simt_words, b_at, NX_SIMT_B_AT);
NX_FIELD(simt_words, b_phase, NX_SIMT_B_PHASE);
NX_FIELD(simt_words, b_dtype, NX_SIMT_B_DTYPE);
NX_FIELD(simt_words, b_z, NX_SIMT_B_Z);
NX_FIELD(simt_words, b_k, NX_SIMT_B_K);
NX_FIELD(simt_words, b_c, NX_SIMT_B_C);
NX_FIELD(simt_words, acc, NX_SIMT_ACC);
NX_FIELD(simt_words, m, NX_SIMT_M);
NX_FIELD(simt_words, n, NX_SIMT_N);
NX_FIELD(simt_words, k, NX_SIMT_K);
NX_FIELD(simt_words, plane, NX_SIMT_PLANE);
NX_FIELD(simt_words, first, NX_SIMT_FIRST);
NX_FIELD(simt_words, last, NX_SIMT_LAST);
NX_FIELD(simt_words, ranges, NX_SIMT_RANGES);
NX_FIELD(simt_words, spans, NX_SIMT_SPANS);
NX_FIELD(simt_words, length, NX_SIMT_LENGTH);
NX_FIELD(simt_words, chunk, NX_SIMT_CHUNK);
NX_FIELD(simt_words, buckets, NX_SIMT_BUCKETS);
NX_FIELD(simt_words, groups, NX_SIMT_GROUPS);
NX_FIELD(simt_words, b_g, NX_SIMT_B_G);
NX_FIELD(simt_words, segmented, NX_SIMT_SEGMENTED);
NX_FIELD(simt_words, koff, NX_SIMT_KOFF);
NX_FIELD(simt_words, klist, NX_SIMT_KLIST);
NX_FIELD(simt_words, kgroups, NX_SIMT_KGROUPS);
NX_FIELD(simt_words, batch, NX_SIMT_BATCH);
NX_FIELD(simt_words, cap, NX_SIMT_CAP);
NX_SIZE(simt_words, NX_SIMT_WORDS);

/* The held value [h] of [dt] as float32 holds it, by its cast. */
static inline float contract_f32(int dt, held h) {
  return as_type<float>(uint(cast(dt, NX_FLOAT32, h).lo));
}

/* An output's sum over terms [c0, c1) of its range, from [sum]: term t's
   elements at a + q a_k and b + q b_k, q the entry t of [list], or t
   where [list] is null. An integer [acc]'s products and sums wrap in 64
   bits; a float32 one's are fused multiply-adds of the casts; a
   complex64 one's sum part [part] of the real form's column. */
struct simt_sum {
  device const uchar *a, *b;
  int da, db, acc;
  uint pha, phb, part;
  long pa, pb, ak, bk;
  device const uint *list;

  inline long term(ulong t) const { return list ? long(list[t]) : long(t); }

  u64 integer(ulong c0, ulong c1, u64 sum) const {
    for (ulong t = c0; t < c1; t++) {
      const long q = term(t);
      const u64 va = contract_value(da, acc, read(a, u64(pa + q * ak), pha, da));
      const u64 vb = contract_value(db, acc, read(b, u64(pb + q * bk), phb, db));
      sum += va * vb;
    }
    return sum;
  }

  float real(ulong c0, ulong c1, float sum) const {
    for (ulong t = c0; t < c1; t++) {
      const long q = term(t);
      const float va = contract_f32(da, read(a, u64(pa + q * ak), pha, da));
      const float vb = contract_f32(db, read(b, u64(pb + q * bk), phb, db));
      sum = metal::fma(va, vb, sum);
    }
    return sum;
  }

  float complex(ulong c0, ulong c1, float sum) const {
    for (ulong t = c0; t < c1; t++) {
      const long q = term(t);
      const u64 x = read(a, u64(pa + q * ak), pha, da).lo;
      const u64 y = read(b, u64(pb + q * bk), phb, db).lo;
      const float ar = as_type<float>(uint(x)), ai = as_type<float>(uint(x >> 32));
      const float br = as_type<float>(uint(y)), bi = as_type<float>(uint(y >> 32));
      if (part == 0) {
        sum = metal::fma(ar, br, sum);
        sum = metal::fma(ai, -bi, sum);
      } else {
        sum = metal::fma(ar, bi, sum);
        sum = metal::fma(ai, br, sum);
      }
    }
    return sum;
  }
};

/* contract_simt: an output's sum over its range's terms, or its chunk
   of them, each element read at its own dtype and entering the
   accumulator by its cast. Its addresses are computed once, so the
   terms' loop reads without a guard. A Rows call's row multiplies its
   group's b, or has no terms in the bucket of no group. A Contracting
   call's output takes its group's terms through the prepass's list, and
   its group's ranges. */
NX_KERNEL(NX_SIMT_THREADS)
void contract_simt(constant simt_words &w [[buffer(0)]],
                   uint3 g [[threadgroup_position_in_grid]],
                   uint t [[thread_index_in_threadgroup]]) {
  /* The outputs number below 2^32, so 32-bit divisions find (z, r, c). */
  const uint o = uint(w.first) + g.x * NX_SIMT_THREADS + t;
  if (o >= uint(w.last)) return;
  const ulong s = g.z;
  const uint n = uint(w.n), mn = uint(w.m) * n;
  const uint z = o / mn, q = o - z * mn;
  const uint r = q / n, c = q - r * n;
  const int acc = int(w.acc);
  const bool whole = !float_dt(acc) && acc != NX_COMPLEX64;
  device ulong *y = w.y + s * w.plane + o;
  device float *yf = (device float *)w.y + s * w.plane + o;

  /* The output's terms: the call's, or its group's entries of the list
     from [first], with their ranges. */
  ulong k = w.k, spans = w.spans, ranges = w.ranges, first = 0;
  uint zb = z;
  if (w.kgroups != 0) {
    const ulong2 at =
        contract_group(w.koff, z, uint(w.batch), uint(w.kgroups));
    const uint e = z / uint(w.batch);
    zb = z - e * uint(w.batch);
    first = at.x;
    k = at.y - at.x;
    spans = (k + contract_span(acc) - 1) / contract_span(acc);
    ranges = contract_ranges(spans, w.cap);
    if (s >= ranges) {
      if (whole && w.chunk == 0) *y = 0;
      return;
    }
  }
  const ulong span = contract_span(acc);
  const ulong k0 = span * (s * spans / ranges);
  const ulong k1 = metal::min(k, span * ((s + 1) * spans / ranges));
  ulong c0 = k0, c1 = k1;
  if (w.length > 0) {
    c0 = k0 + w.chunk * w.length;
    if (c0 >= k1) {
      if (whole && w.chunk == 0) *y = 0;
      return;
    }
    c1 = metal::min(k1, c0 + w.length);
  }

  const uint col = acc == NX_COMPLEX64 ? c >> 1 : c;
  simt_sum f;
  f.a = w.a;
  f.b = w.b;
  f.da = int(w.a_dtype);
  f.db = int(w.b_dtype);
  f.acc = acc;
  f.pha = uint(w.a_phase);
  f.phb = uint(w.b_phase);
  f.part = c & 1;
  f.pa = long(w.a_at) + long(zb) * w.a_z + long(r) * w.a_r;
  f.pb = long(w.b_at) + long(zb) * w.b_z + long(col) * w.b_c;
  f.ak = w.a_k;
  f.bk = w.b_k;
  f.list = w.kgroups != 0 ? w.klist + first : nullptr;
  if (w.segmented != 0) {
    const uint gs = uint(w.groups);
    const uint j = w.buckets[z * uint(w.m) + r] - z * gs;
    if (j < gs)
      f.pb += long(j) * w.b_g;
    else
      c1 = c0;
  }
  if (whole)
    *y = f.integer(c0, c1, w.chunk != 0 ? *y : 0);
  else if (acc == NX_COMPLEX64)
    *yf = f.complex(c0, c1, w.chunk != 0 ? *yf : 0.0f);
  else
    *yf = f.real(c0, c1, w.chunk != 0 ? *yf : 0.0f);
}

/* contract_finish_int: as contract_finish, of 64-bit sums: an output's
   ranges added in order, then init as the accumulator holds it, the
   total wrapped into the accumulator and cast into dst. */
NX_KERNEL(NX_FINISH_THREADS)
void contract_finish_int(constant finish_words &w [[buffer(0)]],
                         uint2 g [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
  const ulong c = ulong(g.x) * NX_FINISH_THREADS + t;
  if (c >= w.cols) return;
  const uint y = uint(w.first) + g.y;
  const uint rows = uint(w.rows);
  const uint z = y / rows, r = y - z * rows;
  const int acc = int(w.acc);
  u64 v = 0;
  if (w.ranges > 0) {
    device const ulong *p =
        (device const ulong *)w.planes + ulong(y) * w.cols + c;
    for (ulong s = 0; s < w.ranges; s++) v += p[s * w.plane];
  }
  if (w.init_dtype >= 0) {
    const int dt = int(w.init_dtype);
    const long at = long(w.init_at) + long(z) * w.init_z + long(r) * w.init_r +
                    long(c) * w.init_c;
    const uint phase = uint(w.init_phase);
    v += contract_value(dt, acc, read(w.init, u64(at), phase, dt));
  }
  const int dt = int(w.dst_dtype);
  const long at = long(w.dst_at) + long(z) * w.dst_z + long(r) * w.dst_r +
                  long(c) * w.dst_c;
  write(w.dst, u64(at), uint(w.dst_phase), dt,
        cast(acc, dt, held_of(of_raw(acc, v), 0)));
}

/* Compaction

   contract_compact, a Contracting call's prepass: each part of a batch
   element's terms counts its groups' terms with device atomics; one
   group a batch element scans its counts, its groups in order and each
   group's parts in order, chunk after chunk by SIMD-group prefix sums,
   into each part's first entry; then a SIMD-group a part lists its terms 32
   at a time, a lane's entry its group's next past the lanes
   before it that share its group, so that a group's terms keep their
   order: the order its products are numbered in. */

struct compact_words {
  device const long *ids;
  device uint *counts;
  device ulong *offsets;
  device uint *list;
  long ids_z;
  long ids_r;
  ulong batch;
  ulong terms;
  ulong inner;
  ulong extent;
  ulong groups;
  ulong parts;
  ulong phase;
};

NX_FIELD(compact_words, ids, NX_COMPACT_IDS);
NX_FIELD(compact_words, counts, NX_COMPACT_COUNTS);
NX_FIELD(compact_words, offsets, NX_COMPACT_OFFSETS);
NX_FIELD(compact_words, list, NX_COMPACT_LIST);
NX_FIELD(compact_words, ids_z, NX_COMPACT_IDS_Z);
NX_FIELD(compact_words, ids_r, NX_COMPACT_IDS_R);
NX_FIELD(compact_words, batch, NX_COMPACT_BATCH);
NX_FIELD(compact_words, terms, NX_COMPACT_TERMS);
NX_FIELD(compact_words, inner, NX_COMPACT_INNER);
NX_FIELD(compact_words, extent, NX_COMPACT_EXTENT);
NX_FIELD(compact_words, groups, NX_COMPACT_GROUPS);
NX_FIELD(compact_words, parts, NX_COMPACT_PARTS);
NX_FIELD(compact_words, phase, NX_COMPACT_PHASE);
NX_SIZE(compact_words, NX_COMPACT_WORDS);

/* Term [q] of batch element [z]'s group, or [groups] for none. Its id is
   tested signed and whole: an id past 2^32 or below 0 never reaches a
   group by its low bits. The terms number below 2^32. */
static inline uint compact_group(constant compact_words &w, uint z, uint q) {
  const uint i = (q / uint(w.inner)) % uint(w.extent);
  const long id = w.ids[long(z) * w.ids_z + long(i) * w.ids_r];
  return id >= 0 && id < long(w.groups) ? uint(id) : uint(w.groups);
}

NX_KERNEL(NX_COMPACT_THREADS)
void contract_compact(constant compact_words &w [[buffer(0)]],
                      threadgroup uchar *shared [[threadgroup(0)]],
                      uint g [[threadgroup_position_in_grid]],
                      uint ng [[threadgroups_per_grid]],
                      uint t [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NT = NX_COMPACT_THREADS;
  constexpr auto relaxed = metal::memory_order_relaxed;
  const uint gs = uint(w.groups), parts = uint(w.parts);
  const ulong k = w.terms, part = (k + parts - 1) / parts;
  const ulong stride = ulong(ng) * NT, at = ulong(g) * NT + t;
  switch (w.phase) {
  case 0:
    for (ulong i = at; i < w.batch * parts * gs; i += stride) w.counts[i] = 0;
    return;
  case 1: {
    const uint z = g / parts, p = g - z * parts;
    device metal::atomic_uint *count =
        (device metal::atomic_uint *)w.counts + ulong(z) * gs * parts + p;
    const ulong end = metal::min(k, (p + 1) * part);
    for (ulong q = p * part + t; q < end; q += NT) {
      const uint e = compact_group(w, z, uint(q));
      if (e < gs)
        metal::atomic_fetch_add_explicit(&count[ulong(e) * parts], 1u,
                                         relaxed);
    }
    return;
  }
  case 2: {
    /* Batch element g's counts, below 2^32 together, in chunks: thread
       t takes PER of them in order, the SIMD-groups' exclusive sums
       and their sums' give its first, past the chunks' before. */
    constexpr uint C = NX_COMPACT_CHUNK, PER = C / NT, NSG = NT / 32;
    constexpr auto group = metal::mem_flags::mem_threadgroup;
    threadgroup uint *sums = (threadgroup uint *)shared;
    threadgroup uint *carry = sums + NSG, *total = sums + NSG + 1;
    const ulong n = ulong(gs) * parts, base = ulong(g) * k;
    device uint *c = w.counts + ulong(g) * n;
    device ulong *o = w.offsets + ulong(g) * (gs + 1);
    if (t == 0) *carry = 0;
    for (ulong c0 = 0; c0 < n; c0 += C) {
      uint v[PER], own = 0;
      for (uint u = 0; u < PER; u++) {
        const ulong i = c0 + t * PER + u;
        v[u] = i < n ? c[i] : 0;
        own += v[u];
      }
      const uint ex = metal::simd_prefix_exclusive_sum(own);
      if (lane == 31) sums[sg] = ex + own;
      metal::threadgroup_barrier(group);
      if (sg == 0) {
        const uint x = lane < NSG ? sums[lane] : 0;
        const uint before = metal::simd_prefix_exclusive_sum(x);
        if (lane < NSG) sums[lane] = before;
        if (lane == NSG - 1) *total = before + x;
      }
      metal::threadgroup_barrier(group);
      uint first = *carry + sums[sg] + ex;
      for (uint u = 0; u < PER; u++) {
        const ulong i = c0 + t * PER + u;
        if (i >= n) break;
        c[i] = first;
        if (i % parts == 0) o[i / parts] = base + first;
        first += v[u];
      }
      metal::threadgroup_barrier(group);
      if (t == 0) *carry += *total;
      metal::threadgroup_barrier(group);
    }
    if (t == 0) o[gs] = base + *carry;
    return;
  }
  default: {
    const ulong zp = ulong(g) * (NT / 32) + sg;
    if (zp >= w.batch * parts) return;
    const uint z = uint(zp / parts), p = uint(zp - ulong(z) * parts);
    device uint *next = w.counts + ulong(z) * gs * parts + p;
    const ulong end = metal::min(k, (p + 1) * part);
    const ulong base = ulong(z) * k;
    for (ulong q0 = p * part; q0 < end; q0 += 32) {
      const ulong q = q0 + lane;
      const uint e = q < end ? compact_group(w, z, uint(q)) : gs;
      /* The lanes before this one in its group, and the group's lanes. */
      uint before = 0, same = 0;
      for (ushort l = 0; l < 32; l++) {
        const uint f = metal::simd_shuffle(e, l);
        same += f == e;
        before += f == e && l < lane;
      }
      device uint *cur = next + ulong(e) * parts;
      if (e < gs) w.list[base + *cur + before] = uint(q);
      metal::simdgroup_barrier(metal::mem_flags::mem_device);
      if (e < gs && before == 0) *cur += same;
      metal::simdgroup_barrier(metal::mem_flags::mem_device);
    }
  }
  }
}

/* Vector kernels

   A call of few rows over a weight of a key: each threadgroup's four
   SIMD-groups hold columns, 32 / S each of c columns a lane, and a lane
   holds R rows of a pass. An output's r = ⌈k / 32⌉ spans of 32 terms go
   to its S lanes, span j to lane j mod S; a lane loads each of its spans
   whole, then for each 8 terms decodes its columns' weights once and,
   for each row, adds the 8 products by fused multiply-adds in increasing
   k, from +0. The lanes' sums combine by halving, lane l adding lane
   l + h's for h = S / 2, ..., 1, and the first lane stores them:

     span loads --> 8 terms decoded --> rows' fmas --> halving --> y

   A span lies within one block of a blocked weight (Q8_0's quants), and
   one that ends past k is loaded element by element, its terms past k
   adding nothing. The host picks one load form for every lane's whole
   spans, by the alignment of every span's start: 16-byte words, or
   2-byte ones (Q8_0's 34-byte blocks). Where neither holds, and for the
   span past k, each lane loads by its own addresses, words of 16, 4 or 2
   bytes, else bytes, so every layout the host passes reads in place. */

struct mv_words {
  device const uchar *a;
  device const uchar *b;
  device uchar *y;
  device const uchar *bs;
  device const long *ids;
  device uint *of;
  long lda;
  long sa;
  long ldb;
  long sb;
  long sj;
  ulong blocks;
  ulong bsz;
  ulong bsn;
  ulong bsk;
  long ldy;
  long cy;
  long sy;
  ulong m;
  ulong n;
  ulong k;
  ulong lanes;
  ulong groups_n;
  ulong first;
  long ids_z;
  long ids_r;
  ulong inner;
  ulong extent;
  ulong groups;
  ulong rows;
  ulong form;
};

NX_FIELD(mv_words, a, NX_MV_A);
NX_FIELD(mv_words, b, NX_MV_B);
NX_FIELD(mv_words, y, NX_MV_Y);
NX_FIELD(mv_words, bs, NX_MV_BS);
NX_FIELD(mv_words, ids, NX_MV_IDS);
NX_FIELD(mv_words, of, NX_MV_OF);
NX_FIELD(mv_words, lda, NX_MV_LDA);
NX_FIELD(mv_words, sa, NX_MV_SA);
NX_FIELD(mv_words, ldb, NX_MV_LDB);
NX_FIELD(mv_words, sb, NX_MV_SB);
NX_FIELD(mv_words, sj, NX_MV_SJ);
NX_FIELD(mv_words, blocks, NX_MV_BLOCKS);
NX_FIELD(mv_words, bsz, NX_MV_BSZ);
NX_FIELD(mv_words, bsn, NX_MV_BSN);
NX_FIELD(mv_words, bsk, NX_MV_BSK);
NX_FIELD(mv_words, ldy, NX_MV_LDY);
NX_FIELD(mv_words, cy, NX_MV_CY);
NX_FIELD(mv_words, sy, NX_MV_SY);
NX_FIELD(mv_words, m, NX_MV_M);
NX_FIELD(mv_words, n, NX_MV_N);
NX_FIELD(mv_words, k, NX_MV_K);
NX_FIELD(mv_words, lanes, NX_MV_LANES);
NX_FIELD(mv_words, groups_n, NX_MV_GROUPS_N);
NX_FIELD(mv_words, first, NX_MV_FIRST);
NX_FIELD(mv_words, ids_z, NX_MV_IDS_Z);
NX_FIELD(mv_words, ids_r, NX_MV_IDS_R);
NX_FIELD(mv_words, inner, NX_MV_INNER);
NX_FIELD(mv_words, extent, NX_MV_EXTENT);
NX_FIELD(mv_words, groups, NX_MV_GROUPS);
NX_FIELD(mv_words, rows, NX_MV_ROWS);
NX_FIELD(mv_words, form, NX_MV_FORM);
NX_SIZE(mv_words, NX_MV_WORDS);

/* The dtype of sums stored as [YT], float32 or the other operand's
   float16 or bfloat16, which contract_finish's finish_write rounds them
   into. A store of a dtype a word chose took MXFP4's call from 70 to
   77 us: the dtype is the kernel's. */
template <typename YT>
constexpr long mv_dtype() {
  return metal::is_same_v<YT, half>     ? NX_FLOAT16
         : metal::is_same_v<YT, bfloat> ? NX_BFLOAT16
                                        : NX_FLOAT32;
}

/* The wider operand's keys. */
#define NX_MV_F32 0
#define NX_MV_F16 1
#define NX_MV_BF16 2
#define NX_MV_E2M1 3
#define NX_MV_I8 4

/* A span's 32-bit words of a column at [KEY], and twice its element's
   bytes. */
constexpr int mv_span_words(int key) {
  return key == NX_MV_F32 ? 32 : key == NX_MV_E2M1 ? 4 : key == NX_MV_I8 ? 8 : 16;
}
constexpr int mv_half_bytes(int key) {
  return key == NX_MV_F32 ? 8 : key == NX_MV_E2M1 ? 1 : key == NX_MV_I8 ? 2 : 4;
}

/* [e] elements of [KEY]'s weight as bytes, [e] even at e2m1, where the
   shift is exact (the host places e2m1's weight at even elements only,
   and spans start at multiples of 32). Apple GPUs divide 64-bit integers
   in software: a signed division by 2 a column took half of e2m1's
   call. */
template <int KEY>
static inline long mv_bytes(long e) {
  constexpr long hb = mv_half_bytes(KEY);
  return hb % 2 == 0 ? e * (hb / 2) : (e * hb) >> 1;
}

/* float4_e2m1fn's values by code: the sign bit, then 0, 0.5, 1, 1.5,
   2, 3, 4 and 6. */
constant float mv_e2m1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                              4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                              -2.0f, -3.0f, -4.0f, -6.0f};

/* [N] words from [p], by the widest loads its address allows: 16
   bytes only where [N] is a multiple of 4. */
template <int N>
NX_INLINE_TILE void mv_load(device const uchar *p, thread uint (&wd)[N]) {
  const ulong at = ulong(p);
  if (N % 4 == 0 && (at & 15) == 0) {
    NX_UNROLL for (int i = 0; i < N / 4; i++) {
      const uint4 v = ((device const uint4 *)p)[i];
      wd[4 * i] = v.x;
      wd[4 * i + 1] = v.y;
      wd[4 * i + 2] = v.z;
      wd[4 * i + 3] = v.w;
    }
  } else if ((at & 3) == 0) {
    NX_UNROLL for (int i = 0; i < N; i++) wd[i] = ((device const uint *)p)[i];
  } else if ((at & 1) == 0) {
    device const ushort *h = (device const ushort *)p;
    NX_UNROLL for (int i = 0; i < N; i++)
      wd[i] = uint(h[2 * i]) | (uint(h[2 * i + 1]) << 16);
  } else {
    NX_UNROLL for (int i = 0; i < N; i++)
      wd[i] = uint(p[4 * i]) | (uint(p[4 * i + 1]) << 8) |
              (uint(p[4 * i + 2]) << 16) | (uint(p[4 * i + 3]) << 24);
  }
}

/* The first [bytes] bytes from [p] as [N] words, the rest 0. */
template <int N>
NX_INLINE_TILE void mv_load_part(device const uchar *p, uint bytes,
                                 thread uint (&wd)[N]) {
  NX_UNROLL for (int i = 0; i < N; i++) {
    uint v = 0;
    for (uint j = 0; j < 4; j++)
      if (4 * uint(i) + j < bytes) v |= uint(p[4 * i + j]) << (8 * j);
    wd[i] = v;
  }
}

/* Term [t] of a span's words [wd] at [KEY], as float32, unscaled. */
template <int KEY, int N>
static inline float mv_term(thread const uint (&wd)[N], int t) {
  if (KEY == NX_MV_F32) return as_type<float>(wd[t]);
  if (KEY == NX_MV_F16)
    return float(as_type<half>(ushort(wd[t >> 1] >> ((t & 1) * 16))));
  if (KEY == NX_MV_BF16)
    return as_type<float>(((wd[t >> 1] >> ((t & 1) * 16)) & 0xFFFFu) << 16);
  if (KEY == NX_MV_I8) return float(char(uchar(wd[t >> 2] >> ((t & 3) * 8))));
  return mv_e2m1[(wd[t >> 3] >> ((t & 7) * 4)) & 15];
}

/* Whether code [e]'s 2^(e - 113) is a normal float32, so a half's
   value times it is exact: codes 2 to 240. */
NX_INLINE_TILE bool mv_e2m1_exact(uint e) { return e >= 2 && e <= 240; }

/* 2^(e - 113) as a float32, for [e] within mv_e2m1_exact. */
NX_INLINE_TILE float mv_e2m1_power(uint e) { return as_type<float>((e + 14) << 23); }

/* The halves of [word]'s 8 e2m1 codes, 2^-14 times their values. */
NX_INLINE_TILE void mv_e2m1_halves(uint word, thread float (&v)[8]) {
  NX_UNROLL for (int i = 0; i < 4; i++) {
    const uint q = (word >> (4 * i)) & 0x000F000Fu;
    const float2 h =
        float2(as_type<half2>(((q & 0x00080008u) << 12) | ((q & 0x00070007u) << 9)));
    v[i] = h.x;
    v[i + 4] = h.y;
  }
}

/* 8 e2m1 codes of [word], each times 2^(e - 127), as contract_exponent
   computes it: within codes 2 to 240 a code's half, 2^-14 times its
   value (subnormal halves included), times 2^(e - 113), a product that
   is exact and normal or zero. Codes i and i + 4 lie 16 bits apart, so
   one half2 of their bits decodes both: a code at a time took MXFP4's
   bfloat16 call 68 us, against 54. */
NX_INLINE_TILE void mv_e2m1_word(uint word, uint e, thread float (&v)[8]) {
  if (mv_e2m1_exact(e)) {
    const float f = mv_e2m1_power(e);
    mv_e2m1_halves(word, v);
    NX_UNROLL for (int i = 0; i < 8; i++) v[i] = v[i] * f;
  } else {
    NX_UNROLL for (int i = 0; i < 8; i++)
      v[i] = contract_exponent(mv_e2m1[(word >> (4 * i)) & 15], e);
  }
}

/* 8 terms from [t] of a span's words, each times its scale [s] (a code
   at e2m1, a Value's float32 cast at int8) as contract_scale computes
   it: e2m1's by mv_e2m1_word, [t] a multiple of 8; a Value's one
   rounded product. */
template <int KEY, int N>
NX_INLINE_TILE void mv_decode(thread const uint (&wd)[N], int t, float s,
                              thread float (&v)[8]) {
  if (KEY == NX_MV_E2M1) {
    mv_e2m1_word(wd[t >> 3], uint(s), v);
  } else if (KEY == NX_MV_I8) {
    NX_UNROLL for (int i = 0; i < 8; i++) v[i] = mv_term<KEY>(wd, t + i) * s;
  } else {
    NX_UNROLL for (int i = 0; i < 8; i++) v[i] = mv_term<KEY>(wd, t + i);
  }
}

/* A scale's element at [p] as float32: a code as its value, a Value of
   [SD] by its cast. */
template <int KEY, int SD>
static inline float mv_scale(device const uchar *p) {
  if (KEY == NX_MV_E2M1) return float(*p);
  if (SD == NX_FLOAT16) return float(*(device const half *)p);
  return *(device const float *)p;
}

template <int KEY, int SD>
constexpr int mv_scale_bytes() {
  return KEY == NX_MV_E2M1 ? 1 : SD == NX_FLOAT16 ? 2 : 4;
}

/* 8 terms of a row at [x], 16-byte aligned, as float32. */
NX_INLINE_TILE void mv_x8(device const float *x, thread float (&v)[8]) {
  const float4 x0 = ((device const float4 *)x)[0];
  const float4 x1 = ((device const float4 *)x)[1];
  v[0] = x0.x; v[1] = x0.y; v[2] = x0.z; v[3] = x0.w;
  v[4] = x1.x; v[5] = x1.y; v[6] = x1.z; v[7] = x1.w;
}
template <typename T>
NX_INLINE_TILE void mv_x8(device const T *x, thread float (&v)[8]) {
  const uint4 h = ((device const uint4 *)x)[0];
  const uint hw[4] = {h.x, h.y, h.z, h.w};
  NX_UNROLL for (int t = 0; t < 8; t++)
    v[t] = float(as_type<T>(ushort(hw[t >> 1] >> ((t & 1) * 16))));
}

/* A lane's whole spans from [sp] on, below [whole], each S apart, as
   the design probed them: each column's weight and scales and each
   row's terms from a pointer set once, a span's offset from it 32-bit
   but for a block's stride, its words loaded together before its
   terms, by 16-byte words where [WIDE], else by 2-byte ones (Q8_0's
   34-byte blocks), and its one scale a column read once. The caller
   has checked every alignment this takes, that blocks and scale blocks
   are a span each or hold all of k, and that e2m1 has no blocks. A
   span's block, and e2m1's scale block, is then [sp] or 0, with no
   division: Apple GPUs divide in software. Where [HALVES], int8's
   scale blocks are of 16 terms (Q6_K's) and its weight has no blocks:
   a span's two, 2 sp and 2 sp + 1, are read once each, its first two
   steps scaled by the first. The terms, their order and their fused
   multiply-adds are mv_body's. */
template <int KEY, int SD, typename VT, int R, int C, bool WIDE, bool HALVES = false>
NX_INLINE_TILE uint mv_spans(constant mv_words &w, thread const long (&cb)[C],
                             thread const long (&sc)[C], thread const long (&ra)[R],
                             uint sp, uint whole, uint S,
                             thread float (&acc)[R][C]) {
  constexpr int N = mv_span_words(KEY);
  constexpr bool scaled = KEY == NX_MV_E2M1 || KEY == NX_MV_I8;
  constexpr uint SB = mv_scale_bytes<KEY, SD>();
  const uint bl = uint(contract_low(w.blocks));
  const long bs = mv_bytes<KEY>(contract_high(w.blocks));
  const long sk = contract_low(w.bsk);
  const uint rk = uint(contract_high(w.bsk));
  const bool bspan = KEY == NX_MV_I8 && bl == 32;
  const long sstep = rk == 32 ? sk * long(SB) : 0;
  device const uchar *wc[C];
  device const uchar *sp0[C];
  NX_UNROLL for (int c = 0; c < C; c++) {
    wc[c] = w.b + mv_bytes<KEY>(cb[c]);
    sp0[c] = w.bs + sc[c] * SB;
  }
  device const VT *xr[R];
  NX_UNROLL for (int i = 0; i < R; i++) xr[i] = (device const VT *)w.a + ra[i];

  if (KEY == NX_MV_E2M1 && WIDE) {
    /* e2m1's span is a uint4 and a scale code a column. A step loads its
       rows' 8 terms, then decodes one column's 32-bit word at a time and
       adds its terms, so one column's 8 weights are live, not every
       column's; mv_e2m1_word's path is chosen once a span and column,
       the fallback rolled. Each sum takes mv_e2m1_word's products in
       the same order. */
    for (; sp < whole; sp += S) {
      uint4 v4[C];
      uint e4[C];
      bool fast[C];
      for (int c = 0; c < C; c++) {
        v4[c] = ((device const uint4 *)wc[c])[sp];
        e4[c] = sp0[c][long(sp) * sstep];
        fast[c] = mv_e2m1_exact(e4[c]);
      }
      for (uint sub = 0; sub < 4; sub++) {
        float xv[R][8];
        for (int i = 0; i < R; i++) mv_x8(xr[i] + sp * 32 + sub * 8, xv[i]);
        for (int c = 0; c < C; c++) {
          const uint word = v4[c][sub];
          if (fast[c]) {
            const float f = mv_e2m1_power(e4[c]);
            float v[8];
            mv_e2m1_halves(word, v);
            for (int i = 0; i < R; i++)
              for (int t = 0; t < 8; t++)
                acc[i][c] = metal::fma(xv[i][t], v[t] * f, acc[i][c]);
          } else {
#pragma unroll(1)
            for (int t = 0; t < 8; t++) {
              const float v = contract_exponent(mv_e2m1[(word >> (4 * t)) & 15], e4[c]);
              for (int i = 0; i < R; i++) acc[i][c] = metal::fma(xv[i][t], v, acc[i][c]);
            }
          }
        }
      }
    }
    return sp;
  }
  for (; sp < whole; sp += S) {
    const uint t0 = sp * 32;
    const long ob = bspan ? long(sp) * bs : mv_bytes<KEY>(t0);
    uint wd[C][N];
    NX_UNROLL for (int c = 0; c < C; c++) {
      device const uchar *p = wc[c] + ob;
      if (WIDE) {
        NX_UNROLL for (int i = 0; i < N / 4; i++) {
          const uint4 v = ((device const uint4 *)p)[i];
          wd[c][4 * i] = v.x;
          wd[c][4 * i + 1] = v.y;
          wd[c][4 * i + 2] = v.z;
          wd[c][4 * i + 3] = v.w;
        }
      } else {
        device const ushort *h = (device const ushort *)p;
        NX_UNROLL for (int i = 0; i < N; i++)
          wd[c][i] = uint(h[2 * i]) | (uint(h[2 * i + 1]) << 16);
      }
    }
    /* The scale block by a division: the index by [sstep], as e2m1's,
       took Q8_0's call from 55 to 64 us, for a cause not found. */
    float sv[C], sv2[C];
    NX_UNROLL for (int c = 0; c < C; c++) {
      if (HALVES) {
        sv[c] = mv_scale<KEY, SD>(sp0[c] + long(2 * sp) * sk * long(SB));
        sv2[c] = mv_scale<KEY, SD>(sp0[c] + long(2 * sp + 1) * sk * long(SB));
      } else {
        sv[c] = scaled ? mv_scale<KEY, SD>(sp0[c] + long(t0 / rk) * sk * long(SB)) : 1.0f;
        sv2[c] = sv[c];
      }
    }
    if (KEY == NX_MV_E2M1) {
      /* e2m1 by 2-byte words: a step selects its span's 32-bit word by
         index, which leaves the loop rolled, so that the compiler keeps
         one step's weights live, not four's. */
      for (int u = 0; u < 32; u += 8) {
        float v[C][8];
        NX_UNROLL for (int c = 0; c < C; c++)
          mv_e2m1_word(wd[c][u >> 3], uint(sv[c]), v[c]);
        NX_UNROLL for (int i = 0; i < R; i++) {
          float xv[8];
          mv_x8(xr[i] + t0 + uint(u), xv);
          NX_UNROLL for (int t = 0; t < 8; t++) NX_UNROLL for (int c = 0; c < C; c++)
            acc[i][c] = metal::fma(xv[t], v[c][t], acc[i][c]);
        }
      }
    } else {
      NX_UNROLL for (int u = 0; u < 32; u += 8) {
        float v[C][8];
        NX_UNROLL for (int c = 0; c < C; c++)
          mv_decode<KEY>(wd[c], u, u < 16 ? sv[c] : sv2[c], v[c]);
        NX_UNROLL for (int i = 0; i < R; i++) {
          float xv[8];
          mv_x8(xr[i] + t0 + uint(u), xv);
          NX_UNROLL for (int t = 0; t < 8; t++) NX_UNROLL for (int c = 0; c < C; c++)
            acc[i][c] = metal::fma(xv[t], v[c][t], acc[i][c]);
        }
      }
    }
  }
  return sp;
}

template <int KEY, int SD, typename VT, typename YT, int R, int C, int TWIN = 0>
NX_INLINE_TILE void mv_body(constant mv_words &w, uint3 g, ushort sg,
                            ushort lane) {
  constexpr int N = mv_span_words(KEY);
  constexpr bool scaled = KEY == NX_MV_E2M1 || KEY == NX_MV_I8;
  /* The host bounds column groups, rows and columns under 2^32, so
     their divisions are 32-bit: a 64-bit one costs as much as a span.
     A twin (kernels.h's NX_CONTRACT_MV_TWIN), whose form TWIN is not 0,
     has 32 lanes. */
  const uint S = TWIN != 0 ? 32u : uint(w.lanes), gpl = 32 / S, j0 = lane % S;
  const uint q = uint(w.first) + g.y, gn = uint(w.groups_n);
  const uint z = q / gn, cg = q - z * gn;
  const ulong col0 = ulong((cg * 4 + sg) * gpl + lane / S) * C;
  const ulong r0 = ulong(g.x) * R;
  const ulong k = w.k, spans = (k + 31) / 32;
  const uint nr = uint(metal::min(ulong(R), w.m - r0));
  const uint gcount = uint(w.groups), none = uint(w.groups >> 32);

  /* The pass's weight: batch element z's, a Rows call's of its row's
     group. */
  long wb = long(z) * w.sb;
  long sbase = long(z) * contract_low(w.bsz);
  bool live = true;
  if ((w.rows & 1) != 0) {
    /* Row r0's index into its ids, without the divisions where they are
       the identity: one row an id (inner 1), r0 within the extent. */
    const uint e = w.inner == 1 && r0 < w.extent
                       ? uint(r0)
                       : (uint(r0) / uint(w.inner)) % uint(w.extent);
    const long id = w.ids[long(z) * w.ids_z + long(e) * w.ids_r];
    live = id >= 0 && id < long(gcount);
    if (live) {
      wb += id * w.sj;
      sbase += id * contract_high(w.bsz);
    }
    if ((w.rows & 2) != 0 && cg == 0 && sg == 0 && lane == 0)
      w.of[z * w.m + r0] = live ? uint(z * gcount + ulong(id)) : none;
  }

  /* Each lane's columns, the last one's where past n. */
  long cb[C], sc[C];
  const long bl = contract_low(w.blocks), bstride = contract_high(w.blocks);
  const long sn = contract_low(w.bsn), rn = contract_high(w.bsn);
  const long sk = contract_low(w.bsk), rk = contract_high(w.bsk);
  NX_UNROLL for (int c = 0; c < C; c++) {
    const uint col = uint(metal::min(col0 + ulong(c), w.n - 1));
    cb[c] = wb + long(col) * w.ldb;
    sc[c] = sbase + long(col / uint(rn)) * sn;
  }
  long ra[R];
  NX_UNROLL for (int i = 0; i < R; i++)
    ra[i] = long(z) * w.sa + long(r0 + metal::min(uint(i), nr - 1)) * w.lda;

  float acc[R][C];
  NX_UNROLL for (int i = 0; i < R; i++) NX_UNROLL for (int c = 0; c < C; c++)
    acc[i][c] = 0.0f;

  /* Terms, blocks and scale blocks number under 2^32 (the host checks
     k), so their divisions are 32-bit, once a span or a step for all
     columns: a 64-bit division a column and step costs several times
     the span's decoding. */
  const uint ubl = uint(bl), urk = uint(rk);

  /* The whole spans by mv_spans in the form the host chose for every
     lane alike, so that no lane's address splits a SIMD-group's loads;
     the rest, and every span at form 0, below, by each lane's
     addresses. A lane's spans stay in increasing k either way. A twin's
     spans are all whole and of its form; a kernel takes form 3, which
     only int8 by float32's twin runs, as 0. */
  const uint whole = uint(k) / 32;
  const uint form = TWIN != 0 ? uint(TWIN) : uint(w.form);
  uint first = j0;
  if (live && TWIN == 3)
    first = mv_spans<KEY, SD, VT, R, C, true, true>(w, cb, sc, ra, first, whole, S, acc);
  else if (live && form == 2)
    first = mv_spans<KEY, SD, VT, R, C, true>(w, cb, sc, ra, first, whole, S, acc);
  else if (live && form == 1)
    first = mv_spans<KEY, SD, VT, R, C, false>(w, cb, sc, ra, first, whole, S, acc);

  for (uint sp = first; TWIN == 0 && live && sp < spans; sp += S) {
    const uint t0 = sp * 32;
    const uint cnt = metal::min(32u, uint(k) - t0);
    const long off = bl > 0 ? long(t0 / ubl) * bstride + long(t0 % ubl) : long(t0);
    /* A span's scale blocks: one where its first and last steps share
       one, as at Q8_0's and MXFP4's blocks of 32. */
    const uint q0 = scaled ? t0 / urk : 0u;
    const bool one = !scaled || q0 == (t0 + 24) / urk;
    uint wd[C][N];
    NX_UNROLL for (int c = 0; c < C; c++) {
      device const uchar *p = w.b + mv_bytes<KEY>(cb[c] + off);
      if (cnt == 32)
        mv_load<N>(p, wd[c]);
      else
        mv_load_part<N>(p, (cnt * uint(mv_half_bytes(KEY)) + 1) / 2, wd[c]);
    }
    /* A span's scales where one block holds it, read once. */
    float sv[C];
    NX_UNROLL for (int c = 0; c < C; c++)
      sv[c] = scaled && one ? mv_scale<KEY, SD>(w.bs + (sc[c] + long(q0) * sk) *
                                                         mv_scale_bytes<KEY, SD>())
                            : 1.0f;
    /* The steps unrolled, so that every index into the span's words is
       a constant and the words stay in registers. */
    NX_UNROLL for (int u = 0; u < 32; u += 8) {
      if (uint(u) >= cnt) break;
      const uint lim = metal::min(8u, cnt - uint(u));
      float v[C][8];
      NX_UNROLL for (int c = 0; c < C; c++) {
        float s = sv[c];
        if (scaled && !one) {
          const long sq = long((t0 + uint(u)) / urk) * sk;
          s = mv_scale<KEY, SD>(w.bs + (sc[c] + sq) * mv_scale_bytes<KEY, SD>());
        }
        mv_decode<KEY>(wd[c], u, s, v[c]);
      }
      NX_UNROLL for (int i = 0; i < R; i++) {
        device const VT *x = (device const VT *)w.a + ra[i] + long(t0 + uint(u));
        float xv[8];
        if (lim == 8 && (ulong(x) & 15) == 0)
          mv_x8(x, xv);
        else
          NX_UNROLL for (int t = 0; t < 8; t++)
            xv[t] = uint(t) < lim ? float(x[t]) : 0.0f;
        /* Past k a term adds nothing: its decoded weight may be NaN
           where its scale is infinite. */
        if (lim == 8) {
          NX_UNROLL for (int t = 0; t < 8; t++) NX_UNROLL for (int c = 0; c < C; c++)
            acc[i][c] = metal::fma(xv[t], v[c][t], acc[i][c]);
        } else {
          NX_UNROLL for (int t = 0; t < 8; t++) {
            if (uint(t) >= lim) break;
            NX_UNROLL for (int c = 0; c < C; c++)
              acc[i][c] = metal::fma(xv[t], v[c][t], acc[i][c]);
          }
        }
      }
    }
  }

  /* Lane l < h adds lane l + h's, for h = S / 2, ..., 1. */
  for (uint h = S / 2; h >= 1; h /= 2)
    NX_UNROLL for (int i = 0; i < R; i++) NX_UNROLL for (int c = 0; c < C; c++)
      acc[i][c] = acc[i][c] + metal::simd_shuffle_down(acc[i][c], ushort(h));
  if (j0 != 0) return;
  NX_UNROLL for (int i = 0; i < R; i++) NX_UNROLL for (int c = 0; c < C; c++) {
    if (uint(i) >= nr || col0 + ulong(c) >= w.n) continue;
    finish_write(w.y, mv_dtype<YT>(),
                 long(z) * w.sy + long(r0 + ulong(i)) * w.ldy + long(col0 + ulong(c)) * w.cy,
                 0, acc[i][c]);
  }
}

/* Element [c] past [p] of a k x n lane's columns, as float32: a plain
   key's value; e2m1's by code [e] as contract_exponent gives it, [p]
   at an even element; int8's times Value [sv], one rounded product. */
template <int KEY>
static inline float mv_kn_elem(device const uchar *p, uint c, uint e, float sv) {
  if (KEY == NX_MV_F32) return ((device const float *)p)[c];
  if (KEY == NX_MV_F16) return float(((device const half *)p)[c]);
  if (KEY == NX_MV_BF16) return as_type<float>(uint(((device const ushort *)p)[c]) << 16);
  if (KEY == NX_MV_I8) return float(char(p[c])) * sv;
  return contract_exponent(mv_e2m1[(p[c >> 1] >> ((c & 1) * 4)) & 15], e);
}

/* Column [c]'s code of a lane's codes packed 4 a word. */
template <int W>
NX_INLINE_TILE uint mv_kn_code(thread const uint (&e)[W], int c) {
  return (e[c >> 2] >> ((c & 3) * 8)) & 255u;
}

/* A scaled k x n lane's scales of its C columns from [col0] for scale
   block [q], as the n x k kernels read them (mv_scale): e2m1's codes
   packed 4 a word into [e], a Value's float32 casts into [s]. Where
   the scale's columns are its elements ([contig]), e2m1's codes are one
   load by mv_load's alignment check; else each column's block is found
   by a division, once a scale block. */
template <int KEY, int SD, int C>
NX_INLINE_TILE void mv_kn_scales(constant mv_words &w, long base, ulong col0, long q,
                                 bool contig, thread uint (&e)[C / 4], thread float (&s)[C]) {
  constexpr int SB = mv_scale_bytes<KEY, SD>();
  const long sn = contract_low(w.bsn), at = base + q * contract_low(w.bsk);
  if (KEY == NX_MV_E2M1 && contig) {
    mv_load<C / 4>(w.bs + at + long(col0), e);
    return;
  }
  const uint rn = uint(contract_high(w.bsn));
  NX_UNROLL for (int i = 0; i < C / 4; i++) e[i] = 0;
  NX_UNROLL for (int c = 0; c < C; c++) {
    const uint col = uint(metal::min(col0 + ulong(c), w.n - 1));
    const float v = mv_scale<KEY, SD>(w.bs + (at + long(col / rn) * sn) * SB);
    if (KEY == NX_MV_E2M1) e[c >> 2] |= uint(v) << ((c & 3) * 8);
    s[c] = v;
  }
}

/* The k x n form: a weight of columns contiguous. A lane holds C
   adjacent columns, 16 bytes of a plain key, 8 columns of a scaled one
   (4 bytes of e2m1, 8 of int8), and reads them one term at a time, so
   [32 / rho'] lanes read a term's row at once, for R rows a pass. Span
   lane j of a column block is j = g + rho' s, g the lane's span row in
   its SIMD-group and s its SIMD-group in the block: the halving's levels
   h >= rho' pair SIMD-groups, through threadgroup memory 8 sums a thread
   at a time, and the levels below pair a SIMD-group's lanes. The terms,
   their order and the halving are the n x k form's, and a scaled term is
   the n x k decode's (mv_e2m1_word's values, int8's one rounded product),
   so are its bits. A scaled lane's scales change every rk terms: it
   takes a span's terms in runs of one scale block, with no division a
   term. The probes: e2m1 at 32 columns a lane, or with 8 terms' words
   loaded first, or with the scale's test in every term, holds more
   values live and ran MXFP4's 16-row call 1.1 to 4x slower. */
template <int KEY, int SD, typename VT, typename YT, int R>
NX_INLINE_TILE void mv_kn_body(constant mv_words &w, threadgroup float *red,
                               uint3 g, ushort sg, ushort lane) {
  constexpr bool scaled = KEY == NX_MV_E2M1 || KEY == NX_MV_I8;
  constexpr int C = KEY == NX_MV_F32 ? 4 : 8;
  constexpr int N = KEY == NX_MV_E2M1 ? 1 : KEY == NX_MV_I8 ? 2 : 4;
  constexpr int RC = R * C, CH = RC < 8 ? RC : 8;
  static_assert(RC % CH == 0, "the halving's chunks of sums");
  const uint S = uint(w.lanes);
  const uint rho = metal::min(uint(NX_MV_RHO), S), sigma = S / rho;
  const uint per = 32 / rho, gr = lane / per, cl = lane % per;
  const uint blocks = (NX_MV_KN_THREADS / 32) / sigma;
  const uint blk = sg / sigma, sgi = sg % sigma;
  const uint j0 = gr + rho * sgi;
  /* The host bounds column groups and rows under 2^32: 32-bit divisions,
     as mv_body's. */
  const uint q = uint(w.first) + g.y, gn = uint(w.groups_n);
  const uint z = q / gn, cg = q - z * gn;
  const ulong col0 = ulong((cg * blocks + blk) * per + cl) * C;
  const ulong r0 = ulong(g.x) * R;
  const uint nr = R == 1 ? 1u : uint(metal::min(ulong(R), w.m - r0));
  const ulong k = w.k, spans = (k + 31) / 32;
  const uint gcount = uint(w.groups), none = uint(w.groups >> 32);

  /* A Rows call takes one row a pass (the host's R = 1), its row's
     group's weight and scales. */
  long wb = long(z) * w.sb;
  long sbase = long(z) * contract_low(w.bsz);
  bool live = true;
  if ((w.rows & 1) != 0) {
    /* Row r0's index into its ids, without the divisions where they are
       the identity: one row an id (inner 1), r0 within the extent. */
    const uint e = w.inner == 1 && r0 < w.extent
                       ? uint(r0)
                       : (uint(r0) / uint(w.inner)) % uint(w.extent);
    const long id = w.ids[long(z) * w.ids_z + long(e) * w.ids_r];
    live = id >= 0 && id < long(gcount);
    if (live) {
      wb += id * w.sj;
      sbase += id * contract_high(w.bsz);
    }
    if ((w.rows & 2) != 0 && cg == 0 && sg == 0 && lane == 0)
      w.of[ulong(z) * w.m + r0] = live ? uint(ulong(z) * gcount + ulong(id)) : none;
  }

  /* The lane's columns within n: all, some, or none. */
  const uint have = col0 >= w.n ? 0u : uint(metal::min(ulong(C), w.n - col0));
  device const VT *x[R];
  NX_UNROLL for (int i = 0; i < R; i++)
    x[i] = (device const VT *)w.a + long(z) * w.sa +
           long(r0 + metal::min(uint(i), nr - 1)) * w.lda;
  float acc[RC];
  NX_UNROLL for (int c = 0; c < RC; c++) acc[c] = 0.0f;

  /* A scaled lane's scale blocks: rk terms each; its columns' scales
     contiguous where the scale's columns are its elements. */
  const uint urk = uint(contract_high(w.bsk));
  const bool contig =
      contract_low(w.bsn) == 1 && contract_high(w.bsn) == 1 && have == uint(C);

  /* A span's terms in runs of one scale block each, the block's scales
     read before its run: one run a span where its blocks hold spans, as
     MXFP4's and Q8_0's do. A run's terms hold only their loads, decodes
     and sums live, as the probes ran fastest. */
  uint e[C / 4];
  float s[C], f[C];
  NX_UNROLL for (int c = 0; c < C; c++) {
    s[c] = 1.0f;
    f[c] = 1.0f;
  }
  NX_UNROLL for (int i = 0; i < C / 4; i++) e[i] = 0;
  bool exact = true;
  for (ulong sp = j0; live && have > 0 && sp < spans; sp += S) {
    /* The host bounds k + 32 under 2^32: 32-bit terms. */
    const uint t0 = uint(sp) * 32, end = uint(metal::min(ulong(t0) + 32, k));
    long sq = scaled ? long(t0 / urk) : 0;
    uint left = scaled ? urk - (t0 - uint(sq) * urk) : 32u;
    for (uint t = t0; t < end;) {
      const uint stop = end - t <= left ? end : t + left;
      if (scaled) {
        mv_kn_scales<KEY, SD, C>(w, sbase, col0, sq, contig, e, s);
        if (KEY == NX_MV_E2M1) {
          exact = true;
          NX_UNROLL for (int c = 0; c < C; c++) {
            const uint ec = mv_kn_code(e, c);
            exact = exact && mv_e2m1_exact(ec);
            f[c] = mv_e2m1_power(ec);
          }
        }
      }
      if (have == uint(C)) {
        for (; t < stop; t++) {
          device const uchar *p = w.b + mv_bytes<KEY>(wb + long(t) * w.ldb + long(col0));
          uint wd[N];
          mv_load<N>(p, wd);
          float v[C];
          if (KEY == NX_MV_E2M1 && exact) {
            float h[8];
            mv_e2m1_halves(wd[0], h);
            NX_UNROLL for (int c = 0; c < C; c++) v[c] = h[c & 7] * f[c];
          } else if (KEY == NX_MV_E2M1) {
            NX_UNROLL for (int c = 0; c < C; c++)
              v[c] = contract_exponent(mv_term<KEY>(wd, c), mv_kn_code(e, c));
          } else if (KEY == NX_MV_I8) {
            NX_UNROLL for (int c = 0; c < C; c++) v[c] = mv_term<KEY>(wd, c) * s[c];
          } else {
            NX_UNROLL for (int c = 0; c < C; c++) v[c] = mv_term<KEY>(wd, c);
          }
          NX_UNROLL for (int i = 0; i < R; i++) {
            const float xv = float(x[i][t]);
            NX_UNROLL for (int c = 0; c < C; c++)
              acc[i * C + c] = metal::fma(xv, v[c], acc[i * C + c]);
          }
        }
      } else {
        for (; t < stop; t++) {
          device const uchar *p = w.b + mv_bytes<KEY>(wb + long(t) * w.ldb + long(col0));
          float v[C];
          NX_UNROLL for (int c = 0; c < C; c++)
            v[c] = uint(c) < have ? mv_kn_elem<KEY>(p, uint(c), mv_kn_code(e, c), s[c]) : 0.0f;
          NX_UNROLL for (int i = 0; i < R; i++) {
            const float xv = float(x[i][t]);
            NX_UNROLL for (int c = 0; c < C; c++)
              acc[i * C + c] = metal::fma(xv, v[c], acc[i * C + c]);
          }
        }
      }
      sq += 1;
      left = urk;
    }
  }

  /* Levels h = S / 2, ..., rho' across SIMD-groups: block member s < h /
     rho' adds member s + h / rho''s, CH sums at a time. */
  threadgroup float *mine = red + (uint(sg) * 32 + lane) * CH;
  NX_UNROLL for (int c0 = 0; c0 < RC; c0 += CH) {
    for (uint hs = sigma / 2; hs >= 1; hs /= 2) {
      NX_UNROLL for (int c = 0; c < CH; c++) mine[c] = acc[c0 + c];
      metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      if (sgi < hs) {
        threadgroup const float *other = red + ((uint(sg) + hs) * 32 + lane) * CH;
        NX_UNROLL for (int c = 0; c < CH; c++) acc[c0 + c] = acc[c0 + c] + other[c];
      }
      metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
  }
  /* Levels h = rho' / 2, ..., 1 within a SIMD-group: span row g < h adds
     row g + h's, h per lanes on. */
  for (uint h = rho / 2; h >= 1; h /= 2)
    NX_UNROLL for (int c = 0; c < RC; c++)
      acc[c] = acc[c] + metal::simd_shuffle_down(acc[c], ushort(h * per));
  if (j0 != 0) return;
  NX_UNROLL for (int i = 0; i < R; i++) NX_UNROLL for (int c = 0; c < C; c++) {
    if (uint(i) >= nr || uint(c) >= have) continue;
    finish_write(w.y, mv_dtype<YT>(),
                 long(z) * w.sy + long(r0 + ulong(i)) * w.ldy + long(col0 + ulong(c)) * w.cy,
                 0, acc[i * C + c]);
  }
}

#define CONTRACT_MV_KN(name, KEY, SD, VT, YT, R)                              \
  NX_KERNEL(NX_MV_KN_THREADS)                                                 \
  void name(constant mv_words &w [[buffer(0)]],                               \
            threadgroup uchar *shared [[threadgroup(0)]],                     \
            uint3 g [[threadgroup_position_in_grid]],                         \
            ushort sg [[simdgroup_index_in_threadgroup]],                     \
            ushort lane [[thread_index_in_simdgroup]]) {                      \
    mv_kn_body<KEY, SD, VT, YT, R>(w, (threadgroup float *)shared, g, sg, lane); \
  }

/* A key's bound on its n x k kernels' threads a group. int8's carry
   none: with one, the compiler gives a lane more registers and a core
   fewer groups, and Q8_0's Rows call took 1.6 us more. The others keep
   it, which e2m1's runs faster with (1.3 us at MXFP4). nx_metal.ml's
   resolve fails where a pipeline holds fewer threads than
   NX_MV_THREADS. */
#define NX_MV_BOUNDED() NX_KERNEL(NX_MV_THREADS)
#define NX_MV_UNBOUNDED() [[kernel]]
#define NX_MV_BOUND_f32 NX_MV_BOUNDED
#define NX_MV_BOUND_f16 NX_MV_BOUNDED
#define NX_MV_BOUND_bf16 NX_MV_BOUNDED
#define NX_MV_BOUND_e2m1 NX_MV_BOUNDED
#define NX_MV_BOUND_i8f16 NX_MV_UNBOUNDED
#define NX_MV_BOUND_i8f32 NX_MV_UNBOUNDED

#define CONTRACT_MV(name, BOUND, KEY, SD, VT, YT, R, C)                       \
  BOUND()                                                                     \
  void name(constant mv_words &w [[buffer(0)]],                               \
            uint3 g [[threadgroup_position_in_grid]],                         \
            ushort sg [[simdgroup_index_in_threadgroup]],                     \
            ushort lane [[thread_index_in_simdgroup]]) {                      \
    mv_body<KEY, SD, VT, YT, R, C>(w, g, sg, lane);                           \
  }

/* Each key's code, a Value scale's dtype and a lane's columns. */
#define NX_MV_KEY_f32 NX_MV_F32, 0, 1
#define NX_MV_KEY_f16 NX_MV_F16, 0, 2
#define NX_MV_KEY_bf16 NX_MV_BF16, 0, 2
#define NX_MV_KEY_e2m1 NX_MV_E2M1, 0, 4
#define NX_MV_KEY_i8f16 NX_MV_I8, NX_FLOAT16, 2
#define NX_MV_KEY_i8f32 NX_MV_I8, NX_FLOAT32, 2

#define CONTRACT_MV_AT(name, key, VT, YT, R)                                  \
  CONTRACT_MV_AT_(name, NX_MV_BOUND_##key, NX_MV_KEY_##key, VT, YT, R)
#define CONTRACT_MV_AT_(name, bound, keyargs, VT, YT, R)                      \
  CONTRACT_MV_ARGS(name, bound, keyargs, VT, YT, R)
#define CONTRACT_MV_ARGS(name, bound, KEY, SD, C, VT, YT, R)                  \
  CONTRACT_MV(name, bound, KEY, SD, VT, YT, R, C)

#define CONTRACT_MV_SET(X, key, family, r)                                    \
  CONTRACT_MV_AT(contract_mv_##key##_f32_r##r##_nk, key, float, float, r)     \
  CONTRACT_MV_AT(contract_mv_##key##_f16_r##r##_nk, key, half, float, r)      \
  CONTRACT_MV_AT(contract_mv_##key##_bf16_r##r##_nk, key, bfloat, float, r)   \
  CONTRACT_MV_AT(contract_mv_##key##_f16_r##r##_nk_typed, key, half, half, r) \
  CONTRACT_MV_AT(contract_mv_##key##_bf16_r##r##_nk_typed, key, bfloat, bfloat, r)

NX_CONTRACT_MV_KEYS(CONTRACT_MV_SET, _)

/* The twins carry no bound on a group's threads: with one, the
   compiler gives a lane more registers and a core fewer threads, and
   MXFP4's bfloat16 call took 46.5 us, against 44.8 (the kernel alone). */
#define CONTRACT_MV_TWIN(name, key, VT, YT, F)                                \
  CONTRACT_MV_TWIN_(name, NX_MV_KEY_##key, VT, YT, F)
#define CONTRACT_MV_TWIN_(name, keyargs, VT, YT, F)                           \
  CONTRACT_MV_TWIN_ARGS(name, keyargs, VT, YT, F)
#define CONTRACT_MV_TWIN_ARGS(name, KEY, SD, C, VT, YT, F)                    \
  NX_MV_UNBOUNDED()                                                           \
  void name(constant mv_words &w [[buffer(0)]],                               \
            uint3 g [[threadgroup_position_in_grid]],                         \
            ushort sg [[simdgroup_index_in_threadgroup]],                     \
            ushort lane [[thread_index_in_simdgroup]]) {                      \
    mv_body<KEY, SD, VT, YT, 1, C, F>(w, g, sg, lane);                        \
  }

#define CONTRACT_MV_TWIN_SET(X, key, family, form)                            \
  CONTRACT_MV_TWIN(contract_mv_##key##_f32_r1_nk_w32, key, float, float, form) \
  CONTRACT_MV_TWIN(contract_mv_##key##_f16_r1_nk_w32, key, half, float, form) \
  CONTRACT_MV_TWIN(contract_mv_##key##_bf16_r1_nk_w32, key, bfloat, float, form) \
  CONTRACT_MV_TWIN(contract_mv_##key##_f16_r1_nk_typed_w32, key, half, half, form) \
  CONTRACT_MV_TWIN(contract_mv_##key##_bf16_r1_nk_typed_w32, key, bfloat, bfloat, form)

NX_CONTRACT_MV_TWINS(CONTRACT_MV_TWIN_SET, _)

/* Each k x n key's code and a Value scale's dtype. */
#define NX_MV_KN_KEY_f32 NX_MV_F32, 0
#define NX_MV_KN_KEY_f16 NX_MV_F16, 0
#define NX_MV_KN_KEY_bf16 NX_MV_BF16, 0
#define NX_MV_KN_KEY_e2m1 NX_MV_E2M1, 0
#define NX_MV_KN_KEY_i8f16 NX_MV_I8, NX_FLOAT16
#define NX_MV_KN_KEY_i8f32 NX_MV_I8, NX_FLOAT32

#define CONTRACT_MV_KN_AT(name, key, VT, YT, R)                               \
  CONTRACT_MV_KN_AT_(name, NX_MV_KN_KEY_##key, VT, YT, R)
#define CONTRACT_MV_KN_AT_(name, keyargs, VT, YT, R) CONTRACT_MV_KN(name, keyargs, VT, YT, R)

#define CONTRACT_MV_KN_SET(X, key, family, r)                                 \
  CONTRACT_MV_KN_AT(contract_mv_##key##_f32_r##r##_kn, key, float, float, r)  \
  CONTRACT_MV_KN_AT(contract_mv_##key##_f16_r##r##_kn, key, half, float, r)   \
  CONTRACT_MV_KN_AT(contract_mv_##key##_bf16_r##r##_kn, key, bfloat, float, r) \
  CONTRACT_MV_KN_AT(contract_mv_##key##_f16_r##r##_kn_typed, key, half, half, r) \
  CONTRACT_MV_KN_AT(contract_mv_##key##_bf16_r##r##_kn_typed, key, bfloat, bfloat, r)

NX_CONTRACT_MV_KN_KEYS(CONTRACT_MV_KN_SET, _)
