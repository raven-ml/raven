/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Sorts and numberings (kernels.h, Sorts and numberings).

   A sort orders each slice of x by its elements' keys, unsigned integers
   in the contract's order (Keys), by radix of 8-bit digits. Three paths,
   the host's choice by the slice's length n and the elements it keeps:

   - sort_block: slices of at most [cap] elements, [per_block] of them a
     group, sorted in shared memory.
   - sort_hist, then sort_pass once a key byte: longer slices, by one-sweep
     LSD radix over global memory (Adinets & Merrill, "Onesweep: A Faster
     Least Significant Digit Radix Sort for GPUs", arXiv 2206.01784).
     sort_hist counts every digit of every slice in one read; each pass
     ranks a tile's digits and finds its bins' places by look-back over
     the earlier tiles of its slice. A pass whose digit every key of a
     slice shares is skipped for that slice.
   - select_round once a digit, select_gather, then sort_block: a long
     slice that keeps at most [cap] / 4 elements. Rounds of an MSD radix
     select on (key, position) find the kept elements' least prefix that
     leaves at most [cap] candidates, which sort_block then sorts.

   A numbering sorts its rows stably, one word at a time from the last, by
   the one-sweep passes, then numbers them: unique_heads, unique_rank,
   unique_ids.

   Determinism. No float arithmetic. Atomics are integer adds into counts,
   tile numbers (which tile a group runs changes nothing written),
   candidate slots (sorted by a distinct key after), ORs of bits, and a
   sub-byte element's store of its own bits. A look-back sums integers.

   Progress. A look-back waits only on tiles numbered before its own, and a
   group takes its tile's number from a counter once it runs: every tile
   it waits on belongs to a group that runs already and waits only on
   tiles before it, down to tile 0, which waits on none. No group assumes
   another is resident with it but one that started before it.

   Keys. An element's key is an unsigned integer in the contract's order:
   an unsigned integer its bits; a signed one its bits with the sign
   flipped; a boolean 1 where not zero; a float its bits with every bit
   flipped where the sign is set and the sign flipped elsewhere, so -0 sits
   below +0, and every NaN, a magnitude past [nan_abs], all ones, above
   +inf and equal; a complex number its real part's key above its
   imaginary part's, all ones where either is a NaN. A descending sort
   complements the keys: an ascending stable sort of them gives the
   reversed order with equal elements in increasing position. The format
   is a parameter, the same for every element, so its branches are
   uniform.

   Ranks. A group ranks its items' 8-bit digits stably in the order warp,
   then item, then lane (rank_digits): within a warp by __match_any_sync,
   then each bin's warps one after another, then the bins. Items past the
   slice take digit 256, after every real one. */

#ifndef NX_CUDA_SORT_CU
#define NX_CUDA_SORT_CU

#include <stdint.h>

#include "kernels.h"

#define SORT_THREADS NX_SORT_THREADS
#define SORT_WARPS (SORT_THREADS / 32)
#define SORT_BINS 257
#define ST_AGG (1ull << 62)
#define ST_PRE (1ull << 63)
#define ST_CNT (ST_AGG - 1)

typedef sort_params SP;

struct El { uint64_t lo, hi; };
struct It { uint64_t lo, hi; int64_t pos; };

/* Elements */

/* Element [off] of [base], [ebits] wide; a sub-byte one's code from the
   low bits of its byte first. A complex number is read a component at a
   time: its array promises only a component's alignment. */
__device__ __forceinline__ El load_el(const SP &P, const void *base,
                                      int64_t off) {
  El e = {0, 0};
  if (P.kind == NX_KEY_COMPLEX && P.ebits == 64) {
    const uint32_t *c = (const uint32_t *)base + 2 * off;
    e.lo = (uint64_t)c[1] << 32 | c[0];
    return e;
  }
  switch (P.ebits) {
    case 1: case 4: {
      int64_t bit = off * P.ebits;
      uint32_t b = ((const uint8_t *)base)[bit >> 3];
      e.lo = (b >> (bit & 7)) & ((1u << P.ebits) - 1);
      break;
    }
    case 8: e.lo = ((const uint8_t *)base)[off]; break;
    case 16: e.lo = ((const uint16_t *)base)[off]; break;
    case 32: e.lo = ((const uint32_t *)base)[off]; break;
    case 64: e.lo = ((const uint64_t *)base)[off]; break;
    default:
      e.lo = ((const uint64_t *)base)[2 * off];
      e.hi = ((const uint64_t *)base)[2 * off + 1];
  }
  return e;
}

/* Stores [e] at element [off] of [base], a complex number a component at
   a time. A sub-byte element shares its byte with others, maybe of other
   slices: it clears and sets its own bits by atomics on their word, which
   commute with its neighbours'. */
__device__ __forceinline__ void store_el(const SP &P, void *base,
                                         int64_t off, El e) {
  if (P.kind == NX_KEY_COMPLEX && P.ebits == 64) {
    uint32_t *c = (uint32_t *)base + 2 * off;
    c[0] = (uint32_t)e.lo;
    c[1] = (uint32_t)(e.lo >> 32);
    return;
  }
  switch (P.ebits) {
    case 1: case 4: {
      int64_t bit = off * P.ebits;
      uintptr_t a = (uintptr_t)base + (bit >> 3);
      uint32_t *word = (uint32_t *)(a & ~(uintptr_t)3);
      int sh = (int)(a & 3) * 8 + (int)(bit & 7);
      uint32_t m = ((1u << P.ebits) - 1) << sh;
      atomicAnd(word, ~m);
      atomicOr(word, (uint32_t)e.lo << sh);
      break;
    }
    case 8: ((uint8_t *)base)[off] = (uint8_t)e.lo; break;
    case 16: ((uint16_t *)base)[off] = (uint16_t)e.lo; break;
    case 32: ((uint32_t *)base)[off] = (uint32_t)e.lo; break;
    case 64: ((uint64_t *)base)[off] = e.lo; break;
    default:
      ((uint64_t *)base)[2 * off] = e.lo;
      ((uint64_t *)base)[2 * off + 1] = e.hi;
  }
}

/* Keys */

struct Key { uint64_t lo, hi; };

__device__ __forceinline__ bool fnan(uint64_t b, int bits, uint64_t nan_abs) {
  return (b & ((1ull << (bits - 1)) - 1)) > nan_abs;
}

__device__ __forceinline__ uint64_t fkey(uint64_t b, int bits) {
  uint64_t sign = 1ull << (bits - 1), mask = sign | (sign - 1);
  return b & sign ? ~b & mask : b | sign;
}

/* A real element's key is its bits xored with flip[1] where its sign is
   set and flip[0] where not, or kmask where its magnitude is past
   nan_abs: flip is {sign, mask} for a float, {sign, sign} for a signed
   integer, zero for the others, and nan_abs all ones for every
   non-float. A boolean's bits are first 1 where not zero. */
__device__ __forceinline__ Key key_of(const SP &P, El e) {
  Key k = {0, 0};
  const int bits = P.ebits;
  if (P.kind != NX_KEY_COMPLEX) {
    uint64_t b = P.kind == NX_KEY_BOOL ? e.lo != 0 : e.lo;
    uint64_t sign = 1ull << (bits - 1);
    k.lo = (b & ~sign) > P.nan_abs ? P.kmask : b ^ P.flip[(b & sign) != 0];
  } else if (bits == 64) {
    uint64_t re = e.lo & 0xFFFFFFFFu, im = e.lo >> 32;
    k.lo = fnan(re, 32, P.nan_abs) || fnan(im, 32, P.nan_abs)
               ? ~0ull
               : fkey(re, 32) << 32 | fkey(im, 32);
  } else if (fnan(e.lo, 64, P.nan_abs) || fnan(e.hi, 64, P.nan_abs)) {
    k.lo = k.hi = ~0ull;
  } else {
    k.hi = fkey(e.lo, 64);
    k.lo = fkey(e.hi, 64);
  }
  k.lo ^= P.dmask[0];
  k.hi ^= P.dmask[1];
  return k;
}

/* Byte [p] of [k], from the least significant. */
__device__ __forceinline__ int kbyte(Key k, int p) {
  return (int)((p < 8 ? k.lo >> (8 * p) : k.hi >> (8 * (p - 8))) & 255);
}

/* Digit [j] of (key, position) from the most significant, a select's. */
__device__ __forceinline__ int cdigit(const SP &P, Key k, int64_t pos,
                                      int j) {
  if (j < P.kbytes) return kbyte(k, P.kbytes - 1 - j);
  return (int)((uint64_t)pos >> (8 * (P.kbytes + P.pbytes - 1 - j)) & 255);
}

/* Slices */

/* Slice [s]'s first elements of x, values and positions. */
__device__ void slice_offsets(const SP &P, int64_t s, int64_t o[3]) {
  o[0] = P.off[0];
  o[1] = P.off[1];
  o[2] = P.off[2];
  for (int i = P.nkept - 1; i >= 0; i--) {
    int64_t e = P.kept[i][0], c = s % e;
    s /= e;
    o[0] += c * P.kept[i][1];
    o[1] += c * P.kept[i][2];
    o[2] += c * P.kept[i][3];
  }
}

/* Writes the element [e] at position [pos] of its slice, rank [r] of the
   sorted order, where the slice keeps that rank. */
__device__ __forceinline__ void emit(const SP &P, const int64_t o[3],
                                     int64_t r, El e, int64_t pos) {
  if (r >= P.keep) return;
  P.positions[o[2] + r * P.sp] = pos;
  if (P.flags & NX_SORT_VALUES) store_el(P, P.values, o[1] + r * P.sv, e);
}

/* Scans */

__device__ __forceinline__ uint32_t lanemask_lt() {
  uint32_t m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

/* The exclusive sum of [v] over the group's threads, and in [total]
   their sum. Every thread calls it. */
template <class T>
__device__ T block_scan(T v, T &total) {
  __shared__ T sums[SORT_WARPS];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  T x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    T y = __shfl_up_sync(~0u, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) sums[w] = x;
  __syncthreads();
  T base = 0, all = 0;
#pragma unroll
  for (int i = 0; i < SORT_WARPS; i++) {
    T s = sums[i];
    base += i < w ? s : 0;
    all += s;
  }
  __syncthreads();
  total = all;
  return base + x - v;
}

/* Stores tile [t]'s count [c] where later tiles see it, and returns the
   sum of the counts of the tiles before [t]: tile u's word is st[u *
   stride], zero until it stores its count (ST_AGG) or the inclusive sum
   (ST_PRE). The sums are of integers, the same in any order the tiles
   run; a tile waits only on tiles numbered before it (Progress). */
__device__ uint64_t lookback(uint64_t *st, int64_t stride, int64_t t,
                             uint64_t c) {
  volatile uint64_t *v = st;
  if (t == 0) {
    v[0] = ST_PRE | c;
    return 0;
  }
  v[t * stride] = ST_AGG | c;
  uint64_t excl = 0;
  for (int64_t u = t - 1;;) {
    uint64_t s = v[u * stride];
    if (s == 0) continue;
    excl += s & ST_CNT;
    if (s & ST_PRE) break;
    u--;
  }
  v[t * stride] = ST_PRE | (excl + c);
  return excl;
}

/* Ranks */

struct Rank {
  uint32_t warp[SORT_WARPS][SORT_BINS]; /* each warp's count, then offset */
  uint32_t start[SORT_BINS], count[SORT_BINS];
};

/* Ranks the first [cnt] of each thread's digits [d], item j of lane l of
   warp w at w * cnt * 32 + j * 32 + l in the group's order: [r] gets
   each item's place in the stable order by digit. [R]'s count and start
   then hold each bin's items and first place. */
template <int N>
__device__ void rank_digits(const int (&d)[N], int cnt, uint32_t (&r)[N],
                            Rank &R) {
  const int w = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < SORT_WARPS * SORT_BINS; i += SORT_THREADS)
    (&R.warp[0][0])[i] = 0;
  __syncthreads();
  const uint32_t lt = lanemask_lt();
#pragma unroll
  for (int j = 0; j < N; j++) {
    if (j >= cnt) break;
    uint32_t m = __match_any_sync(~0u, d[j]);
    uint32_t base = R.warp[w][d[j]];
    __syncwarp();
    if ((m & lt) == 0) R.warp[w][d[j]] = base + __popc(m);
    __syncwarp();
    r[j] = base + __popc(m & lt);
  }
  __syncthreads();
  for (int b = threadIdx.x; b < SORT_BINS; b += SORT_THREADS) {
    uint32_t sum = 0;
    for (int i = 0; i < SORT_WARPS; i++) {
      uint32_t c = R.warp[i][b];
      R.warp[i][b] = sum;
      sum += c;
    }
    R.count[b] = sum;
  }
  __syncthreads();
  uint32_t all;
  uint32_t s = block_scan<uint32_t>(R.count[threadIdx.x], all);
  R.start[threadIdx.x] = s;
  if (threadIdx.x == 0) R.start[256] = all;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < N; j++)
    if (j < cnt) r[j] += R.start[d[j]] + R.warp[w][d[j]];
}

/* Clears */

__device__ void sort_clear_run(const clear_params &C) {
  const int64_t stride = (int64_t)gridDim.x * SORT_THREADS;
  for (int64_t i = (int64_t)blockIdx.x * SORT_THREADS + threadIdx.x;
       i < C.words; i += stride)
    C.p[i] = 0;
}

/* Block sorts */

/* Sorts slices of at most [cap] elements in shared memory: [per_block]
   slices from x a group, or, under NX_SORT_CANDIDATES, one slice's
   candidates (select_gather). Each element's key waits in shared memory,
   kbytes rounded up to words; the items' order moves as 16-bit indices,
   by LSD passes over the candidates' positions, then the keys' bytes,
   then the slices in the group. */
#define BLOCK_ITEMS (NX_SORT_CAP / SORT_THREADS)
__device__ void sort_block_run(const SP &P) {
  extern __shared__ uint32_t dyn[];
  __shared__ Rank R;
  __shared__ int64_t so[256][3];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int cap = P.cap, nb = cap / SORT_THREADS, kw = (P.kbytes + 3) / 4;
  uint32_t *keys = dyn;
  uint16_t *ord = (uint16_t *)(keys + cap * kw);
  const bool cand = P.flags & NX_SORT_CANDIDATES;
  const int64_t n = P.n;
  int64_t s0;
  int b_slices, v;
  if (cand) {
    s0 = blockIdx.x;
    b_slices = 1;
    v = (int)P.sel[s0].cnt;
  } else {
    s0 = (int64_t)blockIdx.x * P.per_block;
    b_slices = (int)min((int64_t)P.per_block, P.slices - s0);
    v = b_slices * (int)n;
  }
  if (threadIdx.x < b_slices)
    slice_offsets(P, s0 + threadIdx.x, so[threadIdx.x]);
  __syncthreads();
  const uint64_t *cb = P.cand + (size_t)s0 * cap * 3;
  for (int i = threadIdx.x; i < v; i += SORT_THREADS) {
    El e;
    if (cand) e = {cb[3 * i], cb[3 * i + 1]};
    else e = load_el(P, P.x, so[i / n][0] + (i % n) * P.sx);
    Key k = key_of(P, e);
    for (int q = 0; q < kw; q++)
      keys[i * kw + q] = (uint32_t)((q < 2 ? k.lo : k.hi) >> (32 * (q & 1)));
  }
  __syncthreads();
  uint16_t idx[BLOCK_ITEMS];
#pragma unroll
  for (int j = 0; j < BLOCK_ITEMS; j++) idx[j] = w * nb * 32 + j * 32 + lane;
  const int pb = cand ? P.pbytes : 0;
  const int passes = pb + P.kbytes + (b_slices > 1);
  for (int p = 0; p < passes; p++) {
    int d[BLOCK_ITEMS];
    uint32_t r[BLOCK_ITEMS];
#pragma unroll
    for (int j = 0; j < BLOCK_ITEMS; j++) {
      int i = idx[j];
      if (j >= nb) break;
      if (i >= v) d[j] = 256;
      else if (p < pb) d[j] = (int)(cb[3 * i + 2] >> (8 * p) & 255);
      else if (p < pb + P.kbytes) {
        int q = p - pb;
        d[j] = (int)(keys[i * kw + q / 4] >> (8 * (q & 3)) & 255);
      } else d[j] = i / (int)n;
    }
    rank_digits(d, nb, r, R);
#pragma unroll
    for (int j = 0; j < BLOCK_ITEMS; j++)
      if (j < nb) ord[r[j]] = idx[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < BLOCK_ITEMS; j++)
      if (j < nb) idx[j] = ord[w * nb * 32 + j * 32 + lane];
  }
  for (int r = threadIdx.x; r < v; r += SORT_THREADS) {
    int i = ord[r], b = cand ? 0 : i / (int)n;
    int64_t pos = cand ? (int64_t)cb[3 * i + 2] : i % n;
    El e = cand ? El{cb[3 * i], cb[3 * i + 1]}
                : load_el(P, P.x, so[b][0] + pos * P.sx);
    emit(P, so[b], r - b * n, e, pos);
  }
}

/* One-sweep passes */

/* Counts every digit of every key of a group's range of its slice into
   the slice's counts; the last group of the slice makes them each bin's
   first place and finds the passes the slice takes: those whose digit
   not every key shares, or the last pass where every one is. */
__device__ void sort_hist_run(const SP &P) {
  __shared__ uint32_t h[16 * 256];
  __shared__ int64_t o[3];
  __shared__ bool last;
  const int64_t s = blockIdx.x / P.hblocks, c = blockIdx.x % P.hblocks;
  const int kb = P.kbytes;
  for (int i = threadIdx.x; i < kb * 256; i += SORT_THREADS) h[i] = 0;
  if (threadIdx.x == 0) slice_offsets(P, s, o);
  __syncthreads();
  const int64_t lo = c * NX_SORT_CHUNK, hi = min(P.n, lo + NX_SORT_CHUNK);
  for (int64_t i = lo + threadIdx.x; i < hi; i += SORT_THREADS) {
    Key k = key_of(P, load_el(P, P.x, o[0] + i * P.sx));
    for (int p = 0; p < kb; p++) atomicAdd(&h[p * 256 + kbyte(k, p)], 1u);
  }
  __syncthreads();
  uint64_t *hs = P.hist + (size_t)s * kb * 256;
  for (int i = threadIdx.x; i < kb * 256; i += SORT_THREADS)
    if (h[i]) atomicAdd((unsigned long long *)&hs[i], h[i]);
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    last = atomicAdd(&P.tickets[s], 1u) == (uint32_t)P.hblocks - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  uint32_t mask = 0;
  for (int p = 0; p < kb; p++) {
    uint64_t x = __ldcg(&hs[p * 256 + threadIdx.x]), all;
    uint64_t e = block_scan<uint64_t>(x, all);
    hs[p * 256 + threadIdx.x] = e;
    if (__syncthreads_or(x == (uint64_t)P.n) == 0) mask |= 1u << p;
  }
  if (threadIdx.x == 0) {
    P.meta[s] = mask ? mask : 1u << (kb - 1);
    P.tickets[s] = 0;
  }
}

/* Items as a pass's buffers hold them, slice s's at s n of [total]:
   8 bytes, a key of at most 32 bits above its 32-bit position; 12, an
   element of at most 64 bits, then apart its 32-bit position; 24, an
   element of at most 128 bits and its 64-bit position. */
template <int F> struct Fmt;
template <> struct Fmt<8> {
  typedef uint64_t T;
  static __device__ T pack(It e) { return e.lo << 32 | (uint32_t)e.pos; }
  static __device__ It unpack(T t) {
    return {t >> 32, 0, (int64_t)(t & 0xFFFFFFFFu)};
  }
  static __device__ T get(const void *b, int64_t, int64_t i) {
    return ((const T *)b)[i];
  }
  static __device__ void put(void *b, int64_t, int64_t i, T t) {
    ((T *)b)[i] = t;
  }
};
template <> struct Fmt<12> {
  typedef ulonglong2 T;
  static __device__ T pack(It e) {
    return make_ulonglong2(e.lo, (uint64_t)e.pos);
  }
  static __device__ It unpack(T t) { return {t.x, 0, (int64_t)t.y}; }
  static __device__ T get(const void *b, int64_t total, int64_t i) {
    const uint64_t *bits = (const uint64_t *)b;
    return make_ulonglong2(bits[i], ((const uint32_t *)(bits + total))[i]);
  }
  static __device__ void put(void *b, int64_t total, int64_t i, T t) {
    uint64_t *bits = (uint64_t *)b;
    bits[i] = t.x;
    ((uint32_t *)(bits + total))[i] = (uint32_t)t.y;
  }
};
struct Three { uint64_t a, b, c; };
template <> struct Fmt<24> {
  typedef Three T;
  static __device__ T pack(It e) { return {e.lo, e.hi, (uint64_t)e.pos}; }
  static __device__ It unpack(T t) { return {t.a, t.b, (int64_t)t.c}; }
  static __device__ T get(const void *b, int64_t, int64_t i) {
    return ((const T *)b)[i];
  }
  static __device__ void put(void *b, int64_t, int64_t i, T t) {
    ((T *)b)[i] = t;
  }
};

/* Pass [pass], over that key byte: each tile of a slice ranks its items
   by the digit, takes each bin's place after the slice's earlier tiles by
   look-back, and moves its items there through shared memory, in runs of
   one bin. The slice's first pass reads x, its last writes the values and
   positions it keeps, and the others move items between buf[0] and
   buf[1], the last pass reading buf[1]. A tile clears its words of the
   other status array, which the next pass uses. */
template <int F, int N>
__device__ void pass(const SP &P) {
  typedef Fmt<F> M;
  typedef typename M::T T;
  constexpr int TILE = SORT_THREADS * N;
  __shared__ Rank R;
  __shared__ T items[TILE];
  __shared__ uint64_t gbase[256];
  __shared__ int64_t o[3];
  __shared__ uint32_t tile;
  const int p = P.pass;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (threadIdx.x == 0) tile = atomicAdd(&P.counter[p], 1u);
  __syncthreads();
  const int64_t s = tile / P.tiles, t = tile % P.tiles;
  const uint32_t mask = P.meta[s];
  if (!(mask >> p & 1)) return;
  const int q = __popc(mask & ((1u << p) - 1)), qs = __popc(mask);
  const bool first = q == 0, final = q == qs - 1;
  void *b0 = P.flags & NX_SORT_REUSE ? (void *)(P.positions + P.off[2])
                                     : P.buf[0];
  void *src = (qs - q) & 1 ? P.buf[1] : b0;
  void *dst = (qs - 1 - q) & 1 ? P.buf[1] : b0;
  if (threadIdx.x == 0) slice_offsets(P, s, o);
  __syncthreads();
  const int64_t n = P.n, t0 = t * TILE, base = s * n, total = P.slices * n;
  const int valid = (int)min((int64_t)TILE, n - t0);
  /* The first pass stages x's elements in shared memory, in a loop that
     inlines load_el once. */
  if (first) {
    for (int l = threadIdx.x; l < valid; l += SORT_THREADS) {
      int64_t e = P.flags & NX_SORT_ORDER ? P.order[base + t0 + l] : t0 + l;
      El x = load_el(P, P.x, o[0] + e * P.sx);
      items[l] = M::pack(It{x.lo, x.hi, e});
    }
    __syncthreads();
  }
  It it[N];
  int d[N];
  uint32_t r[N];
#pragma unroll
  for (int j = 0; j < N; j++) {
    const int l = w * N * 32 + j * 32 + lane;
    it[j] = {0, 0, 0};
    d[j] = 256;
    if (l >= valid) continue;
    it[j] = M::unpack(first ? items[l] : M::get(src, total, base + t0 + l));
    d[j] = kbyte(key_of(P, El{it[j].lo, it[j].hi}), p);
  }
  rank_digits(d, N, r, R);
  const size_t slot = ((size_t)s * P.tiles + t) * 256 + threadIdx.x;
  P.status[(q + 1) & 1][slot] = 0;
  uint64_t excl = lookback(P.status[q & 1] + (size_t)s * P.tiles * 256 +
                               threadIdx.x, 256, t, R.count[threadIdx.x]);
  gbase[threadIdx.x] = P.hist[((size_t)s * P.kbytes + p) * 256 + threadIdx.x] +
                       excl - R.start[threadIdx.x];
#pragma unroll
  for (int j = 0; j < N; j++) items[r[j]] = M::pack(it[j]);
  __syncthreads();
  for (int i = threadIdx.x; i < valid; i += SORT_THREADS) {
    It e = M::unpack(items[i]);
    int dd = kbyte(key_of(P, El{e.lo, e.hi}), p);
    int64_t g = gbase[dd] + i;
    if (final) emit(P, o, g, El{e.lo, e.hi}, e.pos);
    else M::put(dst, total, base + g, M::pack(e));
  }
}

__device__ void sort_pass8_run(const SP &P) { pass<8, NX_SORT_ITEMS8>(P); }
__device__ void sort_pass12_run(const SP &P) { pass<12, NX_SORT_ITEMS12>(P); }
__device__ void sort_pass24_run(const SP &P) { pass<24, NX_SORT_ITEMS24>(P); }

/* Selects */

/* -1, 0 or 1 as the first [rounds] digits of (k, pos) are below, equal to
   or above [prefix]. */
__device__ __forceinline__ int against(const SP &P, Key k, int64_t pos,
                                       const uint8_t *prefix, int rounds) {
  for (int j = 0; j < rounds; j++) {
    int d = cdigit(P, k, pos, j);
    if (d != prefix[j]) return d < prefix[j] ? -1 : 1;
  }
  return 0;
}

/* Round [pass] of each slice's select, unless done: counts that digit of
   the elements whose earlier digits match the prefix. The slice's last
   group finds the bin of the [need]th of them, which extends the prefix,
   and ends the select where the elements below or matching it are at
   most [cap], or no digit is left. */
__device__ void select_round_run(const SP &P) {
  __shared__ uint32_t h[256];
  __shared__ uint8_t prefix[32];
  __shared__ int64_t o[3];
  __shared__ bool last;
  const int round = P.pass;
  const int64_t s = blockIdx.x / P.hblocks, c = blockIdx.x % P.hblocks;
  sort_select *st = &P.sel[s];
  if (st->done) return;
  h[threadIdx.x] = 0;
  if (threadIdx.x < 32) prefix[threadIdx.x] = st->prefix[threadIdx.x];
  if (threadIdx.x == 0) slice_offsets(P, s, o);
  __syncthreads();
  const int64_t lo = c * NX_SORT_CHUNK, hi = min(P.n, lo + NX_SORT_CHUNK);
  for (int64_t i = lo + threadIdx.x; i < hi; i += SORT_THREADS) {
    Key k = key_of(P, load_el(P, P.x, o[0] + i * P.sx));
    if (against(P, k, i, prefix, round) == 0)
      atomicAdd(&h[cdigit(P, k, i, round)], 1u);
  }
  __syncthreads();
  if (h[threadIdx.x])
    atomicAdd((unsigned long long *)&st->hist[threadIdx.x], h[threadIdx.x]);
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    last = atomicAdd(&P.tickets[s], 1u) == (uint32_t)P.hblocks - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const uint64_t need = round == 0 ? (uint64_t)P.keep : st->need;
  uint64_t x = __ldcg(&st->hist[threadIdx.x]), all;
  uint64_t e = block_scan<uint64_t>(x, all);
  st->hist[threadIdx.x] = 0;
  if (e < need && need <= e + x) {
    uint64_t left = need - e, below = (uint64_t)P.keep - left;
    st->prefix[round] = (uint8_t)threadIdx.x;
    st->need = left;
    st->rounds = round + 1;
    st->done = below + x <= (uint64_t)P.cap ||
               round + 1 == P.kbytes + P.pbytes;
  }
  if (threadIdx.x == 0) P.tickets[s] = 0;
}

/* Gathers each slice's elements below or matching its select's prefix
   into its candidates. Their order there depends on the order groups run
   in; sort_block orders them by (key, position), which every candidate
   holds a distinct one of. */
__device__ void select_gather_run(const SP &P) {
  __shared__ uint8_t prefix[32];
  __shared__ int64_t o[3];
  __shared__ int rounds;
  const int64_t s = blockIdx.x / P.hblocks, c = blockIdx.x % P.hblocks;
  sort_select *st = &P.sel[s];
  if (threadIdx.x < 32) prefix[threadIdx.x] = st->prefix[threadIdx.x];
  if (threadIdx.x == 0) {
    rounds = st->rounds;
    slice_offsets(P, s, o);
  }
  __syncthreads();
  uint64_t *cb = P.cand + (size_t)s * P.cap * 3;
  const int64_t lo = c * NX_SORT_CHUNK, hi = min(P.n, lo + NX_SORT_CHUNK);
  for (int64_t i = lo + threadIdx.x; i < hi; i += SORT_THREADS) {
    El e = load_el(P, P.x, o[0] + i * P.sx);
    if (against(P, key_of(P, e), i, prefix, rounds) > 0) continue;
    uint64_t at = atomicAdd((unsigned long long *)&st->cnt, 1ull);
    cb[3 * at] = e.lo;
    cb[3 * at + 1] = e.hi;
    cb[3 * at + 2] = (uint64_t)i;
  }
}

/* Numbering */

#define UNIQUE_ITEMS (NX_UNIQUE_TILE / SORT_THREADS)

/* Whether sorted rows [i - 1] and [i] differ. */
__device__ __forceinline__ bool differs(const unique_params &U, int64_t i) {
  if (U.flags & NX_UNIQUE_SKEYS) return U.skeys[i] != U.skeys[i - 1];
  const bool ordered = U.flags & NX_UNIQUE_ORDER;
  int64_t a = ordered ? U.order[i] : i, b = ordered ? U.order[i - 1] : i - 1;
  for (int64_t j = 0; j < U.w; j++)
    if (U.keys[a * U.s0 + j * U.s1] != U.keys[b * U.s0 + j * U.s1])
      return true;
  return false;
}

/* Each sorted row's key's number in sorted order, gid, by a scan of the
   rows where a key starts; each key's first row, gfirst (the stable sort
   puts it first), and first sorted row, gstart; a bit at each first
   row. */
__device__ void unique_heads_run(const unique_params &U) {
  __shared__ uint32_t tile;
  __shared__ uint64_t before;
  if (threadIdx.x == 0) tile = atomicAdd(&U.counter[0], 1u);
  __syncthreads();
  const int64_t t = tile, i0 = t * NX_UNIQUE_TILE + threadIdx.x * UNIQUE_ITEMS;
  const bool ordered = U.flags & NX_UNIQUE_ORDER;
  uint32_t heads = 0;
  for (int j = 0; j < UNIQUE_ITEMS; j++) {
    int64_t i = i0 + j;
    if (i < U.n && (i == 0 || differs(U, i))) heads |= 1u << j;
  }
  uint64_t all, excl = block_scan<uint64_t>(__popc(heads), all);
  if (threadIdx.x == 0) before = lookback(U.status, 1, t, all);
  __syncthreads();
  int64_t g = (int64_t)(before + excl) - 1;
  for (int j = 0; j < UNIQUE_ITEMS; j++) {
    int64_t i = i0 + j;
    if (i >= U.n) break;
    if (heads >> j & 1) {
      int64_t f = ordered ? U.order[i] : i;
      g++;
      U.gfirst[g] = f;
      U.gstart[g] = i;
      atomicOr((unsigned long long *)&U.bits[f >> 6], 1ull << (f & 63));
    }
    U.gid[i] = g;
  }
  if (t == (U.n - 1) / NX_UNIQUE_TILE && threadIdx.x == 0) {
    *U.k = (int64_t)(before + all);
    U.gstart[before + all] = U.n;
  }
}

/* The first rows set before each word of the bits, by a scan. */
__device__ void unique_rank_run(const unique_params &U) {
  __shared__ uint32_t tile;
  __shared__ uint64_t before;
  if (threadIdx.x == 0) tile = atomicAdd(&U.counter[1], 1u);
  __syncthreads();
  const int64_t words = (U.n + 63) / 64, t = tile,
                i0 = t * NX_UNIQUE_TILE + threadIdx.x * UNIQUE_ITEMS;
  const int64_t heads_tiles = (U.n + NX_UNIQUE_TILE - 1) / NX_UNIQUE_TILE;
  uint64_t c = 0;
  for (int j = 0; j < UNIQUE_ITEMS; j++)
    if (i0 + j < words) c += __popcll(U.bits[i0 + j]);
  uint64_t all, excl = block_scan<uint64_t>(c, all);
  if (threadIdx.x == 0) before = lookback(U.status + heads_tiles, 1, t, all);
  __syncthreads();
  uint64_t at = before + excl;
  for (int j = 0; j < UNIQUE_ITEMS && i0 + j < words; j++) {
    U.below[i0 + j] = at;
    at += __popcll(U.bits[i0 + j]);
  }
}

/* Each row's id, the rank of its key's first row among the first rows;
   each key's first row and count, and the entries past the keys. */
__device__ void unique_ids_run(const unique_params &U) {
  const int64_t k = U.n == 0 ? 0 : *U.k;
  const int64_t stride = (int64_t)gridDim.x * SORT_THREADS;
  const bool ordered = U.flags & NX_UNIQUE_ORDER;
  for (int64_t i = (int64_t)blockIdx.x * SORT_THREADS + threadIdx.x; i < U.n;
       i += stride) {
    int64_t g = U.gid[i], f = U.gfirst[g];
    uint64_t below = U.bits[f >> 6] & ((1ull << (f & 63)) - 1);
    int64_t num = (int64_t)(U.below[f >> 6] + __popcll(below));
    U.ids[ordered ? U.order[i] : i] = num;
    if (U.gstart[g] == i && num < U.m) {
      U.first[num] = f;
      U.counts[num] = U.gstart[g + 1] - i;
    }
  }
  for (int64_t j = k + (int64_t)blockIdx.x * SORT_THREADS + threadIdx.x;
       j < U.m; j += stride) {
    U.first[j] = U.n;
    U.counts[j] = 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) *U.count = k;
}

/* Entries. A pass holds two groups of its threads an SM, so that one ranks
   while the other moves its items. */

#define SORT_GROUPS_sort_block 1
#define SORT_GROUPS_sort_hist 1
#define SORT_GROUPS_sort_pass8 2
#define SORT_GROUPS_sort_pass12 2
#define SORT_GROUPS_sort_pass24 2
#define SORT_GROUPS_select_round 1
#define SORT_GROUPS_select_gather 1

#define CLEAR(name, ...)                                                       \
  extern "C" __global__ void __launch_bounds__(SORT_THREADS)                   \
      name(const __grid_constant__ clear_params p) {                           \
    name##_run(p);                                                             \
  }

#define SORT(name, ...)                                                        \
  extern "C" __global__ void __launch_bounds__(SORT_THREADS,                   \
                                               SORT_GROUPS_##name)             \
      name(const __grid_constant__ sort_params p) {                            \
    name##_run(p);                                                             \
  }

#define UNIQUE(name, ...)                                                      \
  extern "C" __global__ void __launch_bounds__(SORT_THREADS)                   \
      name(const __grid_constant__ unique_params p) {                          \
    name##_run(p);                                                             \
  }

#endif
