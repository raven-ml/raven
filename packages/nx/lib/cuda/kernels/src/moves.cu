/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Moves and their adjoints: gathers, range gathers, copies and fills
   (move), a range gather's runs (runs), assemblies (assemble), folds
   (fold) and scatters (scatter), with their parameters of kernels.h.

   One entry per walk. An element is [count] units of [unit] bits, the
   launch's parameters: 1, 4, 8, 16, 32 or 64 bits, its dtype's alignment,
   and two units for a complex number, whose parts align as one part does.
   The unit is the same for every thread of a launch, so its branches do
   not diverge. Arithmetic (a scatter's Add, Max and Min, a fold) follows
   the dtype's code: float32, with the narrow floats decoded to it by
   nx_dtype.h, float64, complex64, complex128 and the integers. A float
   Add sums in float32 or float64; a complex one sums each part apart.

   A launch is a grid of any size whose threads stride over their work, so
   the host sizes it by the work alone.

   Determinism. No float atomic anywhere. Integer atomics only where the
   result is a function of the set of operations: counts, flags, the
   maximum of positions, bitwise and/or of disjoint bits, integer adds that
   wrap, extremes under a total order (a NaN result canonical), and a lock
   whose holder computes such an extreme.

   A scatter's float Add sums a target's updates by the position tree: the
   complete binary tree over the positions along the scattered axis, an
   absent position taking no part, then adds that sum to the target last,
   x + S. The tree is a function of the positions alone, so where the
   kernels cut the axis into aligned tiles of 2^b positions changes no bit.

     tile (or rows): sort a tile's (target, slot) keys in shared memory,
       merge level by level, one entry a target and tile
     one tile a slice: store x + S
     else spill: a target in one tile stores x + S; the others' entries
       spill, keyed (target, tile)
     hist, scan, place, a pass each 8 bits: radix sort of the keys
     tree1: the tree across a target's tiles, pieces of 2^qb tiles
     tree2: the pieces' tree, then x + S */

#ifndef NX_CUDA_MOVES_CU
#define NX_CUDA_MOVES_CU

#include "nx_dtype.h"
#include "words.cuh"

#define MT NX_MOVE_THREADS

namespace nx_moves {

typedef unsigned long long u64;

/* An element: its first unit in [lo], its second in [hi]. */
struct el {
  u64 lo, hi;
};

/* Elements by unit */

/* Element [p] of [count] units of [unit] bits at [base]. */
__device__ __forceinline__ el ld(const void *base, int64_t p, int unit,
                                 int count) {
  el v = {0, 0};
  switch (unit) {
  case 1:
  case 4: {
    uint64_t bit = (uint64_t)p * unit;
    uint8_t byte = ((const uint8_t *)base)[bit >> 3];
    v.lo = (byte >> (bit & 7)) & ((1u << unit) - 1);
    return v;
  }
  case 8: v.lo = ((const uint8_t *)base)[p]; return v;
  case 16: v.lo = ((const uint16_t *)base)[p]; return v;
  case 32: {
    const uint32_t *q = (const uint32_t *)base + p * count;
    v.lo = q[0];
    if (count == 2) v.hi = q[1];
    return v;
  }
  default: {
    const u64 *q = (const u64 *)base + p * count;
    v.lo = q[0];
    if (count == 2) v.hi = q[1];
    return v;
  }
  }
}

/* A store of one element, a sub-byte one into the word that holds it
   (nx_store_bits). */
__device__ __forceinline__ void st(void *base, int64_t p, int unit, int count,
                                   el v) {
  switch (unit) {
  case 1:
  case 4: nx_store_bits(base, p, unit, (uint32_t)v.lo); return;
  case 8: ((uint8_t *)base)[p] = (uint8_t)v.lo; return;
  case 16: ((uint16_t *)base)[p] = (uint16_t)v.lo; return;
  case 32: {
    uint32_t *q = (uint32_t *)base + p * count;
    q[0] = (uint32_t)v.lo;
    if (count == 2) q[1] = (uint32_t)v.hi;
    return;
  }
  default: {
    u64 *q = (u64 *)base + p * count;
    q[0] = v.lo;
    if (count == 2) q[1] = v.hi;
    return;
  }
  }
}

/* Stores [bits] under [mask] into a word other threads may change. */
__device__ __forceinline__ void merge(unsigned *w, unsigned bits,
                                      unsigned mask) {
  if (mask == ~0u) {
    *w = bits;
    return;
  }
  if (mask == 0) return;
  atomicAnd(w, ~mask);
  atomicOr(w, bits & mask);
}

/* Index arithmetic: 32-bit where the space has fewer than 2^32 indices. */
__device__ __noinline__ int64_t div64(int64_t &i, int64_t n) {
  int64_t q = i / n, c = i - q * n;
  i = q;
  return c;
}

__device__ __forceinline__ int64_t divmod(int64_t &i, int64_t n, int small) {
  if (small) {
    uint32_t a = (uint32_t)i, b = (uint32_t)n, q = a / b;
    i = q;
    return a - q * b;
  }
  return div64(i, n);
}

__device__ __forceinline__ int msb64(u64 x) {
  return 63 - __clzll((long long)x);
}

/* A grid-stride loop. */
#define MOVES_LOOP(i, n)                                                      \
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += (int64_t)gridDim.x * blockDim.x)

/* A sub-byte destination, C-contiguous over [numel] elements of [unit]
   bits from [dst_off], written a 32-bit word a thread, its elements in
   order: [at e v] is false for none, else sets v to element e's bits. The
   edge words merge by and/or with the bits of other arrays, which rig
   orders with dst's (nx_word_at). */
template <class F>
__device__ __forceinline__ void words(void *dst, int64_t dst_off,
                                      int64_t numel, int unit, F at) {
  const int per = 32 / unit;
  const unsigned one = (1u << unit) - 1;
  uintptr_t base = (uintptr_t)dst, abase = base & ~(uintptr_t)3;
  int64_t lead = (int64_t)(base - abase) * 8;
  int64_t first = lead + dst_off * unit, end = first + numel * unit;
  int64_t w0 = first / 32, w1 = (end + 31) / 32;
  MOVES_LOOP(t, w1 - w0) {
    int64_t w = w0 + t, e0 = (32 * w - lead) / unit - dst_off;
    unsigned bits = 0, mask = 0;
#pragma unroll 1
    for (int i = 0; i < per; i++) {
      int64_t e = e0 + i;
      unsigned v;
      if (e < 0 || e >= numel || !at(e, v)) continue;
      bits |= (v & one) << (i * unit);
      mask |= one << (i * unit);
    }
    merge((unsigned *)abase + w, bits, mask);
  }
}

/* Moves: gathers, range gathers, copies, fills */

struct move_pos {
  int64_t pd, ps, pi, ca;
};

__device__ __forceinline__ move_pos move_at(const move_params &p, int64_t e) {
  move_pos q = {p.dst_off, p.src_off, p.idx_off, 0};
#pragma unroll 1
  for (int a = p.rank - 1; a >= 0; a--) {
    const move_axis &x = p.ax[a];
    int64_t c = divmod(e, x.n, p.small);
    q.pd += c * x.sd;
    q.ps += c * x.ss;
    q.pi += c * x.si;
    if (a == p.axis) q.ca = c;
  }
  return q;
}

/* The run of position [pos] in a row of the runs: the first whose maximum
   of the ends up to it exceeds [pos], [k] for none. */
__device__ __forceinline__ int64_t run_of(const int64_t *maxima, int64_t k,
                                          int64_t pos) {
  int64_t lo = 0, hi = k;
  while (lo < hi) {
    int64_t mid = (lo + hi) >> 1;
    if (maxima[mid] > pos)
      hi = mid;
    else
      lo = mid + 1;
  }
  return lo;
}

__device__ __forceinline__ el move_value(const move_params &p,
                                         const move_pos &q) {
  const el none = {0, 0};
  switch (p.mode) {
  case NX_MOVE_FILL: return el{p.fill[0], p.fill[1]};
  case NX_MOVE_COPY: return ld(p.src, q.ps, p.unit, p.count);
  case NX_MOVE_GATHER: {
    int64_t at = p.idx[q.pi];
    if ((uint64_t)at >= (uint64_t)p.d) return none;
    return ld(p.src, q.ps + at * p.sa, p.unit, p.count);
  }
  default: {
    /* q.pi is the row's first run, q.ca the position along the axis; the
       read position wraps as int64 does. */
    int64_t j = run_of(p.maxima + q.pi, p.k, q.ca);
    if (j == p.k) return none;
    uint64_t e = j ? (uint64_t)p.ends[q.pi + j - 1] : 0;
    uint64_t at = (uint64_t)p.starts[q.pi + j] + (uint64_t)q.ca - e;
    if (at >= (uint64_t)p.d) return none;
    return ld(p.src, q.ps + (int64_t)at * p.sa, p.unit, p.count);
  }
  }
}

__device__ void move(const move_params &p) {
  if (p.unit < 8) {
    words(p.dst, p.dst_off, p.numel, p.unit, [&](int64_t e, unsigned &v) {
      v = (unsigned)move_value(p, move_at(p, e)).lo;
      return true;
    });
    return;
  }
  MOVES_LOOP(e, p.numel) {
    move_pos q = move_at(p, e);
    st(p.dst, q.pd, p.unit, p.count, move_value(p, q));
  }
}

/* A range gather's runs */

__device__ __forceinline__ void runs_row(const runs_params &p, int64_t row,
                                         int64_t &ps, int64_t &pe) {
  ps = p.s_off;
  pe = p.e_off;
#pragma unroll 1
  for (int a = p.lead - 1; a >= 0; a--) {
    int64_t c = divmod(row, p.ax[a].n, p.small);
    ps += c * p.ax[a].ss;
    pe += c * p.ax[a].se;
  }
}

/* The block's exclusive running maximum of [v], one value a thread, in
   thread order; [total] the maximum of all. */
__device__ int64_t block_max_excl(int64_t v, int64_t &total) {
  __shared__ int64_t warp_max[MT / 32];
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  int64_t x = v;
  for (int o = 1; o < 32; o <<= 1) {
    int64_t y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x = max(x, y);
  }
  int64_t ex = __shfl_up_sync(0xffffffffu, x, 1);
  if (lane == 0) ex = INT64_MIN;
  if (lane == 31) warp_max[w] = x;
  __syncthreads();
  total = INT64_MIN;
  for (int u = 0; u < MT / 32; u++) {
    if (u < w) ex = max(ex, warp_max[u]);
    total = max(total, warp_max[u]);
  }
  __syncthreads();
  return ex;
}

/* A block a chunk of NX_RUNS_CHUNK runs of one row: the runs copied, their
   ends' running maximum within the chunk, the chunk's maximum. */
__device__ void runs_chunk(const runs_params &p) {
  const int per = NX_RUNS_CHUNK / MT;
  for (int64_t b = blockIdx.x; b < p.rows * p.chunks; b += gridDim.x) {
    int64_t row = b / p.chunks, c = b % p.chunks, ps, pe;
    runs_row(p, row, ps, pe);
    int64_t j0 = c * NX_RUNS_CHUNK + threadIdx.x * per, m = INT64_MIN;
    int64_t *S = p.starts + row * p.k, *E = p.ends + row * p.k,
            *M = p.maxima + row * p.k;
#pragma unroll 1
    for (int i = 0; i < per; i++) {
      int64_t j = j0 + i;
      if (j >= p.k) break;
      S[j] = p.in_s[ps + j * p.s_k];
      int64_t e = p.in_e[pe + j * p.e_k];
      E[j] = e;
      m = max(m, e);
    }
    int64_t total, run = block_max_excl(m, total);
#pragma unroll 1
    for (int i = 0; i < per; i++) {
      int64_t j = j0 + i;
      if (j >= p.k) break;
      run = max(run, E[j]);
      M[j] = run;
    }
    if (threadIdx.x == 0 && p.chunks > 1) p.carry[row * p.chunks + c] = total;
  }
}

/* A block a row: each chunk's maximum becomes the maximum of the chunks
   before it. */
__device__ void runs_carry(const runs_params &p) {
  for (int64_t row = blockIdx.x; row < p.rows; row += gridDim.x) {
    int64_t *c = p.carry + row * p.chunks, in = INT64_MIN;
    for (int64_t t0 = 0; t0 < p.chunks; t0 += MT) {
      int64_t t = t0 + threadIdx.x, total;
      int64_t ex = block_max_excl(t < p.chunks ? c[t] : INT64_MIN, total);
      if (t < p.chunks) c[t] = max(in, ex);
      in = max(in, total);
    }
  }
}

__device__ void runs_merge(const runs_params &p) {
  MOVES_LOOP(i, p.rows * p.k) {
    int64_t row = i / p.k, j = i % p.k, c = j / NX_RUNS_CHUNK;
    if (c == 0) continue;
    int64_t v = p.carry[row * p.chunks + c];
    if (v > p.maxima[i]) p.maxima[i] = v;
  }
}

/* Assemblies */

__device__ __forceinline__ int record(const assemble_params &p) {
  return NX_ASSEMBLE_HEAD + p.rank * NX_ASSEMBLE_AXIS;
}

/* Whether the index [cd] of dst lies in the region of the piece whose
   record is [t]. */
__device__ __forceinline__ bool covers(const int64_t *t, int rank,
                                       const int64_t *cd) {
#pragma unroll 1
  for (int a = 0; a < rank; a++) {
    const int64_t *x = t + NX_ASSEMBLE_HEAD + a * NX_ASSEMBLE_AXIS;
    int64_t n = x[0], start = x[2], step = x[3], d = cd[a] - start;
    if (step > 0 ? d < 0 : d > 0) return false;
    if (d % step != 0) return false;
    if (d / step >= n) return false;
  }
  return true;
}

/* Element e of the launch's pieces one after the other, copied unless a
   later piece of the launch covers its place; later launches store
   after. */
__device__ void assemble_pieces(const assemble_params &p) {
  const int rec = record(p);
  MOVES_LOOP(e, p.total) {
    int k = 0;
#pragma unroll 1
    for (int i = 1; i < p.pieces; i++)
      if (p.tab[i * rec + 2] <= e) k = i;
    const int64_t *t = p.tab + k * rec;
    int64_t i = e - t[2], ps = t[0], pd = p.dst_off + t[1];
    int64_t cd[NX_MOVE_RANK];
#pragma unroll 1
    for (int a = p.rank - 1; a >= 0; a--) {
      const int64_t *x = t + NX_ASSEMBLE_HEAD + a * NX_ASSEMBLE_AXIS;
      int64_t c = divmod(i, x[0], p.small);
      ps += c * x[1];
      pd += c * x[4];
      cd[a] = x[2] + c * x[3];
    }
    bool hidden = false;
    uint64_t later = (uint64_t)t[3];
#pragma unroll 1
    while (later && !hidden) {
      int r = __ffsll((long long)later) - 1;
      later &= later - 1;
      hidden = covers(p.tab + r * rec, p.rank, cd);
    }
    if (!hidden)
      st(p.dst, pd, p.unit, p.count, ld(p.src[k], ps, p.unit, p.count));
  }
}

/* [fill] where no piece covers, with [check]; else everywhere, the pieces'
   launches storing theirs after. */
__device__ void assemble_fill(const assemble_params &p) {
  int64_t numel = 1;
#pragma unroll 1
  for (int a = 0; a < p.rank; a++) numel *= p.dn[a];
  const int rec = record(p);
  auto open = [&](int64_t e) {
    if (!p.check) return true;
    int64_t cd[NX_MOVE_RANK];
#pragma unroll 1
    for (int a = p.rank - 1; a >= 0; a--) cd[a] = divmod(e, p.dn[a], p.small);
#pragma unroll 1
    for (int r = 0; r < p.pieces; r++)
      if (covers(p.tab + r * rec, p.rank, cd)) return false;
    return true;
  };
  if (p.unit < 8) {
    words(p.dst, p.dst_off, numel, p.unit, [&](int64_t e, unsigned &v) {
      v = (unsigned)p.fill[0];
      return open(e);
    });
    return;
  }
  const el v = {p.fill[0], p.fill[1]};
  MOVES_LOOP(e, numel) if (open(e)) st(p.dst, p.dst_off + e, p.unit, p.count, v);
}

__device__ void assemble(const assemble_params &p) {
  if (p.mode == NX_ASSEMBLE_FILL)
    assemble_fill(p);
  else
    assemble_pieces(p);
}

/* Arithmetic */

/* A float element as its format T, float32 (a narrow float decoded) or
   float64: of [unit] bits at [base]. */
template <class T>
__device__ __forceinline__ T load_val(const void *base, int64_t p, int dt,
                                      int unit) {
  if constexpr (sizeof(T) == 8)
    return ((const double *)base)[p];
  else if (dt == NX_FLOAT32)
    return ((const float *)base)[p];
  else
    return nx_bits_to_float(dt, (uint32_t)ld(base, p, unit, 1).lo);
}

template <class T>
__device__ __forceinline__ void store_val(void *base, int64_t p, int dt,
                                          int unit, T v) {
  if constexpr (sizeof(T) == 8)
    ((double *)base)[p] = v;
  else if (dt == NX_FLOAT32)
    ((float *)base)[p] = v;
  else
    st(base, p, unit, 1, el{nx_narrow_code(dt, v), 0});
}

/* The sum of elements [a] and [b] of [dt], as Prog.Add: a float's in its
   format, each part of a complex number's apart, a narrow float's rounded
   to it, an integer's wrapping once stored at its width. */
__device__ __forceinline__ el add_el(int dt, el a, el b) {
  switch (dt) {
  case NX_FLOAT32:
    return el{nx_float_bits(nx_bits_float((uint32_t)a.lo) +
                            nx_bits_float((uint32_t)b.lo)),
              0};
  case NX_FLOAT64:
    return el{nx_double_bits(nx_bits_double(a.lo) + nx_bits_double(b.lo)), 0};
  case NX_COMPLEX64:
    return el{nx_float_bits(nx_bits_float((uint32_t)a.lo) +
                            nx_bits_float((uint32_t)b.lo)),
              nx_float_bits(nx_bits_float((uint32_t)a.hi) +
                            nx_bits_float((uint32_t)b.hi))};
  case NX_COMPLEX128:
    return el{nx_double_bits(nx_bits_double(a.lo) + nx_bits_double(b.lo)),
              nx_double_bits(nx_bits_double(a.hi) + nx_bits_double(b.hi))};
  case NX_FLOAT16:
  case NX_BFLOAT16:
  case NX_FLOAT8_E4M3FN:
  case NX_FLOAT8_E5M2:
  case NX_FLOAT4_E2M1FN:
    return el{nx_narrow_code(dt, nx_bits_to_float(dt, (uint32_t)a.lo) +
                                  nx_bits_to_float(dt, (uint32_t)b.lo)),
              0};
  }
  return el{a.lo + b.lo, 0};
}

/* Folds: dst's element is the sum of the elements of x the Padded load
   reads from its index, left to right in C order of their taps, with no +0
   before the first; a narrow float and a 4-bit integer round each
   addition to the dtype; +0 where no tap reads it. */

/* The first tap at or after [j] of window axis [x] that reads padded
   position [P]: P - j dil = w step, 0 <= w < nwin; [taps] for none. */
__device__ __forceinline__ int64_t next_tap(const fold_axis &x, int64_t P,
                                            int64_t j) {
  for (; j < x.taps; j++) {
    int64_t w = P - j * x.dil;
    if (w < 0) return x.taps;
    if (w % x.step == 0 && w / x.step < x.nwin) return j;
  }
  return x.taps;
}

__device__ void fold(const fold_params &p) {
  MOVES_LOOP(e, p.numel) {
    int64_t i = e, base = p.x_off, P[NX_MOVE_RANK], tap[NX_MOVE_RANK];
    bool none = false;
#pragma unroll 1
    for (int a = p.rank - 1; a >= 0; a--) {
      const fold_axis &x = p.ax[a];
      int64_t c = divmod(i, x.n, p.small);
      int64_t pp = x.lo + c * (x.inter + 1);
      P[a] = pp;
      if (x.win < 0) {
        if (pp < 0 || pp >= x.pext) none = true;
        base += pp * x.sx;
      }
    }
    el acc = {0, 0};
    bool first = true;
    if (!none) {
#pragma unroll 1
      for (int a = 0; a < p.rank && !none; a++)
        if (p.ax[a].win >= 0) {
          tap[a] = next_tap(p.ax[a], P[a], 0);
          if (tap[a] == p.ax[a].taps) none = true;
        }
      while (!none) {
        int64_t at = base;
#pragma unroll 1
        for (int a = 0; a < p.rank; a++) {
          const fold_axis &x = p.ax[a];
          if (x.win >= 0)
            at += (P[a] - tap[a] * x.dil) / x.step * x.sw + tap[a] * x.st;
        }
        el v = ld(p.x, at, p.unit, p.count);
        acc = first ? v : add_el(p.dtype, acc, v);
        first = false;
        /* The odometer, the last windowed axis fastest. */
        int a = p.rank - 1;
        for (; a >= 0; a--) {
          const fold_axis &x = p.ax[a];
          if (x.win < 0) continue;
          tap[a] = next_tap(x, P[a], tap[a] + 1);
          if (tap[a] < x.taps) break;
          tap[a] = next_tap(x, P[a], 0);
        }
        if (a < 0) none = true;
      }
    }
    st(p.dst, p.dst_off + e, p.unit, p.count, acc);
  }
}

/* Scatters of one pass */

struct scatter_pos {
  int64_t pu, pi, pd, j;
};

/* Update e, in C order of the updates' shape: its positions and its index
   along the axis. [pd] is its target's position less the axis'. */
__device__ __forceinline__ scatter_pos scatter_at(const scatter_params &p,
                                                  int64_t e) {
  scatter_pos q = {p.upd_off, p.idx_off, p.dst_off, 0};
#pragma unroll 1
  for (int a = p.rank - 1; a >= 0; a--) {
    const scatter_axis &x = p.ax[a];
    int64_t c = divmod(e, x.n, p.small);
    q.pu += c * x.su;
    q.pi += c * x.si;
    if (a == p.axis)
      q.j = c;
    else
      q.pd += c * x.sd;
  }
  return q;
}

/* Slice s, in C order of the axes off the scattered one, at its index 0
   along it. */
__device__ __forceinline__ scatter_pos slice_at(const scatter_params &p,
                                                int64_t s) {
  scatter_pos q = {p.upd_off, p.idx_off, p.dst_off, 0};
#pragma unroll 1
  for (int a = p.rank - 1; a >= 0; a--) {
    if (a == p.axis) continue;
    const scatter_axis &x = p.ax[a];
    int64_t c = divmod(s, x.n, p.small);
    q.pu += c * x.su;
    q.pi += c * x.si;
    q.pd += c * x.sd;
  }
  return q;
}

/* The order of floats, -0 below +0, NaN apart. */
template <class F> __device__ __forceinline__ bool below(F a, F b) {
  return a < b || (a == b && signbit(a) && !signbit(b));
}

/* The canonical quiet NaN's code: the NaN a Max or Min stores where an
   update is one, a complex number's in each part. */
__device__ __forceinline__ u64 nan_code(int dt) {
  switch (dt) {
  case NX_FLOAT64:
  case NX_COMPLEX128: return 0x7FF8000000000000ull;
  case NX_FLOAT32:
  case NX_COMPLEX64: return 0x7FC00000u;
  case NX_FLOAT16: return 0x7E00u;
  case NX_BFLOAT16: return 0x7FC0u;
  case NX_FLOAT8_E4M3FN: return 0x7Fu;
  case NX_FLOAT8_E5M2: return 0x7Eu;
  }
  return 0;
}

/* The greater (with [mx], else the lesser) of element [a] and update [b],
   codes of [dt] of [bits] bits: a NaN target stays, a NaN update makes the
   canonical NaN, so that the result is a function of the set. A boolean
   stores 0 or 1. */
__device__ u64 extreme(int dt, bool mx, u64 a, u64 b, int bits) {
  switch (dt) {
  case NX_FLOAT64: {
    double x = nx_bits_double(a), y = nx_bits_double(b);
    if (x != x) return a;
    if (y != y) return nan_code(dt);
    return (mx ? below(x, y) : below(y, x)) ? b : a;
  }
  case NX_FLOAT32:
  case NX_FLOAT16:
  case NX_BFLOAT16:
  case NX_FLOAT8_E4M3FN:
  case NX_FLOAT8_E5M2:
  case NX_FLOAT4_E2M1FN: {
    float x = dt == NX_FLOAT32 ? nx_bits_float((uint32_t)a)
                               : nx_bits_to_float(dt, (uint32_t)a);
    float y = dt == NX_FLOAT32 ? nx_bits_float((uint32_t)b)
                               : nx_bits_to_float(dt, (uint32_t)b);
    if (x != x) return a;
    if (y != y) return nan_code(dt);
    return (mx ? below(x, y) : below(y, x)) ? b : a;
  }
  case NX_BOOL: return mx ? ((a | b) != 0) : (a != 0 && b != 0);
  case NX_BIT: return mx ? (a | b) : (a & b);
  }
  if (nx_dtype_row_of(dt).kind == NX_KIND_SIGNED) {
    int64_t x = (int64_t)(a << (64 - bits)) >> (64 - bits);
    int64_t y = (int64_t)(b << (64 - bits)) >> (64 - bits);
    return (mx ? x < y : y < x) ? b : a;
  }
  return (mx ? a < b : b < a) ? b : a;
}

/* A complex number's parts, as doubles: float32 parts in its units of 32
   bits, float64 ones in its units of 64. */
__device__ __forceinline__ void parts(int unit, el v, double &re, double &im) {
  if (unit == 32) {
    re = nx_bits_float((uint32_t)v.lo);
    im = nx_bits_float((uint32_t)v.hi);
  } else {
    re = nx_bits_double(v.lo);
    im = nx_bits_double(v.hi);
  }
}

/* The complex extreme at element [g], under its lock: by the real part,
   then the imaginary one. */
__device__ void extreme_pair(const scatter_params &p, int64_t g, el u,
                             bool mx) {
  uint32_t *lock = p.locks + (g - p.dst_off);
  bool done = false;
  while (!done) {
    if (atomicCAS(lock, 0u, 1u) != 0u) continue;
    __threadfence();
    el v;
    if (p.unit == 32) {
      volatile uint32_t *q = (volatile uint32_t *)p.dst + 2 * g;
      v = el{q[0], q[1]};
    } else {
      volatile u64 *q = (volatile u64 *)p.dst + 2 * g;
      v = el{q[0], q[1]};
    }
    double xr, xi, yr, yi;
    parts(p.unit, v, xr, xi);
    parts(p.unit, u, yr, yi);
    bool xn = xr != xr || xi != xi, yn = yr != yr || yi != yi;
    if (!xn) {
      bool take;
      if (yn) {
        take = true;
        u = el{nan_code(p.dtype), nan_code(p.dtype)};
      } else {
        bool lt = below(xr, yr) || (!below(yr, xr) && below(xi, yi));
        bool gt = below(yr, xr) || (!below(xr, yr) && below(yi, xi));
        take = mx ? lt : gt;
      }
      if (take) {
        if (p.unit == 32) {
          volatile uint32_t *q = (volatile uint32_t *)p.dst + 2 * g;
          q[0] = (uint32_t)u.lo;
          q[1] = (uint32_t)u.hi;
        } else {
          volatile u64 *q = (volatile u64 *)p.dst + 2 * g;
          q[0] = u.lo;
          q[1] = u.hi;
        }
      }
    }
    __threadfence();
    atomicExch(lock, 0u);
    done = true;
  }
}

/* A compare-and-swap of element [g] of [unit] bits, at most 64, by [f]:
   a word of 64 bits, else the 32-bit word that holds it. */
template <class F>
__device__ __forceinline__ void cas_elem(void *dst, int64_t g, int unit, F f) {
  if (unit == 64) {
    u64 *w = (u64 *)dst + g, old = *w;
    for (;;) {
      u64 nv = f(old);
      if (nv == old) return;
      u64 prev = atomicCAS(w, old, nv);
      if (prev == old) return;
      old = prev;
    }
  }
  unsigned sh;
  unsigned *w = nx_word_at(dst, g, unit, sh);
  const unsigned mask = unit == 32 ? ~0u : ((1u << unit) - 1) << sh;
  unsigned old = *w;
  for (;;) {
    u64 e = (old & mask) >> sh;
    u64 ne = f(e);
    if (ne == e) return;
    unsigned nw = (old & ~mask) | (((unsigned)ne << sh) & mask);
    unsigned prev = atomicCAS(w, old, nw);
    if (prev == old) return;
    old = prev;
  }
}

/* x + s into element [g], rounded once to the dtype. */
template <class T>
__device__ __forceinline__ void add_final(const scatter_params &p, int64_t g,
                                          T s) {
  T x = load_val<T>(p.dst, g, p.dtype, p.unit);
  store_val<T>(p.dst, g, p.dtype, p.unit, x + s);
}

__device__ void scatter(const scatter_params &p) {
  const int unit = p.unit, count = p.count;
  MOVES_LOOP(e, p.numel_u) {
    scatter_pos q = scatter_at(p, e);
    int64_t at = p.idx[q.pi];
    if ((uint64_t)at >= (uint64_t)p.d) continue;
    int64_t g = q.pd + at * p.sda;
    switch (p.op) {
    case NX_SCATTER_SET:
      st(p.dst, g, unit, count, ld(p.upd, q.pu, unit, count));
      break;
    case NX_SCATTER_OWN:
      if (p.wide)
        atomicMax((u64 *)p.owner + (g - p.dst_off), (u64)q.j + 1);
      else
        atomicMax((unsigned *)p.owner + (g - p.dst_off), (unsigned)q.j + 1);
      break;
    case NX_SCATTER_OWNED: {
      u64 o = p.wide ? ((const u64 *)p.owner)[g - p.dst_off]
                     : ((const unsigned *)p.owner)[g - p.dst_off];
      if (o == (u64)q.j + 1)
        st(p.dst, g, unit, count, ld(p.upd, q.pu, unit, count));
      break;
    }
    case NX_SCATTER_ADD_INT: {
      u64 u = ld(p.upd, q.pu, unit, 1).lo;
      if (unit == 32)
        atomicAdd((unsigned *)p.dst + g, (unsigned)u);
      else if (unit == 64)
        atomicAdd((u64 *)p.dst + g, u);
      else
        cas_elem(p.dst, g, unit,
                 [&](u64 x) { return (x + u) & ((1ull << unit) - 1); });
      break;
    }
    case NX_SCATTER_MAX:
    case NX_SCATTER_MIN: {
      bool mx = p.op == NX_SCATTER_MAX;
      if (count == 2) {
        extreme_pair(p, g, ld(p.upd, q.pu, unit, 2), mx);
        break;
      }
      u64 u = ld(p.upd, q.pu, unit, 1).lo;
      if (p.dtype == NX_BIT) {
        unsigned sh, *w = nx_word_at(p.dst, g, 1, sh);
        if (mx && u) atomicOr(w, 1u << sh);
        if (!mx && !u) atomicAnd(w, ~(1u << sh));
        break;
      }
      int dt = p.dtype;
      cas_elem(p.dst, g, unit,
               [&](u64 x) { return extreme(dt, mx, x, u, unit); });
      break;
    }
    case NX_SCATTER_ADD_UNIQUE:
      st(p.dst, g, unit, count,
         add_el(p.dtype, ld(p.dst, g, unit, count),
                ld(p.upd, q.pu, unit, count)));
      break;
    }
  }
}

/* A float Add's position tree */

/* The position tree over leaves in position order, by a stack: [gap] is
   the level at which a leaf meets the one before, the highest bit in which
   their positions differ. Gaps on the stack decrease from bottom to top,
   so the top pair is always the next to meet. */
template <class T> struct tree_stack {
  T v[66];
  int g[66];
  int n;
  __device__ void init() { n = 0; }
  __device__ void push(T x, int gap) {
    while (n >= 2 && g[n - 1] < gap) {
      v[n - 2] = v[n - 2] + v[n - 1];
      n--;
    }
    g[n] = gap;
    v[n++] = x;
  }
  __device__ T total() {
    while (n >= 2) {
      v[n - 2] = v[n - 2] + v[n - 1];
      n--;
    }
    return v[0];
  }
};

#define NO_KEY (~0ull)

/* Sorts [n] keys, a power of two, ascending, unless they are. */
__device__ void sort_keys(u64 *key, int n) {
  bool ok = true;
  for (int q = threadIdx.x; q + 1 < n; q += blockDim.x)
    if (key[q] > key[q + 1]) ok = false;
  if (__syncthreads_and(ok)) return;
#pragma unroll 1
  for (int k = 2; k <= n; k <<= 1)
#pragma unroll 1
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int r = threadIdx.x; r < n / 2; r += blockDim.x) {
        int i = ((r & ~(j - 1)) << 1) | (r & (j - 1)), l = i | j;
        u64 a = key[i], b = key[l];
        if ((a > b) == ((i & k) == 0)) {
          key[i] = b;
          key[l] = a;
        }
      }
      __syncthreads();
    }
}

/* The block's exclusive sum of [mine], in thread order, and its total. */
__device__ int block_excl(int mine, int &total) {
  __shared__ int ws[MT / 32];
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, x = mine;
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) ws[w] = x;
  __syncthreads();
  int before = x - mine;
  total = 0;
  for (int u = 0; u < MT / 32; u++) {
    if (u < w) before += ws[u];
    total += ws[u];
  }
  __syncthreads();
  return before;
}

/* The next sorted key of the same target, or -1. */
__device__ __forceinline__ int16_t link(const u64 *key, int i, int n,
                                        int bits) {
  u64 k = key[i], k2 = i + 1 < n ? key[i + 1] : NO_KEY;
  return k != NO_KEY && k2 != NO_KEY && (k2 >> bits) == (k >> bits)
             ? (int16_t)(i + 1)
             : (int16_t)-1;
}

/* Level l of the tree: entry i takes the next of its target where both
   lie in one aligned block of 2^(l+1) positions, the left first. An entry
   is never both a left and a right one in a level, so the merges are
   independent. */
__device__ __forceinline__ bool merges(const u64 *key, const int16_t *nx,
                                       const uint8_t *alive, int i, int l,
                                       unsigned slots) {
  if (!alive[i] || nx[i] < 0) return false;
  unsigned qi = key[i] & slots, qn = key[nx[i]] & slots;
  return (qi >> (l + 1)) == (qn >> (l + 1));
}

/* The flag slot of the row whose first target is [g]: a row block's in row
   tiles, the target's otherwise. */
__device__ __forceinline__ u64 flag_slot(const scatter_params &p, u64 g) {
  if (p.last == 1) return g;
  return g / (u64)p.last * (u64)p.nc + ((g % (u64)p.last) >> p.lcols);
}

/* Marks a slot's tiles: 1 for one, 2 for several. */
__device__ __forceinline__ void mark(uint32_t *flag) {
  if (atomicCAS(flag, 0u, 1u) != 0u) atomicExch(flag, 2u);
}

template <class T> __device__ void scatter_tile(const scatter_params &p) {
  __shared__ u64 key[NX_SCATTER_TILE];
  __shared__ T va[NX_SCATTER_TILE];
  __shared__ int16_t nx[NX_SCATTER_TILE];
  __shared__ uint8_t alive[NX_SCATTER_TILE];
  const int per = NX_SCATTER_TILE / MT, tid = threadIdx.x;
  const int rb = msb64((u64)p.run);
  const unsigned slots = NX_SCATTER_TILE - 1;
  const scatter_axis &ax = p.ax[p.axis];
  for (int64_t t = blockIdx.x; t < p.tiles; t += gridDim.x) {
    int64_t jb = t / p.nsg, sg = t % p.nsg;
    /* Slot q holds slice sg group + q / run at position jb run + q % run. */
    for (int r = 0; r < per; r++) {
      int q = tid + r * MT;
      int64_t s = sg * p.group + (q >> rb), j = jb * p.run + (q & (p.run - 1));
      u64 k = NO_KEY;
      if (s < p.slices && j < p.m) {
        scatter_pos o = slice_at(p, s);
        int64_t at = p.idx[o.pi + j * ax.si];
        if ((uint64_t)at < (uint64_t)p.d) {
          k = ((u64)(o.pd + at * p.sda - p.dst_off) << NX_SCATTER_TILE_BITS) |
              q;
          va[q] = load_val<T>(p.upd, o.pu + j * ax.su, p.dtype, p.unit);
        }
      }
      key[q] = k;
    }
    __syncthreads();
    sort_keys(key, NX_SCATTER_TILE);
    for (int r = 0; r < per; r++) {
      int i = tid + r * MT;
      nx[i] = link(key, i, NX_SCATTER_TILE, NX_SCATTER_TILE_BITS);
      alive[i] = key[i] != NO_KEY;
    }
    __syncthreads();
    /* Values stay at their slots: entry i's is va[key[i] & slots]. */
#pragma unroll 1
    for (int l = 0; l < NX_SCATTER_TILE_BITS; l++) {
      for (int r = 0; r < per; r++) {
        int i = tid + r * MT;
        if (!merges(key, nx, alive, i, l, slots)) continue;
        int n = nx[i];
        va[key[i] & slots] = va[key[i] & slots] + va[key[n] & slots];
        nx[i] = nx[n];
        alive[n] = 0;
      }
      __syncthreads();
    }
    if (p.nb == 1) {
      for (int r = 0; r < per; r++) {
        int i = tid + r * MT;
        if (alive[i])
          add_final<T>(p, (int64_t)(key[i] >> NX_SCATTER_TILE_BITS) + p.dst_off,
                       va[key[i] & slots]);
      }
    } else {
      int mine = 0, total;
      for (int r = 0; r < per; r++) mine += alive[tid * per + r];
      int at = block_excl(mine, total);
      for (int r = 0; r < per; r++) {
        int i = tid * per + r;
        if (!alive[i]) continue;
        u64 g = key[i] >> NX_SCATTER_TILE_BITS;
        p.ent_g[t * p.est + at] = g;
        ((T *)p.ent_v)[t * p.est + at] = va[key[i] & slots];
        mark(p.tiles_of + g);
        at++;
      }
      if (tid == 0) p.ent_n[t] = total;
    }
    __syncthreads();
  }
}

/* Row tiles: NX_SCATTER_ROWS positions by 2^lcols columns, column c0 + c
   of the last axis at value (slot << lcols) + c. */
template <class T> __device__ void scatter_rows(const scatter_params &p) {
  __shared__ u64 key[NX_SCATTER_ROWS];
  __shared__ int64_t ub[NX_SCATTER_ROWS];
  __shared__ int16_t nx[NX_SCATTER_ROWS], slot[NX_SCATTER_ROWS];
  __shared__ uint8_t alive[NX_SCATTER_ROWS];
  __shared__ __align__(16) unsigned char raw[NX_SCATTER_ROW_BYTES];
  T *va = (T *)raw;
  const int lc = p.lcols, C = 1 << lc, tid = threadIdx.x;
  const int rb = msb64((u64)p.run), cells = NX_SCATTER_ROWS << lc;
  const unsigned slots = NX_SCATTER_ROWS - 1;
  const scatter_axis &ax = p.ax[p.axis], &last = p.ax[p.rank - 1];
  for (int64_t t = blockIdx.x; t < p.tiles; t += gridDim.x) {
    int64_t cb = t % p.nc, rest = t / p.nc, jb = rest / p.nsg,
            sg = rest % p.nsg, c0 = cb << lc;
    int ncol = (int)min((int64_t)C, p.last - c0);
    for (int q = tid; q < NX_SCATTER_ROWS; q += MT) {
      int64_t s = sg * p.group + (q >> rb), j = jb * p.run + (q & (p.run - 1));
      u64 k = NO_KEY;
      if (s < p.slices && j < p.m) {
        scatter_pos o = slice_at(p, s * p.last);
        int64_t at = p.idx[o.pi + j * ax.si];
        if ((uint64_t)at < (uint64_t)p.d) {
          int64_t g = o.pd + at * p.sda + c0 * last.sd - p.dst_off;
          k = ((u64)g << NX_SCATTER_ROWS_BITS) | q;
          ub[q] = o.pu + j * ax.su + c0 * last.su;
        }
      }
      key[q] = k;
    }
    __syncthreads();
    for (int e = tid; e < cells; e += MT) {
      int q = e >> lc, c = e & (C - 1);
      if (key[q] != NO_KEY && c < ncol)
        va[e] = load_val<T>(p.upd, ub[q] + c * last.su, p.dtype, p.unit);
    }
    __syncthreads();
    sort_keys(key, NX_SCATTER_ROWS);
    for (int i = tid; i < NX_SCATTER_ROWS; i += MT) {
      nx[i] = link(key, i, NX_SCATTER_ROWS, NX_SCATTER_ROWS_BITS);
      alive[i] = key[i] != NO_KEY;
    }
    __syncthreads();
#pragma unroll 1
    for (int l = 0; l < NX_SCATTER_ROWS_BITS; l++) {
      for (int e = tid; e < cells; e += MT) {
        int i = e >> lc, c = e & (C - 1);
        if (!merges(key, nx, alive, i, l, slots)) continue;
        unsigned qi = key[i] & slots, qn = key[nx[i]] & slots;
        va[(qi << lc) + c] = va[(qi << lc) + c] + va[(qn << lc) + c];
      }
      __syncthreads();
      for (int i = tid; i < NX_SCATTER_ROWS; i += MT) {
        if (!merges(key, nx, alive, i, l, slots)) continue;
        int n = nx[i];
        nx[i] = nx[n];
        alive[n] = 0;
      }
      __syncthreads();
    }
    if (p.nb == 1) {
      for (int e = tid; e < cells; e += MT) {
        int i = e >> lc, c = e & (C - 1);
        if (!alive[i] || c >= ncol) continue;
        int64_t g = (int64_t)(key[i] >> NX_SCATTER_ROWS_BITS) + c * last.sd;
        add_final<T>(p, g + p.dst_off, va[((key[i] & slots) << lc) + c]);
      }
    } else {
      bool live = tid < NX_SCATTER_ROWS && alive[tid];
      int total, at = block_excl(live, total);
      if (live) {
        u64 g = key[tid] >> NX_SCATTER_ROWS_BITS;
        slot[tid] = (int16_t)at;
        p.ent_g[t * p.est + at] = g;
        mark(p.tiles_of + flag_slot(p, g));
      }
      __syncthreads();
      for (int e = tid; e < cells; e += MT) {
        int i = e >> lc, c = e & (C - 1);
        if (!alive[i]) continue;
        ((T *)p.ent_v)[((t * p.est + slot[i]) << lc) + c] =
            va[((key[i] & slots) << lc) + c];
      }
      if (tid == 0) p.ent_n[t] = total;
    }
    __syncthreads();
  }
}

/* The columns of the row whose first target is [g]. */
__device__ __forceinline__ int row_cols(const scatter_params &p, u64 g) {
  return (int)min((int64_t)(1 << p.lcols),
                  p.last - (int64_t)(g % (u64)p.last));
}

/* A target in one tile stores x + S; the others' entries spill, keyed
   (target, tile), with their rows' indices: a thread a column of an
   entry. The spill order is the atomics', and the keys, which are
   distinct, carry what orders them. */
template <class T> __device__ void scatter_spill(const scatter_params &p) {
  const int lc = p.lcols, C = 1 << lc;
  for (int64_t t = blockIdx.x; t < p.tiles; t += gridDim.x) {
    int64_t n = (int64_t)p.ent_n[t] << lc;
    u64 jb = (u64)(t / p.nc / p.nsg);
    for (int64_t e = threadIdx.x; e < n; e += MT) {
      int64_t i = e >> lc, row = t * p.est + i;
      int c = (int)(e & (C - 1));
      u64 g = p.ent_g[row];
      if (p.tiles_of[flag_slot(p, g)] == 1) {
        if (c < row_cols(p, g))
          add_final<T>(p, (int64_t)g + c + p.dst_off,
                       ((const T *)p.ent_v)[(row << lc) + c]);
      } else if (c == 0) {
        u64 at = atomicAdd((u64 *)p.ctl, 1ull);
        p.keys[0][at] = (g << p.kb_bits) | jb;
        p.index[0][at] = (u64)row;
      }
    }
  }
}

/* The spilled keys' radix sort: a pass sorts by the 8 bits from [shift],
   stably, from keys[c] to keys[c ^ 1], c the pass's parity. hist counts
   each chunk's digits, scan turns each digit's counts into the chunks'
   offsets and its total, place moves each key and its index. */

__device__ __forceinline__ int pass_of(const scatter_params &p) {
  return (p.shift / 8) & 1;
}

__device__ void scatter_hist(const scatter_params &p) {
  __shared__ uint32_t count[256];
  const u64 S = p.ctl[0], nch = (S + NX_SCATTER_CHUNK - 1) / NX_SCATTER_CHUNK;
  const u64 *ks = (const u64 *)p.keys[pass_of(p)];
  const int tid = threadIdx.x;
  for (u64 c = blockIdx.x; c < nch; c += gridDim.x) {
    count[tid] = 0;
    __syncthreads();
    for (int r = 0; r < NX_SCATTER_CHUNK / MT; r++) {
      u64 i = c * NX_SCATTER_CHUNK + r * MT + tid;
      if (i < S) atomicAdd(&count[(ks[i] >> p.shift) & 255], 1u);
    }
    __syncthreads();
    p.hist[(u64)tid * nch + c] = count[tid];
    __syncthreads();
  }
}

__device__ void scatter_scan(const scatter_params &p) {
  __shared__ u64 warp_sum[MT / 32];
  const u64 S = p.ctl[0], nch = (S + NX_SCATTER_CHUNK - 1) / NX_SCATTER_CHUNK;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  for (u64 d = blockIdx.x; d < 256; d += gridDim.x) {
    u64 *h = (u64 *)p.hist + d * nch, in = 0;
    for (u64 c0 = 0; c0 < nch; c0 += MT) {
      u64 c = c0 + tid, v = c < nch ? h[c] : 0, x = v;
      for (int o = 1; o < 32; o <<= 1) {
        u64 y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
      }
      if (lane == 31) warp_sum[w] = x;
      __syncthreads();
      u64 before = in + x - v, all = 0;
      for (int u = 0; u < MT / 32; u++) {
        if (u < w) before += warp_sum[u];
        all += warp_sum[u];
      }
      if (c < nch) h[c] = before;
      in += all;
      __syncthreads();
    }
    if (tid == 0) p.ctl[1 + d] = in;
  }
}

/* A warp walks its NX_SCATTER_CHUNK / 8 keys of a chunk in order, 32 a
   round, so equal digits keep their order. */
__device__ void scatter_place(const scatter_params &p) {
  __shared__ uint32_t wcnt[MT / 32][256];
  __shared__ u64 base[256];
  const u64 S = p.ctl[0], nch = (S + NX_SCATTER_CHUNK - 1) / NX_SCATTER_CHUNK;
  const int c0 = pass_of(p), tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const u64 *ks = (const u64 *)p.keys[c0], *ps = (const u64 *)p.index[c0];
  u64 *kd = (u64 *)p.keys[c0 ^ 1], *pd = (u64 *)p.index[c0 ^ 1];
  const int rounds = NX_SCATTER_CHUNK / MT;
  if (tid == 0) {
    u64 s = 0;
    for (int d = 0; d < 256; d++) {
      base[d] = s;
      s += p.ctl[1 + d];
    }
  }
  __syncthreads();
  for (u64 c = blockIdx.x; c < nch; c += gridDim.x) {
    for (int d = lane; d < 256; d += 32) wcnt[w][d] = 0;
    __syncwarp();
#pragma unroll 1
    for (int r = 0; r < rounds; r++) {
      u64 i = c * NX_SCATTER_CHUNK + w * (NX_SCATTER_CHUNK / 8) + r * 32 + lane;
      unsigned d = i < S ? (unsigned)((ks[i] >> p.shift) & 255) : 256u;
      unsigned peers = __match_any_sync(0xffffffffu, d);
      if (d < 256 && lane == __ffs(peers) - 1) wcnt[w][d] += __popc(peers);
      __syncwarp();
    }
    __syncthreads();
    {
      uint32_t run = 0;
      for (int u = 0; u < MT / 32; u++) {
        uint32_t t2 = wcnt[u][tid];
        wcnt[u][tid] = run;
        run += t2;
      }
    }
    __syncthreads();
#pragma unroll 1
    for (int r = 0; r < rounds; r++) {
      u64 i = c * NX_SCATTER_CHUNK + w * (NX_SCATTER_CHUNK / 8) + r * 32 + lane;
      unsigned d = i < S ? (unsigned)((ks[i] >> p.shift) & 255) : 256u;
      unsigned peers = __match_any_sync(0xffffffffu, d);
      uint32_t rank = 0;
      if (d < 256) rank = wcnt[w][d] + __popc(peers & ((1u << lane) - 1));
      __syncwarp();
      if (d < 256 && lane == __ffs(peers) - 1) wcnt[w][d] += __popc(peers);
      __syncwarp();
      if (d < 256) {
        u64 at = base[d] + p.hist[(u64)d * nch + c] + rank;
        kd[at] = ks[i];
        pd[at] = ps[i];
      }
    }
    __syncthreads();
  }
}

/* The sorted keys' parity: the passes' count's. */
__device__ __forceinline__ int sorted_of(const scatter_params &p) {
  return ((p.key_bits + 7) / 8) & 1;
}

/* The tree across tiles, first level: each piece of 2^qb tiles of a
   target, a thread a column; the piece's end where it starts. */
template <class T> __device__ void scatter_tree1(const scatter_params &p) {
  const int64_t S = (int64_t)p.ctl[0];
  const int lc = p.lcols, C = 1 << lc, cur = sorted_of(p);
  const int qb = (p.kb_bits + 1) / 2;
  const u64 *ks = (const u64 *)p.keys[cur], kmask = (1ull << p.kb_bits) - 1;
  const u64 *ix = (const u64 *)p.index[cur];
  MOVES_LOOP(e, S << lc) {
    int64_t i = e >> lc;
    int c = (int)(e & (C - 1));
    u64 piece = ks[i] >> qb;
    if (i > 0 && (ks[i - 1] >> qb) == piece) continue;
    tree_stack<T> s;
    s.init();
    int64_t j = i;
    u64 prev = 0;
    for (; j < S && (ks[j] >> qb) == piece; j++) {
      u64 kj = ks[j] & kmask;
      s.push(((const T *)p.ent_v)[((int64_t)ix[j] << lc) + c],
             j == i ? 0 : msb64(prev ^ kj));
      prev = kj;
    }
    ((T *)p.piece)[(i << lc) + c] = s.total();
    if (c == 0) p.ent_g[i] = (u64)j;
  }
}

/* Second level: a target's pieces, then x + S. */
template <class T> __device__ void scatter_tree2(const scatter_params &p) {
  const int64_t S = (int64_t)p.ctl[0];
  const int lc = p.lcols, C = 1 << lc, cur = sorted_of(p);
  const int qb = (p.kb_bits + 1) / 2;
  const u64 *ks = (const u64 *)p.keys[cur], kmask = (1ull << p.kb_bits) - 1;
  MOVES_LOOP(e, S << lc) {
    int64_t i = e >> lc;
    int c = (int)(e & (C - 1));
    u64 g = ks[i] >> p.kb_bits;
    if (i > 0 && (ks[i - 1] >> p.kb_bits) == g) continue;
    if (c >= row_cols(p, g)) continue;
    tree_stack<T> s;
    s.init();
    u64 prev = 0;
    for (int64_t h = i; h < S && (ks[h] >> p.kb_bits) == g;
         h = (int64_t)p.ent_g[h]) {
      u64 pc = (ks[h] & kmask) >> qb;
      s.push(((const T *)p.piece)[(h << lc) + c],
             h == i ? 0 : msb64(prev ^ pc) + qb);
      prev = pc;
    }
    add_final<T>(p, (int64_t)g + c + p.dst_off, s.total());
  }
}

} // namespace nx_moves

/* Entries */

#define MOVE(name, ...)                                                       \
  extern "C" __global__ void __launch_bounds__(MT)                            \
      name(const __grid_constant__ move_params p) {                           \
    nx_moves::move(p);                                                        \
  }
#define RUNS(name, step)                                                      \
  extern "C" __global__ void __launch_bounds__(MT)                            \
      name(const __grid_constant__ runs_params p) {                           \
    nx_moves::runs_##step(p);                                                 \
  }
#define ASSEMBLE(name, ...)                                                   \
  extern "C" __global__ void __launch_bounds__(MT)                            \
      name(const __grid_constant__ assemble_params p) {                       \
    nx_moves::assemble(p);                                                    \
  }
#define FOLD(name, ...)                                                       \
  extern "C" __global__ void __launch_bounds__(MT)                            \
      name(const __grid_constant__ fold_params p) {                           \
    nx_moves::fold(p);                                                        \
  }
#define SCATTER(name, ...)                                                    \
  extern "C" __global__ void __launch_bounds__(MT)                            \
      name(const __grid_constant__ scatter_params p) {                        \
    nx_moves::scatter(p);                                                     \
  }
#define MOVES_f32 float
#define MOVES_f64 double
#define SCATTER_ADD(name, step, T)                                            \
  extern "C" __global__ void __launch_bounds__(MT)                            \
      name(const __grid_constant__ scatter_params p) {                        \
    nx_moves::scatter_##step<MOVES_##T>(p);                                   \
  }
#define SCATTER_SORT(name, step)                                              \
  extern "C" __global__ void __launch_bounds__(MT)                            \
      name(const __grid_constant__ scatter_params p) {                        \
    nx_moves::scatter_##step(p);                                              \
  }

#endif
