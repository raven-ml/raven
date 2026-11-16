/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Moves and their adjoints: gathers, range gathers and copies (move), a range gather's runs (runs_*), assemblies (assemble), folds
   (fold) and scatters (scatter*), with their words of kernels.h.

   A group first stages the words it reads into its LDS, each read once by
   one work-item, and its loops read them there. Its work-items stride over
   their work, [groups] groups of them, so the host sizes a grid by the work
   alone and no bit depends on it.

   An element is [count] units of [unit] bits, the launch's words: 1, 4, 8,
   16, 32 or 64 bits, its dtype's alignment, and two units for a complex
   number, whose parts align as one part does. Arithmetic (a scatter's Add,
   Max and Min, a fold) follows the dtype's code: float32, with the narrow
   floats decoded to it by nx_dtype.h, float64, complex64, complex128 and
   the integers. A float Add sums in float32 or float64; a complex one sums
   each part apart.

   Determinism. No float atomic anywhere, and no work-item waits on
   another's. Integer atomics only where the result is a function of the
   set of operations: counts, flags, the maximum of positions or of order
   keys, bitwise and/or of disjoint bits, integer adds that wrap, and
   compare-and-swaps that store an extreme under a total order, a NaN
   result canonical.

   A scatter's float Add sums a target's updates by the position tree: the
   complete binary tree over the positions along the scattered axis, an
   absent position taking no part, then adds that sum to the target last,
   x + S. The tree is a function of the positions alone, so where the
   kernels cut the axis into aligned blocks of positions changes no bit.

     axis of one tile: tile (or rows) sorts a tile's (target, slot) keys
       in LDS, merges them level by level, and stores x + S
     else, entries: the updates, or where a slice has at most a tile's
       targets, the tiles' entries, one a target and tile
     tally, pack: the tiles' entries as items, in tile order
     count, move, a pass each key byte: a stable radix sort of the
       entries by target alone
     sum, a launch each level of 2^LB positions: each target's run of
       entries by the tree, then x + S

   A tree's work-item sums its leaves through a stack, which lives in LDS:
   a local array indexed at run time would live in scratch. */

#include "device.h"
#include "dtypes.h"

#define MT NX_MOVE_THREADS
#define TT NX_TREE_THREADS

typedef unsigned long long u64;

/* Words */

/* The address the staged word [i] of [w] holds, which may differ by
   work-item, as an assembly's piece records do: device.h's address reads
   the wave's first work-item's. */
template <typename T> DEVICE T *ptr(const int64_t *w, int i) {
  return (T *)(uintptr_t)w[i];
}

/* The grid-stride loop of a launch of [groups] groups of [threads]. */
#define STRIDE(i, n, groups, threads)                                        \
  for (int64_t i = (int64_t)group_x() * (threads) + item(); i < (n);          \
       i += (int64_t)(groups) * (threads))

DEVICE int msb64(u64 x) { return 63 - __builtin_clzll(x); }
DEVICE int lsb64(u64 x) { return __builtin_ctzll(x); }

/* Stores [nv] at [p] if it holds [old], which becomes what it held. */
template <typename T> DEVICE bool a_cas(T *p, T &old, T nv) {
  return __hip_atomic_compare_exchange_strong(p, &old, nv, __ATOMIC_RELAXED,
                                              __ATOMIC_RELAXED,
                                              __HIP_MEMORY_SCOPE_AGENT);
}

/* Elements by unit */

/* An element: its first unit in [lo], its second in [hi]. */
struct el {
  u64 lo, hi;
};

/* Element [p] of [count] units of [unit] bits at [base]. */
DEVICE el ld(const void *base, int64_t p, int unit, int count) {
  el v = {0, 0};
  switch (unit) {
  case 1:
  case 4: {
    const uint64_t bit = (uint64_t)p * unit;
    const uint8_t byte = ((const gbyte *)base)[bit >> 3];
    v.lo = (byte >> (bit & 7)) & ((1u << unit) - 1);
    return v;
  }
  case 8: v.lo = ((const gbyte *)base)[p]; return v;
  case 16: v.lo = ((const GLOBAL uint16_t *)base)[p]; return v;
  case 32: {
    const GLOBAL uint32_t *q = (const GLOBAL uint32_t *)base + p * count;
    v.lo = q[0];
    if (count == 2) v.hi = q[1];
    return v;
  }
  default: {
    const GLOBAL u64 *q = (const GLOBAL u64 *)base + p * count;
    v.lo = q[0];
    if (count == 2) v.hi = q[1];
    return v;
  }
  }
}

/* A store of one element, a sub-byte one into the word that holds it. */
DEVICE void st(void *base, int64_t p, int unit, int count, el v) {
  switch (unit) {
  case 1:
  case 4: store_bits(base, p, unit, (uint32_t)v.lo); return;
  case 8: ((gbyte *)base)[p] = (uint8_t)v.lo; return;
  case 16: ((GLOBAL uint16_t *)base)[p] = (uint16_t)v.lo; return;
  case 32: {
    GLOBAL uint32_t *q = (GLOBAL uint32_t *)base + p * count;
    q[0] = (uint32_t)v.lo;
    if (count == 2) q[1] = (uint32_t)v.hi;
    return;
  }
  default: {
    GLOBAL u64 *q = (GLOBAL u64 *)base + p * count;
    q[0] = v.lo;
    if (count == 2) q[1] = v.hi;
    return;
  }
  }
}

/* Stores [bits] under [mask] into a word other work-items may change. */
DEVICE void merge(uint32_t *w, uint32_t bits, uint32_t mask) {
  if (mask == ~0u) {
    *w = bits;
    return;
  }
  if (mask == 0) return;
  a_and(w, ~mask);
  a_or(w, bits & mask);
}

/* A sub-byte destination, C-contiguous over [numel] elements of [unit]
   bits from [dst_off], written a 32-bit word a work-item, its elements in
   order: [f.at(e, v)] is false for none, else sets [v] to element e's
   bits. The edge words merge by and/or with the bits of other arrays,
   which rig orders with dst's (word_at). */
template <class F>
DEVICE void packed(void *dst, int64_t dst_off, int64_t numel, int unit,
                   int64_t groups, const F &f) {
  const int per = 32 / unit;
  const uint32_t one = (1u << unit) - 1;
  const uintptr_t base = (uintptr_t)dst, abase = base & ~(uintptr_t)3;
  const int64_t lead = (int64_t)(base - abase) * 8;
  const int64_t first = lead + dst_off * unit, end = first + numel * unit;
  const int64_t w0 = first / 32, w1 = (end + 31) / 32;
  STRIDE(t, w1 - w0, groups, MT) {
    const int64_t w = w0 + t, e0 = (32 * w - lead) / unit - dst_off;
    uint32_t bits = 0, mask = 0;
#pragma unroll 1
    for (int i = 0; i < per; i++) {
      const int64_t e = e0 + i;
      uint32_t v;
      if (e < 0 || e >= numel || !f.at(e, v)) continue;
      bits |= (v & one) << (i * unit);
      mask |= one << (i * unit);
    }
    merge((uint32_t *)abase + w, bits, mask);
  }
}

/* Moves: gathers, range gathers, copies, fills */

struct move_pos {
  int64_t pd, ps, pi, ca;
};

DEVICE move_pos move_at(const int64_t *w, int64_t e) {
  move_pos q = {w[NX_MOVE_DST_OFF], w[NX_MOVE_SRC_OFF], w[NX_MOVE_IDX_OFF],
                0};
  const int axis = (int)w[NX_MOVE_AXIS];
  const bool small = w[NX_MOVE_SMALL];
#pragma unroll 1
  for (int a = (int)w[NX_MOVE_RANK] - 1; a >= 0; a--) {
    const int64_t *x = w + NX_MOVE_AX + a * NX_MOVE_AXIS_WORDS;
    const int64_t c = split(e, x, small);
    q.pd += c * x[NX_MOVE_AXIS_SD];
    q.ps += c * x[NX_MOVE_AXIS_SS];
    q.pi += c * x[NX_MOVE_AXIS_SI];
    if (a == axis) q.ca = c;
  }
  return q;
}

/* The run of position [pos] in a row of the runs: the first whose maximum
   of the ends up to it exceeds [pos], [k] for none. */
DEVICE int64_t run_of(const int64_t *maxima, int64_t k, int64_t pos) {
  int64_t lo = 0, hi = k;
  while (lo < hi) {
    const int64_t mid = (lo + hi) >> 1;
    if (maxima[mid] > pos)
      hi = mid;
    else
      lo = mid + 1;
  }
  return lo;
}

DEVICE el move_value(const int64_t *w, const move_pos &q) {
  const el none = {0, 0};
  const int unit = (int)w[NX_MOVE_UNIT], count = (int)w[NX_MOVE_COUNT];
  const void *src = ptr<const void>(w, NX_MOVE_SRC);
  if (w[NX_MOVE_MODE] == NX_MOVE_COPY) return ld(src, q.ps, unit, count);
  const int64_t d = w[NX_MOVE_D], sa = w[NX_MOVE_SA];
  switch (w[NX_MOVE_MODE]) {
  case NX_MOVE_GATHER: {
    const int64_t at = ptr<const int64_t>(w, NX_MOVE_IDX)[q.pi];
    if ((uint64_t)at >= (uint64_t)d) return none;
    return ld(src, q.ps + at * sa, unit, count);
  }
  default: {
    /* q.pi is the row's first run, q.ca the position along the axis; the
       read position wraps as int64 does. */
    const int64_t k = w[NX_MOVE_K];
    const int64_t j = run_of(ptr<const int64_t>(w, NX_MOVE_MAXIMA) + q.pi, k,
                             q.ca);
    if (j == k) return none;
    const uint64_t e =
        j ? (uint64_t)ptr<const int64_t>(w, NX_MOVE_ENDS)[q.pi + j - 1] : 0;
    const uint64_t at =
        (uint64_t)ptr<const int64_t>(w, NX_MOVE_STARTS)[q.pi + j] +
        (uint64_t)q.ca - e;
    if (at >= (uint64_t)d) return none;
    return ld(src, q.ps + (int64_t)at * sa, unit, count);
  }
  }
}

struct move_bits {
  const int64_t *w;
  DEVICE_MEMBER bool at(int64_t e, uint32_t &v) const {
    v = (uint32_t)move_value(w, move_at(w, e)).lo;
    return true;
  }
};

/* Runs

   A run is [bytes] bytes of dst at [d], each the byte at the same place
   of [s], or zero where [s] is null; [ub], 1, 2, 4 or 8, divides the
   bytes and both addresses. Its chunks are NX_MOVE_CHUNK bytes of dst
   from [d], the last partial. A wave copies a chunk, where [s] and [d]
   lie alike modulo 16 its lanes NX_MOVE_CHUNK / 512 accesses of 16 bytes
   each, all loads issued before any store, and the units before and after
   them; else a unit each in turn. Arrays are reached as global memory,
   from a scalar base. */

typedef uint32_t v4 __attribute__((ext_vector_type(4)));

#define RUN_LANES 32
#define RUN_VECTORS (NX_MOVE_CHUNK / (16 * RUN_LANES))

/* Unit [i] of [ub] bytes from [s] to [d], zero where [s] is null. */
DEVICE void unit_copy(GLOBAL uint8_t *d, const GLOBAL uint8_t *s, int ub,
                      int64_t i) {
  switch (ub) {
  case 1: d[i] = s ? s[i] : 0; return;
  case 2:
    ((GLOBAL uint16_t *)d)[i] = s ? ((const GLOBAL uint16_t *)s)[i] : 0;
    return;
  case 4:
    ((GLOBAL uint32_t *)d)[i] = s ? ((const GLOBAL uint32_t *)s)[i] : 0;
    return;
  default:
    ((GLOBAL u64 *)d)[i] = s ? ((const GLOBAL u64 *)s)[i] : 0;
    return;
  }
}

/* Chunk [c] of a run of [bytes] at [d]: its bytes [lo, hi), the 16-byte
   accesses from [vlo] to [vhi], and the units before them, [lo, head), and
   after, [tail, hi), at most 15 bytes each. */
struct chunk {
  uintptr_t lo, hi, vlo, vhi, head, tail;
};

DEVICE chunk chunk_of(uintptr_t d, int64_t bytes, int64_t c) {
  chunk k;
  k.lo = d + (uintptr_t)c * NX_MOVE_CHUNK;
  const uintptr_t end = d + (uintptr_t)bytes;
  k.hi = k.lo + NX_MOVE_CHUNK < end ? k.lo + NX_MOVE_CHUNK : end;
  k.vlo = (k.lo + 15) & ~(uintptr_t)15;
  k.vhi = k.hi & ~(uintptr_t)15;
  k.head = k.vlo < k.hi ? k.vlo : k.hi;
  k.tail = k.vhi > k.head ? k.vhi : k.head;
  return k;
}

/* Chunk [c] of the run, by the calling wave. */
DEVICE void run_chunk(uintptr_t d, uintptr_t s, int64_t bytes, int ub,
                      int64_t c) {
  const uint32_t l = lane();
  const chunk k = chunk_of(d, bytes, c);
  const intptr_t gap = (intptr_t)(s - d);
  if (s && (gap & 15) != 0) {
    /* Unalike modulo 16: a unit a lane in turn. */
    GLOBAL uint8_t *dp = (GLOBAL uint8_t *)k.lo;
    const GLOBAL uint8_t *sp = (const GLOBAL uint8_t *)(k.lo + gap);
#pragma unroll 1
    for (int64_t i = l; i < (int64_t)(k.hi - k.lo) / ub; i += RUN_LANES)
      unit_copy(dp, sp, ub, i);
    return;
  }
  const int64_t nh = (int64_t)(k.head - k.lo) / ub;
  const int64_t nt = (int64_t)(k.hi - k.tail) / ub;
  if ((int64_t)l < nh)
    unit_copy((GLOBAL uint8_t *)k.lo,
              s ? (const GLOBAL uint8_t *)(k.lo + gap) : nullptr, ub, l);
  if ((int64_t)l >= 16 && (int64_t)l - 16 < nt)
    unit_copy((GLOBAL uint8_t *)k.tail,
              s ? (const GLOBAL uint8_t *)(k.tail + gap) : nullptr, ub,
              l - 16);
  const int64_t nv = k.vhi > k.vlo ? (int64_t)(k.vhi - k.vlo) / 16 : 0;
  GLOBAL v4 *dv = (GLOBAL v4 *)k.vlo;
  if (!s) {
#pragma unroll
    for (int i = 0; i < RUN_VECTORS; i++) {
      const int64_t v = l + i * RUN_LANES;
      if (v < nv) dv[v] = (v4){0, 0, 0, 0};
    }
    return;
  }
  if (nv == 0) return;
  /* Every lane loads, past [nv] the last vector again, so that the loads
     issue together, unguarded. */
  const GLOBAL v4 *sv = (const GLOBAL v4 *)(k.vlo + gap);
  v4 x[RUN_VECTORS];
#pragma unroll
  for (int i = 0; i < RUN_VECTORS; i++) {
    const int64_t v = l + i * RUN_LANES;
    x[i] = sv[v < nv ? v : nv - 1];
  }
#pragma unroll
  for (int i = 0; i < RUN_VECTORS; i++) {
    const int64_t v = l + i * RUN_LANES;
    if (v < nv) dv[v] = x[i];
  }
}

/* A move whose last axis is a run ([chunks] > 0): a wave a chunk of a
   run, the run's source found once, by the wave, in scalar registers. */
DEVICE void move_runs(const int64_t *w) {
  const int64_t chunks = uniform(w[NX_MOVE_CHUNKS]);
  const int last = (int)uniform(w[NX_MOVE_RANK]) - 1;
  const int64_t n =
      uniform(w[NX_MOVE_AX + last * NX_MOVE_AXIS_WORDS + NX_AXIS_N]);
  const bool small = uniform(w[NX_MOVE_SMALL]);
  const int unit = (int)uniform(w[NX_MOVE_UNIT]);
  const int count = (int)uniform(w[NX_MOVE_COUNT]);
  const int64_t mode = uniform(w[NX_MOVE_MODE]);
  const int64_t es = unit / 8 * count;
  const int64_t tasks = uniform(w[NX_MOVE_NUMEL]) / n * chunks;
  const int64_t waves = uniform(w[NX_MOVE_GROUPS]) * (MT / RUN_LANES);
  const uintptr_t dst = (uintptr_t)uniform(w[NX_MOVE_DST]);
  const uintptr_t src = (uintptr_t)uniform(w[NX_MOVE_SRC]);
  for (int64_t t = uniform((int64_t)group_x() * (MT / RUN_LANES) + wave());
       t < tasks; t += waves) {
    int64_t r = t;
    const int64_t c =
        split(r, chunks, (uint64_t)uniform(w[NX_MOVE_CHUNKS_DIV]), small);
    int64_t ps = uniform(w[NX_MOVE_SRC_OFF]), pi = uniform(w[NX_MOVE_IDX_OFF]);
    int64_t ca = 0, rest = r;
#pragma unroll 1
    for (int a = last - 1; a >= 0; a--) {
      const int64_t *x = w + NX_MOVE_AX + a * NX_MOVE_AXIS_WORDS;
      const int64_t k = split(rest, uniform(x[NX_AXIS_N]),
                              (uint64_t)uniform(x[NX_AXIS_DIV]), small);
      ps += k * uniform(x[NX_MOVE_AXIS_SS]);
      pi += k * uniform(x[NX_MOVE_AXIS_SI]);
      if (a == uniform(w[NX_MOVE_AXIS])) ca = k;
    }
    /* The run's source row, none for zeros. */
    bool none = false;
    if (mode != NX_MOVE_COPY) {
      const int64_t d = uniform(w[NX_MOVE_D]);
      uint64_t at;
      if (mode == NX_MOVE_GATHER)
        at = (uint64_t)uniform(address<const int64_t>(w, NX_MOVE_IDX)[pi]);
      else {
        const int64_t k = uniform(w[NX_MOVE_K]);
        const int64_t *maxima = address<const int64_t>(w, NX_MOVE_MAXIMA) + pi;
        int64_t lo = 0, hi = k;
#pragma unroll 1
        while (lo < hi) {
          const int64_t mid = (lo + hi) >> 1;
          if (uniform(maxima[mid]) > ca)
            hi = mid;
          else
            lo = mid + 1;
        }
        const int64_t j = lo;
        none = j == k;
        const uint64_t e =
            j && !none
                ? (uint64_t)uniform(
                      address<const int64_t>(w, NX_MOVE_ENDS)[pi + j - 1])
                : 0;
        at = none ? 0
                  : (uint64_t)uniform(
                        address<const int64_t>(w, NX_MOVE_STARTS)[pi + j]) +
                        (uint64_t)ca - e;
      }
      none = none || at >= (uint64_t)d;
      if (!none) ps += (int64_t)at * uniform(w[NX_MOVE_SA]);
    }
    const int64_t pd = uniform(w[NX_MOVE_DST_OFF]) + r * n;
    run_chunk(dst + (uintptr_t)(pd * es), none ? 0 : src + (uintptr_t)(ps * es),
              n * es, unit / 8, c);
  }
}

KERNEL SIZED(MT) void move(Words<NX_MOVE_WORDS>) {
  static SHARED int64_t w[NX_MOVE_WORDS];
  const int64_t *k = words();
  const int64_t n = NX_MOVE_AX + k[NX_MOVE_RANK] * NX_MOVE_AXIS_WORDS;
  if (k[NX_MOVE_MODE] == NX_MOVE_COPY)
    stage(w, n, MT, NX_MOVE_GATHER_WORDS, NX_MOVE_AX);
  else
    stage(w, n, MT);
  if (w[NX_MOVE_CHUNKS]) {
    move_runs(w);
    return;
  }
  void *dst = ptr<void>(w, NX_MOVE_DST);
  const int unit = (int)w[NX_MOVE_UNIT], count = (int)w[NX_MOVE_COUNT];
  const int64_t numel = w[NX_MOVE_NUMEL], groups = w[NX_MOVE_GROUPS];
  if (unit < 8) {
    packed(dst, w[NX_MOVE_DST_OFF], numel, unit, groups, move_bits{w});
    return;
  }
  STRIDE(e, numel, groups, MT) {
    const move_pos q = move_at(w, e);
    st(dst, q.pd, unit, count, move_value(w, q));
  }
}

/* A range gather's runs */

DEVICE void runs_row(const int64_t *w, int64_t row, int64_t &ps,
                     int64_t &pe) {
  const bool small = w[NX_RUNS_SMALL];
  ps = w[NX_RUNS_S_OFF];
  pe = w[NX_RUNS_E_OFF];
#pragma unroll 1
  for (int a = (int)w[NX_RUNS_LEAD] - 1; a >= 0; a--) {
    const int64_t *x = w + NX_RUNS_AX + a * NX_RUNS_AXIS_WORDS;
    const int64_t c = split(row, x, small);
    ps += c * x[NX_RUNS_AXIS_SS];
    pe += c * x[NX_RUNS_AXIS_SE];
  }
}

DEVICE int64_t runs_words(void) {
  return NX_RUNS_AX + words()[NX_RUNS_LEAD] * NX_RUNS_AXIS_WORDS;
}

/* A group a chunk of NX_RUNS_CHUNK runs of one row: the runs copied, their
   ends' running maximum within the chunk, the chunk's maximum. */
KERNEL SIZED(MT) void runs_chunk(Words<NX_RUNS_WORDS>) {
  static SHARED int64_t w[NX_RUNS_WORDS];
  stage(w, runs_words(), MT);
  const int per = NX_RUNS_CHUNK / MT;
  const int64_t rows = w[NX_RUNS_ROWS], k = w[NX_RUNS_K];
  const int64_t chunks = w[NX_RUNS_CHUNKS];
  const int64_t s_k = w[NX_RUNS_S_K], e_k = w[NX_RUNS_E_K];
  const int64_t *in_s = ptr<const int64_t>(w, NX_RUNS_IN_S);
  const int64_t *in_e = ptr<const int64_t>(w, NX_RUNS_IN_E);
  for (int64_t b = group_x(); b < rows * chunks; b += w[NX_RUNS_GROUPS]) {
    int64_t ps, pe;
    const int64_t row = b / chunks, c = b % chunks;
    runs_row(w, row, ps, pe);
    const int64_t j0 = c * NX_RUNS_CHUNK + item() * per;
    int64_t m = INT64_MIN;
    int64_t *S = ptr<int64_t>(w, NX_RUNS_STARTS) + row * k;
    int64_t *E = ptr<int64_t>(w, NX_RUNS_ENDS) + row * k;
    int64_t *M = ptr<int64_t>(w, NX_RUNS_MAXIMA) + row * k;
#pragma unroll 1
    for (int i = 0; i < per; i++) {
      const int64_t j = j0 + i;
      if (j >= k) break;
      S[j] = in_s[ps + j * s_k];
      const int64_t e = in_e[pe + j * e_k];
      E[j] = e;
      m = imax(m, e);
    }
    int64_t total;
    int64_t run =
        group_excl<MT>(m, (int64_t)INT64_MIN, Max_op<int64_t>{}, total);
#pragma unroll 1
    for (int i = 0; i < per; i++) {
      const int64_t j = j0 + i;
      if (j >= k) break;
      run = imax(run, E[j]);
      M[j] = run;
    }
    if (item() == 0 && chunks > 1)
      ptr<int64_t>(w, NX_RUNS_CARRY)[row * chunks + c] = total;
  }
}

/* A group a row: each chunk's maximum becomes the maximum of the chunks
   before it. */
KERNEL SIZED(MT) void runs_carry(Words<NX_RUNS_WORDS>) {
  static SHARED int64_t w[NX_RUNS_WORDS];
  stage(w, NX_RUNS_AX, MT);
  const int64_t chunks = w[NX_RUNS_CHUNKS];
  for (int64_t row = group_x(); row < w[NX_RUNS_ROWS];
       row += w[NX_RUNS_GROUPS]) {
    int64_t *c = ptr<int64_t>(w, NX_RUNS_CARRY) + row * chunks;
    int64_t in = INT64_MIN;
    for (int64_t t0 = 0; t0 < chunks; t0 += MT) {
      const int64_t t = t0 + item();
      int64_t total;
      const int64_t ex = group_excl<MT>(t < chunks ? c[t] : INT64_MIN,
                                    (int64_t)INT64_MIN, Max_op<int64_t>{},
                                    total);
      if (t < chunks) c[t] = imax(in, ex);
      in = imax(in, total);
    }
  }
}

KERNEL SIZED(MT) void runs_merge(Words<NX_RUNS_WORDS>) {
  static SHARED int64_t w[NX_RUNS_WORDS];
  stage(w, NX_RUNS_AX, MT);
  const int64_t k = w[NX_RUNS_K], chunks = w[NX_RUNS_CHUNKS];
  const int64_t *carry = ptr<const int64_t>(w, NX_RUNS_CARRY);
  int64_t *maxima = ptr<int64_t>(w, NX_RUNS_MAXIMA);
  STRIDE(i, w[NX_RUNS_ROWS] * k, w[NX_RUNS_GROUPS], MT) {
    const int64_t row = i / k, c = (i % k) / NX_RUNS_CHUNK;
    if (c == 0) continue;
    const int64_t v = carry[row * chunks + c];
    if (v > maxima[i]) maxima[i] = v;
  }
}

/* Assemblies */

DEVICE int record(const int64_t *w) {
  return NX_ASSEMBLE_HEAD + (int)w[NX_ASSEMBLE_RANK] * NX_ASSEMBLE_AXIS;
}

/* The pieces' records, after dst's extents. */
DEVICE const int64_t *table(const int64_t *w) {
  return w + NX_ASSEMBLE_DN + NX_ASSEMBLE_DN_AXIS * w[NX_ASSEMBLE_RANK];
}

/* Whether dst's element [lin], in C order, lies in the region of the piece
   whose record is [t]. */
DEVICE bool covers(const int64_t *w, const int64_t *t, int64_t lin) {
  const bool small = w[NX_ASSEMBLE_SMALL];
#pragma unroll 1
  for (int a = (int)w[NX_ASSEMBLE_RANK] - 1; a >= 0; a--) {
    const int64_t *x = t + NX_ASSEMBLE_HEAD + a * NX_ASSEMBLE_AXIS;
    const int64_t c =
        split(lin, w + NX_ASSEMBLE_DN + NX_ASSEMBLE_DN_AXIS * a, small);
    const int64_t n = x[NX_AXIS_N], step = x[NX_ASSEMBLE_AXIS_STEP];
    const int64_t d = c - x[NX_ASSEMBLE_AXIS_START];
    if (step > 0 ? d < 0 : d > 0) return false;
    if (d % step != 0) return false;
    if (d / step >= n) return false;
  }
  return true;
}

/* Whether no piece of the launch covers dst's element [lin], where
   [check]. */
DEVICE bool open(const int64_t *w, int64_t lin) {
  if (!w[NX_ASSEMBLE_CHECK]) return true;
  const int rec = record(w);
  const int64_t *tab = table(w);
#pragma unroll 1
  for (int r = 0; r < (int)w[NX_ASSEMBLE_N]; r++)
    if (covers(w, tab + r * rec, lin)) return false;
  return true;
}

struct fill_bits {
  const int64_t *w;
  DEVICE_MEMBER bool at(int64_t e, uint32_t &v) const {
    v = (uint32_t)w[NX_ASSEMBLE_FILL_LO];
    return open(w, e);
  }
};

/* [fill] where no piece covers, with [check]; else everywhere, the pieces'
   launches storing theirs after. */
DEVICE void assemble_fill(const int64_t *w) {
  int64_t numel = 1;
#pragma unroll 1
  for (int a = 0; a < (int)w[NX_ASSEMBLE_RANK]; a++)
    numel *= w[NX_ASSEMBLE_DN + NX_ASSEMBLE_DN_AXIS * a];
  void *dst = ptr<void>(w, NX_ASSEMBLE_DST);
  const int unit = (int)w[NX_ASSEMBLE_UNIT], count = (int)w[NX_ASSEMBLE_COUNT];
  const int64_t groups = w[NX_ASSEMBLE_GROUPS], off = w[NX_ASSEMBLE_DST_OFF];
  if (unit < 8) {
    packed(dst, off, numel, unit, groups, fill_bits{w});
    return;
  }
  const el v = {(u64)w[NX_ASSEMBLE_FILL_LO], (u64)w[NX_ASSEMBLE_FILL_HI]};
  STRIDE(e, numel, groups, MT) {
    if (open(w, e)) st(dst, off + e, unit, count, v);
  }
}

/* Element e of the launch's pieces one after the other, copied unless a
   later piece of the launch covers its place; later launches store
   after. */
DEVICE void assemble_pieces(const int64_t *w) {
  const int rec = record(w), n = (int)w[NX_ASSEMBLE_N];
  const int rank = (int)w[NX_ASSEMBLE_RANK];
  const bool small = w[NX_ASSEMBLE_SMALL];
  const int unit = (int)w[NX_ASSEMBLE_UNIT], count = (int)w[NX_ASSEMBLE_COUNT];
  void *dst = ptr<void>(w, NX_ASSEMBLE_DST);
  const int64_t *tab = table(w);
  STRIDE(e, w[NX_ASSEMBLE_TOTAL], w[NX_ASSEMBLE_GROUPS], MT) {
    int k = 0;
#pragma unroll 1
    for (int i = 1; i < n; i++)
      if (tab[i * rec + NX_ASSEMBLE_REC_PREFIX] <= e) k = i;
    const int64_t *t = tab + k * rec;
    int64_t i = e - t[NX_ASSEMBLE_REC_PREFIX];
    int64_t ps = t[NX_ASSEMBLE_REC_OFF], pd = t[NX_ASSEMBLE_REC_BASE];
    /* cs is dst's C-order stride along axis a. */
    int64_t cs = 1;
#pragma unroll 1
    for (int a = rank - 1; a >= 0; a--) {
      const int64_t *x = t + NX_ASSEMBLE_HEAD + a * NX_ASSEMBLE_AXIS;
      const int64_t c = split(i, x, small);
      ps += c * x[NX_ASSEMBLE_AXIS_SX];
      pd += c * x[NX_ASSEMBLE_AXIS_STEP] * cs;
      cs *= w[NX_ASSEMBLE_DN + NX_ASSEMBLE_DN_AXIS * a];
    }
    /* pd is the element's place in dst in C order, from its first. */
    bool hidden = false;
    uint64_t later = (uint64_t)t[NX_ASSEMBLE_REC_LATER];
#pragma unroll 1
    while (later && !hidden) {
      const int r = lsb64(later);
      later &= later - 1;
      hidden = covers(w, tab + r * rec, pd);
    }
    if (!hidden)
      st(dst, w[NX_ASSEMBLE_DST_OFF] + pd, unit, count,
         ld(ptr<const void>(t, NX_ASSEMBLE_REC_SRC), ps, unit, count));
  }
}

/* Assemblies by runs ([runs]): a wave a chunk of a row, the row's piece,
   places and spans found once, by the wave, in scalar registers. */

/* The launch's chunks of pieces' rows, each piece's from its [prefix]. */
DEVICE void assemble_copy_runs(const int64_t *w) {
  const int rank = (int)uniform(w[NX_ASSEMBLE_RANK]), last = rank - 1;
  const int n = (int)uniform(w[NX_ASSEMBLE_N]), rec = record(w);
  const bool small = uniform(w[NX_ASSEMBLE_SMALL]);
  const int ub = (int)uniform(w[NX_ASSEMBLE_UNIT]) / 8;
  const int64_t es = ub * uniform(w[NX_ASSEMBLE_COUNT]);
  const int64_t waves = uniform(w[NX_ASSEMBLE_GROUPS]) * (MT / RUN_LANES);
  const uintptr_t dst = (uintptr_t)uniform(w[NX_ASSEMBLE_DST]) +
                        (uintptr_t)(uniform(w[NX_ASSEMBLE_DST_OFF]) * es);
  const int64_t *tab = table(w);
  for (int64_t t = uniform((int64_t)group_x() * (MT / RUN_LANES) + wave());
       t < uniform(w[NX_ASSEMBLE_TOTAL]); t += waves) {
    int k = 0;
#pragma unroll 1
    for (int i = 1; i < n; i++)
      if (uniform(tab[i * rec + NX_ASSEMBLE_REC_PREFIX]) <= t) k = i;
    const int64_t *p = tab + k * rec;
    const int64_t *ax = p + NX_ASSEMBLE_HEAD;
    const int64_t len = uniform(ax[last * NX_ASSEMBLE_AXIS + NX_AXIS_N]);
    int64_t row = t - uniform(p[NX_ASSEMBLE_REC_PREFIX]);
    const int64_t c =
        split(row, uniform(p[NX_ASSEMBLE_REC_CHUNKS]),
              (uint64_t)uniform(p[NX_ASSEMBLE_REC_CHUNKS_DIV]), small);
    int64_t ps = uniform(p[NX_ASSEMBLE_REC_OFF]);
    int64_t pd = uniform(p[NX_ASSEMBLE_REC_BASE]);
    int64_t cs = uniform(w[NX_ASSEMBLE_DN + NX_ASSEMBLE_DN_AXIS * last]);
#pragma unroll 1
    for (int a = last - 1; a >= 0; a--) {
      const int64_t *x = ax + a * NX_ASSEMBLE_AXIS;
      const int64_t i = split(row, uniform(x[NX_AXIS_N]),
                              (uint64_t)uniform(x[NX_AXIS_DIV]), small);
      ps += i * uniform(x[NX_ASSEMBLE_AXIS_SX]);
      pd += i * uniform(x[NX_ASSEMBLE_AXIS_STEP]) * cs;
      cs *= uniform(w[NX_ASSEMBLE_DN + NX_ASSEMBLE_DN_AXIS * a]);
    }
    const uintptr_t src = (uintptr_t)uniform(p[NX_ASSEMBLE_REC_SRC]);
    run_chunk(dst + (uintptr_t)(pd * es), src + (uintptr_t)(ps * es),
              len * es, ub, c);
  }
}

/* Whether element [e] of a row lies in one of the [m] spans [s]. */
DEVICE bool spanned(const int64_t (*s)[2], int m, int64_t e) {
#pragma unroll 1
  for (int i = 0; i < m; i++)
    if (e >= s[i][0] && e < s[i][1]) return true;
  return false;
}

/* The fill's 16 bytes at [a], [a] 16-aligned and [row] the row's first
   byte: [lo], [hi] its units, of [ub] bytes, [count] an element. */
DEVICE v4 fill_vector(uint64_t lo, uint64_t hi, int ub, int count,
                      uintptr_t a, uintptr_t row) {
  switch (ub) {
  case 1: {
    const uint32_t b = (uint32_t)lo & 0xff;
    const uint32_t x = b * 0x01010101u;
    return (v4){x, x, x, x};
  }
  case 2: {
    const uint32_t h = (uint32_t)lo & 0xffff;
    const uint32_t x = h | h << 16;
    return (v4){x, x, x, x};
  }
  case 4: {
    if (count == 1) {
      const uint32_t x = (uint32_t)lo;
      return (v4){x, x, x, x};
    }
    const bool odd = ((a - row) / 4) & 1;
    const uint32_t x = (uint32_t)(odd ? hi : lo), y = (uint32_t)(odd ? lo : hi);
    return (v4){x, y, x, y};
  }
  default: {
    const bool odd = count == 2 && ((a - row) / 8) & 1;
    const uint64_t x = odd ? hi : lo, y = count == 1 ? lo : odd ? lo : hi;
    return (v4){(uint32_t)x, (uint32_t)(x >> 32), (uint32_t)y,
                (uint32_t)(y >> 32)};
  }
  }
}

/* The fill's unit [i] of [ub] bytes at [d], [row] the row's first byte. */
DEVICE void fill_unit(GLOBAL uint8_t *d, int64_t i, uint64_t lo, uint64_t hi,
                      int ub, int count, uintptr_t row) {
  const uint64_t v =
      count == 2 && (((uintptr_t)d + i * ub - row) / ub) & 1 ? hi : lo;
  switch (ub) {
  case 1: d[i] = (uint8_t)v; return;
  case 2: ((GLOBAL uint16_t *)d)[i] = (uint16_t)v; return;
  case 4: ((GLOBAL uint32_t *)d)[i] = (uint32_t)v; return;
  default: ((GLOBAL u64 *)d)[i] = v; return;
  }
}

/* The fill of dst's rows but the spans the launch's pieces cover, which
   the wave keeps in LDS. */
DEVICE void assemble_fill_runs(const int64_t *w) {
  static SHARED int64_t spans[MT / RUN_LANES][NX_ASSEMBLE_PIECES][2];
  const int rank = (int)uniform(w[NX_ASSEMBLE_RANK]), last = rank - 1;
  const int n = (int)uniform(w[NX_ASSEMBLE_N]), rec = record(w);
  const bool check = uniform(w[NX_ASSEMBLE_CHECK]);
  const bool small = uniform(w[NX_ASSEMBLE_SMALL]);
  const int ub = (int)uniform(w[NX_ASSEMBLE_UNIT]) / 8;
  const int count = (int)uniform(w[NX_ASSEMBLE_COUNT]);
  const int64_t es = ub * count;
  const uint64_t lo = (uint64_t)uniform(w[NX_ASSEMBLE_FILL_LO]);
  const uint64_t hi = (uint64_t)uniform(w[NX_ASSEMBLE_FILL_HI]);
  const int64_t *dn = w + NX_ASSEMBLE_DN;
  const int64_t len = uniform(dn[NX_ASSEMBLE_DN_AXIS * last]);
  const int64_t chunks = uniform(w[NX_ASSEMBLE_CHUNKS]);
  const uint64_t chunks_div = (uint64_t)uniform(w[NX_ASSEMBLE_CHUNKS_DIV]);
  int64_t runs = 1;
#pragma unroll 1
  for (int a = 0; a < last; a++) runs *= uniform(dn[NX_ASSEMBLE_DN_AXIS * a]);
  const int64_t waves = uniform(w[NX_ASSEMBLE_GROUPS]) * (MT / RUN_LANES);
  const uintptr_t dst = (uintptr_t)uniform(w[NX_ASSEMBLE_DST]) +
                        (uintptr_t)(uniform(w[NX_ASSEMBLE_DST_OFF]) * es);
  const int64_t *tab = table(w);
  const uint32_t l = lane();
  int64_t(*s)[2] = spans[wave()];
  for (int64_t t = uniform((int64_t)group_x() * (MT / RUN_LANES) + wave());
       t < runs * chunks; t += waves) {
    int64_t row = t;
    const int64_t c = split(row, chunks, chunks_div, small);
    /* The spans of the row the pieces cover, written by lane 0 once the
       wave has read the last row's. */
    wave_barrier();
    int m = 0;
#pragma unroll 1
    for (int k = 0; check && k < n; k++) {
      const int64_t *ax = tab + k * rec + NX_ASSEMBLE_HEAD;
      int64_t rest = row;
      bool in = true;
#pragma unroll 1
      for (int a = last - 1; a >= 0 && in; a--) {
        const int64_t *x = ax + a * NX_ASSEMBLE_AXIS;
        const int64_t *y = dn + NX_ASSEMBLE_DN_AXIS * a;
        const int64_t i = split(rest, uniform(y[NX_AXIS_N]),
                                (uint64_t)uniform(y[NX_AXIS_DIV]), small);
        const int64_t step = uniform(x[NX_ASSEMBLE_AXIS_STEP]);
        const int64_t d = i - uniform(x[NX_ASSEMBLE_AXIS_START]);
        in = (step > 0 ? d >= 0 : d <= 0) && d % step == 0 &&
             d / step < uniform(x[NX_AXIS_N]);
      }
      if (!in) continue;
      const int64_t *x = ax + last * NX_ASSEMBLE_AXIS;
      const int64_t start = uniform(x[NX_ASSEMBLE_AXIS_START]);
      const int64_t cnt = uniform(x[NX_AXIS_N]);
      const bool up = uniform(x[NX_ASSEMBLE_AXIS_STEP]) > 0;
      if (l == 0) {
        s[m][0] = up ? start : start - cnt + 1;
        s[m][1] = up ? start + cnt : start + 1;
      }
      m++;
    }
    wave_barrier();
    const uintptr_t r0 = dst + (uintptr_t)(row * len * es);
    const chunk k = chunk_of(r0, len * es, c);
    /* A unit's element is its byte's from the row's first over [es]. */
    const int64_t nh = (int64_t)(k.head - k.lo) / ub;
    const int64_t nt = (int64_t)(k.hi - k.tail) / ub;
    if ((int64_t)l < nh &&
        !spanned(s, m, (int64_t)(k.lo + l * ub - r0) / es))
      fill_unit((GLOBAL uint8_t *)k.lo, l, lo, hi, ub, count, r0);
    if ((int64_t)l >= 16 && (int64_t)l - 16 < nt &&
        !spanned(s, m, (int64_t)(k.tail + (l - 16) * ub - r0) / es))
      fill_unit((GLOBAL uint8_t *)k.tail, l - 16, lo, hi, ub, count, r0);
    const int64_t nv = k.vhi > k.vlo ? (int64_t)(k.vhi - k.vlo) / 16 : 0;
#pragma unroll 1
    for (int64_t v = l; v < nv; v += RUN_LANES) {
      const uintptr_t a = k.vlo + (uintptr_t)v * 16;
      const int64_t f = (int64_t)(a - r0) / es;
      const int64_t g = (int64_t)(a + 15 - r0) / es;
      /* Whether a span meets the vector's elements, and one holds them. */
      bool meets = false, within = false;
#pragma unroll 1
      for (int i = 0; i < m; i++) {
        const int64_t x0 = s[i][0], x1 = s[i][1];
        if (f < x1 && g >= x0) meets = true;
        if (f >= x0 && g < x1) within = true;
      }
      if (!meets)
        *(GLOBAL v4 *)a = fill_vector(lo, hi, ub, count, a, r0);
      else if (!within)
#pragma unroll 1
        for (int u = 0; u < 16 / ub; u++)
          if (!spanned(s, m, (int64_t)(a + u * ub - r0) / es))
            fill_unit((GLOBAL uint8_t *)a, u, lo, hi, ub, count, r0);
    }
  }
}

KERNEL SIZED(MT) void assemble(Words<NX_ASSEMBLE_WORDS>) {
  static SHARED int64_t w[NX_ASSEMBLE_WORDS];
  const int64_t *k = words();
  const int64_t rank = k[NX_ASSEMBLE_RANK];
  stage(w,
        NX_ASSEMBLE_DN + NX_ASSEMBLE_DN_AXIS * rank +
            k[NX_ASSEMBLE_N] * (NX_ASSEMBLE_HEAD + rank * NX_ASSEMBLE_AXIS),
        MT);
  const bool runs = w[NX_ASSEMBLE_RUNS];
  if (w[NX_ASSEMBLE_MODE] == NX_ASSEMBLE_FILL) {
    if (runs)
      assemble_fill_runs(w);
    else
      assemble_fill(w);
  } else if (runs)
    assemble_copy_runs(w);
  else
    assemble_pieces(w);
}

/* Arithmetic */

/* A float element as its format T, float32 (a narrow float decoded) or
   float64: of [unit] bits at [base]. */
template <class T>
DEVICE T load_val(const void *base, int64_t p, int dt, int unit) {
  if constexpr (sizeof(T) == 8)
    return ((const GLOBAL double *)base)[p];
  else if (dt == NX_FLOAT32)
    return ((const GLOBAL float *)base)[p];
  else
    return nx_bits_to_float(dt, (uint32_t)ld(base, p, unit, 1).lo);
}

template <class T>
DEVICE void store_val(void *base, int64_t p, int dt, int unit, T v) {
  if constexpr (sizeof(T) == 8)
    ((GLOBAL double *)base)[p] = v;
  else if (dt == NX_FLOAT32)
    ((GLOBAL float *)base)[p] = v;
  else
    st(base, p, unit, 1, el{narrow_code(dt, v), 0});
}

/* The sum of elements [a] and [b] of [dt], as Prog.Add: a float's in its
   format, each part of a complex number's apart, a narrow float's rounded
   to it, an integer's wrapping once stored at its width. */
DEVICE el add_el(int dt, el a, el b) {
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
    return el{narrow_code(dt, nx_bits_to_float(dt, (uint32_t)a.lo) +
                                  nx_bits_to_float(dt, (uint32_t)b.lo)),
              0};
  }
  return el{a.lo + b.lo, 0};
}

/* Folds: dst's element is the sum of the elements of x the Padded load
   reads from its index, left to right in C order of their taps, with no +0
   before the first; a narrow float and a 4-bit integer round each
   addition to the dtype; +0 where no tap reads it. A work-item keeps each
   window's padded position and tap in the group's dynamic LDS. */

extern SHARED int64_t fold_lds[];

/* The first tap at or after [j] of window axis [x] that reads padded
   position [P]: P - j dil = w step, 0 <= w < nwin; [taps] for none. */
DEVICE int64_t next_tap(const int64_t *x, int64_t P, int64_t j) {
  const int64_t taps = x[NX_FOLD_AXIS_TAPS], dil = x[NX_FOLD_AXIS_DIL];
  const int64_t step = x[NX_FOLD_AXIS_STEP], nwin = x[NX_FOLD_AXIS_NWIN];
  for (; j < taps; j++) {
    const int64_t v = P - j * dil;
    if (v < 0) return taps;
    if (v % step == 0 && v / step < nwin) return j;
  }
  return taps;
}

/* A window's words, in scalar registers. */
struct window {
  int32_t taps, dil, step, nwin;
  int64_t ms, sw, st;
};

/* The first tap at or after [j] of window [x] that reads padded position
   [P], its window's index in [q]; [x.taps] for none. In 31 bits. */
DEVICE int32_t next_few(const window &x, int32_t P, int32_t j, int32_t &q) {
#pragma unroll 1
  for (; j < x.taps; j++) {
    const int32_t v = P - j * x.dil;
    if (v < 0) return x.taps;
    const int32_t k = (int32_t)divide((uint32_t)v, (uint64_t)x.ms);
    if (k * x.step == v && k < x.nwin) {
      q = k;
      return j;
    }
  }
  return x.taps;
}

/* fold with [narrow] and W windows: as fold, each window's position, tap
   and index in registers, slot u the u-th windowed axis. */
template <int W> DEVICE void fold_few(const int64_t *w) {
  const int rank = (int)uniform(w[NX_FOLD_RANK]);
  const int dt = (int)uniform(w[NX_FOLD_DTYPE]);
  const int unit = (int)uniform(w[NX_FOLD_UNIT]);
  const int count = (int)uniform(w[NX_FOLD_COUNT]);
  const int64_t *xs = address<const int64_t>(w, NX_FOLD_X);
  void *dst = address<void>(w, NX_FOLD_DST);
  window win[W];
#pragma unroll 1
  for (int a = 0, k = 0; a < rank; a++) {
    const int64_t *x = w + NX_FOLD_AX + a * NX_FOLD_AXIS_WORDS;
    if (uniform(x[NX_FOLD_AXIS_WIN]) < 0) continue;
#pragma unroll
    for (int u = 0; u < W; u++)
      if (u == k)
        win[u] = window{(int32_t)uniform(x[NX_FOLD_AXIS_TAPS]),
                        (int32_t)uniform(x[NX_FOLD_AXIS_DIL]),
                        (int32_t)uniform(x[NX_FOLD_AXIS_STEP]),
                        (int32_t)uniform(x[NX_FOLD_AXIS_NWIN]),
                        uniform(x[NX_FOLD_AXIS_MS]),
                        uniform(x[NX_FOLD_AXIS_SW]),
                        uniform(x[NX_FOLD_AXIS_ST])};
    k++;
  }
  const int64_t off = uniform(w[NX_FOLD_DST_OFF]);
  STRIDE(e, w[NX_FOLD_NUMEL], w[NX_FOLD_GROUPS], NX_FOLD_THREADS) {
    uint32_t i = (uint32_t)e;
    int64_t base = uniform(w[NX_FOLD_X_OFF]);
    bool none = false;
    int32_t P[W];
#pragma unroll 1
    for (int a = rank - 1, k = W - 1; a >= 0; a--) {
      const int64_t *x = w + NX_FOLD_AX + a * NX_FOLD_AXIS_WORDS;
      int64_t l = i;
      const int32_t c = (int32_t)split(l, x, true);
      i = (uint32_t)l;
      const int32_t inter = (int32_t)x[NX_FOLD_AXIS_INTER];
      const int32_t pp = (int32_t)x[NX_FOLD_AXIS_LO] + c * (inter + 1);
      if (x[NX_FOLD_AXIS_WIN] >= 0) {
#pragma unroll
        for (int u = 0; u < W; u++)
          if (u == k) P[u] = pp;
        k--;
      } else {
        if (pp < 0 || pp >= (int32_t)x[NX_FOLD_AXIS_PEXT]) none = true;
        base += pp * x[NX_FOLD_AXIS_SX];
      }
    }
    el acc = {0, 0};
    bool first = true;
    int32_t tap[W], q[W];
#pragma unroll
    for (int u = 0; u < W; u++) {
      tap[u] = none ? win[u].taps : next_few(win[u], P[u], 0, q[u]);
      none = none || tap[u] == win[u].taps;
    }
#pragma unroll 1
    while (!none) {
      int64_t at = base;
#pragma unroll
      for (int u = 0; u < W; u++)
        at += q[u] * win[u].sw + tap[u] * win[u].st;
      const el y = ld(xs, at, unit, count);
      acc = first ? y : add_el(dt, acc, y);
      first = false;
      /* The odometer, the last windowed axis fastest. */
      bool carry = true;
#pragma unroll
      for (int u = W - 1; u >= 0; u--) {
        if (!carry) continue;
        tap[u] = next_few(win[u], P[u], tap[u] + 1, q[u]);
        carry = tap[u] == win[u].taps;
        if (carry) tap[u] = next_few(win[u], P[u], 0, q[u]);
      }
      none = carry;
    }
    st(dst, off + e, unit, count, acc);
  }
}

KERNEL SIZED(NX_FOLD_THREADS) void fold(Words<NX_FOLD_WORDS>) {
  static SHARED int64_t w[NX_FOLD_WORDS];
  stage(w, NX_FOLD_AX + words()[NX_FOLD_RANK] * NX_FOLD_AXIS_WORDS,
        NX_FOLD_THREADS);
  if (w[NX_FOLD_NARROW]) switch (w[NX_FOLD_WINDOWS]) {
    case 1: fold_few<1>(w); return;
    case 2: fold_few<2>(w); return;
    case 3: fold_few<3>(w); return;
    }
  const int rank = (int)w[NX_FOLD_RANK], dt = (int)w[NX_FOLD_DTYPE];
  const int unit = (int)w[NX_FOLD_UNIT], count = (int)w[NX_FOLD_COUNT];
  const bool small = w[NX_FOLD_SMALL];
  const int64_t windows = w[NX_FOLD_WINDOWS];
  const int64_t *xs = ptr<const int64_t>(w, NX_FOLD_X);
  void *dst = ptr<void>(w, NX_FOLD_DST);
  /* Window v's padded position and tap, of this work-item. */
  int64_t *P = fold_lds + item(), *tap = fold_lds + windows * NX_FOLD_THREADS
                                         + item();
  const int T = NX_FOLD_THREADS;
  STRIDE(e, w[NX_FOLD_NUMEL], w[NX_FOLD_GROUPS], NX_FOLD_THREADS) {
    int64_t i = e, base = w[NX_FOLD_X_OFF];
    bool none = false;
#pragma unroll 1
    for (int a = rank - 1; a >= 0; a--) {
      const int64_t *x = w + NX_FOLD_AX + a * NX_FOLD_AXIS_WORDS;
      const int64_t c = split(i, x, small);
      const int64_t pp = x[NX_FOLD_AXIS_LO] + c * (x[NX_FOLD_AXIS_INTER] + 1);
      const int64_t v = x[NX_FOLD_AXIS_WIN];
      if (v >= 0)
        P[v * T] = pp;
      else {
        if (pp < 0 || pp >= x[NX_FOLD_AXIS_PEXT]) none = true;
        base += pp * x[NX_FOLD_AXIS_SX];
      }
    }
    el acc = {0, 0};
    bool first = true;
    if (!none) {
#pragma unroll 1
      for (int a = 0; a < rank && !none; a++) {
        const int64_t *x = w + NX_FOLD_AX + a * NX_FOLD_AXIS_WORDS;
        const int64_t v = x[NX_FOLD_AXIS_WIN];
        if (v < 0) continue;
        tap[v * T] = next_tap(x, P[v * T], 0);
        if (tap[v * T] == x[NX_FOLD_AXIS_TAPS]) none = true;
      }
#pragma unroll 1
      while (!none) {
        int64_t at = base;
#pragma unroll 1
        for (int a = 0; a < rank; a++) {
          const int64_t *x = w + NX_FOLD_AX + a * NX_FOLD_AXIS_WORDS;
          const int64_t v = x[NX_FOLD_AXIS_WIN];
          if (v < 0) continue;
          const int64_t t = tap[v * T];
          at += (P[v * T] - t * x[NX_FOLD_AXIS_DIL]) / x[NX_FOLD_AXIS_STEP] *
                    x[NX_FOLD_AXIS_SW] +
                t * x[NX_FOLD_AXIS_ST];
        }
        const el y = ld(xs, at, unit, count);
        acc = first ? y : add_el(dt, acc, y);
        first = false;
        /* The odometer, the last windowed axis fastest. */
        int a = rank - 1;
#pragma unroll 1
        for (; a >= 0; a--) {
          const int64_t *x = w + NX_FOLD_AX + a * NX_FOLD_AXIS_WORDS;
          const int64_t v = x[NX_FOLD_AXIS_WIN];
          if (v < 0) continue;
          tap[v * T] = next_tap(x, P[v * T], tap[v * T] + 1);
          if (tap[v * T] < x[NX_FOLD_AXIS_TAPS]) break;
          tap[v * T] = next_tap(x, P[v * T], 0);
        }
        if (a < 0) none = true;
      }
    }
    st(dst, w[NX_FOLD_DST_OFF] + e, unit, count, acc);
  }
}

/* Scatters of one pass */

struct scatter_pos {
  int64_t pu, pi, pd, j;
};

/* A scatter's words, its rank's axes included. */
DEVICE int64_t scatter_words(void) {
  return NX_SCATTER_AX + words()[NX_SCATTER_RANK] * NX_SCATTER_AXIS_WORDS;
}

DEVICE const int64_t *scatter_axis(const int64_t *w, int a) {
  return w + NX_SCATTER_AX + a * NX_SCATTER_AXIS_WORDS;
}

/* Update e, in C order of the updates' shape: its positions and its index
   along the axis. [pd] is its target's position less the axis'. */
DEVICE scatter_pos scatter_at(const int64_t *w, int64_t e) {
  scatter_pos q = {w[NX_SCATTER_UPD_OFF], w[NX_SCATTER_IDX_OFF],
                   w[NX_SCATTER_DST_OFF], 0};
  const int axis = (int)w[NX_SCATTER_AXIS];
  const bool small = w[NX_SCATTER_SMALL];
#pragma unroll 1
  for (int a = (int)w[NX_SCATTER_RANK] - 1; a >= 0; a--) {
    const int64_t *x = scatter_axis(w, a);
    const int64_t c = split(e, x, small);
    q.pu += c * x[NX_SCATTER_AXIS_SU];
    q.pi += c * x[NX_SCATTER_AXIS_SI];
    if (a == axis)
      q.j = c;
    else
      q.pd += c * x[NX_SCATTER_AXIS_SD];
  }
  return q;
}

/* Slice s, in C order of the axes off the scattered one, at its index 0
   along it. */
DEVICE scatter_pos slice_at(const int64_t *w, int64_t s) {
  scatter_pos q = {w[NX_SCATTER_UPD_OFF], w[NX_SCATTER_IDX_OFF],
                   w[NX_SCATTER_DST_OFF], 0};
  const int axis = (int)w[NX_SCATTER_AXIS];
  const bool small = w[NX_SCATTER_SMALL];
#pragma unroll 1
  for (int a = (int)w[NX_SCATTER_RANK] - 1; a >= 0; a--) {
    if (a == axis) continue;
    const int64_t *x = scatter_axis(w, a);
    const int64_t c = split(s, x, small);
    q.pu += c * x[NX_SCATTER_AXIS_SU];
    q.pi += c * x[NX_SCATTER_AXIS_SI];
    q.pd += c * x[NX_SCATTER_AXIS_SD];
  }
  return q;
}

/* The order of floats, -0 below +0, NaN apart. */
template <class F> DEVICE bool below(F a, F b) {
  return a < b ||
         (a == b && __builtin_signbit(a) && !__builtin_signbit(b));
}

/* The canonical quiet NaN's code: the NaN a Max or Min stores where an
   update is one, a complex number's in each part. */
DEVICE u64 nan_code(int dt) {
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
DEVICE u64 extreme(int dt, bool mx, u64 a, u64 b, int bits) {
  switch (dt) {
  case NX_FLOAT64: {
    const double x = nx_bits_double(a), y = nx_bits_double(b);
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
    const float x = dt == NX_FLOAT32 ? nx_bits_float((uint32_t)a)
                                     : nx_bits_to_float(dt, (uint32_t)a);
    const float y = dt == NX_FLOAT32 ? nx_bits_float((uint32_t)b)
                                     : nx_bits_to_float(dt, (uint32_t)b);
    if (x != x) return a;
    if (y != y) return nan_code(dt);
    return (mx ? below(x, y) : below(y, x)) ? b : a;
  }
  case NX_BOOL: return mx ? ((a | b) != 0) : (a != 0 && b != 0);
  case NX_BIT: return mx ? (a | b) : (a & b);
  }
  if (is_signed(dt)) {
    const int64_t x = (int64_t)(a << (64 - bits)) >> (64 - bits);
    const int64_t y = (int64_t)(b << (64 - bits)) >> (64 - bits);
    return (mx ? x < y : y < x) ? b : a;
  }
  return (mx ? a < b : b < a) ? b : a;
}

/* A compare-and-swap of element [g] of [unit] bits, at most 64, by
   [f.next]: a word of 64 bits, else the 32-bit word that holds it. */
template <class F>
DEVICE void cas_elem(void *dst, int64_t g, int unit, const F &f) {
  if (unit == 64) {
    u64 *w = (u64 *)dst + g, old = *w;
    for (;;) {
      const u64 nv = f.next(old);
      if (nv == old || a_cas(w, old, nv)) return;
    }
  }
  unsigned sh;
  uint32_t *w = word_at((uintptr_t)dst, g, unit, sh);
  const uint32_t mask = unit == 32 ? ~0u : ((1u << unit) - 1) << sh;
  uint32_t old = *w;
  for (;;) {
    const u64 e = (old & mask) >> sh;
    const u64 ne = f.next(e);
    if (ne == e) return;
    const uint32_t nw = (old & ~mask) | (((uint32_t)ne << sh) & mask);
    if (a_cas(w, old, nw)) return;
  }
}

struct add_wrap {
  u64 u;
  int unit;
  DEVICE_MEMBER u64 next(u64 x) const {
    return (x + u) & ((1ull << unit) - 1);
  }
};

struct extreme_of {
  int dt;
  bool mx;
  u64 u;
  int bits;
  DEVICE_MEMBER u64 next(u64 x) const { return extreme(dt, mx, x, u, bits); }
};

/* Complex extremes by keys

   A part's order key, of a float of [unit] bits: its bits mapped onto the
   integers in its order, -0 below +0, complemented for a Min, so that the
   greatest key is the extreme; never 0, which no update leaves, nor
   NAN_KEY, which a NaN update leaves. A target's extreme is the greatest
   real part's key, then the greatest imaginary part's among updates of
   that real part: a function of the set of updates. A complex64's two
   32-bit keys make one 64-bit key, the real part's above, whose greatest
   is that extreme; a real key of 0xFFFFFFFF comes only from a NaN, so
   NAN_KEY stays above every number's. */

#define NAN_KEY (~0ull)

DEVICE u64 order_key(u64 bits, int unit, bool mx) {
  const u64 mask = unit == 64 ? ~0ull : 0xFFFFFFFFull;
  const u64 sign = 1ull << (unit - 1);
  const u64 o = bits & sign ? ~bits & mask : bits | sign;
  return mx ? o : ~o & mask;
}

DEVICE u64 key_bits(u64 key, int unit, bool mx) {
  const u64 mask = unit == 64 ? ~0ull : 0xFFFFFFFFull;
  const u64 sign = 1ull << (unit - 1);
  const u64 o = mx ? key : ~key & mask;
  return o & sign ? o & ~sign : ~o & mask;
}

/* A complex64's key from its parts' bits. */
DEVICE u64 pair_key(el v, bool mx) {
  return order_key(v.lo, 32, mx) << 32 | order_key(v.hi, 32, mx);
}

DEVICE bool part_nan(u64 bits, int unit) {
  return unit == 64 ? nx_double_nan(nx_bits_double(bits))
                    : nx_float_nan(nx_bits_float((uint32_t)bits));
}

DEVICE bool el_nan(el v, int unit) {
  return part_nan(v.lo, unit) || part_nan(v.hi, unit);
}

/* x + s into element [g], rounded once to the dtype. */
template <class T>
DEVICE void add_final(const int64_t *w, int64_t g, T s) {
  void *dst = ptr<void>(w, NX_SCATTER_DST);
  const int dt = (int)w[NX_SCATTER_DTYPE], unit = (int)w[NX_SCATTER_UNIT];
  const T x = load_val<T>(dst, g, dt, unit);
  store_val<T>(dst, g, dt, unit, x + s);
}

/* Each target's last update from its packed word, a work-item a target. */
DEVICE void packed_store(const int64_t *w) {
  void *dst = ptr<void>(w, NX_SCATTER_DST);
  const u64 *words = ptr<const u64>(w, NX_SCATTER_OWNER);
  const int64_t nd = w[NX_SCATTER_NUMEL_D], off = w[NX_SCATTER_DST_OFF];
  const int unit = (int)w[NX_SCATTER_UNIT];
  STRIDE(t, nd, w[NX_SCATTER_GROUPS], MT) {
    const u64 k = words[t];
    if (k != 0) st(dst, off + t, unit, 1, el{k & 0xFFFFFFFFull, 0});
  }
}

/* Each target's complex extreme from its keys, a work-item a target. */
DEVICE void top_store(const int64_t *w) {
  void *dst = ptr<void>(w, NX_SCATTER_DST);
  const u64 *keys = ptr<const u64>(w, NX_SCATTER_OWNER);
  const int64_t nd = w[NX_SCATTER_NUMEL_D], off = w[NX_SCATTER_DST_OFF];
  const int unit = (int)w[NX_SCATTER_UNIT];
  const bool mx = w[NX_SCATTER_GREATER];
  STRIDE(t, nd, w[NX_SCATTER_GROUPS], MT) {
    const u64 kre = keys[t];
    if (kre == 0) continue;
    const el x = ld(dst, off + t, unit, 2);
    if (el_nan(x, unit)) continue;
    el v;
    if (kre == NAN_KEY) {
      const u64 nan = nan_code(unit == 64 ? NX_COMPLEX128 : NX_COMPLEX64);
      v = el{nan, nan};
    } else if (unit == 32) {
      if (pair_key(x, mx) >= kre) continue;
      v = el{key_bits(kre >> 32, 32, mx),
             key_bits(kre & 0xFFFFFFFFull, 32, mx)};
    } else {
      const u64 xr = order_key(x.lo, unit, mx), xi = order_key(x.hi, unit, mx);
      const u64 kim = keys[nd + t];
      if (xr > kre || (xr == kre && xi >= kim)) continue;
      v = el{key_bits(kre, unit, mx), key_bits(kim, unit, mx)};
    }
    st(dst, off + t, unit, 2, v);
  }
}

KERNEL SIZED(MT) void scatter(Words<NX_SCATTER_WORDS>) {
  static SHARED int64_t w[NX_SCATTER_WORDS];
  stage(w, scatter_words(), MT, NX_SCATTER_TILED_WORDS, NX_SCATTER_AX);
  const int unit = (int)w[NX_SCATTER_UNIT], count = (int)w[NX_SCATTER_COUNT];
  const int op = (int)w[NX_SCATTER_OP], dt = (int)w[NX_SCATTER_DTYPE];
  const bool wide = w[NX_SCATTER_WIDE], mx = w[NX_SCATTER_GREATER];
  const int64_t d = w[NX_SCATTER_D], sda = w[NX_SCATTER_SDA];
  const int64_t off = w[NX_SCATTER_DST_OFF], nd = w[NX_SCATTER_NUMEL_D];
  void *dst = ptr<void>(w, NX_SCATTER_DST);
  const void *upd = ptr<const void>(w, NX_SCATTER_UPD);
  const int64_t *idx = ptr<const int64_t>(w, NX_SCATTER_IDX);
  void *owner = ptr<void>(w, NX_SCATTER_OWNER);
  if (op == NX_SCATTER_TOP_STORE) {
    top_store(w);
    return;
  }
  if (op == NX_SCATTER_PACKED_STORE) {
    packed_store(w);
    return;
  }
  STRIDE(e, w[NX_SCATTER_NUMEL_U], w[NX_SCATTER_GROUPS], MT) {
    const scatter_pos q = scatter_at(w, e);
    const int64_t at = idx[q.pi];
    if ((uint64_t)at >= (uint64_t)d) continue;
    const int64_t g = q.pd + at * sda;
    switch (op) {
    case NX_SCATTER_SET: st(dst, g, unit, count, ld(upd, q.pu, unit, count)); break;
    case NX_SCATTER_OWN:
      if (wide)
        a_max((u64 *)owner + (g - off), (u64)q.j + 1);
      else
        a_max((uint32_t *)owner + (g - off), (uint32_t)q.j + 1);
      break;
    case NX_SCATTER_OWN_PACKED:
      a_max((u64 *)owner + (g - off),
            ((u64)q.j + 1) << 32 | ld(upd, q.pu, unit, 1).lo);
      break;
    case NX_SCATTER_OWNED: {
      const u64 o = wide ? ((const u64 *)owner)[g - off]
                         : ((const uint32_t *)owner)[g - off];
      if (o == (u64)q.j + 1)
        st(dst, g, unit, count, ld(upd, q.pu, unit, count));
      break;
    }
    case NX_SCATTER_ADD_INT: {
      const u64 u = ld(upd, q.pu, unit, 1).lo;
      if (unit == 32)
        a_add((uint32_t *)dst + g, (uint32_t)u);
      else if (unit == 64)
        a_add((u64 *)dst + g, u);
      else
        cas_elem(dst, g, unit, add_wrap{u, unit});
      break;
    }
    case NX_SCATTER_MAX:
    case NX_SCATTER_MIN: {
      const bool greater = op == NX_SCATTER_MAX;
      const u64 u = ld(upd, q.pu, unit, 1).lo;
      if (dt == NX_BIT) {
        unsigned sh;
        uint32_t *word = word_at((uintptr_t)dst, g, 1, sh);
        if (greater && u) a_or(word, 1u << sh);
        if (!greater && !u) a_and(word, ~(1u << sh));
        break;
      }
      cas_elem(dst, g, unit, extreme_of{dt, greater, u, unit});
      break;
    }
    case NX_SCATTER_ADD_UNIQUE:
      st(dst, g, unit, count,
         add_el(dt, ld(dst, g, unit, count), ld(upd, q.pu, unit, count)));
      break;
    case NX_SCATTER_TOP_RE: {
      const el u = ld(upd, q.pu, unit, 2);
      const u64 k = el_nan(u, unit) ? NAN_KEY
                    : unit == 32    ? pair_key(u, mx)
                                    : order_key(u.lo, unit, mx);
      a_max((u64 *)owner + (g - off), k);
      break;
    }
    case NX_SCATTER_TOP_IM: {
      const el u = ld(upd, q.pu, unit, 2);
      const u64 kre = ((const u64 *)owner)[g - off];
      if (kre == NAN_KEY || order_key(u.lo, unit, mx) != kre) break;
      a_max((u64 *)owner + nd + (g - off), order_key(u.hi, unit, mx));
      break;
    }
    }
  }
}

/* A float Add's position tree */

#define NO_KEY (~0ull)

/* Sorts [n] keys, a power of two, ascending, unless they are. */
DEVICE void sort_keys(u64 *key, int n) {
  static SHARED int unsorted;
  barrier();
  if (item() == 0) unsorted = 0;
  barrier();
  for (int q = item(); q + 1 < n; q += MT)
    if (key[q] > key[q + 1]) unsorted = 1;
  barrier();
  if (!unsorted) return;
#pragma unroll 1
  for (int k = 2; k <= n; k <<= 1)
#pragma unroll 1
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int r = item(); r < n / 2; r += MT) {
        const int i = ((r & ~(j - 1)) << 1) | (r & (j - 1)), l = i | j;
        const u64 a = key[i], b = key[l];
        if ((a > b) == ((i & k) == 0)) {
          key[i] = b;
          key[l] = a;
        }
      }
      barrier();
    }
}

/* The next sorted key of the same target, or -1. */
DEVICE int16_t link(const u64 *key, int i, int n, int bits) {
  const u64 k = key[i], k2 = i + 1 < n ? key[i + 1] : NO_KEY;
  return k != NO_KEY && k2 != NO_KEY && (k2 >> bits) == (k >> bits)
             ? (int16_t)(i + 1)
             : (int16_t)-1;
}

/* Level l of the tree: entry i takes the next of its target where both
   lie in one aligned block of 2^(l+1) positions, the left first. An entry
   is never both a left and a right one in a level, so the merges are
   independent. */
DEVICE bool merges(const u64 *key, const int16_t *nx, const uint8_t *alive,
                   int i, int l, unsigned slots) {
  if (!alive[i] || nx[i] < 0) return false;
  const unsigned qi = key[i] & slots, qn = key[nx[i]] & slots;
  return (qi >> (l + 1)) == (qn >> (l + 1));
}

/* The key of the row tile entry whose first target is [g]: its row's
   index, then its block of columns. */
DEVICE u64 row_key(const int64_t *w, u64 g) {
  int64_t r = (int64_t)g;
  const int64_t c = split(r, w + NX_SCATTER_LAST, w[NX_SCATTER_SMALL]);
  return (u64)r * (u64)w[NX_SCATTER_NC] + ((u64)c >> w[NX_SCATTER_LCOLS]);
}

template <class T> DEVICE void scatter_tile(const int64_t *w) {
  static SHARED u64 key[NX_SCATTER_TILE];
  static SHARED T va[NX_SCATTER_TILE];
  static SHARED int16_t nx[NX_SCATTER_TILE];
  static SHARED uint8_t alive[NX_SCATTER_TILE];
  const int per = NX_SCATTER_TILE / MT, tid = item();
  const int64_t run = w[NX_SCATTER_RUN], nsg = w[NX_SCATTER_NSG];
  const int rb = msb64((u64)run);
  const unsigned slots = NX_SCATTER_TILE - 1;
  const int64_t *ax = scatter_axis(w, (int)w[NX_SCATTER_AXIS]);
  const int64_t *idx = ptr<const int64_t>(w, NX_SCATTER_IDX);
  const void *upd = ptr<const void>(w, NX_SCATTER_UPD);
  const int dt = (int)w[NX_SCATTER_DTYPE], unit = (int)w[NX_SCATTER_UNIT];
  const int64_t d = w[NX_SCATTER_D], sda = w[NX_SCATTER_SDA];
  const int64_t off = w[NX_SCATTER_DST_OFF], est = w[NX_SCATTER_EST];
  for (int64_t t = group_x(); t < w[NX_SCATTER_TILES];
       t += w[NX_SCATTER_GROUPS]) {
    const int64_t jb = t / nsg, sg = t % nsg;
    /* Slot q holds slice sg group + q / run at position jb run + q % run. */
    for (int r = 0; r < per; r++) {
      const int q = tid + r * MT;
      const int64_t s = sg * w[NX_SCATTER_GROUP] + (q >> rb);
      const int64_t j = jb * run + (q & (run - 1));
      u64 k = NO_KEY;
      if (s < w[NX_SCATTER_SLICES] && j < w[NX_SCATTER_M]) {
        const scatter_pos o = slice_at(w, s);
        const int64_t at = idx[o.pi + j * ax[NX_SCATTER_AXIS_SI]];
        if ((uint64_t)at < (uint64_t)d) {
          k = ((u64)(o.pd + at * sda - off) << NX_SCATTER_TILE_BITS) | q;
          va[q] = load_val<T>(upd, o.pu + j * ax[NX_SCATTER_AXIS_SU], dt, unit);
        }
      }
      key[q] = k;
    }
    barrier();
    sort_keys(key, NX_SCATTER_TILE);
    for (int r = 0; r < per; r++) {
      const int i = tid + r * MT;
      nx[i] = link(key, i, NX_SCATTER_TILE, NX_SCATTER_TILE_BITS);
      alive[i] = key[i] != NO_KEY;
    }
    barrier();
    /* Values stay at their slots: entry i's is va[key[i] & slots]. */
#pragma unroll 1
    for (int l = 0; l < NX_SCATTER_TILE_BITS; l++) {
      for (int r = 0; r < per; r++) {
        const int i = tid + r * MT;
        if (!merges(key, nx, alive, i, l, slots)) continue;
        const int n = nx[i];
        va[key[i] & slots] = va[key[i] & slots] + va[key[n] & slots];
        nx[i] = nx[n];
        alive[n] = 0;
      }
      barrier();
    }
    if (w[NX_SCATTER_NB] == 1) {
      for (int r = 0; r < per; r++) {
        const int i = tid + r * MT;
        if (alive[i])
          add_final<T>(w, (int64_t)(key[i] >> NX_SCATTER_TILE_BITS) + off,
                       va[key[i] & slots]);
      }
    } else {
      int mine = 0, total;
      for (int r = 0; r < per; r++) mine += alive[tid * per + r];
      int at = group_excl<MT>(mine, 0, Add_op<int>{}, total);
      for (int r = 0; r < per; r++) {
        const int i = tid * per + r;
        if (!alive[i]) continue;
        const u64 g = key[i] >> NX_SCATTER_TILE_BITS;
        ptr<u64>(w, NX_SCATTER_ENT_G)[t * est + at] = g;
        ptr<T>(w, NX_SCATTER_ENT_V)[t * est + at] = va[key[i] & slots];
        at++;
      }
      if (tid == 0) ptr<uint32_t>(w, NX_SCATTER_ENT_N)[t] = total;
    }
    barrier();
  }
}

/* Row tiles: NX_SCATTER_ROWS positions by 2^lcols columns, column c0 + c
   of the last axis at value (slot << lcols) + c. */
template <class T> DEVICE void scatter_rows(const int64_t *w) {
  static SHARED u64 key[NX_SCATTER_ROWS];
  static SHARED int64_t ub[NX_SCATTER_ROWS];
  static SHARED int16_t nx[NX_SCATTER_ROWS], slot[NX_SCATTER_ROWS];
  static SHARED uint8_t alive[NX_SCATTER_ROWS];
  static SHARED __attribute__((aligned(16))) unsigned char
      raw[NX_SCATTER_ROW_BYTES];
  T *va = (T *)raw;
  const int lc = (int)w[NX_SCATTER_LCOLS], C = 1 << lc, tid = item();
  const int64_t run = w[NX_SCATTER_RUN], nsg = w[NX_SCATTER_NSG];
  const int64_t nc = w[NX_SCATTER_NC], last_n = w[NX_SCATTER_LAST];
  const int rb = msb64((u64)run), cells = NX_SCATTER_ROWS << lc;
  const unsigned slots = NX_SCATTER_ROWS - 1;
  const int64_t *ax = scatter_axis(w, (int)w[NX_SCATTER_AXIS]);
  const int64_t *last = scatter_axis(w, (int)w[NX_SCATTER_RANK] - 1);
  const int64_t *idx = ptr<const int64_t>(w, NX_SCATTER_IDX);
  const void *upd = ptr<const void>(w, NX_SCATTER_UPD);
  const int dt = (int)w[NX_SCATTER_DTYPE], unit = (int)w[NX_SCATTER_UNIT];
  const int64_t d = w[NX_SCATTER_D], sda = w[NX_SCATTER_SDA];
  const int64_t off = w[NX_SCATTER_DST_OFF], est = w[NX_SCATTER_EST];
  for (int64_t t = group_x(); t < w[NX_SCATTER_TILES];
       t += w[NX_SCATTER_GROUPS]) {
    const int64_t cb = t % nc, rest = t / nc, jb = rest / nsg,
                  sg = rest % nsg, c0 = cb << lc;
    const int ncol = (int)imin((int64_t)C, last_n - c0);
    for (int q = tid; q < NX_SCATTER_ROWS; q += MT) {
      const int64_t s = sg * w[NX_SCATTER_GROUP] + (q >> rb);
      const int64_t j = jb * run + (q & (run - 1));
      u64 k = NO_KEY;
      if (s < w[NX_SCATTER_SLICES] && j < w[NX_SCATTER_M]) {
        const scatter_pos o = slice_at(w, s * last_n);
        const int64_t at = idx[o.pi + j * ax[NX_SCATTER_AXIS_SI]];
        if ((uint64_t)at < (uint64_t)d) {
          const int64_t g = o.pd + at * sda + c0 * last[NX_SCATTER_AXIS_SD] - off;
          k = ((u64)g << NX_SCATTER_ROWS_BITS) | q;
          ub[q] = o.pu + j * ax[NX_SCATTER_AXIS_SU] +
                  c0 * last[NX_SCATTER_AXIS_SU];
        }
      }
      key[q] = k;
    }
    barrier();
    for (int e = tid; e < cells; e += MT) {
      const int q = e >> lc, c = e & (C - 1);
      if (key[q] != NO_KEY && c < ncol)
        va[e] = load_val<T>(upd, ub[q] + c * last[NX_SCATTER_AXIS_SU], dt,
                            unit);
    }
    barrier();
    sort_keys(key, NX_SCATTER_ROWS);
    for (int i = tid; i < NX_SCATTER_ROWS; i += MT) {
      nx[i] = link(key, i, NX_SCATTER_ROWS, NX_SCATTER_ROWS_BITS);
      alive[i] = key[i] != NO_KEY;
    }
    barrier();
#pragma unroll 1
    for (int l = 0; l < NX_SCATTER_ROWS_BITS; l++) {
      for (int e = tid; e < cells; e += MT) {
        const int i = e >> lc, c = e & (C - 1);
        if (!merges(key, nx, alive, i, l, slots)) continue;
        const unsigned qi = key[i] & slots, qn = key[nx[i]] & slots;
        va[(qi << lc) + c] = va[(qi << lc) + c] + va[(qn << lc) + c];
      }
      barrier();
      for (int i = tid; i < NX_SCATTER_ROWS; i += MT) {
        if (!merges(key, nx, alive, i, l, slots)) continue;
        const int n = nx[i];
        nx[i] = nx[n];
        alive[n] = 0;
      }
      barrier();
    }
    if (w[NX_SCATTER_NB] == 1) {
      for (int e = tid; e < cells; e += MT) {
        const int i = e >> lc, c = e & (C - 1);
        if (!alive[i] || c >= ncol) continue;
        const int64_t g = (int64_t)(key[i] >> NX_SCATTER_ROWS_BITS) +
                          c * last[NX_SCATTER_AXIS_SD];
        add_final<T>(w, g + off, va[((key[i] & slots) << lc) + c]);
      }
    } else {
      const bool live = tid < NX_SCATTER_ROWS && alive[tid];
      int total;
      const int at = group_excl<MT>((int)live, 0, Add_op<int>{}, total);
      if (live) {
        const u64 g = key[tid] >> NX_SCATTER_ROWS_BITS;
        slot[tid] = (int16_t)at;
        ptr<u64>(w, NX_SCATTER_ENT_G)[t * est + at] = row_key(w, g);
      }
      barrier();
      for (int e = tid; e < cells; e += MT) {
        const int i = e >> lc, c = e & (C - 1);
        if (!alive[i]) continue;
        ptr<T>(w, NX_SCATTER_ENT_V)[((t * est + slot[i]) << lc) + c] =
            va[((key[i] & slots) << lc) + c];
      }
      if (tid == 0) ptr<uint32_t>(w, NX_SCATTER_ENT_N)[t] = total;
    }
    barrier();
  }
}

/* The columns of the row whose first target is [g]. */
DEVICE int row_cols(const int64_t *w, u64 g) {
  const int64_t last = w[NX_SCATTER_LAST];
  return (int)imin((int64_t)1 << w[NX_SCATTER_LCOLS],
                   last - (int64_t)(g % (u64)last));
}

/* The sort by target

   An item is an entry's key and its index among the entries. The first
   pass reads the entries in index order, which within a target is
   position order, and each pass is stable, so the sorted items keep each
   target's entries in position order. A narrow item holds the key above
   [pb] bits of index; a wide one, both in words of their own. */

struct Narrow {
  typedef u64 T;
  static constexpr int items = NX_SCATTER_ITEMS_N;
  static DEVICE_MEMBER T make(u64 k, u64 i, int pb) { return k << pb | i; }
  static DEVICE_MEMBER u64 key(T t, int pb) { return t >> pb; }
  static DEVICE_MEMBER u64 index(T t, int pb) {
    return t & ((1ull << pb) - 1);
  }
};

struct __attribute__((aligned(16))) pair {
  u64 k, i;
};

struct Wide {
  typedef pair T;
  static constexpr int items = NX_SCATTER_ITEMS_W;
  static DEVICE_MEMBER T make(u64 k, u64 i, int) { return pair{k, i}; }
  static DEVICE_MEMBER u64 key(T t, int) { return t.k; }
  static DEVICE_MEMBER u64 index(T t, int) { return t.i; }
};

/* The key of update [i], its target, or with rows of updates its
   target's row; false for an update out of range. */
DEVICE bool source(const int64_t *w, int64_t i, u64 &k) {
  const bool small = w[NX_SCATTER_SMALL];
  int64_t s = i;
  const int64_t j = split(s, w + NX_SCATTER_M, small);
  const int64_t last = w[NX_SCATTER_LAST];
  const scatter_pos o = slice_at(w, s * last);
  const int64_t *ax = scatter_axis(w, (int)w[NX_SCATTER_AXIS]);
  const int64_t at =
      ptr<const int64_t>(w, NX_SCATTER_IDX)[o.pi + j * ax[NX_SCATTER_AXIS_SI]];
  if ((uint64_t)at >= (uint64_t)w[NX_SCATTER_D]) return false;
  int64_t g = o.pd + at * w[NX_SCATTER_SDA] - w[NX_SCATTER_DST_OFF];
  if (last > 1) split(g, w + NX_SCATTER_LAST, small);
  k = (u64)g;
  return true;
}

/* The items a pass reads, the source's before the first, and writes:
   pass q writes items[q mod 2]. */
template <class I> DEVICE const typename I::T *items_in(const int64_t *w) {
  return ptr<const typename I::T>(
      w, w[NX_SCATTER_PASS] & 1 ? NX_SCATTER_ITEMS0 : NX_SCATTER_ITEMS1);
}

template <class I> DEVICE typename I::T *items_out(const int64_t *w) {
  return ptr<typename I::T>(
      w, w[NX_SCATTER_PASS] & 1 ? NX_SCATTER_ITEMS1 : NX_SCATTER_ITEMS0);
}

/* A pass's items: the updates in the first, else the sorted count, and
   their tiles of [tile] items cut into the parts. */
struct pass_span {
  int64_t n, parts, span, tiles;
};

/* Whether a pass reads the updates: the first, sorting the updates. The
   others read items, the tiles' packed ones in the first. */
DEVICE bool from_updates(const int64_t *w) {
  return w[NX_SCATTER_PASS] == 0 &&
         w[NX_SCATTER_FROM] == NX_SCATTER_FROM_UPDATES;
}

DEVICE pass_span span_of(const int64_t *w, int tile) {
  pass_span p;
  p.n = from_updates(w) ? w[NX_SCATTER_ENTRIES]
                        : (int64_t)ptr<const u64>(w, NX_SCATTER_CTL)[0];
  p.parts = w[NX_SCATTER_NPARTS];
  p.tiles = (p.n + tile - 1) / tile;
  p.span = (p.tiles + p.parts - 1) / p.parts;
  return p;
}

/* A pass as radix_count and radix_move take it: item [l] the update's
   in the first pass over the updates, else the items' in turn. */
template <class I> struct Key_pass {
  typedef typename I::T T;
  const int64_t *w;
  const T *in;
  T *out;
  int pb, sh;
  bool updates;

  DEVICE_MEMBER int digit(T t) const {
    return (int)((I::key(t, pb) >> sh) & 255);
  }
  DEVICE_MEMBER bool fetch(int64_t l, T &t) const {
    if (!updates) {
      t = in[l];
      return true;
    }
    u64 k;
    if (!source(w, l, k)) return false;
    t = I::make(k, (u64)l, pb);
    return true;
  }
  DEVICE_MEMBER bool stage(int64_t, int, T *) const { return false; }
  DEVICE_MEMBER void store(int64_t at, T t) const { out[at] = t; }
};

template <class I> DEVICE Key_pass<I> key_pass(const int64_t *w) {
  return {w, items_in<I>(w), items_out<I>(w), (int)w[NX_SCATTER_PB],
          8 * (int)w[NX_SCATTER_PASS], from_updates(w)};
}

/* The tiles of part [g] of [nparts]: [tiles] cut evenly. */
DEVICE void pack_span(const int64_t *w, int64_t g, int64_t &t0, int64_t &t1) {
  const int64_t n = w[NX_SCATTER_TILES], parts = w[NX_SCATTER_NPARTS];
  const int64_t span = (n + parts - 1) / parts;
  t0 = imin(n, g * span);
  t1 = imin(n, t0 + span);
}

/* Counts each part's tiles' entries into [counts]; the last part to take
   the ticket makes them each part's first place, and stores the sorted
   count. */
DEVICE void scatter_tally(const int64_t *w) {
  const uint32_t *ent_n = ptr<const uint32_t>(w, NX_SCATTER_ENT_N);
  u64 *counts = ptr<u64>(w, NX_SCATTER_COUNTS);
  uint32_t *ticket = ptr<uint32_t>(w, NX_SCATTER_TICKET);
  const int64_t parts = w[NX_SCATTER_NPARTS];
  const uint32_t d = item();
  for (int64_t g = group_x(); g < parts; g += w[NX_SCATTER_GROUPS]) {
    int64_t t0, t1;
    pack_span(w, g, t0, t1);
    u64 n = 0, all;
    for (int64_t t = t0 + d; t < t1; t += MT) n += ent_n[t];
    group_excl<MT>(n, 0ull, Add_op<u64>{}, all);
    if (d == 0) counts[g] = all;
    if (!last_of(ticket, (uint32_t)parts)) continue;
    const u64 c = d < parts ? counts[d] : 0;
    const u64 base = group_excl<MT>(c, 0ull, Add_op<u64>{}, all);
    if (d < parts) counts[d] = base;
    if (d == 0) ptr<u64>(w, NX_SCATTER_CTL)[0] = all;
  }
}

/* Writes each part's tiles' entries as items into [items1] from its first
   place, in tile order then slot order, an item's index its slot: MT
   tiles at a time, each entry's tile found among their first places. */
template <class I> DEVICE void scatter_pack(const int64_t *w) {
  static SHARED u64 first[MT];
  const uint32_t *ent_n = ptr<const uint32_t>(w, NX_SCATTER_ENT_N);
  const u64 *ent_g = ptr<const u64>(w, NX_SCATTER_ENT_G);
  const u64 *counts = ptr<const u64>(w, NX_SCATTER_COUNTS);
  typename I::T *out = ptr<typename I::T>(w, NX_SCATTER_ITEMS1);
  const int64_t est = w[NX_SCATTER_EST];
  const int pb = (int)w[NX_SCATTER_PB];
  const uint32_t d = item();
  for (int64_t g = group_x(); g < w[NX_SCATTER_NPARTS];
       g += w[NX_SCATTER_GROUPS]) {
    int64_t t0, t1;
    pack_span(w, g, t0, t1);
    u64 base = counts[g];
    for (int64_t c0 = t0; c0 < t1; c0 += MT) {
      const int64_t t = c0 + d;
      u64 all;
      const u64 at = group_excl<MT>(t < t1 ? (u64)ent_n[t] : 0ull, 0ull,
                                    Add_op<u64>{}, all);
      first[d] = at;
      barrier();
      for (u64 e = d; e < all; e += MT) {
        /* The last tile whose first place is at most [e] holds it. */
        int lo = 0, hi = MT - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) >> 1;
          if (first[mid] <= e) lo = mid;
          else hi = mid - 1;
        }
        const int64_t slot = (c0 + lo) * est + (int64_t)(e - first[lo]);
        out[base + e] = I::make(ent_g[slot], (u64)slot, pb);
      }
      base += all;
      barrier();
    }
  }
}

/* Counts each part's digits into [counts]; the last part to take the
   ticket makes them each part's first place for each digit: the places of
   the digits before, then of the parts before. A pass over the updates
   stores the sorted count. */
template <class I> DEVICE void scatter_count(const int64_t *w) {
  static SHARED uint32_t h[256];
  const pass_span p = span_of(w, MT * I::items);
  const int64_t tile = MT * I::items;
  const Key_pass<I> kp = key_pass<I>(w);
  u64 *counts = ptr<u64>(w, NX_SCATTER_COUNTS);
  uint32_t *ticket = ptr<uint32_t>(w, NX_SCATTER_TICKET);
  const uint32_t d = item();
  for (int64_t g = group_x(); g < p.parts; g += w[NX_SCATTER_GROUPS]) {
    h[d] = 0;
    barrier();
    radix_count<MT>(kp, imin(p.n, g * p.span * tile),
                    imin(p.n, (g + 1) * p.span * tile), h);
    barrier();
    counts[g * 256 + d] = h[d];
    if (!last_of(ticket, (uint32_t)p.parts)) continue;
    u64 run = 0;
    for (int64_t q = 0; q < p.parts; q++) {
      const u64 c = counts[q * 256 + d];
      counts[q * 256 + d] = run;
      run += c;
    }
    u64 all;
    const u64 base = group_excl<MT>(run, 0ull, Add_op<u64>{}, all);
    for (int64_t q = 0; q < p.parts; q++) counts[q * 256 + d] += base;
    if (kp.updates && d == 0) ptr<u64>(w, NX_SCATTER_CTL)[0] = all;
  }
}

/* Moves each part's items to their places, from its first place in each
   bin. */
template <class I> DEVICE void scatter_move(const int64_t *w) {
  typedef typename I::T T;
  constexpr int tile = MT * I::items;
  static SHARED Rank<MT> R;
  static SHARED T held[tile];
  static SHARED uint64_t place[256];
  const pass_span p = span_of(w, tile);
  const Key_pass<I> kp = key_pass<I>(w);
  const u64 *counts = ptr<const u64>(w, NX_SCATTER_COUNTS);
  for (int64_t g = group_x(); g < p.parts; g += w[NX_SCATTER_GROUPS]) {
    place[item()] = counts[g * 256 + item()];
    barrier();
    radix_move<MT, I::items>(kp, g * p.span, imin(p.tiles, (g + 1) * p.span),
                             p.n, place, R, held);
  }
}

/* A work-item's stack of the position tree over leaves in position order:
   [gap] is the level at which a leaf meets the one before, the highest bit
   in which their positions differ, counted from [low]. Gaps on the stack
   decrease from bottom to top, each distinct, so the top pair is always
   the next to meet, and an entry above the bottom is known by its level:
   [mask] holds the levels on the stack, the value of level l at
   [slab[l TT]], and the bottom's in [bottom]. */
template <class T> struct stack {
  T *slab;
  T bottom;
  u64 mask;
};

/* Merges the entries of the levels of [m], lowest first, each into the one
   below it, the left first. */
template <class T> DEVICE void meet(stack<T> &s, u64 m) {
  m &= s.mask;
  if (!m) return;
  s.mask &= ~m;
  int l = lsb64(m);
  T acc = s.slab[l * TT];
  m &= m - 1;
#pragma unroll 1
  while (m) {
    l = lsb64(m);
    acc = s.slab[l * TT] + acc;
    m &= m - 1;
  }
  if (s.mask) {
    l = lsb64(s.mask);
    s.slab[l * TT] = s.slab[l * TT] + acc;
  } else
    s.bottom = s.bottom + acc;
}

template <class T> DEVICE void push(stack<T> &s, T x, int gap) {
  meet(s, (1ull << gap) - 1);
  s.slab[gap * TT] = x;
  s.mask |= 1ull << gap;
}

template <class T> DEVICE T total(stack<T> &s) {
  meet(s, ~0ull);
  return s.bottom;
}

/* Run sums

   The sorted items hold each target's entries, a run, in position order.
   Each level of the position tree's blocks of 2^LB positions is a
   launch: level l sums each run's entries, at level 1, or its blocks of
   level l - 1, at higher levels, that share their positions' bits from
   LB l up, by a work-item at the first of them, through its stack, and
   keeps the sum there with the end of the block, or adds it to the
   target where it is the run's whole sum. A run of at most [whole]
   entries is summed whole at level 1 by the work-item at its first.

     level 1:   a b | c d e | f        blocks of 2^LB positions
     level 2:   (a+b) (c+(d+e)) | f    their blocks of 2^LB blocks
     ...        until one block holds the run: x + S

   A work-item takes an entry and a column of its row: the levels' marks
   and ends are its entry's, written by its column 0 alone, so another
   column reads a mark of this level or the one before, and reads the
   ends of the level before from their own array. */

#define LB NX_SCATTER_LEVEL_BITS

/* How a run's entries are found: its first target and columns, and an
   update's slice, its first index and positions. */
struct run {
  u64 g;
  int cols;
  int64_t base;
  scatter_pos o;
};

template <class I>
DEVICE run run_at(const int64_t *w, bool tiles, typename I::T e,
                     int pb) {
  run r = {0, 1, 0, {0, 0, 0, 0}};
  const u64 k = I::key(e, pb), last = (u64)w[NX_SCATTER_LAST];
  if (tiles) {
    if (last == 1) {
      r.g = k;
      return r;
    }
    int64_t row = (int64_t)k;
    const int64_t c = split(row, w + NX_SCATTER_NC, w[NX_SCATTER_SMALL]);
    r.g = (u64)row * last + ((u64)c << w[NX_SCATTER_LCOLS]);
    r.cols = row_cols(w, r.g);
    return r;
  }
  r.g = k * last;
  r.cols = (int)last;
  int64_t s = 0;
  if (w[NX_SCATTER_SLICES] > 1) {
    s = (int64_t)I::index(e, pb);
    split(s, w + NX_SCATTER_M, w[NX_SCATTER_SMALL]);
  }
  r.base = s * w[NX_SCATTER_M];
  r.o = slice_at(w, s * (int64_t)last);
  return r;
}

/* An entry's position along the axis: a tile entry's, its tile's. */
template <class I>
DEVICE u64 pos_of(const int64_t *w, bool tiles, const run &r,
                  typename I::T e, int pb) {
  int64_t i = (int64_t)I::index(e, pb);
  if (!tiles) return (u64)(i - r.base);
  split(i, w + NX_SCATTER_SPAN, w[NX_SCATTER_SMALL]);
  return (u64)i;
}

/* Column [c] of an entry's value. */
template <class T, class I>
DEVICE T value_of(const int64_t *w, bool tiles, const run &r,
                  typename I::T e, int pb, int c) {
  const int64_t i = (int64_t)I::index(e, pb);
  if (tiles)
    return ptr<const T>(w, NX_SCATTER_ENT_V)[(i << w[NX_SCATTER_LCOLS]) + c];
  const int64_t *ax = scatter_axis(w, (int)w[NX_SCATTER_AXIS]);
  const int64_t *last = scatter_axis(w, (int)w[NX_SCATTER_RANK] - 1);
  const int64_t at = r.o.pu + (i - r.base) * ax[NX_SCATTER_AXIS_SU] +
                     c * last[NX_SCATTER_AXIS_SU];
  return load_val<T>(ptr<const void>(w, NX_SCATTER_UPD), at,
                     (int)w[NX_SCATTER_DTYPE], (int)w[NX_SCATTER_UNIT]);
}

/* Whether the run of the entry [i], not its first, of key [k] holds at
   most [whole] entries. */
template <class I>
DEVICE bool short_run(const typename I::T *it, int64_t n, int64_t i, u64 k,
                      int pb, int64_t whole) {
  if (i >= whole && I::key(it[i - whole], pb) == k) return false;
  int64_t s = i - 1;
  while (s > 0 && I::key(it[s - 1], pb) == k) s--;
  return s + whole >= n || I::key(it[s + whole], pb) != k;
}

template <class T, class I> DEVICE void scatter_sum(const int64_t *w, T *slab) {
  typedef typename I::T It;
  const It *it = ptr<const It>(w, NX_SCATTER_ITEMS0);
  const int64_t n = (int64_t)ptr<const u64>(w, NX_SCATTER_CTL)[0];
  const int pb = (int)w[NX_SCATTER_PB], lc = (int)w[NX_SCATTER_LCOLS];
  const int lvl = (int)w[NX_SCATTER_LEVEL], sh = LB * (lvl - 1);
  const int hs = (int)imin(sh + LB, 63);
  const bool tiles = w[NX_SCATTER_FROM] == NX_SCATTER_FROM_TILES;
  const int64_t whole = lvl == 1 ? w[NX_SCATTER_WHOLE] : 0;
  const int64_t off = w[NX_SCATTER_DST_OFF];
  T *part = ptr<T>(w, NX_SCATTER_PART);
  const u64 *ends_in = ptr<const u64>(w, lvl & 1 ? NX_SCATTER_PLEN0 : NX_SCATTER_PLEN1);
  u64 *ends = ptr<u64>(w, lvl & 1 ? NX_SCATTER_PLEN1 : NX_SCATTER_PLEN0);
  uint8_t *mark = ptr<uint8_t>(w, NX_SCATTER_MARK);
  STRIDE(e, n << lc, w[NX_SCATTER_GROUPS], TT) {
    const int64_t i = e >> lc;
    const int c = (int)(e & ((1 << lc) - 1));
    if (lvl > 1 && mark[i] + 1 < lvl) continue;
    const It ei = it[i];
    const u64 k = I::key(ei, pb);
    const run r = run_at<I>(w, tiles, ei, pb);
    if (c >= r.cols) continue;
    const u64 p = pos_of<I>(w, tiles, r, ei, pb);
    const bool first = i == 0 || I::key(it[i - 1], pb) != k;
    const bool lead =
        first || pos_of<I>(w, tiles, r, it[i - 1], pb) >> hs != p >> hs;
    if (!lead) {
      if (lvl == 1 && c == 0) mark[i] = 0;
      continue;
    }
    if (whole && (first ? i + whole >= n || I::key(it[i + whole], pb) != k
                        : short_run<I>(it, n, i, k, pb, whole))) {
      if (c == 0) mark[i] = 0;
      if (!first) continue;
      stack<T> s = {slab, value_of<T, I>(w, tiles, r, ei, pb, c), 0};
      u64 prev = p;
      for (int64_t j = i + 1; j < n; j++) {
        const It ej = it[j];
        if (I::key(ej, pb) != k) break;
        const u64 pj = pos_of<I>(w, tiles, r, ej, pb);
        push(s, value_of<T, I>(w, tiles, r, ej, pb, c), msb64(prev ^ pj));
        prev = pj;
      }
      add_final<T>(w, (int64_t)r.g + c + off, total(s));
      continue;
    }
    const u64 low = (1ull << LB) - 1;
    stack<T> s = {slab,
                  lvl == 1 ? value_of<T, I>(w, tiles, r, ei, pb, c)
                           : part[(i << lc) + c],
                  0};
    u64 prev = (p >> sh) & low;
    int64_t j = lvl == 1 ? i + 1 : (int64_t)ends_in[i];
    while (j < n) {
      const It ej = it[j];
      if (I::key(ej, pb) != k) break;
      const u64 pj = pos_of<I>(w, tiles, r, ej, pb);
      if (pj >> hs != p >> hs) break;
      const u64 qj = (pj >> sh) & low;
      push(s,
           lvl == 1 ? value_of<T, I>(w, tiles, r, ej, pb, c)
                    : part[(j << lc) + c],
           msb64(prev ^ qj));
      prev = qj;
      j = lvl == 1 ? j + 1 : (int64_t)ends_in[j];
    }
    const T sum = total(s);
    if (first && (j == n || I::key(it[j], pb) != k)) {
      add_final<T>(w, (int64_t)r.g + c + off, sum);
      if (lvl == 1 && c == 0) mark[i] = 0;
      continue;
    }
    part[(i << lc) + c] = sum;
    if (c == 0) {
      ends[i] = (u64)j;
      mark[i] = (uint8_t)lvl;
    }
  }
}

/* Entries */

/* The tiles and a pass over the updates read the axes. */
#define SCATTER_ADD(name, step, T)                                            \
  KERNEL SIZED(MT) void name(Words<NX_SCATTER_WORDS>) {                       \
    static SHARED int64_t w[NX_SCATTER_WORDS];                                \
    stage(w, scatter_words(), MT);                                            \
    step<T>(w);                                                               \
  }

SCATTER_ADD(scatter_tile_f32, scatter_tile, float)
SCATTER_ADD(scatter_tile_f64, scatter_tile, double)
SCATTER_ADD(scatter_rows_f32, scatter_rows, float)
SCATTER_ADD(scatter_rows_f64, scatter_rows, double)
KERNEL SIZED(MT) void scatter_tally(Words<NX_SCATTER_WORDS>) {
  static SHARED int64_t w[NX_SCATTER_WORDS];
  stage(w, scatter_words(), MT);
  scatter_tally(w);
}
SCATTER_ADD(scatter_pack_n, scatter_pack, Narrow)
SCATTER_ADD(scatter_pack_w, scatter_pack, Wide)
SCATTER_ADD(scatter_count_n, scatter_count, Narrow)
SCATTER_ADD(scatter_count_w, scatter_count, Wide)
SCATTER_ADD(scatter_move_n, scatter_move, Narrow)
SCATTER_ADD(scatter_move_w, scatter_move, Wide)

#define SCATTER_SUM(name, T, I)                                               \
  KERNEL SIZED(TT) void name(Words<NX_SCATTER_WORDS>) {                       \
    static SHARED int64_t w[NX_SCATTER_WORDS];                                \
    static SHARED T slab[NX_SCATTER_LEVELS * TT];                             \
    stage(w, scatter_words(), TT);                                            \
    scatter_sum<T, I>(w, slab + item());                                      \
  }

SCATTER_SUM(scatter_sum_f32_n, float, Narrow)
SCATTER_SUM(scatter_sum_f32_w, float, Wide)
SCATTER_SUM(scatter_sum_f64_n, double, Narrow)
SCATTER_SUM(scatter_sum_f64_w, double, Wide)
