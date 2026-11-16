/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What nx.amd's device code is written with: HIP compiled with no HIP
   header, so the attributes are spelled out and a work-item's place is read
   from the compiler's builtins.

   A kernel takes its parameters as one argument of 64-bit words, which the
   host writes at the indices kernels.h names, and reads them through the
   kernarg pointer: a by-value array indexed at run time would otherwise be
   copied to scratch. The argument segment lies in host memory, and every
   group's read of it crosses PCIe, shared with no other group through L2:
   a kernel reads the few words it needs by scalar loads issued together,
   or copies the words it uses into LDS once a group ({!stage}), and never
   reads the segment in a loop. It reads no implicit argument, so it knows
   its group's size only as the constant it is compiled for, and its grid
   only from its words.

   No function but a kernel survives compilation: every device function is
   inlined, since a call keeps its frame in scratch. gen.py refuses an
   object that holds another. */

#ifndef NX_AMD_DEVICE_H
#define NX_AMD_DEVICE_H

#include <stdint.h>

#include "kernels.h"
#define KERNEL extern "C" __attribute__((global))
/* A kernel whose groups have [n] work-items. */
#define SIZED(n) __attribute__((amdgpu_flat_work_group_size(n, n)))
#define DEVICE static inline __attribute__((device, always_inline))
#define SHARED __attribute__((shared))
/* Global memory: a load or store through such a pointer is a global
   instruction, from a scalar base where it is one, where a generic
   pointer's is a flat one, which also counts against LDS's accesses. */
#define GLOBAL __attribute__((address_space(1)))
typedef GLOBAL uint8_t gbyte;

/* [n] / [d] for an unsigned 32-bit [n], by [d]'s word [m] from
   Plan.divisor: a multiply-high, an add and a shift, where RDNA4's
   division takes a reciprocal and its corrections. */
DEVICE uint32_t divide(uint32_t n, uint64_t m) {
  uint64_t t = ((uint64_t)n * (uint32_t)m) >> 32;
  return (uint32_t)((t + n) >> (m >> 32));
}

/* A kernel's argument: [n] words, read through {!words}. */
template <int n> struct Words {
  int64_t w[n];
};

/* The kernel's words. */
DEVICE const int64_t *words(void) {
  return (const int64_t *)__builtin_amdgcn_kernarg_segment_ptr();
}

/* The address word [i] holds: a buffer's address plus an offset, as rig
   writes it. */
template <typename T> DEVICE T *address(int i) {
  return (T *)(uintptr_t)words()[i];
}

/* [x] as the wave's first active work-item holds it: a value the same in
   every work-item, which the compiler then keeps in scalar registers. A
   word read from LDS ({!stage}) is a vector value until then. */
DEVICE int64_t uniform(int64_t x) {
  const uint32_t lo = (uint32_t)__builtin_amdgcn_readfirstlane((int)x);
  const uint32_t hi =
      (uint32_t)__builtin_amdgcn_readfirstlane((int)((uint64_t)x >> 32));
  return (int64_t)(((uint64_t)hi << 32) | lo);
}

/* Word [i] of [p] as the wave's first active work-item reads it
   ({!uniform}): every work-item reads the same words. */
DEVICE int64_t wd(const int64_t *p, int64_t i) { return uniform(p[i]); }
DEVICE uint64_t wd(const uint64_t *p, int64_t i) {
  return (uint64_t)uniform((int64_t)p[i]);
}
DEVICE int wi(const int64_t *p, int64_t i) { return (int)uniform(p[i]); }

/* Axes

   A kernel's index space is C order over its axes, the last varying
   fastest. The host writes each axis as a record (Plan.axis): its extent
   and divisor word at NX_AXIS_N and NX_AXIS_DIV, then a stride for each
   operand that walks it from NX_AXIS_STRIDES, and a family's own words
   after them. A small space, of fewer than 2^32 indices, divides an index
   in 32 bits by the divisor words, a multiply-high, an add and a shift
   an axis; any other space by 64-bit division. An axis whose extent the
   device derives has no divisor word: a small space divides by it in 32
   bits. Every index is at least 0, and below the space's count. */

/* The coordinate of index [i] along an axis of extent [n] and divisor
   word [m], [i] becoming the index over the axes before it. */
DEVICE int64_t split(int64_t &i, int64_t n, uint64_t m, bool small) {
  if (small) {
    const uint32_t l = (uint32_t)i, q = divide(l, m);
    i = q;
    return l - q * (uint32_t)n;
  }
  const uint64_t q = (uint64_t)i / (uint64_t)n;
  const int64_t c = i - (int64_t)q * n;
  i = (int64_t)q;
  return c;
}

/* Whether a space of [count] indices is small. */
DEVICE bool is_small(int64_t count) { return (uint64_t)count >> 32 == 0; }

/* {!split} along an axis without a divisor word, whose extent the device
   derives: in 32 bits by division where [small]. */
DEVICE int64_t split(int64_t &i, int64_t n, bool small) {
  if (small) {
    const uint32_t l = (uint32_t)i, b = (uint32_t)n, q = l / b;
    i = q;
    return l - q * b;
  }
  return split(i, n, 0, false);
}

/* {!split} along the axis whose record starts at [a]. */
DEVICE int64_t split(int64_t &i, const int64_t *a, bool small) {
  return split(i, a[NX_AXIS_N], (uint64_t)a[NX_AXIS_DIV], small);
}

/* Adds to [at] the positions of index [i] in [K] operands, over the space
   of [rank] axes whose records of [per] words start at [ax]: each axis's
   coordinate times operand k's stride at NX_AXIS_STRIDES + k. */
template <int K>
DEVICE void place(const int64_t *ax, int per, int rank, int64_t i, bool small,
                  int64_t (&at)[K]) {
#pragma unroll 1
  for (int a = rank - 1; a >= 0; a--) {
    const int64_t *r = ax + a * per;
    const int64_t c = split(i, r, small);
#pragma unroll
    for (int k = 0; k < K; k++) at[k] += c * r[NX_AXIS_STRIDES + k];
  }
}

/* The position of index [k], counted in C order, of the layout whose
   words start at [l] (NX_LAYOUT_*). [l]'s words must be the same in every
   work-item of the wave: they are read from its first active work-item
   into scalar registers, which keeps each divisor scalar, so a work-item
   whose layout differs gets the first one's. [k] may differ. */
DEVICE int64_t position(const int64_t *l, int64_t k) {
  const int rank = (int)uniform(l[NX_LAYOUT_RANK]);
  const bool small = uniform(l[NX_LAYOUT_SMALL]);
  int64_t at = uniform(l[NX_LAYOUT_AT]);
#pragma unroll 1
  for (int j = rank - 1; j >= 0; j--) {
    const int64_t *a = l + NX_LAYOUT_AXES + NX_LAYOUT_AXIS * j;
    const int64_t n = uniform(a[NX_AXIS_N]);
    const uint64_t m = (uint64_t)uniform(a[NX_AXIS_DIV]);
    at += split(k, n, m, small) * uniform(a[NX_AXIS_STRIDES]);
  }
  return at;
}

DEVICE uint32_t item(void) { return __builtin_amdgcn_workitem_id_x(); }
DEVICE uint32_t group_x(void) { return __builtin_amdgcn_workgroup_id_x(); }
DEVICE uint32_t group_y(void) { return __builtin_amdgcn_workgroup_id_y(); }
DEVICE uint32_t group_z(void) { return __builtin_amdgcn_workgroup_id_z(); }

/* Waits for the group's work-items, each then seeing the writes the others
   made to the group's LDS and, through the agent-scope fences, to global
   memory before it. */
DEVICE void barrier(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

/* Words */

/* The address word [i] of [w], the launch's words as {!stage} copies
   them, holds, in scalar registers. */
template <typename T> DEVICE T *address(const int64_t *w, int i) {
  return (T *)(uintptr_t)uniform(w[i]);
}

/* Copies the launch's first [n] words into [to], but those from [gap] to
   [gap_end], which the launch does not read, then waits for the group.
   The group's first [threads] work-items, [threads] at most the group's
   work-items, copy a word each in turn, so the words cross from the
   argument segment in one coalesced read a wave. Every work-item of the
   group calls it with the same values. */
DEVICE void stage(int64_t *to, int64_t n, int threads, int64_t gap = 0,
                  int64_t gap_end = 0) {
  const int64_t *w = words();
  const int64_t skip = gap_end - gap;
  if (item() < (uint32_t)threads)
    for (int64_t i = item(); i < n - skip; i += threads) {
      const int64_t j = i < gap ? i : i + skip;
      to[j] = w[j];
    }
  barrier();
}

/* {!stage}s the launch's words up to the layout at word [l], its last
   words: those before it, then the words of the layout's rank. */
DEVICE void stage_layout(int64_t *to, int l, int threads) {
  const int64_t rank = words()[l + NX_LAYOUT_RANK];
  stage(to, l + NX_LAYOUT_AXES + NX_LAYOUT_AXIS * rank, threads);
}

/* Tickets

   A ticket is a 32-bit word of the tickets, zero before a launch. Each of
   [n] groups takes it once, after storing what it computed; the last to
   take it stores it back to zero and reads what the others stored. Every
   work-item's release before the take publishes its stores at the agent's
   scope; every work-item's acquire after it, in the last group, makes the
   others' visible to it. No group waits on another. */
DEVICE bool last_of(uint32_t *ticket, uint32_t n) {
  static SHARED bool is_last;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (item() == 0) {
    is_last = __hip_atomic_fetch_add(ticket, 1u, __ATOMIC_ACQ_REL,
                                     __HIP_MEMORY_SCOPE_AGENT) == n - 1;
    if (is_last)
      __hip_atomic_store(ticket, 0u, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
  }
  barrier();
  if (!is_last) return false;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return true;
}

/* Whether the calling work-item, which alone stored what it counts on
   [ticket] of [n], is the last to take it: then it stores the ticket back
   to zero and sees what the others stored, as {!last_of} for a group. */
DEVICE bool last_alone(uint32_t *ticket, uint32_t n) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  if (__hip_atomic_fetch_add(ticket, 1u, __ATOMIC_ACQ_REL,
                             __HIP_MEMORY_SCOPE_AGENT) != n - 1)
    return false;
  __hip_atomic_store(ticket, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return true;
}

/* Arithmetic */

DEVICE int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }
DEVICE int64_t imax(int64_t a, int64_t b) { return a < b ? b : a; }
template <typename T> DEVICE T min(T a, T b) { return a < b ? a : b; }
template <typename T> DEVICE T max(T a, T b) { return a < b ? b : a; }

/* Lanes */

/* The work-item's lane in its wave of 32, and its wave in the group. */
DEVICE uint32_t lane(void) { return item() & 31; }
DEVICE uint32_t wave(void) { return item() >> 5; }

/* Orders a wave's LDS writes before its lanes' reads that follow. */
DEVICE void wave_barrier(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

/* The lanes of the wave whose [d], of 9 bits, is the lane's own: a radix
   digit's, or 256 for a lane of no digit. The wave's lanes all call it. */
DEVICE uint32_t peers_of(uint32_t d) {
  uint32_t peers = ~0u;
#pragma unroll
  for (int b = 0; b < 9; b++) {
    const bool bit = (d >> b) & 1;
    const uint32_t m = __builtin_amdgcn_ballot_w32(bit);
    peers &= bit ? m : ~m;
  }
  return peers;
}

/* [x] of lane [lane () - o] of the wave, any value where that is below 0,
   for a value of 4 or 8 bytes. The wave's lanes all call it. */
DEVICE uint32_t up32(uint32_t x, int o) {
  const int from = (int)lane() - o;
  return (uint32_t)__builtin_amdgcn_ds_bpermute((from < 0 ? 0 : from) << 2,
                                                (int)x);
}

template <class T> DEVICE T up(T x, int o) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "a value of 4 or 8 bytes");
  if constexpr (sizeof(T) == 4) {
    return __builtin_bit_cast(T, up32(__builtin_bit_cast(uint32_t, x), o));
  } else {
    const uint64_t u = __builtin_bit_cast(uint64_t, x);
    const uint64_t hi = up32((uint32_t)(u >> 32), o);
    return __builtin_bit_cast(T, hi << 32 | up32((uint32_t)u, o));
  }
}

/* Group scans */

/* A member function of the device, inlined as every function is. */
#define DEVICE_MEMBER __attribute__((device, always_inline))

template <class T> struct Max_op {
  DEVICE_MEMBER T operator()(T a, T b) const { return a > b ? a : b; }
};

template <class T> struct Add_op {
  DEVICE_MEMBER T operator()(T a, T b) const { return a + b; }
};

/* The exclusive combination by [op], of identity [id], of [v] over the
   group's [threads] work-items in item order, and in [total] the whole
   one: a scan within each wave by lane shuffles, then over the waves'
   combinations through LDS. [op] is associative. Every work-item of the
   group calls it. */
template <int threads, class T, class Op>
DEVICE T group_excl(T v, T id, const Op &op, T &total) {
  static_assert(threads % 32 == 0, "whole waves");
  static SHARED T sums[threads / 32];
  T x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const T y = up(x, o);
    if (lane() >= (uint32_t)o) x = op(y, x);
  }
  const T y = up(x, 1);
  const T in_wave = lane() == 0 ? id : y;
  barrier();
  if (lane() == 31) sums[wave()] = x;
  barrier();
  T before = id, all = id;
#pragma unroll
  for (int i = 0; i < threads / 32; i++) {
    const T s = sums[i];
    if ((uint32_t)i < wave()) before = op(before, s);
    all = op(all, s);
  }
  total = all;
  return op(before, in_wave);
}

/* Ranks

   A group of [threads] work-items, 256, ranks 8-bit digits, 256 for an
   item of no digit, which takes a bin after every other. */

template <int threads> struct Rank {
  static_assert(threads == 256, "a work-item a bin");
  uint32_t wave[threads / 32][257]; /* each wave's count, then offset */
  uint32_t start[257], count[257];
};

/* Ranks the first [cnt] of each work-item's digits [d], item j of lane l
   of wave w at w * cnt * 32 + j * 32 + l in the group's order: [r] gets
   each item's place in the stable order by digit. [R]'s count and start
   then hold each bin's items and first place. Every work-item calls it. A
   wave's lanes read a bin's count before its first lane of that digit
   stores it, in the order of their instructions. */
template <int threads, int N>
DEVICE void rank_digits(const int (&d)[N], int cnt, uint32_t (&r)[N],
                        Rank<threads> &R) {
  constexpr int waves = threads / 32, bins = 257;
  const uint32_t w = wave(), lt = (1u << lane()) - 1;
  barrier();
  for (int i = item(); i < waves * bins; i += threads) (&R.wave[0][0])[i] = 0;
  barrier();
#pragma unroll
  for (int j = 0; j < N; j++) {
    if (j < cnt) {
      const uint32_t m = peers_of((uint32_t)d[j]);
      const uint32_t base = R.wave[w][d[j]];
      __builtin_amdgcn_wave_barrier();
      if ((m & lt) == 0) R.wave[w][d[j]] = base + __builtin_popcount(m);
      __builtin_amdgcn_wave_barrier();
      r[j] = base + __builtin_popcount(m & lt);
    }
  }
  barrier();
  for (int b = item(); b < bins; b += threads) {
    uint32_t sum = 0;
    for (int i = 0; i < waves; i++) {
      const uint32_t c = R.wave[i][b];
      R.wave[i][b] = sum;
      sum += c;
    }
    R.count[b] = sum;
  }
  barrier();
  uint32_t all;
  const uint32_t s =
      group_excl<threads>(R.count[item()], 0u, Add_op<uint32_t>{}, all);
  R.start[item()] = s;
  if (item() == 0) R.start[256] = all;
  barrier();
#pragma unroll
  for (int j = 0; j < N; j++)
    if (j < cnt) r[j] += R.start[d[j]] + R.wave[w][d[j]];
}

/* Radix passes

   A pass of an LSD radix sort moves a sequence's items stably by one
   8-bit digit, a part of consecutive tiles of [threads] N items a group:
   radix_count counts a part's digits, and once the parts' counts are each
   part's first place in each bin, radix_move ranks each tile of the part
   in turn and moves its items there. A pass [P] gives:

   - [T], an item as LDS holds it;
   - [fetch(l, it)], item [l] of the sequence, false for none, which takes
     no place;
   - [digit(it)], an item's digit;
   - [stage(t0, valid, held)], whether it staged the [valid] items of the
     tile from [t0] into [held], which [fetch] then reads no more;
   - [store(at, it)], an item's store at its place [at]. */

/* Counts the digits of items [l0] to [l1] into the LDS bins [h]. */
template <int threads, class P>
DEVICE void radix_count(const P &p, int64_t l0, int64_t l1, uint32_t *h) {
  for (int64_t l = l0 + item(); l < l1; l += threads) {
    typename P::T it;
    if (p.fetch(l, it))
      __hip_atomic_fetch_add(&h[p.digit(it)], 1u, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_WORKGROUP);
  }
}

/* Moves tiles [t0] to [t1] of a sequence of [n] items to their places,
   [place] holding each bin's next one, which it advances. */
template <int threads, int N, class P>
DEVICE void radix_move(const P &p, int64_t t0, int64_t t1, int64_t n,
                       uint64_t *place, Rank<threads> &R, typename P::T *held) {
  typedef typename P::T T;
  constexpr int tile = threads * N;
  const int ln = (int)lane(), wv = (int)wave();
  for (int64_t t = t0; t < t1; t++) {
    const int64_t l0 = t * tile;
    const int valid = (int)imin(tile, n - l0);
    const bool staged = p.stage(l0, valid, held);
    if (staged) barrier();
    T it[N];
    int d[N];
    uint32_t r[N];
#pragma unroll
    for (int j = 0; j < N; j++) {
      const int l = wv * N * 32 + j * 32 + ln;
      it[j] = T{};
      d[j] = 256;
      if (l >= valid) continue;
      bool some = true;
      if (staged) it[j] = held[l];
      else some = p.fetch(l0 + l, it[j]);
      if (some) d[j] = p.digit(it[j]);
    }
    /* An item of no digit takes a place past the kept ones. */
    rank_digits(d, N, r, R);
#pragma unroll
    for (int j = 0; j < N; j++) held[r[j]] = it[j];
    barrier();
    const int kept = (int)R.start[256];
    for (int i = item(); i < kept; i += threads) {
      const T e = held[i];
      const int dd = p.digit(e);
      p.store((int64_t)place[dd] + i - R.start[dd], e);
    }
    barrier();
    place[item()] += R.count[item()];
    barrier();
  }
}

/* Floats

   RDNA4 documents no choice among an instruction's NaN operands, and the
   compiler may commute an add's or a multiply's operands, so a site that
   may combine two NaNs chooses its NaN itself. */

/* A combine's result [r], or, where [r] is a NaN, the first NaN of [a]
   then [b], else [r]: of an integer, [r]. */
template <class T> DEVICE T kept(T a, T b, T r) {
  return r == r ? r : a != a ? a : b != b ? b : r;
}

/* Atomics

   Of no order, at the agent's scope but lds_add's, the group's: each
   result is a function of the set of operations. */

template <typename T> DEVICE T a_and(T *p, T v) {
  return __hip_atomic_fetch_and(p, v, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T> DEVICE T a_or(T *p, T v) {
  return __hip_atomic_fetch_or(p, v, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T> DEVICE T a_add(T *p, T v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T> DEVICE T a_max(T *p, T v) {
  return __hip_atomic_fetch_max(p, v, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT);
}

DEVICE void lds_add(uint32_t *p, uint32_t v) {
  __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}

/* Elements */

/* The aligned 32-bit word holding element [p] of [bits] bits at [base],
   and the element's shift in it. The word may hold bytes outside the
   array: rig orders memory in 4-byte words (Rig.word), so work may write
   each whole word holding a byte of its buffer. */
DEVICE uint32_t *word_at(uintptr_t base, int64_t p, int bits,
                         unsigned &shift) {
  const uint64_t bit = (uint64_t)p * bits;
  const uintptr_t a = base + (bit >> 3);
  shift = (unsigned)((a & 3) * 8 + (bit & 7));
  return (uint32_t *)(a & ~(uintptr_t)3);
}

/* Stores the low [bits] bits of [v] as element [p] of [bits] bits, 1 or 4,
   at [base], by and then or on its word, which commute with other
   work-items' on its other bits. */
template <typename P>
DEVICE void store_bits(P *base, int64_t p, int bits, uint32_t v) {
  unsigned shift;
  uint32_t *w = word_at((uintptr_t)base, p, bits, shift);
  const uint32_t mask = ((1u << bits) - 1) << shift;
  a_and(w, ~mask);
  a_or(w, (v << shift) & mask);
}

#endif

