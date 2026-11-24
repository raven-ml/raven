/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What nx.metal's device code is written with: MSL 3.2, compiled offline
   with -fno-fast-math and -ffp-contract=off, and the pragma below, so that
   no product fuses into a sum the source does not write as an fma.

   A kernel reads its parameters through one argument at buffer 0, a
   reference to a struct of 64-bit fields that the host writes as words
   (kernels.h). A field that addresses a buffer is declared as a pointer
   into the device or the constant address space: the struct is then an
   argument buffer whose pointers rig writes as GPU addresses, which the
   Metal Shading Language defines (§2.13, §5.2), where it defines no cast
   of an integer to a pointer. The struct's size is the parameters' size
   rig checks each launch against. The value rig writes into a pointer
   field is its buffer's MTLBuffer.gpuAddress plus the ref's offset, which
   Metal's API defines as the buffer's address on the GPU. NX_FIELD checks
   that a field lies at its word's index.

   A kernel declares the threads of its groups, which gen.py checks
   against its row of kernels.h, or declares no bound, so that its
   registers set its pipeline's limit, which the host checks against its
   row once Metal makes the pipeline. It takes no static threadgroup memory,
   which gen.py refuses: a group's threadgroup memory is the one dynamic
   binding [[threadgroup(0)]], of the bytes its launch states, so that the
   host bounds it by arithmetic. */

#ifndef NX_METAL_DEVICE_H
#define NX_METAL_DEVICE_H

#pragma METAL fp contract(off)

#include <metal_stdlib>

#include "kernels.h"

/* A kernel whose groups have at most [n] threads. */
#define NX_KERNEL(n) [[kernel, max_total_threads_per_threadgroup(n)]]

/* Checks that field [f] of the parameters [s] lies at word [i]. */
#define NX_FIELD(s, f, i)                                                     \
  static_assert(__builtin_offsetof(s, f) == 8 * (i), #s "." #f)

/* Checks that the parameters [s] are [n] words. */
#define NX_SIZE(s, n) static_assert(sizeof(s) == 8 * (n), #s " size")

/* The index, from the grid's first thread, of thread [t] of group [g] of
   [n] threads: 64-bit, since a grid may hold more than 2^32 threads. */
static inline ulong nx_thread(uint g, uint t, uint n) {
  return ulong(g) * n + t;
}

/* [n / d] for every 64-bit [n] and [d >= 1], [m] [d]'s divisor: the 64-bit
   m = ⌊2^64 (2^l - d) / d⌋ + 1 mod 2^64 for l = ⌈log₂ d⌉ (Granlund and
   Montgomery 1994, figure 4.1). A product's high half, an add and two
   shifts, since the GPU divides integers in software. */
static inline ulong nx_divide(ulong n, ulong d, ulong m) {
  if (d == 1) return n;
  const uint l = 64 - metal::clz(d - 1);
  const ulong t = metal::mulhi(n, m);
  return (t + ((n - t) >> 1)) >> (l - 1);
}

/* float16

   binary16 by the GPU's conversions: half to float is exact, and float
   to half rounds ties to even and keeps the subnormals it makes (Metal
   Shading Language Specification 4.1, §8.6), as nx_dtype.h's codec does,
   in fewer instructions. A NaN goes by its bits, since the GPU's
   conversion gives every NaN the canonical one: it is quieted, its sign
   and top payload bits kept, as the codec gives it. */

/* The float32 of the binary16 code [h]. */
static inline float nx_metal_f16_to_float(ushort h) {
  if ((h & 0x7fffu) <= 0x7c00u) return float(as_type<half>(h));
  return as_type<float>((uint(h) & 0x8000u) << 16 | 0x7fc00000u |
                        (uint(h) & 0x3ffu) << 13);
}

/* The binary16 code nearest the float32 [f]. */
static inline ushort nx_metal_float_to_f16(float f) {
  const uint u = as_type<uint>(f);
  if ((u & 0x7fffffffu) <= 0x7f800000u) return as_type<ushort>(half(f));
  return ushort((u >> 16 & 0x8000u) | 0x7e00u | (u >> 13 & 0x3ffu));
}

/* Tickets

   A last-group combine: each group writes its part to device memory, then
   counts itself on a ticket, a 32-bit word that is zero before the launch;
   the group that takes the last count reads every part. Metal states no
   forward progress across groups on Apple7, so no group waits for
   another. The protocol is the most explicit form the Metal Shading
   Language's text supports for this (§4.8, §6.10.1, §6.16): it states a
   subset of C++17's atomics without naming it, so the form below is
   consistent with the text and probed, not specified:

   1. every thread that wrote a part fences (seq_cst, device scope);
   2. a barrier on device and threadgroup memory;
   3. thread 0 fences, takes the ticket, relaxed, and the last taker
      stores 0 and fences again;
   4. a second such barrier, then each thread of the last group fences;
   5. the parts are read through coherent(device) pointers.

   C++17 pairs a release fence with an acquire fence through an atomic
   that the first's thread writes and the second's thread reads after it:
   both of thread 0's fences, sequenced around its ticket. The other
   threads' writes happen before the first through the first barrier, and
   the second happens before their reads through the second barrier, so
   the pairing covers every thread of the group. The fences of steps 1
   and 4 pair with no atomic of their thread.

   Coherent pointers alone fail on Apple7 from 2 to 128 groups; fences
   alone at every size. Without thread 0's fences the protocol passes on
   Apple7 as well, so no test shows them missing. */

static inline void nx_fence() {
  metal::atomic_thread_fence(metal::mem_flags::mem_device,
                             metal::memory_order_seq_cst,
                             metal::thread_scope_device);
}

/* Whether the calling group is the last of [groups] to take [ticket],
   in every thread of it. Every thread of the group calls it, after its
   writes of its part, with [t] its index in the group and [flag] a word of
   the group's threadgroup memory. The last group leaves [ticket] zero.
   Inlined into every kernel that calls it, so that one kernel's code does
   not depend on how many others call it. */
__attribute__((always_inline)) static inline bool
nx_last(device metal::atomic_uint *ticket, uint groups, uint t,
        threadgroup uint *flag) {
  nx_fence();
  metal::threadgroup_barrier(metal::mem_flags::mem_device |
                             metal::mem_flags::mem_threadgroup);
  if (t == 0) {
    nx_fence();
    const uint taken = metal::atomic_fetch_add_explicit(
        ticket, 1u, metal::memory_order_relaxed);
    const bool last = taken == groups - 1;
    if (last) {
      metal::atomic_store_explicit(ticket, 0u, metal::memory_order_relaxed);
      nx_fence();
    }
    *flag = last ? 1u : 0u;
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_device |
                             metal::mem_flags::mem_threadgroup);
  const bool last = *flag != 0;
  if (last) nx_fence();
  return last;
}

/* [n] / d, n below 2^32, by d's divisor [md] from the host: its multiplier
   in the low 32 bits, its shift above. */
__attribute__((always_inline)) static inline uint
divide(uint n, ulong md) {
  return uint((ulong(metal::mulhi(n, uint(md))) + n) >> (md >> 32));
}

/* Counts across groups

   A count that many groups add into and that may pass 2^32 lives in two
   32-bit words, as Apple7 has no 64-bit atomics: [lo] takes each add, and
   [hi] the multiples of 2^k the add crosses, from [lo]'s value before it,
   so the count is hi 2^k + (lo mod 2^k). Integer adds commute and each
   multiple of 2^k is crossed by one add, where lo wraps too, since 2^32 is
   a multiple of 2^k, so the sum is exact for any k up to 32. */
static inline void nx_carry_add(device metal::atomic_uint *lo,
                                device metal::atomic_uint *hi, uint v,
                                uint k) {
  const uint old =
      metal::atomic_fetch_add_explicit(lo, v, metal::memory_order_relaxed);
  const ulong crossed = ((ulong(old) + v) >> k) - (ulong(old) >> k);
  if (crossed != 0)
    metal::atomic_fetch_add_explicit(hi, uint(crossed),
                                     metal::memory_order_relaxed);
}

/* The count of a low word [lo] and a high word [hi] (nx_carry_add). */
static inline ulong nx_carried(uint lo, uint hi, uint k) {
  const ulong low = k == 32 ? ulong(lo) : ulong(lo) & ((1ul << k) - 1);
  return (ulong(hi) << k) + low;
}

/* Ranks

   A group of NX_SORT_THREADS threads, NX_SIMDS SIMD-groups of 32 lanes,
   holds a tile of N items a thread: item j of lane l of SIMD-group w is
   the tile's item w N 32 + j 32 + l, so a SIMD-group's loads of
   consecutive items are 32-lane runs. nx_rank gives each item its place
   in the tile's stable order by an 8-bit digit, items past the tile's
   end taking digit 256, after every real one: within a SIMD-group by
   the lanes of its digit (nx_peers), then each bin's SIMD-groups one
   after another, then the bins (Satish, Harris and Garland, "Designing
   Efficient Sorting Algorithms for Manycore GPUs", 2009, §III: a tile is
   ranked on chip before it moves). A SIMD-group's count of a bin is
   written by the lowest lane of its digit alone, so each count takes one
   write a distinct digit of a step, whatever the keys. */

#define NX_SIMDS (NX_SORT_THREADS / 32)

/* The lanes of the calling SIMD-group whose digit [d], of [B] bits, is
   the calling lane's: B ballots. Every lane of the SIMD-group calls it. */
template <int B> static inline uint nx_peers(uint d) {
  uint peers = 0xFFFFFFFFu;
#pragma unroll
  for (int b = 0; b < B; b++) {
    const bool bit = (d >> b & 1) != 0;
    const uint m = uint(ulong(metal::simd_ballot(bit)));
    peers &= bit ? m : ~m;
  }
  return peers;
}

/* The lanes below the calling one. */
static inline uint nx_below(uint lane) { return (1u << lane) - 1; }

/* The exclusive sum of [v] over the group's threads in order, and their
   sum in [all]. Every thread calls it; [sums] holds NX_SIMDS words. */
static inline uint nx_group_excl(uint v, uint t, threadgroup uint *sums,
                                 thread uint &all) {
  const uint inc = metal::simd_prefix_inclusive_sum(v);
  if ((t & 31) == 31) sums[t / 32] = inc;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  uint before = 0, a = 0;
#pragma unroll
  for (uint s = 0; s < NX_SIMDS; s++) {
    const uint x = sums[s];
    if (s < t / 32) before += x;
    a += x;
  }
  all = a;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  return before + inc - v;
}

/* The 64-bit exclusive sum of [v] over the group's threads in order, and
   their sum in [all], by doubling steps through [buf] of NX_SORT_THREADS
   words: for combines, once a slice. Every thread calls it. */
static inline ulong nx_group_excl64(ulong v, uint t, threadgroup ulong *buf,
                                    thread ulong &all) {
  buf[t] = v;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint o = 1; o < NX_SORT_THREADS; o <<= 1) {
    const ulong a = t >= o ? buf[t - o] : 0;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    buf[t] += a;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  const ulong inc = buf[t];
  all = buf[NX_SORT_THREADS - 1];
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  return inc - v;
}

/* Each item's place [r] in the tile's stable order by its digit [d],
   0 to 256, with each bin's first place in [start] (257 words) and its
   items in [count] (257 words). [cnt] holds NX_SIMDS rows of 257 counts
   of type C, which hold a SIMD-group's items of a bin; [sums] NX_SIMDS
   words. Every thread of the group calls it. */
template <int N, typename C>
static inline void nx_rank(thread const uint (&d)[N], thread uint (&r)[N],
                           uint t, threadgroup C *cnt,
                           threadgroup uint *start, threadgroup uint *count,
                           threadgroup uint *sums) {
  const uint w = t / 32, below = nx_below(t & 31);
  for (uint i = t; i < NX_SIMDS * 257; i += NX_SORT_THREADS) cnt[i] = 0;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
#pragma unroll
  for (int j = 0; j < N; j++) {
    const uint m = nx_peers<9>(d[j]);
    const uint base = cnt[w * 257 + d[j]];
    metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    if ((m & below) == 0) cnt[w * 257 + d[j]] = C(base + metal::popcount(m));
    metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    r[j] = base + metal::popcount(m & below);
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint b = t; b < 257; b += NX_SORT_THREADS) {
    uint s = 0;
#pragma unroll
    for (uint i = 0; i < NX_SIMDS; i++) {
      const uint c = cnt[i * 257 + b];
      cnt[i * 257 + b] = C(s);
      s += c;
    }
    count[b] = s;
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  uint all;
  start[t] = nx_group_excl(count[t], t, sums, all);
  if (t == 0) start[256] = all;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
#pragma unroll
  for (int j = 0; j < N; j++) r[j] += start[d[j]] + cnt[w * 257 + d[j]];
}

#endif
