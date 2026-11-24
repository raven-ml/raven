/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The Metal suite's harness: kernels that check what nx.metal's device
   code relies on, compiled by nx.metal's gen.py with its flags and its
   device.h. A kernel reads its parameters as nx.metal's do: a struct of
   64-bit fields at buffer 0. */

#include "device.h"

/* combine: a last-group combine by device.h's protocol. Each thread of
   each of the [groups] groups writes its word of [parts], a function of
   its group, its thread and [round]; then the group takes [ticket]. The
   last group reads every word of [parts] and adds, at [out], the count
   of words that differ from what their thread wrote, and 1 at [out] + 1,
   the count of last groups. A word read stale, from an earlier round,
   differs. Threadgroup memory: COMBINE_THREADS + 1 words. */

#define COMBINE_THREADS 256

struct combine_words {
  device metal::atomic_uint *out;
  device coherent(device) uint *parts;
  device metal::atomic_uint *ticket;
  ulong groups;
  ulong round;
};

NX_SIZE(combine_words, 5);

static inline uint pattern(uint g, uint t, uint round) {
  return (g * 2654435761u) ^ (t * 40503u) ^ (round * 97u) ^ 0x5bd1e995u;
}

NX_KERNEL(COMBINE_THREADS)
void combine(constant combine_words &w [[buffer(0)]],
             threadgroup uint *s [[threadgroup(0)]],
             uint g [[threadgroup_position_in_grid]],
             uint t [[thread_index_in_threadgroup]]) {
  const uint groups = uint(w.groups), round = uint(w.round);
  w.parts[g * COMBINE_THREADS + t] = pattern(g, t, round);
  if (!nx_last(w.ticket, groups, t, s + COMBINE_THREADS)) return;

  uint misses = 0;
  for (uint k = t; k < groups * COMBINE_THREADS; k += COMBINE_THREADS)
    if (w.parts[k] != pattern(k / COMBINE_THREADS, k % COMBINE_THREADS, round))
      misses++;
  s[t] = misses;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (t != 0) return;
  uint all = 0;
  for (uint k = 0; k < COMBINE_THREADS; k++) all += s[k];
  metal::atomic_fetch_add_explicit(w.out, all, metal::memory_order_relaxed);
  metal::atomic_fetch_add_explicit(w.out + 1, 1u, metal::memory_order_relaxed);
}

/* divide: [out[i] = nx_divide(n[i], d[i], m[i])] for each i < [count], a
   thread each. */

#define DIVIDE_THREADS 256

struct divide_words {
  device ulong *out;
  device const ulong *n;
  device const ulong *d;
  device const ulong *m;
  ulong count;
};

NX_SIZE(divide_words, 5);

NX_KERNEL(DIVIDE_THREADS)
void divide(constant divide_words &w [[buffer(0)]],
            uint g [[threadgroup_position_in_grid]],
            uint t [[thread_index_in_threadgroup]]) {
  const ulong i = nx_thread(g, t, DIVIDE_THREADS);
  if (i < w.count) w.out[i] = nx_divide(w.n[i], w.d[i], w.m[i]);
}

/* arith: for each i < [count], [out] gets a[i] b[i], a[i] + b[i] and
   fma(a[i], b[i], c[i]) at 3 i, 3 i + 1 and 3 i + 2: float32 arithmetic as
   nx.metal's kernels compile it. */

#define ARITH_THREADS 256

struct arith_words {
  device float *out;
  device const float *a;
  device const float *b;
  device const float *c;
  ulong count;
};

NX_SIZE(arith_words, 5);

NX_KERNEL(ARITH_THREADS)
void arith(constant arith_words &w [[buffer(0)]],
           uint g [[threadgroup_position_in_grid]],
           uint t [[thread_index_in_threadgroup]]) {
  const ulong i = nx_thread(g, t, ARITH_THREADS);
  if (i >= w.count) return;
  const float a = w.a[i], b = w.b[i], c = w.c[i];
  w.out[3 * i] = a * b;
  w.out[3 * i + 1] = a + b;
  w.out[3 * i + 2] = metal::fma(a, b, c);
}
