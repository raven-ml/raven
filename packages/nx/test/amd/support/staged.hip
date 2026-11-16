/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* staged, for tests of nx.amd's device.h: the launch's first [n] words
   but those from [gap] to [gap_end], as device.h's stage copies them,
   each staged word i stored at [dst] + 8 i. Its words are [dst], [n],
   [gap] and [gap_end], then any. A unit of its own, so that device.h's
   definitions meet none of harness.hip's. */

#include "device.h"

enum { DST, N, GAP, GAP_END };

/* Fewer work-items than words, so that staging takes several rounds. */
enum { THREADS = 64, WORDS = 256 };

KERNEL SIZED(THREADS) void staged(Words<WORDS>) {
  static SHARED int64_t w[WORDS];
  const int64_t *k = words();
  const int64_t n = k[N];
  const int64_t gap = k[GAP], gap_end = k[GAP_END];
  stage(w, n, THREADS, gap, gap_end);
  int64_t *dst = address<int64_t>(DST);
  for (int64_t i = item(); i < n; i += THREADS)
    if (i < gap || i >= gap_end) dst[i] = w[i];
}

/* divided, for device.h's divide: [count] quotients, the 32-bit word i
   at [quot] the 32-bit numerator i at [num] divided by Plan.divisor's
   64-bit word i at [divisor]. One group. */

enum { QUOT, NUM, DIVISOR, COUNT };

KERNEL SIZED(THREADS) void divided(Words<4>) {
  const int64_t *k = words();
  uint32_t *quot = address<uint32_t>(QUOT);
  const uint32_t *num = address<uint32_t>(NUM);
  const uint64_t *divisor = address<uint64_t>(DIVISOR);
  for (int64_t i = item(); i < k[COUNT]; i += THREADS)
    quot[i] = divide(num[i], divisor[i]);
}

/* split_axes, for device.h's split: [count] indices at [idx], each split
   along the axis whose head Plan.axis wrote at [heads] + 2 i, in a small
   space where [small]. Index i's coordinate and quotient by its head go
   to [coord] and [quot] + 2 i, by its extent alone to 2 i + 1. One
   group. */

enum { COORD, QUOTS, IDX, HEADS, SPLITS, SMALL };

KERNEL SIZED(THREADS) void split_axes(Words<6>) {
  const int64_t *k = words();
  int64_t *coord = address<int64_t>(COORD), *quot = address<int64_t>(QUOTS);
  const int64_t *idx = address<const int64_t>(IDX);
  const int64_t *heads = address<const int64_t>(HEADS);
  const bool small = k[SMALL];
  for (int64_t i = item(); i < k[SPLITS]; i += THREADS) {
    const int64_t *a = heads + 2 * i;
    int64_t by_head = idx[i], by_extent = idx[i];
    coord[2 * i] = split(by_head, a, small);
    quot[2 * i] = by_head;
    coord[2 * i + 1] = split(by_extent, a[NX_AXIS_N], small);
    quot[2 * i + 1] = by_extent;
  }
}
