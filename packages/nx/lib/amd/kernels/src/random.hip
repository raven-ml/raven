/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* random: the Threefry-2x32 hash of each pair of counter words under its
   pair of key words, 20 rounds with Random123's rotation constants and key
   schedule, as nx.cpu computes it (cpu/nx_c_random.c). Words are int32
   storage computed as uint32, so additions wrap.

   Pair i of the result is dst[2i], dst[2i + 1]; the key's pair i is at
   [at(0, i)] and [at(0, i) + kstep], the counter's at [at(1, i)] and
   [at(1, i) + cstep]. THREEFRY() defines the kernel.
   - [s (dst, key, ctr, meta, kstep, cstep)] */

#include "common.h"

DEVICE uint32_t rotl(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

DEVICE void threefry(uint32_t *dst, const uint32_t *key, const uint32_t *ctr,
                     const meta &m, i64 kstep, i64 cstep) {
  const int rots[8] = {13, 15, 26, 6, 17, 29, 16, 24};
  strided index{&m};
  for (i64 i = first(); i < m.n; i += m.groups * THREADS) {
    i64 k = index.at(0, i), c = index.at(1, i);
    uint32_t k0 = key[k], k1 = key[k + kstep];
    uint32_t ks[3] = {k0, k1, 0x1BD11BDAu ^ k0 ^ k1};
    uint32_t x0 = ctr[c] + k0, x1 = ctr[c + cstep] + k1;
    for (int r = 0; r < 20; r++) {
      x0 += x1;
      x1 = rotl(x1, rots[r % 8]);
      x1 ^= x0;
      if ((r + 1) % 4 == 0) {
        int j = (r + 1) / 4;
        x0 += ks[j % 3];
        x1 += ks[(j + 1) % 3] + (uint32_t)j;
      }
    }
    dst[2 * i] = x0;
    dst[2 * i + 1] = x1;
  }
}

#define THREEFRY()                                                             \
  KERNEL void NAME(s)(uint32_t *dst, const uint32_t *key, const uint32_t *ctr, \
                      meta m, i64 kstep, i64 cstep) {                          \
    threefry(dst, key, ctr, m, kstep, cstep);                                  \
  }
