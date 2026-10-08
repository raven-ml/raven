/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Calls another program through rig_host_call. Its values are the
   program's entry f, its number of values n, a split's four values
   (extent, blocks, lo, hi), the first iteration and the one after the
   last of its own block, whether it runs as a block of a split, then the n
   values of f. Its buffers are f's.

   Called whole, it calls f on its values, split by the split if its extent
   is not negative. As a block, it splits its own iterations: the split's
   extent is the block's length, and f's value at the offset index of
   blocks.c's meta, its third buffer, is the block's first iteration. */

#include <stdint.h>

void rig_host_call(void (*f)(void **, const int64_t *), void **buffers,
                      const int64_t *values, int64_t n, const int64_t *split);

void nested(void **b, const int64_t *v) {
  void (*f)(void **, const int64_t *) = (void (*)(void **, const int64_t *))v[0];
  int64_t n = v[1];
  if (!v[8]) {
    rig_host_call(f, b, v + 9, n, v[2] < 0 ? 0 : v + 2);
    return;
  }
  int64_t w[64], split[4] = {v[7] - v[6], v[3], v[4], v[5]};
  const int64_t *meta = b[2];
  for (int64_t i = 0; i < n; i++) w[i] = v[9 + i];
  w[meta[2]] = v[6];
  rig_host_call(f, b, w, n, split);
}
