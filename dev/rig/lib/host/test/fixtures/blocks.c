/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A call that records what it ran. Its buffers are counts, log and meta,
   and meta holds the index of the values' first iteration, the index of the
   one after the last, the index of an offset added to both (-1 for none),
   and the number of values n.

   It counts each iteration in counts, at the offset, and appends to log a
   record of n + 2 words: its first iteration, the one after its last, and
   the n values it saw. log[0] is the length of the log after it, and
   log[1] its capacity: a record past it is dropped. */

#include <stdint.h>

void blocks(void **b, const int64_t *v) {
  int64_t *counts = b[0], *log = b[1];
  const int64_t *meta = b[2];
  int64_t first = v[meta[0]], last = v[meta[1]], n = meta[3];
  int64_t offset = meta[2] < 0 ? 0 : v[meta[2]];
  for (int64_t i = first; i < last; i++)
    __atomic_fetch_add(&counts[offset + i], 1, __ATOMIC_RELAXED);
  int64_t at = __atomic_fetch_add(&log[0], n + 2, __ATOMIC_RELAXED) + 2;
  if (at + n + 2 > log[1]) return;
  log[at] = first;
  log[at + 1] = last;
  for (int64_t i = 0; i < n; i++) log[at + 2 + i] = v[i];
}
