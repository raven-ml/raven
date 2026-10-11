/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A launch that records what it ran. Its parameters are counts and log.

   Each call counts each thread of its range in counts, and appends to log
   a record of two words: its first thread and the one after its last.
   log[0] is the length of the log after it, and log[1] its capacity: a
   record past it is dropped. */

#include <stdint.h>

struct params {
  int64_t *counts, *log;
};

void blocks(const struct params *p, int64_t first, int64_t end) {
  for (int64_t i = first; i < end; i++)
    __atomic_fetch_add(&p->counts[i], 1, __ATOMIC_RELAXED);
  int64_t at = __atomic_fetch_add(&p->log[0], 2, __ATOMIC_RELAXED) + 2;
  if (at + 2 > p->log[1]) return;
  p->log[at] = first;
  p->log[at + 1] = end;
}
