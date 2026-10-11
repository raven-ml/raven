/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* With the parameters out, count, dst and src: stores at out the 64-bit count
   at count, read with acquire order, and once it is at least 1, copies the
   64-bit word at src to dst, as work that reads a rail's arrival does. */

#include <stdint.h>

struct params {
  int64_t *out;
  const int64_t *count;
  int64_t *dst;
  const int64_t *src;
};

void arrival(const struct params *p, int64_t first, int64_t end) {
  (void)first;
  (void)end;
  int64_t c = __atomic_load_n(p->count, __ATOMIC_ACQUIRE);
  p->out[0] = c;
  if (c >= 1) p->dst[0] = p->src[0];
}
