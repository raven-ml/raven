/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* With the parameters w and n: sets w[1], then spins until w[0] is set, n
   rounds at most. */

#include <stdint.h>

struct params {
  int64_t *w;
  int64_t n;
};

void waiting(const struct params *p, int64_t first, int64_t end) {
  (void)first;
  (void)end;
  __atomic_store_n(&p->w[1], 1, __ATOMIC_RELEASE);
  for (int64_t i = 0; i < p->n; i++)
    if (__atomic_load_n(&p->w[0], __ATOMIC_ACQUIRE) != 0) return;
}
