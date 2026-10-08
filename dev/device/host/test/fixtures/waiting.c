/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* With the buffer w and the value n: sets w[1], then spins until w[0] is
   set, n rounds at most. */

#include <stdint.h>

void waiting(void **b, const int64_t *v) {
  int64_t *w = b[0];
  __atomic_store_n(&w[1], 1, __ATOMIC_RELEASE);
  for (int64_t i = 0; i < v[0]; i++)
    if (__atomic_load_n(&w[0], __ATOMIC_ACQUIRE) != 0) return;
}
