/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Stores value 2 as the 64-bit word at buffer 0, then calls the function at
   value 0 with value 1 and value 3: a rail's ready function, its argument and
   a count. */

#include <stdint.h>

typedef void (*ready_fn)(void *arg, uint64_t c);

void ready(void **b, const int64_t *v) {
  int64_t *out = b[0];
  out[0] = v[2];
  ((ready_fn)v[0])((void *)v[1], (uint64_t)v[3]);
}
