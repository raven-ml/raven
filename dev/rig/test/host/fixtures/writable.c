/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A counter in .bss, which the loader refuses. */

#include <stdint.h>

static int64_t counter;

void writable(void **b, const int64_t *v) {
  (void)b;
  counter += v[0];
}
