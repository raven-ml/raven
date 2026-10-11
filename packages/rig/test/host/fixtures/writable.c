/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A counter in .bss, which the loader refuses. */

#include <stdint.h>

static int64_t counter;

void writable(const void *p, int64_t first, int64_t end) {
  (void)p;
  counter += end - first;
}
