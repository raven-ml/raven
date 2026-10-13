/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A table in a section of its own, .rodata, whose alignment the host suite
   raises by patching the object. out[0] is the table's address, and
   out[1 + i] = table[i] for each thread i of the range. Its parameter is
   out. */

#include <stdint.h>

static const int64_t table[4] __attribute__((aligned(16))) = {3, 1, 4, 1};

void aligned(int64_t *const *p, int64_t first, int64_t end) {
  int64_t *out = *p;
  for (int64_t i = first; i < end; i++) out[1 + i] = table[i];
  if (first == 0) out[0] = (int64_t)(uintptr_t)table;
}
