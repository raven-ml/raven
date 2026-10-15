/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Stores the words at bytes 4,096 and 8,188 of its 8,192 bytes of
   parameters into words 0 and 1 at out, its first 8. */

#include <stdint.h>

struct params {
  uint32_t *out;
  uint32_t w[2046];
};

void wide(const struct params *p, int64_t first, int64_t end) {
  (void)first;
  (void)end;
  p->out[0] = p->w[1022];
  p->out[1] = p->w[2045];
}
