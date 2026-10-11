/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A function that does nothing. */

#include <stdint.h>

void empty(const void *p, int64_t first, int64_t end) {
  (void)p;
  (void)first;
  (void)end;
}
