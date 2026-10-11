/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A symbol of data beside a function. */

#include <stdint.h>

const int64_t data_entry[2] = {1, 2};

void code(const void *p, int64_t first, int64_t end) {
  (void)p;
  (void)first;
  (void)end;
}
