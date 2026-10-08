/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A symbol of data beside a function, named as the entry. */

#include <stdint.h>

const int64_t data_entry[2] = {1, 2};

void code(void **b, const int64_t *v) {
  (void)b;
  (void)v;
}
