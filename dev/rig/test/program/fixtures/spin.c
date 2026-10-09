/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A fill that counts to the 64-bit word its argument points at, as work that
   takes a device a known while: returns 0. */

#include <stdint.h>

int spin(void *queue, void *arg, uint64_t v) {
  (void)queue, (void)v;
  uint64_t n = *(const uint64_t *)arg;
  volatile uint64_t x = 0;
  for (uint64_t i = 0; i < n; i++) x += i;
  return 0;
}
