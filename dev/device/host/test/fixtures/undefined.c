/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A call of a function no library defines. */

#include <stdint.h>

void device_host_test_nowhere(int64_t);

void undefined(void **b, const int64_t *v) {
  (void)b;
  device_host_test_nowhere(v[0]);
}
