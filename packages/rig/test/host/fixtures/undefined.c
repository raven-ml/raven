/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A call of a function no library defines. */

#include <stdint.h>

void rig_host_test_nowhere(int64_t);

void undefined(const void *p, int64_t first, int64_t end) {
  (void)p;
  (void)end;
  rig_host_test_nowhere(first);
}
