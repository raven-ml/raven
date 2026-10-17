/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A jump to the address 2^61, which the object defines as an absolute
   symbol: no address a process maps code at is within reach of a branch's
   field, a 28-bit one on arm64 and a 32-bit one on x86_64, and a symbol the
   object defines gets no stub. */

#include <stdint.h>

__asm__(".globl rig_host_test_far\n"
        ".hidden rig_host_test_far\n"
        ".set rig_host_test_far, 0x2000000000000000\n");

void rig_host_test_far(int64_t) __attribute__((visibility("hidden")));

void far(const void *p, int64_t first, int64_t end) {
  (void)p;
  (void)end;
  rig_host_test_far(first);
}
