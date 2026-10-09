/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The targets' tables, each filled when the program starts on a host that
   runs it, and the one the kernels run: the best whose instructions the
   host has. A table's fill is compiled for its target's instructions too,
   so only a host that has them fills it.

   On x86-64, v3 needs AVX, AVX2, FMA, F16C and BMI2 (CPUID leaves 1 and 7),
   and an operating system that saves the 256-bit registers across a switch
   (XCR0's bits 1 and 2, read with XGETBV once CPUID reports OSXSAVE). Intel
   SDM vol. 2A, CPUID; vol. 1, §14.3. */

#include "cpu.h"

#if defined(__x86_64__)
#include <cpuid.h>

static int has_v3(void) {
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
  int fma = (c >> 12) & 1, osxsave = (c >> 27) & 1, avx = (c >> 28) & 1;
  int f16c = (c >> 29) & 1;
  if (!(fma && osxsave && avx && f16c)) return 0;
  unsigned lo, hi;
  __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  if ((lo & 6) != 6) return 0;
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
  return ((b >> 5) & 1) && ((b >> 8) & 1); /* AVX2, BMI2 */
}
#endif

nx_cpu_target nx_cpu_base = {.name = "base"};
#if defined(__x86_64__)
nx_cpu_target nx_cpu_v3 = {.name = "v3"};
#endif

const nx_cpu_target *nx_cpu_runs = &nx_cpu_base;

__attribute__((constructor)) static void init(void) {
  nx_cpu_fill_base(&nx_cpu_base);
  nx_cpu_fill_generic_base(&nx_cpu_base);
#if defined(__aarch64__)
  nx_cpu_fill_neon(&nx_cpu_base);
#endif
#if defined(__x86_64__)
  if (!has_v3()) return;
  nx_cpu_fill_v3(&nx_cpu_v3);
  nx_cpu_fill_generic_v3(&nx_cpu_v3);
  nx_cpu_fill_avx2(&nx_cpu_v3);
  nx_cpu_runs = &nx_cpu_v3;
#endif
}
