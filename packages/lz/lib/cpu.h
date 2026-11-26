/* The x86-64 extensions Zstandard's second paths are compiled for, read
   from CPUID: BMI1 is leaf 7 EBX bit 3, BMI2 leaf 7 EBX bit 8, LZCNT leaf
   0x80000001 ECX bit 5 (Intel SDM vol. 2A, CPUID). GCC before 11 has no
   "lzcnt" for __builtin_cpu_supports, so the bits are read here. */

#ifndef LZ_CPU_H
#define LZ_CPU_H

#if defined(__x86_64__) && defined(__GNUC__)
#include <cpuid.h>

/* [lz_cpu_bmi_lzcnt ()] is whether the CPU has BMI1, BMI2 and LZCNT. */
static inline int lz_cpu_bmi_lzcnt(void) {
  unsigned a, b, c, d;
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
  int bmi = (b >> 3) & 1, bmi2 = (b >> 8) & 1;
  if (!__get_cpuid(0x80000001, &a, &b, &c, &d)) return 0;
  return bmi && bmi2 && ((c >> 5) & 1);
}
#endif

#endif
