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

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#endif

/* The bytes of L1 data cache of the cores a job runs on: the performance
   cores' where cores differ, macOS's perflevel0, on Linux the largest
   level-1 data cache of any CPU (sysfs-devices-system-cpu's
   cache/indexN). 32 KiB, the smallest of the hosts here, where the host
   does not say. */
#define L1_UNKNOWN (32 * 1024)

#if defined(__linux__)
/* Reads the one line of cpu/cpuC/cache/indexI/[file] into [buf]. */
static int sysfs(int c, int i, const char *file, char *buf, int n) {
  char path[96];
  snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%d/%s",
           c, i, file);
  FILE *f = fopen(path, "r");
  if (f == NULL) return 0;
  int ok = fgets(buf, n, f) != NULL;
  fclose(f);
  buf[strcspn(buf, "\n")] = 0;
  return ok;
}
#endif

static int64_t l1_bytes(void) {
  int64_t most = 0;
#if defined(__APPLE__)
  size_t n = sizeof most;
  if (sysctlbyname("hw.perflevel0.l1dcachesize", &most, &n, NULL, 0) != 0)
    most = 0;
#elif defined(__linux__)
  long cpus = sysconf(_SC_NPROCESSORS_CONF);
  for (int c = 0; c < cpus; c++)
    for (int i = 0; i < 4; i++) {
      char level[8], type[16], size[16];
      if (!sysfs(c, i, "level", level, sizeof level)) break;
      if (strcmp(level, "1") != 0) continue;
      if (!sysfs(c, i, "type", type, sizeof type)) break;
      if (strcmp(type, "Data") != 0) continue;
      if (!sysfs(c, i, "size", size, sizeof size)) break;
      int64_t kib = strtoll(size, NULL, 10);
      if (kib * 1024 > most) most = kib * 1024;
    }
#endif
  return most > 0 ? most : L1_UNKNOWN;
}

#if defined(__x86_64__)
#include <cpuid.h>
#endif

#if defined(__x86_64__)

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

/* AVX-VNNI-INT8: CPUID leaf 7 subleaf 1, EDX bit 4 (Intel SDM vol. 2A), its
   state the AVX state has_v3 checks. */
static int has_vnni_int8(void) {
  unsigned a, b, c, d;
  if (!__get_cpuid_count(7, 1, &a, &b, &c, &d)) return 0;
  return (d >> 4) & 1;
}
#endif

#if defined(__x86_64__)
nx_cpu_target nx_cpu_base = {.name = "base", .copy_strided = 1};
nx_cpu_target nx_cpu_v3 = {.name = "v3", .copy_strided = 1};
#else
nx_cpu_target nx_cpu_base = {.name = "base"};
#endif
#if defined(__APPLE__) && defined(__aarch64__)
nx_cpu_target nx_cpu_amx;
#endif

const nx_cpu_target *nx_cpu_table = &nx_cpu_base;

/* A core's L2 where the host does not tell: 256 KiB, the least of the
   x86-64 and Apple cores nx.cpu tunes for. */
#define L2_BYTES (256 * 1024)

/* x86-64 reports, in KiB, the L2 of the core the program starts on
   (CPUID leaf 0x80000006, ECX bits 31-16; Intel SDM vol. 2A, AMD APM
   vol. 3): a core's own, which its hardware threads share, on AMD's cores
   and Intel's performance cores; an Intel efficient core reports the L2
   its cluster of four shares, so that on a hybrid host the figure changes
   with that core, and speed with it, never bits. Apple's arm64 reports
   its performance cores' L2 and how many cores share it, and a core's
   share is taken. */
static int64_t l2_bytes(void) {
  int64_t bytes = 0;
#if defined(__x86_64__)
  unsigned a, b, c, d;
  if (__get_cpuid(0x80000006, &a, &b, &c, &d))
    bytes = (int64_t)(c >> 16) * 1024;
#elif defined(__APPLE__)
  int64_t size = 0;
  int32_t cores = 0;
  size_t n = sizeof size, m = sizeof cores;
  if (sysctlbyname("hw.perflevel0.l2cachesize", &size, &n, NULL, 0) == 0 &&
      sysctlbyname("hw.perflevel0.cpusperl2", &cores, &m, NULL, 0) == 0 &&
      cores > 0)
    bytes = size / cores;
#endif
  return bytes > 0 ? bytes : L2_BYTES;
}

__attribute__((constructor)) static void init(void) {
  nx_cpu_base.l2 = l2_bytes();
  nx_cpu_base.l1 = l1_bytes();
  nx_cpu_set_convert_base(&nx_cpu_base);
  nx_cpu_set_gemm_base(&nx_cpu_base);
  nx_cpu_set_rows_base(&nx_cpu_base);
  nx_cpu_set_reduce_base(&nx_cpu_base);
#if defined(__aarch64__)
  nx_cpu_set_neon(&nx_cpu_base);
#endif
#if defined(__APPLE__) && defined(__aarch64__)
  /* amx is base with the matrix unit's kernels. */
  if (!nx_cpu_has_amx()) return;
  nx_cpu_amx = nx_cpu_base;
  nx_cpu_amx.name = "amx";
  nx_cpu_set_amx(&nx_cpu_amx);
  nx_cpu_table = &nx_cpu_amx;
#endif
#if defined(__x86_64__)
  if (!has_v3()) return;
  nx_cpu_v3.l2 = nx_cpu_base.l2;
  nx_cpu_v3.l1 = nx_cpu_base.l1;
  nx_cpu_set_convert_v3(&nx_cpu_v3);
  nx_cpu_set_gemm_v3(&nx_cpu_v3);
  nx_cpu_set_rows_v3(&nx_cpu_v3);
  nx_cpu_set_reduce_v3(&nx_cpu_v3);
  nx_cpu_set_avx2(&nx_cpu_v3);
  if (has_vnni_int8()) nx_cpu_set_vnni(&nx_cpu_v3);
  nx_cpu_table = &nx_cpu_v3;
#endif
}
