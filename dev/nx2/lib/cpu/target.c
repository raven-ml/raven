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

/* The bytes of the largest cache the host reports: on Linux the largest
   of CPU 0's caches (sysfs-devices-system-cpu's cache/indexN), the last
   level; on macOS the performance cores' L2, the last level it reports
   (the M1 Max's 48 MB system cache it does not). 8 MiB, the least of the
   hosts here, where the host does not say. */
#define LLC_UNKNOWN (8 * 1024 * 1024)

static int64_t llc_bytes(void) {
  int64_t most = 0;
#if defined(__APPLE__)
  size_t n = sizeof most;
  if (sysctlbyname("hw.perflevel0.l2cachesize", &most, &n, NULL, 0) != 0)
    most = 0;
#elif defined(__linux__)
  for (int i = 0;; i++) {
    char size[16];
    if (!sysfs(0, i, "size", size, sizeof size)) break;
    int64_t kib = strtoll(size, NULL, 10);
    if (kib * 1024 > most) most = kib * 1024;
  }
#endif
  return most > 0 ? most : LLC_UNKNOWN;
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
  nx_cpu_base.llc = llc_bytes();
  nx_cpu_set_convert_base(&nx_cpu_base);
  nx_cpu_set_gemm_base(&nx_cpu_base);
  nx_cpu_set_rows_base(&nx_cpu_base);
  nx_cpu_set_reduce_base(&nx_cpu_base);
  nx_cpu_set_fft_base(&nx_cpu_base);
  nx_cpu_set_linalg_base(&nx_cpu_base);
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
  nx_cpu_v3.llc = nx_cpu_base.llc;
  nx_cpu_set_convert_v3(&nx_cpu_v3);
  nx_cpu_set_gemm_v3(&nx_cpu_v3);
  nx_cpu_set_rows_v3(&nx_cpu_v3);
  nx_cpu_set_reduce_v3(&nx_cpu_v3);
  nx_cpu_set_fft_v3(&nx_cpu_v3);
  nx_cpu_set_linalg_v3(&nx_cpu_v3);
  nx_cpu_set_avx2(&nx_cpu_v3);
  if (has_vnni_int8()) nx_cpu_set_vnni(&nx_cpu_v3);
  nx_cpu_table = &nx_cpu_v3;
#endif
}

/* Kinds' costs

   What computing a kind at one element takes (cpu.h, Calls, scalar), its
   operands' loads and its result's store included, by the element's
   class: a real one of up to 4 bytes, one of 8, and a complex one, each
   the most any dtype of its class took over 4096 elements on one thread.
   The worst: pow, 1003 a float64 element (M1); atan2, 658 (M1); mod, 454
   (fudge); tan, 415 (M1). */
typedef struct {
  uint16_t narrow, wide, complex;
} kind_cost;

static const kind_cost op1_costs[NX_OP1_COUNT] = {
    [NX_OP1_COPY] = {16, 32, 64},       [NX_OP1_CAST] = {16, 128, 32},
    [NX_OP1_BITCAST] = {16, 32, 64},    [NX_OP1_NEG] = {64, 32, 64},
    [NX_OP1_RECIP] = {64, 64, 256},     [NX_OP1_ABS] = {64, 32, 64},
    [NX_OP1_SIGN] = {64, 64, 64},       [NX_OP1_SQRT] = {64, 64, 64},
    [NX_OP1_EXP] = {128, 256, 256},     [NX_OP1_EXP2] = {128, 256, 256},
    [NX_OP1_LOG] = {256, 512, 512},     [NX_OP1_LOG2] = {256, 512, 512},
    [NX_OP1_LOG1P] = {256, 512, 512},   [NX_OP1_EXPM1] = {256, 512, 512},
    [NX_OP1_SIN] = {256, 512, 512},     [NX_OP1_COS] = {256, 512, 512},
    [NX_OP1_TAN] = {256, 1024, 1024},   [NX_OP1_ASIN] = {256, 512, 512},
    [NX_OP1_ACOS] = {256, 512, 512},    [NX_OP1_ATAN] = {128, 512, 512},
    [NX_OP1_SINH] = {512, 1024, 1024},  [NX_OP1_COSH] = {512, 512, 512},
    [NX_OP1_TANH] = {256, 512, 512},    [NX_OP1_ERF] = {256, 512, 512},
    [NX_OP1_FLOOR] = {64, 128, 128},    [NX_OP1_CEIL] = {64, 128, 128},
    [NX_OP1_ROUND] = {128, 128, 128},   [NX_OP1_TRUNC] = {64, 64, 64},
};

static const kind_cost op2_costs[NX_OP2_COUNT] = {
    [NX_OP2_ADD] = {64, 32, 128},          [NX_OP2_SUB] = {64, 32, 128},
    [NX_OP2_MUL] = {64, 64, 128},          [NX_OP2_FDIV] = {64, 64, 256},
    [NX_OP2_IDIV] = {128, 256, 256},       [NX_OP2_MOD] = {512, 1024, 1024},
    [NX_OP2_POW] = {1024, 2048, 2048},     [NX_OP2_ATAN2] = {256, 1024, 1024},
    [NX_OP2_MAXIMUM] = {64, 64, 512},      [NX_OP2_MINIMUM] = {64, 64, 512},
    [NX_OP2_AND] = {16, 32, 32},           [NX_OP2_OR] = {16, 32, 32},
    [NX_OP2_XOR] = {16, 32, 32},           [NX_OP2_THREEFRY] = {256, 256, 256},
    [NX_OP2_EQUAL] = {16, 16, 64},         [NX_OP2_NOT_EQUAL] = {16, 16, 64},
    [NX_OP2_LESS] = {16, 16, 128},         [NX_OP2_LESS_EQUAL] = {16, 16, 128},
};

static const kind_cost op3_costs[NX_OP3_COUNT] = {
    [NX_OP3_WHERE] = {8, 32, 64},
    [NX_OP3_FMA] = {64, 128, 128},
};

double nx_cpu_kind_cost(int arity, int k, int dt) {
  const kind_cost *c = arity == 1   ? &op1_costs[k]
                       : arity == 2 ? &op2_costs[k]
                                    : &op3_costs[k];
  int kind = nx_dtype_row_of(dt).kind;
  if (kind == NX_KIND_COMPLEX) return c->complex;
  return nx_dtype_row_of(dt).bits > 32 ? c->wide : c->narrow;
}
