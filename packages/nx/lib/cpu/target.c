/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The targets' tables, each filled when the program starts on a host that
   runs it, and the one the kernels run: the best whose instructions the
   host has.

   On x86-64 every other unit of nx.cpu is compiled for its floor, AVX2,
   FMA, F16C and BMI2 (probe/avx2.ml), under which the C compiler also
   takes SSE3 to SSE4.2 and POPCNT, and so needs AVX, AVX2, FMA, F16C,
   BMI2, SSE4.2 and POPCNT (CPUID leaves 1 and 7), and an operating system
   that saves the 256-bit registers across a switch (XCR0's bits 1 and 2,
   read with XGETBV once CPUID reports OSXSAVE). Intel SDM vol. 2A, CPUID;
   vol. 1, §14.3. AVX gates the SSE instructions the compiler encodes with
   VEX; SSE4.2's CRC32 and POPCNT have no such encoding, and the x86-64
   psABI asks for their detection "even if all current CPUs implement
   certain CPU features together". This is no level of the psABI:
   x86-64-v3 also has BMI1, LZCNT and MOVBE, which these flags do not take.
   This unit is compiled for the architecture's baseline, so that its check
   runs on any host. A table's fill is code of the other units, so on a
   host without the floor the constructor fills none, and nx_cpu_require
   raises when Nx_cpu initialises: the constructor runs before the OCaml
   runtime, and cannot raise. */

#include <string.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <stdio.h>
#include <stdlib.h>
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

/* Whether a processor whose CPUID.01H:ECX is [ecx1] and CPUID.07H.00H:EBX
   [ebx7], under XCR0 [xcr0], runs the other units. */
static int floor_holds(uint32_t ecx1, uint32_t ebx7, uint64_t xcr0) {
  const uint32_t leaf1 = (1u << 12)   /* FMA */
                         | (1u << 20) /* SSE4_2 */
                         | (1u << 23) /* POPCNT */
                         | (1u << 27) /* OSXSAVE */
                         | (1u << 28) /* AVX */
                         | (1u << 29) /* F16C */;
  const uint32_t leaf7 = (1u << 5) /* AVX2 */ | (1u << 8) /* BMI2 */;
  return (ecx1 & leaf1) == leaf1 && (ebx7 & leaf7) == leaf7 &&
         (xcr0 & 6) == 6; /* XMM and YMM state */
}

value nx_cpu_floor_holds(value ecx1, value ebx7, value xcr0) {
  return Val_bool(floor_holds((uint32_t)Long_val(ecx1),
                              (uint32_t)Long_val(ebx7),
                              (uint64_t)Long_val(xcr0)));
}

#if defined(__x86_64__)

static int has_avx2(void) {
  unsigned a, b, c, d;
  uint32_t ecx1 = 0, ebx7 = 0;
  uint64_t xcr0 = 0;
  if (__get_cpuid(1, &a, &b, &c, &d)) ecx1 = c;
  if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) ebx7 = b;
  /* XGETBV runs only where OSXSAVE enables it. */
  if ((ecx1 >> 27) & 1) {
    unsigned lo, hi;
    __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    xcr0 = ((uint64_t)hi << 32) | lo;
  }
  return floor_holds(ecx1, ebx7, xcr0);
}

/* AVX-VNNI-INT8: CPUID leaf 7 subleaf 1, EDX bit 4 (Intel SDM vol. 2A), its
   state the AVX state has_avx2 checks. */
static int has_vnni_int8(void) {
  unsigned a, b, c, d;
  if (!__get_cpuid_count(7, 1, &a, &b, &c, &d)) return 0;
  return (d >> 4) & 1;
}
#endif

int64_t nx_cpu_l2, nx_cpu_l1, nx_cpu_llc;

nx_cpu_target nx_cpu_base = {.name = "base"};

#if defined(__x86_64__)
/* Whether the host has nx.cpu's floor, as the constructor found. */
static int avx2;
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
#if defined(__x86_64__)
  avx2 = has_avx2();
  if (!avx2) return;
#endif
  nx_cpu_l2 = l2_bytes();
  nx_cpu_l1 = l1_bytes();
  nx_cpu_llc = llc_bytes();
  nx_cpu_set_gemm(&nx_cpu_base);
#if defined(__aarch64__)
  nx_cpu_set_neon(&nx_cpu_base);
#endif
#if defined(__x86_64__)
  nx_cpu_set_avx2(&nx_cpu_base);
  if (has_vnni_int8()) nx_cpu_set_vnni(&nx_cpu_base);
#endif
#if defined(__APPLE__) && defined(__aarch64__)
  /* amx is base with the matrix unit's kernels. */
  if (!nx_cpu_has_amx()) return;
  nx_cpu_amx = nx_cpu_base;
  nx_cpu_amx.name = "amx";
  nx_cpu_set_amx(&nx_cpu_amx);
  nx_cpu_table = &nx_cpu_amx;
#endif
}

/* Raises Failure where the host lacks the instructions nx.cpu is compiled
   for: Nx_cpu's initialiser calls it before any other code of nx.cpu
   runs. */
value nx_cpu_require(value unit) {
  (void)unit;
#if defined(__x86_64__)
  if (!avx2)
    caml_failwith("nx.cpu needs AVX2, FMA, F16C, BMI2, SSE4.2 and POPCNT; "
                  "this host lacks one");
#endif
  return Val_unit;
}

/* Tables for tests */

#define MAX_TABLES 2

/* Fills [t] with the tables the host runs, base first and the best last,
   and answers how many. */
static int host_tables(const nx_cpu_target **t) {
  int n = 0;
  t[n++] = &nx_cpu_base;
#if defined(__APPLE__) && defined(__aarch64__)
  if (nx_cpu_amx.name != NULL) t[n++] = &nx_cpu_amx;
#endif
  return n;
}

value nx_cpu_targets(value unit) {
  CAMLparam1(unit);
  CAMLlocal2(l, cell);
  const nx_cpu_target *t[MAX_TABLES];
  int n = host_tables(t);
  l = Val_emptylist;
  for (int i = n - 1; i >= 0; i--) {
    cell = caml_alloc(2, Tag_cons);
    Store_field(cell, 0, caml_copy_string(t[i]->name));
    Store_field(cell, 1, l);
    l = cell;
  }
  CAMLreturn(l);
}

/* Sets nx_cpu_llc to [bytes] and answers its value before: with_cache's
   scope. */
value nx_cpu_swap_llc(value bytes) {
  value was = Val_long(nx_cpu_llc);
  nx_cpu_llc = Long_val(bytes);
  return was;
}

/* Makes the host's table named [name] the one the kernels run, and answers
   the name of the one they ran before. */
value nx_cpu_swap_target(value name) {
  CAMLparam1(name);
  const nx_cpu_target *t[MAX_TABLES];
  int n = host_tables(t);
  for (int i = 0; i < n; i++) {
    if (strcmp(t[i]->name, String_val(name)) != 0) continue;
    const char *was = nx_cpu_table->name;
    nx_cpu_table = t[i];
    CAMLreturn(caml_copy_string(was));
  }
  caml_invalid_argument("Nx_cpu.with_target: not a table the host runs");
}
