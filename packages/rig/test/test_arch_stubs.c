/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* An instruction of each feature a host's arch states, run in a forked
   child, so that one the processor lacks kills the child alone.

   On x86-64, the psABI's example instruction of each feature of a level
   (table "Micro-Architecture Levels"). AVX512VL has no example: a 256-bit
   EVEX instruction stands for it. LZCNT runs as BSR on a processor without
   it, so its result is checked. On arm64, FEAT_FP16's half arithmetic,
   scalar and Advanced SIMD. */

#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

#if !defined(_WIN32) && (defined(__x86_64__) || defined(__aarch64__))
#include <errno.h>
#include <sys/wait.h>
#include <unistd.h>

/* The status of a child that exits with [body(arg)]: its exit code, or 128
   plus the signal that killed it; -1 if no child forks, and -2 if it cannot
   be waited for. */
static int child(int (*body)(int), int arg) {
  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) _exit(body(arg));
  int status;
  while (waitpid(pid, &status, 0) < 0)
    if (errno != EINTR) return -2;
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return WEXITSTATUS(status);
}
#endif

#if defined(__x86_64__) && !defined(_WIN32)

/* The exit status of a child whose LZCNT counted as BSR. */
#define BSR 3

__attribute__((noinline)) static void v2(void) {
  _Alignas(16) uint64_t m[2] = {0, 0};
  uint64_t a = 0, d = 0;
  __asm__ volatile("lock cmpxchg16b %2"
                   : "+a"(a), "+d"(d), "+m"(m[0])
                   : "b"((uint64_t)1), "c"((uint64_t)1), "m"(m[1])
                   : "memory", "cc");
  __asm__ volatile("lahf" ::: "rax", "cc");
  __asm__ volatile("popcnt %%rcx, %%rax" ::: "rax", "rcx", "cc");
  __asm__ volatile("addsubpd %%xmm1, %%xmm0" ::: "xmm0", "xmm1");
  __asm__ volatile("blendpd $1, %%xmm1, %%xmm0" ::: "xmm0", "xmm1");
  __asm__ volatile("pcmpestri $0, %%xmm1, %%xmm0"
                   :
                   :
                   : "rax", "rcx", "rdx", "xmm0", "xmm1", "cc");
  __asm__ volatile("phaddd %%xmm1, %%xmm0" ::: "xmm0", "xmm1");
}

/* Whether LZCNT counted 63 leading zeros of 1. */
__attribute__((noinline)) static int v3(void) {
  uint32_t word = 0;
  uint64_t one = 1, zeros = 0;
  __asm__ volatile("vpermd %%ymm1, %%ymm2, %%ymm0" ::: "xmm0", "xmm1", "xmm2");
  __asm__ volatile("andn %%rax, %%rcx, %%rdx" ::: "rax", "rcx", "rdx", "cc");
  __asm__ volatile("bzhi %%rax, %%rcx, %%rdx" ::: "rax", "rcx", "rdx", "cc");
  __asm__ volatile("vcvtph2ps %%xmm1, %%ymm0" ::: "xmm0", "xmm1");
  __asm__ volatile("vfmadd132pd %%ymm1, %%ymm2, %%ymm0"
                   :::"xmm0", "xmm1", "xmm2");
  __asm__ volatile("lzcnt %1, %0" : "=r"(zeros) : "r"(one) : "cc");
  __asm__ volatile("movbe %1, %0" : "=r"(word) : "m"(word));
  __asm__ volatile("xgetbv" : : "c"(0) : "rax", "rdx");
  __asm__ volatile("vzeroall" ::: "xmm0", "xmm1", "xmm2");
  return zeros == 63;
}

__attribute__((noinline)) static void v4(void) {
  __asm__ volatile("kmovw %%eax, %%k1" ::: "rax");
  __asm__ volatile("vdbpsadbw $0, %%zmm1, %%zmm2, %%zmm0"
                   :::"xmm0", "xmm1", "xmm2");
  __asm__ volatile("vplzcntd %%zmm1, %%zmm0" ::: "xmm0", "xmm1");
  __asm__ volatile("vpmullq %%zmm1, %%zmm2, %%zmm0" ::: "xmm0", "xmm1", "xmm2");
  __asm__ volatile("vpmullq %%ymm1, %%ymm2, %%ymm0" ::: "xmm0", "xmm1", "xmm2");
  __asm__ volatile("vzeroupper" ::: "xmm0", "xmm1", "xmm2");
}

/* The instructions of the levels above the baseline up to [n], an index of
   the psABI's levels. */
static int levels(int n) {
  if (n >= 1) v2();
  if (n >= 2 && !v3()) return BSR;
  if (n >= 3) v4();
  return 0;
}

/* The status of a child that runs the instructions of the levels up to
   [level] ([child]). */
value rig_test_runs_level(value level) {
  return Val_int(child(levels, Int_val(level)));
}

#else

value rig_test_runs_level(value level) {
  (void)level;
  return Val_int(-1);
}

#endif

#if defined(__aarch64__) && !defined(_WIN32)

static int fp16(int unused) {
  (void)unused;
  __asm__ volatile(".arch_extension fp16\n\t"
                   "fmul v0.8h, v1.8h, v2.8h\n\t"
                   "fadd h0, h1, h2" ::: "v0", "v1", "v2");
  return 0;
}

/* The status of a child that runs FEAT_FP16's instructions ([child]). */
value rig_test_runs_fp16(value unit) {
  (void)unit;
  return Val_int(child(fp16, 0));
}

#else

value rig_test_runs_fp16(value unit) {
  (void)unit;
  return Val_int(-1);
}

#endif
