/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contraction kernels on the matrix unit of Apple's M1 (amx.h), and the amx
   table: base's, with these kernels for chain order.

   float32 adds 32 × 32 outputs per step of k as four 16 × 16 accumulators
   in Z, q = 2·(row half) + (column half): output (i, j) lives in Z row
   4·(i mod 16) + q, lane j mod 16. A step loads a's 32 elements into Y and
   b's 32 into X, 128 bytes each, and runs four fma32, one per accumulator.
   Four steps load at a time, into all eight X and Y registers. Each output
   adds a[i]·b[j] fused, in increasing k: the chain NEON's kernels compute,
   bit for bit on every result but a NaN, which the unit gives as the
   default NaN.

   A call makes the unit's state live and ends it, so a thread holds none
   between calls. A call of a whole KC block is 4,096 fma32, against which
   the tile's 64 stores from Z cost about 3%; a first block starts from the
   zeros SET leaves in Z, so that a product of at most KC steps loads no
   tile. An M1 has one unit per cluster of cores, which every core of the
   cluster feeds: products run on all the performance cores, which on the
   M1 Max beat one thread per unit (1024³: 1.09 against 2.02 ms).

   A kernel's speed prices its jobs (cpu.h): at the unit's, a 128³ product
   runs on one thread, which beats eight (5.9 against 8.0 us). The cores of
   a cluster share its unit, so that the M1 Max's eight performance cores
   hold two: products of at most 8 rows, as decoding is, run faster on the
   cores' vector units (base's kernels, the table's other), and the unit
   packs only in jobs of at most one thread per unit (gemm.c). */

#include "cpu.h"

#if defined(__APPLE__) && defined(__aarch64__)

#include <sys/sysctl.h>

#include "amx.h"

/* The unit's speed (cpu.h): fma32 at 1.36 TFLOP/s from one thread of the
   M1 Max, against the 100 GB/s a performance core's memcpy moves. */
#define SPEED 13

/* The Z row of output (i, j)'s half [h] of a row of float32: row i's
   accumulator for column half h. */
#define ROW32(i, h) (4 * ((i) % 16) + 2 * ((i) / 16) + (h))

/* Moves the 32 × 32 tile at [c] into Z and back: a row of the tile is the
   rows ROW32(i, 0) and ROW32(i, 1), adjacent, so that where its rows start
   on 128 bytes [paired] moves each in one instruction, else in two. */
static void zload(const float *c, int64_t ldc, int paired) {
  for (int i = 0; i < 32; i++)
    if (paired) NX_AMX_LDZ((uint64_t)(c + i * ldc) | NX_AMX_PAIR, ROW32(i, 0));
    else
      for (int h = 0; h < 2; h++) NX_AMX_LDZ(c + i * ldc + 16 * h, ROW32(i, h));
}

static void zstore(float *c, int64_t ldc, int paired) {
  for (int i = 0; i < 32; i++)
    if (paired) NX_AMX_STZ((uint64_t)(c + i * ldc) | NX_AMX_PAIR, ROW32(i, 0));
    else
      for (int h = 0; h < 2; h++) NX_AMX_STZ(c + i * ldc + 16 * h, ROW32(i, h));
}

static void kernel_f32(int64_t k, const void *va, int64_t lda,
                       const void *vb, void *vc, int64_t ldc,
                       nx_cpu_from from) {
  const float *a = va, *b = vb;
  float *c = vc;
  int paired = ((uintptr_t)c | (uintptr_t)(ldc * 4)) % 128 == 0;
  NX_AMX_SET();
  if (from == NX_CPU_FROM_TILE) zload(c, ldc, paired);
  int64_t p = 0;
  for (; p + 4 <= k; p += 4) {
    for (int s = 0; s < 4; s++) {
      NX_AMX_LDY((uint64_t)(a + (p + s) * lda) | NX_AMX_PAIR, 2 * s);
      NX_AMX_LDX((uint64_t)(b + (p + s) * 32) | NX_AMX_PAIR, 2 * s);
    }
    for (int s = 0; s < 4; s++) {
      int o = 128 * s;
      NX_AMX_FMA32(o, o, 0);
      NX_AMX_FMA32(o + 64, o, 1);
      NX_AMX_FMA32(o, o + 64, 2);
      NX_AMX_FMA32(o + 64, o + 64, 3);
    }
  }
  for (; p < k; p++) {
    NX_AMX_LDY((uint64_t)(a + p * lda) | NX_AMX_PAIR, 0);
    NX_AMX_LDX((uint64_t)(b + p * 32) | NX_AMX_PAIR, 0);
    NX_AMX_FMA32(0, 0, 0);
    NX_AMX_FMA32(64, 0, 1);
    NX_AMX_FMA32(0, 64, 2);
    NX_AMX_FMA32(64, 64, 3);
  }
  zstore(c, ldc, paired);
  NX_AMX_CLR();
}

/* Transposes 32 rows × 32 steps from [s] into [d] on the unit. Row i of
   row half h loads into the Z rows 4i + 2h and the next, its steps 0-15 and
   16-31: Z holds four quarters, q = 2·(row half) + (step half), each row i
   of a quarter in Z row 4i + q. Step k is then column k mod 16 of quarters
   k / 16 and 2 + k / 16, moved into Y and stored. Rows and steps that start
   on 128 bytes move in one instruction each, others in two. A move rounds
   nothing, so every bit moves, a NaN's payload included. */
static void transpose32(const float *s, int64_t ld, float *d, int64_t pitch) {
  int in_pairs = ((uintptr_t)s | (uintptr_t)(ld * 4)) % 128 == 0;
  int out_pairs = ((uintptr_t)d | (uintptr_t)(pitch * 4)) % 128 == 0;
  for (int h = 0; h < 2; h++)
    for (int i = 0; i < 16; i++) {
      const float *r = s + (16 * h + i) * ld;
      if (in_pairs) NX_AMX_LDZ((uint64_t)r | NX_AMX_PAIR, 4 * i + 2 * h);
      else {
        NX_AMX_LDZ(r, 4 * i + 2 * h);
        NX_AMX_LDZ(r + 16, 4 * i + 2 * h + 1);
      }
    }
  for (int k = 0; k < 32; k++) {
    int c = 4 * (k % 16) + k / 16;
    float *r = d + k * pitch;
    NX_AMX_EXTRV32(0, c);
    NX_AMX_EXTRV32(64, c + 2);
    if (out_pairs) NX_AMX_STY((uint64_t)r | NX_AMX_PAIR, 0);
    else {
      NX_AMX_STY(r, 0);
      NX_AMX_STY(r + 16, 1);
    }
  }
}

/* The pack. Contiguous rows move 32 elements at a time through X, four
   rows to the eight registers; rows contiguous across, 32 × 32 at a time
   through Z (transpose32). The rest moves element by element. */
static void pack_f32(int64_t n0, int64_t n1, const void *vs, int64_t s0,
                     int64_t s1, void *vd, int64_t pitch) {
  const float *s = vs;
  float *d = vd;
  int64_t i32 = n0 / 32 * 32, j32 = n1 / 32 * 32;
  if (s0 == 1) {
    int64_t j4 = n1 / 4 * 4;
    NX_AMX_SET();
    for (int64_t i = 0; i < i32; i += 32)
      for (int64_t j = 0; j < j4; j += 4) {
        for (int u = 0; u < 4; u++) {
          NX_AMX_LDX(s + (j + u) * s1 + i, 2 * u);
          NX_AMX_LDX(s + (j + u) * s1 + i + 16, 2 * u + 1);
        }
        for (int u = 0; u < 4; u++) {
          NX_AMX_STX(d + (j + u) * pitch + i, 2 * u);
          NX_AMX_STX(d + (j + u) * pitch + i + 16, 2 * u + 1);
        }
      }
    NX_AMX_CLR();
    for (int64_t j = 0; j < n1; j++)
      for (int64_t i = j < j4 ? i32 : 0; i < n0; i++)
        d[j * pitch + i] = s[j * s1 + i];
    return;
  }
  if (i32 > 0 && j32 > 0) {
    NX_AMX_SET();
    for (int64_t i = 0; i < i32; i += 32)
      for (int64_t j = 0; j < j32; j += 32)
        transpose32(s + i * s0 + j, s0, d + j * pitch + i, pitch);
    NX_AMX_CLR();
  }
  for (int64_t j = 0; j < n1; j++)
    for (int64_t i = j < j32 ? i32 : 0; i < n0; i++)
      d[j * pitch + i] = s[i * s0 + j];
}

static int sysctl_int(const char *name) {
  int v = 0;
  size_t n = sizeof v;
  return sysctlbyname(name, &v, &n, NULL, 0) == 0 ? v : 0;
}

/* Whether the host has the matrix unit these kernels use: the M1's cores,
   hw.cpufamily CPUFAMILY_ARM_FIRESTORM_ICESTORM of <mach/machine.h>. M2
   and M3 carry the unit too and join when a machine here runs the suite on
   them; M4 and later have SME instead. */
int nx_cpu_has_amx(void) {
  return (uint32_t)sysctl_int("hw.cpufamily") == 0x1b588bb3u;
}

/* The table's float32 kernels: the unit's, priced against base's. */
void nx_cpu_set_amx(nx_cpu_target *t) {
  /* The cores that share a unit: a cluster's, which share its L2. */
  int shared = sysctl_int("hw.perflevel0.cpusperl2");
  t->gemm[NX_FLOAT32] = (nx_cpu_gemm){
      .kernel = {kernel_f32, 32, 32, SPEED, shared > 0 ? shared : 1},
      .mc = 128,
      .kc = 1024,
      .nc = 2048,
      .pack = pack_f32,
      .other = &nx_cpu_base.gemm[NX_FLOAT32]};
}

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_amx;

#endif
