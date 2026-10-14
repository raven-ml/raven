/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* x86-64's quads kernels and dots by AVX-VNNI-INT8, in place of
   gemm_avx2.c's in the v3 table where the host has it (target.c): dune
   compiles this file with v3's instructions, and AVX-VNNI-INT8's where the
   compiler has them. A compiler without them builds a no-op, and v3 keeps
   AVX2's quads.

   VPDPBSSD (int8) and VPDPBUUD (uint8) add, into each 32-bit lane, the
   four products of the lane's bytes in two vectors, 32 products an
   instruction, each sum exact modulo 2^32. They are the MAC of
   gemm_avx2.h's tile, 6 rows by two vectors of words, a's word broadcast,
   and of its dots, eight words a step. On one fudge core, in cache, the
   tile runs 302 G byte products a second, and oneDNN's 4 × 24 blocking
   for AVX2 VNNI 294. On fudge's six performance cores a decode step
   against a 4096 × 4096 weight stored [n × k] takes 26 µs by these dots,
   41 by AVX2's and 142 in chain order. */

#include <string.h>

#include "cpu.h"

#if defined(__x86_64__) && defined(__AVXVNNIINT8__)

#include "gemm_avx2.h"

#define DOT_S(a, b, c) _mm256_dpbssd_epi32(c, a, b)
#define DOT_U(a, b, c) _mm256_dpbuud_epi32(c, a, b)

INTS(kernel_s8, uint32_t, 16, LOADB32, BCAST32, DOT_S, START32, FINISH32)
INTS(kernel_u8, uint32_t, 16, LOADB32, BCAST32, DOT_U, START32, FINISH32)
QDOT(quads_dot_s8, VLOAD, 8, DOT_S)
QDOT(quads_dot_u8, VLOAD, 8, DOT_U)

/* The tile's speed (cpu.h): 302 G products of bytes a second, 151 G of
   words counted as flops, on a performance core of fudge, against the
   151 GB/s its memcpy moves. */
#define SPEED 1.0

/* The rows of a block of a (cpu.h's mc): on fudge's six performance
   cores, contract-i8-i32-1024 took 0.76 ms at 96 rows, 0.69 at 192 and
   0.71 at 288: 192 makes a's blocks as many as the threads. A KC of 128
   to 512 words gave the same 0.69 within 1%. */
#define MC 192

static nx_cpu_gemm quads(nx_cpu_kernel f) {
  return (nx_cpu_gemm){.kernel = {.f = f,
                                  .mr = MR,
                                  .nr = 16,
                                  .a = NX_CPU_A_ELEMENTS,
                                  .b = NX_CPU_B_SLIVER,
                                  .flops_per_byte = SPEED,
                                  .shared = 1},
                       .mc = MC,
                       .kc = 384,
                       .nc = 3072,
                       .few = FEW};
}

void nx_cpu_set_vnni(nx_cpu_target *t) {
  t->quads[NX_INT8] = quads(kernel_s8);
  t->quads[NX_UINT8] = quads(kernel_u8);
  t->quads_dot[NX_INT8] = quads_dot_s8;
  t->quads_dot[NX_UINT8] = quads_dot_u8;
  /* From this kernel's tile, the zips run it in place of AVX2's axpys. */
  t->quads_zip_rows = MR;
}

#elif defined(__x86_64__)

void nx_cpu_set_vnni(nx_cpu_target *t) { (void)t; }

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_vnni;

#endif
