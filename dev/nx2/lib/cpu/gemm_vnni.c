/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* x86-64's quads kernels, dots and axpys by AVX-VNNI-INT8, in place of
   gemm_avx2.c's in x86-64's table where the host has it (target.c): dune
   compiles this file with AVX-VNNI-INT8's instructions on top of nx.cpu's
   where the compiler has them. A compiler without them builds a no-op, and
   the table keeps AVX2's quads.

   VPDPBSSD (int8) and VPDPBUUD (uint8) add, into each 32-bit lane, the
   four products of the lane's bytes in two vectors, 32 products an
   instruction, each sum exact modulo 2^32. They are the MAC of
   gemm_avx2.h's tile, 6 rows by two vectors of words, a's word broadcast,
   of its dots, eight words a step, and of the axpys below. On one fudge
   core, in cache, the tile runs 302 G byte products a second, and
   oneDNN's 4 × 24 blocking for AVX2 VNNI 294. On fudge's six performance
   cores a decode step against a 4096 × 4096 weight stored [n × k] takes
   26 µs by these dots, 41 by AVX2's and 142 in chain order. */

#include <limits.h>
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

/* Quads' axpys, as gemm_avx2.c's, with b's rows zipped four at a time:
   VPUNPCKLBW and VPUNPCKHBW interleave rows t, t + 1 and t + 2, t + 3
   into pairs, VPUNPCKLWD and VPUNPCKHWD the pairs into words, each a
   column's four elements along k, against a's word broadcast. Eight
   shuffles zip 32 columns, 128 products, where AVX2's widening takes 16.
   Within 128-bit halves the words come out of column order: a row alone
   sums in that order and puts its sums in order at the end; the first of
   several rows puts the words it keeps in order, four VPERM2I128 more,
   for the others to read with no shuffle.

   On fudge's six performance cores by a 4096 × 4096 weight stored
   [k × n], against AVX2's axpys below 6 rows and the quads kernel over
   zipped rows from 6: 1 row took 114 µs against 147, 4 133 against 205,
   8 163 against 262, 32 435 against 552, 128 1.56 ms against 1.60, and
   MKL's s8u8s32 223 µs, 238, 276, 525 and 1.53 ms. One row from memory
   took 215 µs against 235, at the read's floor. No reference was
   ported. */
/* Rows t to t + 3 of b from column j, a row past k repeating the last,
   which a's zeros there cancel, and a's word at t: a's words lie on 4
   bytes, in place (gemm_quads.c's words_in_place) or copied, so the int32 load
   is aligned, here and in VWORD. */
#define VROWS                                    \
  const uint8_t *r0 = b + t * ldb + j;           \
  const uint8_t *r1 = t + 1 < k ? r0 + ldb : r0; \
  const uint8_t *r2 = t + 2 < k ? r1 + ldb : r1; \
  const uint8_t *r3 = t + 3 < k ? r2 + ldb : r2; \
  __m256i x = _mm256_set1_epi32(*(const int32_t *)(a + t));
/* 32 columns from [at] zipped: q0 holds columns 0-3 and 16-19, q1 4-7
   and 20-23, q2 8-11 and 24-27, q3 12-15 and 28-31. */
#define VZIP(at, q0, q1, q2, q3)                          \
  {                                                       \
    __m256i v0 = VLOAD(r0 + (at)), v1 = VLOAD(r1 + (at)); \
    __m256i v2 = VLOAD(r2 + (at)), v3 = VLOAD(r3 + (at)); \
    __m256i l01 = _mm256_unpacklo_epi8(v0, v1);           \
    __m256i h01 = _mm256_unpackhi_epi8(v0, v1);           \
    __m256i l23 = _mm256_unpacklo_epi8(v2, v3);           \
    __m256i h23 = _mm256_unpackhi_epi8(v2, v3);           \
    q0 = _mm256_unpacklo_epi16(l01, l23);                 \
    q1 = _mm256_unpackhi_epi16(l01, l23);                 \
    q2 = _mm256_unpacklo_epi16(h01, h23);                 \
    q3 = _mm256_unpackhi_epi16(h01, h23);                 \
  }
/* VZIP's order to column order. */
#define VORDER(q0, q1, q2, q3)                            \
  {                                                       \
    __m256i o0 = _mm256_permute2x128_si256(q0, q1, 0x20); \
    __m256i o1 = _mm256_permute2x128_si256(q2, q3, 0x20); \
    __m256i o2 = _mm256_permute2x128_si256(q0, q1, 0x31); \
    q3 = _mm256_permute2x128_si256(q2, q3, 0x31);         \
    q0 = o0, q1 = o1, q2 = o2;                            \
  }
/* 32 columns from [at] into [c0] to [c3], in column order and kept at
   [p] where KEEP. */
#define VSTEP(DOT, KEEP, at, c0, c1, c2, c3)        \
  {                                                 \
    __m256i q0, q1, q2, q3;                         \
    VZIP(at, q0, q1, q2, q3)                        \
    if (KEEP) {                                     \
      VORDER(q0, q1, q2, q3)                        \
      p[(at) / 8] = q0, p[(at) / 8 + 1] = q1;       \
      p[(at) / 8 + 2] = q2, p[(at) / 8 + 3] = q3;   \
    }                                               \
    c0 = DOT(x, q0, c0), c1 = DOT(x, q1, c1);       \
    c2 = DOT(x, q2, c2), c3 = DOT(x, q3, c3);       \
  }
#define VADD(p, c) VSTORE(p, _mm256_add_epi32(VLOAD(p), c));
#define VAHEAD(r) _mm_prefetch((const char *)((r) + ahead), _MM_HINT_T0);
/* A row: 64 columns at a time in 8 accumulators from zero, added to y at
   the end, then 16 in 128-bit vectors, which zip in column order, then
   one at a time. Where KEEP, at most QK rows of b and QCOLS columns,
   whose words it keeps at [pb] for the other rows. */
#define VAXPY_1(name, DOT, DOTX, T, KEEP)                                  \
  static void name(const uint8_t *a, const uint8_t *b, int64_t ldb,        \
                   int64_t k, int64_t n, uint32_t *y,                      \
                   const uint8_t *next, __m256i *pb) {                     \
    int64_t ahead = next - b;                                              \
    int64_t j = 0;                                                         \
    for (; j + 64 <= n; j += 64) {                                         \
      __m256i c0 = _mm256_setzero_si256(), c1 = c0, c2 = c0, c3 = c0;      \
      __m256i c4 = c0, c5 = c0, c6 = c0, c7 = c0;                          \
      for (int64_t t = 0; t < k; t += 4) {                                 \
        VROWS                                                              \
        __m256i *p = KEEP ? pb + t / 4 * (QCOLS / 8) + j / 8 : NULL;      \
        VAHEAD(r0) VAHEAD(r1) VAHEAD(r2) VAHEAD(r3)                        \
        VSTEP(DOT, KEEP, 0, c0, c1, c2, c3)                                \
        VSTEP(DOT, KEEP, 32, c4, c5, c6, c7)                               \
      }                                                                    \
      if (!KEEP) {                                                         \
        VORDER(c0, c1, c2, c3)                                             \
        VORDER(c4, c5, c6, c7)                                             \
      }                                                                    \
      VADD(y + j, c0) VADD(y + j + 8, c1) VADD(y + j + 16, c2)             \
      VADD(y + j + 24, c3) VADD(y + j + 32, c4) VADD(y + j + 40, c5)       \
      VADD(y + j + 48, c6) VADD(y + j + 56, c7)                            \
    }                                                                      \
    for (; j + 16 <= n; j += 16) {                                         \
      __m128i *s = (__m128i *)(y + j), c[4];                               \
      for (int v = 0; v < 4; v++) c[v] = _mm_loadu_si128(s + v);           \
      for (int64_t t = 0; t < k; t += 4) {                                 \
        VROWS                                                              \
        __m128i v0 = _mm_loadu_si128((const __m128i *)r0);                 \
        __m128i v1 = _mm_loadu_si128((const __m128i *)r1);                 \
        __m128i v2 = _mm_loadu_si128((const __m128i *)r2);                 \
        __m128i v3 = _mm_loadu_si128((const __m128i *)r3);                 \
        __m128i l01 = _mm_unpacklo_epi8(v0, v1);                           \
        __m128i h01 = _mm_unpackhi_epi8(v0, v1);                           \
        __m128i l23 = _mm_unpacklo_epi8(v2, v3);                           \
        __m128i h23 = _mm_unpackhi_epi8(v2, v3);                           \
        __m128i q[4] = {_mm_unpacklo_epi16(l01, l23),                      \
                        _mm_unpackhi_epi16(l01, l23),                      \
                        _mm_unpacklo_epi16(h01, h23),                      \
                        _mm_unpackhi_epi16(h01, h23)};                     \
        __m256i *p = KEEP ? pb + t / 4 * (QCOLS / 8) + j / 8 : NULL;      \
        for (int v = 0; v < 4; v++) {                                      \
          c[v] = DOTX(_mm256_castsi256_si128(x), q[v], c[v]);              \
          if (KEEP) _mm_storeu_si128((__m128i *)p + v, q[v]);              \
        }                                                                  \
      }                                                                    \
      for (int v = 0; v < 4; v++) _mm_storeu_si128(s + v, c[v]);           \
    }                                                                      \
    for (; j < n; j++)                                                     \
      for (int64_t t = 0; t < k; t++)                                      \
        y[j] += (uint32_t)((T)a[t] * (T)b[t * ldb + j]);                   \
  }
/* A group reads its rows' words in place. */
#define VPREP(R)
#define VWORD(i, t) (*(const int32_t *)(a + (i) * lda + 4 * (t)))
#define DOTX_S(a, b, c) _mm_dpbssd_epi32(c, a, b)
#define DOTX_U(a, b, c) _mm_dpbuud_epi32(c, a, b)
#define VAXPY(name, DOT, DOTX, T)              \
  VAXPY_1(name##_1, DOT, DOTX, T, 0)           \
  VAXPY_1(name##_keep, DOT, DOTX, T, 1)        \
  QAXPYS(name, T, 4, VPREP, VWORD, DOT)
VAXPY(quads_axpy_s8, DOT_S, DOTX_S, int8_t)
VAXPY(quads_axpy_u8, DOT_U, DOTX_U, uint8_t)

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
  t->quads_axpy[NX_INT8] = quads_axpy_s8;
  t->quads_axpy[NX_UINT8] = quads_axpy_u8;
  /* The axpys outran the zips at every count of rows they take (above,
     at the quads' axpys). */
  t->quads_zip_rows = INT_MAX;
}

#elif defined(__x86_64__)

void nx_cpu_set_vnni(nx_cpu_target *t) { (void)t; }

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_vnni;

#endif
