/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* x86-64's contraction kernels, in the v3 table: dune compiles this file
   with AVX2 and FMA.

   A step of k adds 6 rows × two vectors of outputs, 6 × 16 of float32 or
   6 × 8 of float64: 12 accumulators, two vectors of b and one broadcast of
   a, 15 of the 16 vector registers. Each output adds a[i] · b[j] by VFMADD,
   one rounding, in increasing k. Two FMA ports with a latency of four want
   eight independent accumulators; twelve leave room for the loads. The
   accumulators are named variables: gcc keeps an array of them in memory
   and stores each at every step.

   a's elements are broadcast one at a time, so that its rows are read in
   place wherever they lie. A tile of fewer rows points its missing rows at
   its last and stores only its own; one of fewer columns loads b and its
   outputs and stores them through masks, so that it reads no column past
   its own (NX_CPU_B_COLUMNS). That rests on VMASKMOV's masked-off lanes
   never faulting, which Intel's SDM states and AMD's APM leaves to the
   implementation; no AMD core has run it. Before a tile, the next tile's
   first lines of a and of the outputs are prefetched: its rows start
   cold, where the hardware's prefetchers follow each row once it
   streams. */

#include <math.h>
#include <string.h>

#include "cpu.h"

#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)

#include "gemm_avx2.h"

/* The kernels' speed (cpu.h): 151 GFLOP/s of float32 and 75.7 of float64
   on a performance core of fudge, against the 151 GB/s its memcpy moves. */
#define SPEED32 1.0
#define SPEED64 0.5

/* The integers' kernels' speed, a product counted as float's two flops: on
   one i9-9900K core, 1024³ products of 32-bit integers ran 30.7 G
   products/s, of 16-bit ones 81.9, of 8-bit ones 83.7, and int8 quads
   75.8, 19 G products of their words, against 138 GFLOP/s of float32. */
#define SPEED32I 0.4
#define SPEED16 1.2
#define SPEED8 1.2
#define SPEEDQ 0.25

/* The masks of a vector's first q lanes, q from 0 to its lanes: a run of
   ones then zeros, read from q lanes before the zeros. */
static const int32_t ones32[16] = {-1, -1, -1, -1, -1, -1, -1, -1};
static const int64_t ones64[8] = {-1, -1, -1, -1};
#define MASK32(q) _mm256_loadu_si256((const __m256i *)(ones32 + 8 - (q)))
#define MASK64(q) _mm256_loadu_si256((const __m256i *)(ones64 + 4 - (q)))

#define LOADC(LOAD, MLOAD, ZERO, W, i)                             \
  if (from == NX_CPU_FROM_ZERO || i >= rows) {                     \
    c##i##0 = ZERO;                                                \
    c##i##1 = ZERO;                                                \
  } else if (n == 2 * W) {                                         \
    c##i##0 = LOAD(y + i * ldc);                                   \
    c##i##1 = LOAD(y + i * ldc + W);                               \
  } else {                                                         \
    c##i##0 = MLOAD(y + i * ldc, m0);                              \
    c##i##1 = MLOAD(y + i * ldc + W, m1);                          \
  }
#define STOREC(STORE, MSTORE, W, i)            \
  if (i < rows && n == 2 * W) {                \
    STORE(y + i * ldc, c##i##0);               \
    STORE(y + i * ldc + W, c##i##1);           \
  } else if (i < rows) {                       \
    MSTORE(y + i * ldc, m0, c##i##0);          \
    MSTORE(y + i * ldc + W, m1, c##i##1);      \
  }

#define KERNEL(name, T, V, W, MASK, LOAD, MLOAD, BLOAD, STORE, MSTORE,      \
               BCAST, FMA, ZERO)                                            \
  static void name(int64_t m, int64_t n, int64_t k, const void *va,         \
                   int64_t a_row, int64_t a_k, const void *vb, int64_t b_k, \
                   void *vc, int64_t ldc, nx_cpu_from from) {               \
    __m256i m0 = MASK(n < W ? n : W), m1 = MASK(n > W ? n - W : 0);         \
    for (int64_t i0 = 0; i0 < m; i0 += MR) {                                \
      int64_t rows = m - i0 < MR ? m - i0 : MR;                             \
      const T *a = (const T *)va + i0 * a_row, *b = vb;                     \
      T *y = (T *)vc + i0 * ldc;                                            \
      for (int64_t i = MR; i < 2 * MR && i0 + i < m; i++) {                 \
        _mm_prefetch((const char *)(a + i * a_row), _MM_HINT_T0);           \
        _mm_prefetch((const char *)(y + i * ldc), _MM_HINT_T0);             \
      }                                                                     \
      ROWS(DECL, V, T);                                                     \
      ROWS(LOADC, LOAD, MLOAD, ZERO, W);                                    \
      for (int64_t p = 0, q = 0; p < k; p++, q += a_k, b += b_k) {          \
        V b0 = BLOAD(b, m0), b1 = BLOAD(b + W, m1), ai;                     \
        ROWS(STEP, BCAST, FMA);                                             \
      }                                                                     \
      ROWS(STOREC, STORE, MSTORE, W);                                       \
    }                                                                       \
  }

/* A tile of every column loads b whole, one of fewer through masks: two
   instances, so that the masks cost the whole tile's loop nothing. */
#define WHOLE32(p, m) _mm256_loadu_ps(p)
#define WHOLE64(p, m) _mm256_loadu_pd(p)
#define TILES(name, T, NR)                                                   \
  static void name(int64_t m, int64_t n, int64_t k, const void *a,          \
                   int64_t a_row, int64_t a_k, const void *b, int64_t b_k,  \
                   void *c, int64_t ldc, nx_cpu_from from) {                \
    (n == NR ? whole_##T : edge_##T)(m, n, k, a, a_row, a_k, b, b_k, c, ldc, \
                                     from);                                 \
  }

KERNEL(whole_f32, float, __m256, 8, MASK32, _mm256_loadu_ps,
       _mm256_maskload_ps, WHOLE32, _mm256_storeu_ps, _mm256_maskstore_ps,
       _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_setzero_ps())
KERNEL(edge_f32, float, __m256, 8, MASK32, _mm256_loadu_ps,
       _mm256_maskload_ps, _mm256_maskload_ps, _mm256_storeu_ps,
       _mm256_maskstore_ps, _mm256_broadcast_ss, _mm256_fmadd_ps,
       _mm256_setzero_ps())
TILES(kernel_f32, f32, 16)
KERNEL(whole_f64, double, __m256d, 4, MASK64, _mm256_loadu_pd,
       _mm256_maskload_pd, WHOLE64, _mm256_storeu_pd, _mm256_maskstore_pd,
       _mm256_broadcast_sd, _mm256_fmadd_pd, _mm256_setzero_pd())
KERNEL(edge_f64, double, __m256d, 4, MASK64, _mm256_loadu_pd,
       _mm256_maskload_pd, _mm256_maskload_pd, _mm256_storeu_pd,
       _mm256_maskstore_pd, _mm256_broadcast_sd, _mm256_fmadd_pd,
       _mm256_setzero_pd())
TILES(kernel_f64, f64, 8)

/* The integers' kernels, modulo their width, as unsigned numbers, which
   gives a signed accumulator's bits too; 64-bit ones have no vector
   multiply and keep the portable kernel. A step adds MR rows × two
   vectors of outputs, as the floats' kernels do, each by a multiply and an
   add: twelve accumulators, b's two vectors, a's broadcast and a product,
   the 16 vector registers. b is read in packed slivers of NR columns; a
   tile of fewer columns moves its outputs through a tile on the stack,
   since AVX2 masks no 8- or 16-bit stores.

   32-bit outputs multiply by VPMULLD, 16-bit ones by VPMULLW. AVX2 has no
   8-bit multiply: an 8-bit output's product is the low byte of the 16-bit
   product of the low bytes, so a vector of 32 bytes of b multiplies as 16
   words, which give the even columns' bytes, and shifted right by a byte,
   the odd columns'; the stores merge them back.

   Quads (gemm.c) of int8 or uint8 follow XNNPACK's qs8 GEMM for AVX2
   (src/qs8-gemm/MRx8c8-avx2.c.in, BSD-3-Clause; no code of it is copied
   here): each operand widened to 16 bits by VPMOVSXBW or VPMOVZXBW, then
   VPMADDWD adds pairs of their products into 32 bits, exactly: a product
   of 8-bit numbers is at most 2^14 in magnitude, an unsigned one 255^2.
   XNNPACK groups eight elements of k per column, nx's quads four: a
   vector of b widens four columns' words, a's word broadcast four times
   multiplies it, and each column's two pair sums add once, when the tile
   is stored. */

#define LOADB16(b, b0, b1) LOAD2(b, b0, b1, 16)
#define START16(y, c0, c1) LOAD2(y, c0, c1, 16)
#define FINISH16(y, c0, c1, from) STORE2(y, c0, c1, 16)
#define BCAST16(p) _mm256_set1_epi16((int16_t) * (p))
#define MAC32(a, b, c) _mm256_add_epi32(c, _mm256_mullo_epi32(a, b))
#define MAC16(a, b, c) _mm256_add_epi16(c, _mm256_mullo_epi16(a, b))

/* 8-bit outputs: even columns in the low bytes of b0's and c0's words,
   odd ones in c1's and b1's. a's byte is broadcast into both bytes of a
   word, by one VPBROADCASTB: the high byte changes no low byte of a
   product. */
#define BCAST8(p) _mm256_set1_epi8((char)*(p))
#define LOADB8(b, b0, b1) (b0 = VLOAD(b), b1 = _mm256_srli_epi16(b0, 8))
#define START8(y, c0, c1) (c0 = VLOAD(y), c1 = _mm256_srli_epi16(c0, 8))
#define FINISH8(y, c0, c1, from)                                         \
  VSTORE(y, _mm256_or_si256(_mm256_and_si256(c0, _mm256_set1_epi16(255)), \
                            _mm256_slli_epi16(c1, 8)))

INTS(kernel_u32, uint32_t, 16, LOADB32, BCAST32, MAC32, START32, FINISH32)
INTS(kernel_u16, uint16_t, 32, LOADB16, BCAST16, MAC16, START16, FINISH16)
INTS(kernel_u8, uint8_t, 32, LOADB8, BCAST8, MAC16, START8, FINISH8)

/* Quads: c0 holds columns 0 to 3, two pair sums each, c1 columns 4 to 7.
   The tile starts from zero and adds its outputs at the end. */
#define WIDEN_S(p) _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(p)))
#define WIDEN_U(p) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(p)))
#define LOADB_S(b, b0, b1) (b0 = WIDEN_S(b), b1 = WIDEN_S((b) + 4))
#define LOADB_U(b, b0, b1) (b0 = WIDEN_U(b), b1 = WIDEN_U((b) + 4))
#define BCAST_S(p) _mm256_cvtepi8_epi16(_mm_set1_epi32((int32_t) * (p)))
#define BCAST_U(p) _mm256_cvtepu8_epi16(_mm_set1_epi32((int32_t) * (p)))
#define MADD(a, b, c) _mm256_add_epi32(c, _mm256_madd_epi16(a, b))
#define START_Q(y, c0, c1) (c0 = c1 = _mm256_setzero_si256())
/* VPHADDD sums each column's pairs within 128-bit halves, giving columns
   0 1 4 5 | 2 3 6 7; VPERMQ puts them in order. */
#define FINISH_Q(y, c0, c1, from)                                         \
  {                                                                       \
    __m256i s = _mm256_permute4x64_epi64(_mm256_hadd_epi32(c0, c1), 0xD8); \
    if (from == NX_CPU_FROM_TILE) s = _mm256_add_epi32(s, VLOAD(y));      \
    VSTORE(y, s);                                                         \
  }

INTS(kernel_sq, uint32_t, 8, LOADB_S, BCAST_S, MADD, START_Q, FINISH_Q)
INTS(kernel_uq, uint32_t, 8, LOADB_U, BCAST_U, MADD, START_Q, FINISH_Q)

QDOT(quads_dot_s8, WIDEN_S, 4, MADD)
QDOT(quads_dot_u8, WIDEN_U, 4, MADD)

/* Lane order's dots and axpys: 16 lanes are 2 vectors of float32, 4 of
   float64; 4 rows of float32 or 2 of float64 hold 8 accumulators and b's
   vectors. */
#define VT __m256
#define LOAD _mm256_loadu_ps
#define STORE _mm256_storeu_ps
#define BCAST _mm256_set1_ps
#define FMA(c, a, b) _mm256_fmadd_ps(a, b, c)
#include "gemm_lanes.h"
DOT(dot1_f32, float, 8, 2, 1, fmaf)
DOT(dot2_f32, float, 8, 2, 2, fmaf)
DOT(dot4_f32, float, 8, 2, 4, fmaf)
DOTS(dot_f32, float, 4, dot4_f32, dot2_f32, dot1_f32)
AXPY(axpy_f32, float, 8, fmaf)
#undef VT
#undef LOAD
#undef STORE
#undef BCAST
#undef FMA
#define VT __m256d
#define LOAD _mm256_loadu_pd
#define STORE _mm256_storeu_pd
#define BCAST _mm256_set1_pd
#define FMA(c, a, b) _mm256_fmadd_pd(a, b, c)
DOT(dot1_f64, double, 4, 4, 1, fma)
DOT(dot2_f64, double, 4, 4, 2, fma)
DOTS(dot_f64, double, 2, dot2_f64, dot2_f64, dot1_f64)
AXPY(axpy_f64, double, 4, fma)

/* The rows of a block of a (cpu.h's mc). */
#define MC 96

/* The integers' kernels of [nr] columns at [speed] (cpu.h). They stream
   b past as many rows as the floats' do: on the i9-9900K an int32 product
   of 128 rows by a 4096 × 4096 weight stored n × k took 10.2 ms so, 11.6
   in chain order (few MC); int8 quads 4.09 against 4.22. */
static nx_cpu_gemm integers(nx_cpu_kernel f, int nr, double speed) {
  return (nx_cpu_gemm){.kernel = {.f = f,
                                  .mr = MR,
                                  .nr = nr,
                                  .a = NX_CPU_A_ELEMENTS,
                                  .b = NX_CPU_B_SLIVER,
                                  .flops_per_byte = speed,
                                  .shared = 1},
                       .mc = MC,
                       .kc = 384,
                       .nc = 3072,
                       .few = FEW};
}

void nx_cpu_set_avx2(nx_cpu_target *t) {
  t->gemm[NX_FLOAT32] = (nx_cpu_gemm){
      .kernel = {.f = kernel_f32,
                 .mr = MR,
                 .nr = 16,
                 .a = NX_CPU_A_ELEMENTS,
                 .b = NX_CPU_B_COLUMNS,
                 .flops_per_byte = SPEED32,
                 .shared = 1},
      .mc = MC,
      .kc = 384,
      .nc = 3072,
      .few = FEW};
  t->gemm[NX_FLOAT64] = (nx_cpu_gemm){
      .kernel = {.f = kernel_f64,
                 .mr = MR,
                 .nr = 8,
                 .a = NX_CPU_A_ELEMENTS,
                 .b = NX_CPU_B_COLUMNS,
                 .flops_per_byte = SPEED64,
                 .shared = 1},
      .mc = MC,
      .kc = 256,
      .nc = 3072,
      .few = FEW};
  t->dot[NX_FLOAT32] = dot_f32;
  t->dot[NX_FLOAT64] = dot_f64;
  t->axpy[NX_FLOAT32] = axpy_f32;
  t->axpy[NX_FLOAT64] = axpy_f64;
  t->gemm[NX_INT8] = t->gemm[NX_UINT8] = integers(kernel_u8, 32, SPEED8);
  t->gemm[NX_INT16] = t->gemm[NX_UINT16] = integers(kernel_u16, 32, SPEED16);
  t->gemm[NX_INT32] = t->gemm[NX_UINT32] = integers(kernel_u32, 16, SPEED32I);
  t->quads[NX_INT8] = integers(kernel_sq, 8, SPEEDQ);
  t->quads[NX_UINT8] = integers(kernel_uq, 8, SPEEDQ);
  t->quads_dot[NX_INT8] = quads_dot_s8;
  t->quads_dot[NX_UINT8] = quads_dot_u8;
}

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_avx2;

#endif
