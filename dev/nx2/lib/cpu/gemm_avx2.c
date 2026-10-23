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

#include <limits.h>
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

/* Quads' axpys: b's rows read in place, two at a time interleaved by
   VPUNPCKLBW and VPUNPCKHBW into pairs of elements along k per column,
   widened to 16 bits, each column's pair against the broadcast pair of
   a's elements by VPMADDWD, exact as in the quads kernel. Rows past k
   repeat the last, which a's elements there, zeros, cancel. Each row of b
   prefetches the same row of [next] (gemm.c says why, above AXPY_ROWS):
   on the i9-9900K one row by a 4096 × 4096 weight took 141 µs against
   263 without, from memory 508 against 964.

   The interleaving and widening run on port 5 alone, a shuffle for each
   VPMADDWD, which bounds a row alone. Several rows widen b once: the
   first runs as a row alone and keeps the pairs it widens, and the others
   broadcast a's pairs, widened on the stack, against them with no
   shuffle. On the i9-9900K at 16 threads by a 4096 × 4096 weight stored
   [k × n], against a row at a time below 6 rows and the quads kernel over
   zipped rows from 6: 2 rows took 200 µs against 208, 4 243 against 377,
   5 263 against 461, 8 358 against 575, 32 1.04 ms against 1.38, 128
   4.18 against 4.80. Groups of 6 rather than 4 ran 6 rows on one core
   1.16 times as fast. No reference was ported. */
#define CVT_S _mm256_cvtepi8_epi16
#define CVT_U _mm256_cvtepu8_epi16
/* 16 columns from [at] of rows [r0] and [r1] times a's pair [x], their
   pairs kept at [p] where KEEP. */
#define QPAIR(S, KEEP, c0, c1, x, r0, r1, at, p)                \
  {                                                             \
    __m128i v0 = _mm_loadu_si128((const __m128i *)((r0) + at)); \
    __m128i v1 = _mm_loadu_si128((const __m128i *)((r1) + at)); \
    __m256i lo = CVT_##S(_mm_unpacklo_epi8(v0, v1));            \
    __m256i hi = CVT_##S(_mm_unpackhi_epi8(v0, v1));            \
    if (KEEP) (p)[(at) / 8] = lo, (p)[(at) / 8 + 1] = hi;       \
    c0 = MADD(x, lo, c0);                                       \
    c1 = MADD(x, hi, c1);                                       \
  }
/* 16 columns from [at]: rows t and t + 1, then t + 2 and t + 3. */
#define QZIP(S, KEEP, c0, c1, at)              \
  QPAIR(S, KEEP, c0, c1, x01, r0, r1, at, p01) \
  QPAIR(S, KEEP, c0, c1, x23, r2, r3, at, p01 + QCOLS / 8)
/* Rows t to t + 3 of b from column j, a's pairs at t and t + 2, and
   where the pairs of b's rows are kept. */
#define QROWS(S, KEEP)                                           \
  const uint8_t *r0 = b + t * ldb + j;                           \
  const uint8_t *r1 = t + 1 < k ? r0 + ldb : r0;                 \
  const uint8_t *r2 = t + 2 < k ? r1 + ldb : r1;                 \
  const uint8_t *r3 = t + 3 < k ? r2 + ldb : r2;                 \
  __m256i x = BCAST_##S((const uint32_t *)(a + t));              \
  __m256i x01 = _mm256_shuffle_epi32(x, 0x00);                   \
  __m256i x23 = _mm256_shuffle_epi32(x, 0x55);                   \
  __m256i *p01 = KEEP ? pb + t / 2 * (QCOLS / 8) + j / 8 : NULL;
/* A row of [next] where [r] is a row of b. */
#define QAHEAD(r) _mm_prefetch((const char *)((r) + ahead), _MM_HINT_T0)
/* A row: 64 columns at a time in 8 accumulators, then 16 in 2, then one
   at a time. Where KEEP, at most QK rows of b and QCOLS columns, whose
   pairs it keeps at [pb] for the other rows. */
#define QAXPY_1(name, S, T, KEEP)                                   \
  static void name(const uint8_t *a, const uint8_t *b, int64_t ldb, \
                   int64_t k, int64_t n, uint32_t *y,               \
                   const uint8_t *next, __m256i *pb) {              \
    int64_t ahead = next - b;                                       \
    int64_t j = 0;                                                  \
    for (; j + 64 <= n; j += 64) {                                  \
      __m256i c[8];                                                 \
      for (int v = 0; v < 8; v++) c[v] = VLOAD(y + j + 8 * v);      \
      for (int64_t t = 0; t < k; t += 4) {                          \
        QROWS(S, KEEP)                                              \
        QAHEAD(r0);                                                 \
        QAHEAD(r1);                                                 \
        QAHEAD(r2);                                                 \
        QAHEAD(r3);                                                 \
        QZIP(S, KEEP, c[0], c[1], 0)                                \
        QZIP(S, KEEP, c[2], c[3], 16)                               \
        QZIP(S, KEEP, c[4], c[5], 32)                               \
        QZIP(S, KEEP, c[6], c[7], 48)                               \
      }                                                             \
      for (int v = 0; v < 8; v++) VSTORE(y + j + 8 * v, c[v]);      \
    }                                                               \
    for (; j + 16 <= n; j += 16) {                                  \
      __m256i c0 = VLOAD(y + j), c1 = VLOAD(y + j + 8);             \
      for (int64_t t = 0; t < k; t += 4) {                          \
        QROWS(S, KEEP)                                              \
        QZIP(S, KEEP, c0, c1, 0)                                    \
      }                                                             \
      VSTORE(y + j, c0);                                            \
      VSTORE(y + j + 8, c1);                                        \
    }                                                               \
    for (; j < n; j++)                                              \
      for (int64_t t = 0; t < k; t++)                               \
        y[j] += (uint32_t)((T)a[t] * (T)b[t * ldb + j]);            \
  }
/* A group's rows' pairs of elements, widened into words at [w]. An odd
   k's last pair ends on a's zero past it. */
#define QPREP(S, T, R)                                                 \
  _Alignas(32) uint32_t w[R][QK / 2];                                  \
  for (int i = 0; i < R; i++) {                                        \
    const T *e = (const T *)(a + i * lda);                             \
    if (kw == QK / 2) {                                                \
      VSTORE(w[i], CVT_##S(_mm_loadu_si128((const __m128i *)e)));      \
      continue;                                                        \
    }                                                                  \
    for (int64_t t = 0; t < kw; t++)                                   \
      w[i][t] =                                                        \
          (uint16_t)e[2 * t] | (uint32_t)(uint16_t)e[2 * t + 1] << 16; \
  }
#define QPREP_S(R) QPREP(S, int8_t, R)
#define QPREP_U(R) QPREP(U, uint8_t, R)
#define QWORD(i, t) w[i][t]
#define QAXPY(name, S, T)                    \
  QAXPY_1(name##_1, S, T, 0)                 \
  QAXPY_1(name##_keep, S, T, 1)              \
  QAXPYS(name, T, 2, QPREP_##S, QWORD, MADD)
QAXPY(quads_axpy_s8, S, int8_t)
QAXPY(quads_axpy_u8, U, uint8_t)

/* Lane order's dots and axpys: 16 lanes are 2 vectors of float32, 4 of
   float64; 4 rows of float32 or 2 of float64 hold 8 accumulators and b's
   vectors. */
#define VT __m256
#define LOAD _mm256_loadu_ps
#define STORE _mm256_storeu_ps
#define BCAST _mm256_set1_ps
#define ADD _mm256_add_ps
#define FMA(c, a, b) _mm256_fmadd_ps(a, b, c)
#include "gemm_lanes.h"
DOT(dot1_f32, float, 8, 2, 1, fmaf)
DOT(dot2_f32, float, 8, 2, 2, fmaf)
DOT(dot4_f32, float, 8, 2, 4, fmaf)
DOTS(dot_f32, float, 4, dot4_f32, dot2_f32, dot1_f32)
AXPY(axpy_f32, float, 8, fmaf)
TILE_PAIRS(tile8_f32, float, 8, fmaf)
GEMV(gemv_f32, float, 8, fmaf)
#undef VT
#undef LOAD
#undef STORE
#undef BCAST
#undef ADD
#undef FMA
#define VT __m256d
#define LOAD _mm256_loadu_pd
#define STORE _mm256_storeu_pd
#define BCAST _mm256_set1_pd
#define ADD _mm256_add_pd
#define FMA(c, a, b) _mm256_fmadd_pd(a, b, c)
DOT(dot1_f64, double, 4, 4, 1, fma)
DOT(dot2_f64, double, 4, 4, 2, fma)
DOTS(dot_f64, double, 2, dot2_f64, dot2_f64, dot1_f64)
AXPY(axpy_f64, double, 4, fma)
TILE_PAIRS(tile4_f64, double, 4, fma)
GEMV(gemv_f64, double, 4, fma)
#undef VT
#undef LOAD
#undef STORE
#undef BCAST
#undef ADD
#undef FMA

/* Tiles of fewer columns than a 256-bit vector holds run on 128-bit
   vectors, whose lanes take them where the wider tile takes them one at a
   time: on fudge, 256-bit tiles alone took a 4³ float32 product from 0.37
   µs through rows to 0.39. */
#define VT __m128
#define LOAD _mm_loadu_ps
#define STORE _mm_storeu_ps
#define BCAST _mm_set1_ps
#define ADD _mm_add_ps
#define FMA(c, a, b) _mm_fmadd_ps(a, b, c)
TILE_PAIRS(tile4_f32, float, 4, fmaf)
#undef VT
#undef LOAD
#undef STORE
#undef BCAST
#undef ADD
#undef FMA
#define VT __m128d
#define LOAD _mm_loadu_pd
#define STORE _mm_storeu_pd
#define BCAST _mm_set1_pd
#define ADD _mm_add_pd
#define FMA(c, a, b) _mm_fmadd_pd(a, b, c)
TILE_PAIRS(tile2_f64, double, 2, fma)

/* cpu.h's lane tile of [c] columns: 256-bit vectors from a vector's
   columns, 128-bit ones below. */
#define LANE_TILE(name, W, wide, narrow)                                  \
  static void name(const void *a, int64_t a_k, const void *b, int64_t b_k, \
                   int64_t k, int64_t c, void *y) {                       \
    if (c >= W) wide(a, a_k, b, b_k, k, c, y);                            \
    else narrow(a, a_k, b, b_k, k, c, y);                                 \
  }
LANE_TILE(tile_f32, 8, tile8_f32, tile4_f32)
LANE_TILE(tile_f64, 4, tile4_f64, tile2_f64)

/* cpu.h's float4 stage, 8 codes a vector: each code's magnitude, its low
   three bits, by VPERMPS from the 8 magnitudes, its sign bit moved to
   the float's, then times its scale, rounded once as the element's rule
   rounds it. A line's 4 bytes, broadcast and shifted 4 bits more a
   vector lane, give its 8 codes; where the lines lie adjacent in y, 8
   lines' 4 bytes, one a lane, give the 8 lines' codes of a step of k,
   then of the next as they shift 4 bits. On fudge a token through a
   4096 × 4096 MXFP4 weight took 0.55 ms staged through the portable
   stage, 0.32 so. */
static inline __m256 e2m1x8(__m256i x) {
  const __m256 mag = _mm256_setr_ps(
      nx_e2m1fn_to_float(0), nx_e2m1fn_to_float(1), nx_e2m1fn_to_float(2),
      nx_e2m1fn_to_float(3), nx_e2m1fn_to_float(4), nx_e2m1fn_to_float(5),
      nx_e2m1fn_to_float(6), nx_e2m1fn_to_float(7));
  __m256i sign = _mm256_and_si256(_mm256_slli_epi32(x, 28),
                                  _mm256_set1_epi32(INT32_MIN));
  return _mm256_or_ps(_mm256_permutevar8x32_ps(mag, x),
                      _mm256_castsi256_ps(sign));
}

static inline uint32_t word_at(const uint8_t *b) {
  uint32_t x;
  memcpy(&x, b, sizeof x);
  return x;
}

/* A vector of 8 scales in acc, and [y]'s 8 elements set to the values
   [v] times them. */
typedef __m256 scales32;
typedef struct {
  __m256d lo, hi;
} scales64;
static inline scales32 bcast32(const float *s) { return _mm256_set1_ps(*s); }
static inline scales32 load32(const float *s) { return _mm256_loadu_ps(s); }
static inline void put32(float *y, __m256 v, scales32 s) {
  _mm256_storeu_ps(y, _mm256_mul_ps(v, s));
}
static inline scales64 bcast64(const double *s) {
  __m256d x = _mm256_set1_pd(*s);
  return (scales64){x, x};
}
static inline scales64 load64(const double *s) {
  return (scales64){_mm256_loadu_pd(s), _mm256_loadu_pd(s + 4)};
}
static inline void put64(double *y, __m256 v, scales64 s) {
  __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
  __m256d hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
  _mm256_storeu_pd(y, _mm256_mul_pd(lo, s.lo));
  _mm256_storeu_pd(y + 4, _mm256_mul_pd(hi, s.hi));
}

#define F4(name, T, S, BCAST, LOADS, PUT)                                     \
  static void name(const uint8_t *codes, int64_t sl, int64_t lines,          \
                   int64_t runs, const void *vs, void *vy, int64_t du,       \
                   int64_t dl) {                                             \
    const T *s = vs;                                                         \
    T *y = vy;                                                               \
    int64_t l = 0;                                                           \
    for (; dl == 1 && l + 8 <= lines; l += 8)                                \
      for (int64_t r = 0; r < runs; r++) {                                   \
        S sc = LOADS(s + r * lines + l);                                     \
        const uint8_t *b = codes + l * sl + r * 16;                          \
        for (int q = 0; q < 4; q++) {                                        \
          __m256i x = _mm256_setr_epi32(                                     \
              (int)word_at(b + 4 * q), (int)word_at(b + sl + 4 * q),         \
              (int)word_at(b + 2 * sl + 4 * q),                              \
              (int)word_at(b + 3 * sl + 4 * q),                              \
              (int)word_at(b + 4 * sl + 4 * q),                              \
              (int)word_at(b + 5 * sl + 4 * q),                              \
              (int)word_at(b + 6 * sl + 4 * q),                              \
              (int)word_at(b + 7 * sl + 4 * q));                             \
          for (int t = 0; t < 8; t++) {                                      \
            PUT(y + (r * 32 + q * 8 + t) * du + l, e2m1x8(x), sc);           \
            x = _mm256_srli_epi32(x, 4);                                     \
          }                                                                  \
        }                                                                    \
      }                                                                      \
    const __m256i shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);  \
    _Alignas(32) T tmp[8];                                                   \
    for (; l < lines; l++)                                                   \
      for (int64_t r = 0; r < runs; r++) {                                   \
        S sc = BCAST(s + r * lines + l);                                     \
        const uint8_t *b = codes + l * sl + r * 16;                          \
        for (int q = 0; q < 4; q++) {                                        \
          __m256i x = _mm256_srlv_epi32(                                     \
              _mm256_set1_epi32((int)word_at(b + 4 * q)), shifts);           \
          T *d = y + (r * 32 + q * 8) * du + l * dl;                         \
          if (du == 1) {                                                     \
            PUT(d, e2m1x8(x), sc);                                           \
            continue;                                                        \
          }                                                                  \
          PUT(tmp, e2m1x8(x), sc);                                           \
          for (int i = 0; i < 8; i++) d[i * du] = tmp[i];                    \
        }                                                                    \
      }                                                                      \
  }
F4(f4_f32, float, scales32, bcast32, load32, put32)
F4(f4_f64, double, scales64, bcast64, load64, put64)

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
  t->tile[NX_FLOAT32] = tile_f32;
  t->tile[NX_FLOAT64] = tile_f64;
  t->gemv[NX_FLOAT32] = gemv_f32;
  t->gemv[NX_FLOAT64] = gemv_f64;
  t->f4[NX_FLOAT32] = f4_f32;
  t->f4[NX_FLOAT64] = f4_f64;
  t->gemm[NX_INT8] = t->gemm[NX_UINT8] = integers(kernel_u8, 32, SPEED8);
  t->gemm[NX_INT16] = t->gemm[NX_UINT16] = integers(kernel_u16, 32, SPEED16);
  t->gemm[NX_INT32] = t->gemm[NX_UINT32] = integers(kernel_u32, 16, SPEED32I);
  t->quads[NX_INT8] = integers(kernel_sq, 8, SPEEDQ);
  t->quads[NX_UINT8] = integers(kernel_uq, 8, SPEEDQ);
  t->quads_dot[NX_INT8] = quads_dot_s8;
  t->quads_dot[NX_UINT8] = quads_dot_u8;
  t->quads_axpy[NX_INT8] = quads_axpy_s8;
  t->quads_axpy[NX_UINT8] = quads_axpy_u8;
  /* The axpys outran the zips at every count of rows they take (above,
     at the quads' axpys). */
  t->quads_zip_rows = INT_MAX;
}

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_avx2;

#endif
