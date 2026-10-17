/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* x86-64's integer tiles and quads' dots and axpys (cpu.h), written once
   over AVX2 vectors: gemm_avx2.c instantiates them with AVX2's
   multiplies, gemm_vnni.c with AVX-VNNI-INT8's dot products.

   INTS is a tile of MR rows by two vectors of outputs: LOADB loads a step
   of b's sliver into two vectors, BCAST a's element into one, MAC adds
   their product into an accumulator, START and FINISH load and store a
   row's two accumulators. QDOT is a quads dot: W loads a vector of S
   words, MAC adds its products into an accumulator of 32-bit lanes.
   QAXPYS is a quads axpy of several rows over a file's axpy of a row
   alone. A file includes this header after <string.h> and cpu.h, with AVX2
   enabled. */

#ifndef NX_CPU_GEMM_AVX2_H
#define NX_CPU_GEMM_AVX2_H

#include <immintrin.h>

/* The rows of a tile. */
#define MR 6

/* The rows at most which a product may stream b past all of a (gemm.c's
   streams), each sliver of b reading all of a's. On the i9-9900K's 16
   threads a 4096 × 4096 weight stored n × k takes 5.1 ms so against 128
   rows, 6.7 in chain order. A bound from the threads instead (stream
   where b's slivers are enough units and a's KC rows fit in L2) took no
   less time on the i9, and on fudge's six cores lost to this one: 64³
   float32 took 1.08 times as long, 1024³ float64 1.12. */
#define FEW 256

/* Row i's two accumulators and its row of a. */
#define DECL(V, T, i)                                       \
  V c##i##0, c##i##1;                                       \
  const T *a##i = a + (i < rows ? i : rows - 1) * a_row
#define STEP(BCAST, FMA, i)          \
  ai = BCAST(a##i + q);              \
  c##i##0 = FMA(ai, b0, c##i##0);    \
  c##i##1 = FMA(ai, b1, c##i##1)
#define ROWS(M, ...)                                                     \
  M(__VA_ARGS__, 0); M(__VA_ARGS__, 1); M(__VA_ARGS__, 2);               \
  M(__VA_ARGS__, 3); M(__VA_ARGS__, 4); M(__VA_ARGS__, 5)

#define VLOAD(p) _mm256_loadu_si256((const __m256i *)(p))
#define VSTORE(p, x) _mm256_storeu_si256((__m256i *)(p), x)
#define LOAD2(p, x0, x1, W) (x0 = VLOAD(p), x1 = VLOAD((p) + W))
#define STORE2(p, x0, x1, W) (VSTORE(p, x0), VSTORE((p) + W, x1))

#define LOADB32(b, b0, b1) LOAD2(b, b0, b1, 8)
#define START32(y, c0, c1) LOAD2(y, c0, c1, 8)
#define FINISH32(y, c0, c1, from) STORE2(y, c0, c1, 8)
#define BCAST32(p) _mm256_set1_epi32((int32_t) * (p))

/* Row i's accumulators from y, or zero; stored, where the row is the
   tile's. */
#define ISTART(START, i)                        \
  if (from == NX_CPU_FROM_ZERO || i >= rows)    \
    c##i##0 = c##i##1 = _mm256_setzero_si256(); \
  else                                          \
    START(y + i * ld, c##i##0, c##i##1)
#define IFINISH(FINISH, i) \
  if (i < rows) FINISH(y + i * ld, c##i##0, c##i##1, from)

#define INTS(name, T, NR, LOADB, BCAST, MAC, START, FINISH)                 \
  static void name(int64_t m, int64_t n, int64_t k, const void *va,         \
                   int64_t a_row, int64_t a_k, const void *vb, int64_t b_k, \
                   void *vc, int64_t ldc, nx_cpu_from from) {               \
    _Alignas(32) T t[MR * NR];                                              \
    size_t row = (size_t)n * sizeof(T);                                     \
    for (int64_t i0 = 0; i0 < m; i0 += MR) {                                \
      int64_t rows = m - i0 < MR ? m - i0 : MR, ld = ldc;                   \
      const T *a = (const T *)va + i0 * a_row, *b = vb;                     \
      T *y = (T *)vc + i0 * ldc;                                            \
      if (n < NR) {                                                         \
        for (int64_t i = 0; i < rows && from == NX_CPU_FROM_TILE; i++)      \
          memcpy(t + i * NR, y + i * ldc, row);                             \
        y = t;                                                              \
        ld = NR;                                                            \
      }                                                                     \
      ROWS(DECL, __m256i, T);                                               \
      ROWS(ISTART, START);                                                  \
      for (int64_t p = 0, q = 0; p < k; p++, q += a_k, b += b_k) {          \
        __m256i b0, b1, ai;                                                 \
        LOADB(b, b0, b1);                                                   \
        ROWS(STEP, BCAST, MAC);                                             \
      }                                                                     \
      ROWS(IFINISH, FINISH);                                                \
      for (int64_t i = 0; i < rows && n < NR; i++)                          \
        memcpy((T *)vc + (i0 + i) * ldc, t + i * NR, row);                  \
    }                                                                       \
  }

/* Quads' dots, as gemm_neon.c's: a tile of two rows of a by four columns
   of b, a step of S words of each, which W loads as a vector for MAC,
   eight accumulators whose lanes add at the end. A tail of fewer than S
   words goes through words zeroed on the stack. A tile whose rows are one
   (lda 0, a product's last row alone) computes it once: decoding a token
   is one row, and its second would double the work. */
#define QDOT_ROW(MAC, i, x, v0, v1, v2, v3) \
  c##i##0 = MAC(x, v0, c##i##0);            \
  c##i##1 = MAC(x, v1, c##i##1);            \
  c##i##2 = MAC(x, v2, c##i##2);            \
  c##i##3 = MAC(x, v3, c##i##3);
#define QDOT_STEP(W, MAC, R, r, q)                                      \
  {                                                                     \
    __m256i x = W(r[0] + q), v0 = W(r[2] + q), v1 = W(r[3] + q),        \
            v2 = W(r[4] + q), v3 = W(r[5] + q);                         \
    QDOT_ROW(MAC, 0, x, v0, v1, v2, v3)                                 \
    if (R > 1) {                                                        \
      x = W(r[1] + q);                                                  \
      QDOT_ROW(MAC, 1, x, v0, v1, v2, v3)                               \
    }                                                                   \
  }

/* A row's four sums: VPHADDD twice leaves each column's two halves, one
   per 128-bit lane. */
static inline __m128i sums4(__m256i c0, __m256i c1, __m256i c2, __m256i c3) {
  __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(c0, c1),
                                _mm256_hadd_epi32(c2, c3));
  return _mm_add_epi32(_mm256_castsi256_si128(h),
                       _mm256_extracti128_si256(h, 1));
}

/* The dot of [R] rows, 1 or 2; a row alone stores its sums twice. */
#define QDOT_R(name, W, S, MAC, R)                                         \
  static void name(const uint32_t *a, int64_t lda, const uint32_t *b,     \
                   int64_t ldb, int64_t n, uint32_t *y, int64_t ldy) {    \
    __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c03 = c00; \
    __m256i c10 = c00, c11 = c00, c12 = c00, c13 = c00;                   \
    const uint32_t *r[6] = {a, a + lda, b, b + ldb, b + 2 * ldb,          \
                            b + 3 * ldb};                                 \
    int64_t q = 0;                                                        \
    for (; q + S <= n; q += S) QDOT_STEP(W, MAC, R, r, q)                 \
    if (q < n) {                                                          \
      uint32_t t[6][S] = {{0}};                                           \
      const uint32_t *z[6];                                               \
      for (int i = 0; i < 6; i++) {                                       \
        memcpy(t[i], r[i] + q, (size_t)(n - q) * 4);                      \
        z[i] = t[i];                                                      \
      }                                                                   \
      QDOT_STEP(W, MAC, R, z, 0)                                          \
    }                                                                     \
    _mm_storeu_si128((__m128i *)y, sums4(c00, c01, c02, c03));            \
    _mm_storeu_si128((__m128i *)(y + ldy),                                \
                     R > 1 ? sums4(c10, c11, c12, c13)                    \
                           : sums4(c00, c01, c02, c03));                  \
  }
#define QDOT(name, W, S, MAC)                                             \
  QDOT_R(name##_1, W, S, MAC, 1)                                          \
  QDOT_R(name##_2, W, S, MAC, 2)                                          \
  static void name(const uint32_t *a, int64_t lda, const uint32_t *b,     \
                   int64_t ldb, int64_t n, uint32_t *y, int64_t ldy) {    \
    (lda == 0 ? name##_1 : name##_2)(a, lda, b, ldb, n, y, ldy);          \
  }

/* Quads' axpys of several rows: by steps of QK rows and QCOLS columns of
   b, the first row by [name]_keep, then the others over the words it
   keeps, in groups of at most QG whose sizes differ by at most one: 6
   others as 6, 7 as 3 and 4. [name]_1 is a row alone, and [name]_keep
   the same row, which also stores the word of b's rows from t to
   t + PER - 1 at column j in pb + t / PER * (QCOLS / 8) + j / 8, 8
   columns a vector in column order, for the columns up to the last 16.
   pb holds QK / PER × QCOLS words on the stack: 8 KiB of AVX2's pairs, 4
   KiB of AVX-VNNI-INT8's quads. Each line of b is read once: a step's
   rows, 4 KiB apart, fall in one set of the L1 cache, whose 8 ways a
   second read of 16 rows would miss. Columns past the last 16 go one at
   a time. */
#define QK 16
#define QCOLS 256
/* The most rows of a group: 12 accumulators, b's 2 vectors and a's
   broadcast. */
#define QG 6
/* Row i's accumulators, named: gcc -O2 keeps an array indexed by a loop
   over the rows in memory. */
#define QLOAD(R, i)                   \
  if (R > i)                          \
    c##i##0 = VLOAD(y + i * ldy + j), \
    c##i##1 = VLOAD(y + i * ldy + j + 8);
#define QSTORE(R, i)                  \
  if (R > i)                          \
    VSTORE(y + i * ldy + j, c##i##0), \
    VSTORE(y + i * ldy + j + 8, c##i##1);
/* Row i's word [t], WORD(i, t), times b's. */
#define QMAC(MAC, WORD, R, i)                       \
  if (R > i) {                                      \
    __m256i x = _mm256_set1_epi32((int)WORD(i, t)); \
    c##i##0 = MAC(x, lo, c##i##0);                  \
    c##i##1 = MAC(x, hi, c##i##1);                  \
  }
/* [R] rows, 1 to QG, from [a], [lda] apart, over the [kw] words of b's
   rows kept at [pb], into the 16-column groups of [c] columns of y, [ldy]
   apart: PREP(R) makes the rows' words that WORD reads, then 2
   accumulators a row. */
#define QGROUP(name, R, PREP, WORD, MAC)                                  \
  static void name(const uint8_t *a, int64_t lda, const __m256i *pb,      \
                   int64_t kw, int64_t c, uint32_t *y, int64_t ldy) {     \
    PREP(R)                                                               \
    for (int64_t j = 0; j < c; j += 16) {                                 \
      __m256i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51; \
      QLOAD(R, 0) QLOAD(R, 1) QLOAD(R, 2)                                 \
      QLOAD(R, 3) QLOAD(R, 4) QLOAD(R, 5)                                 \
      const __m256i *p = pb + j / 8;                                      \
      for (int64_t t = 0; t < kw; t++, p += QCOLS / 8) {                  \
        __m256i lo = p[0], hi = p[1];                                     \
        QMAC(MAC, WORD, R, 0) QMAC(MAC, WORD, R, 1)                       \
        QMAC(MAC, WORD, R, 2) QMAC(MAC, WORD, R, 3)                       \
        QMAC(MAC, WORD, R, 4) QMAC(MAC, WORD, R, 5)                       \
      }                                                                   \
      QSTORE(R, 0) QSTORE(R, 1) QSTORE(R, 2)                              \
      QSTORE(R, 3) QSTORE(R, 4) QSTORE(R, 5)                              \
    }                                                                     \
  }
#define QAXPYS(name, T, PER, PREP, WORD, MAC)                                \
  QGROUP(name##_g1, 1, PREP, WORD, MAC)                                      \
  QGROUP(name##_g2, 2, PREP, WORD, MAC)                                      \
  QGROUP(name##_g3, 3, PREP, WORD, MAC)                                      \
  QGROUP(name##_g4, 4, PREP, WORD, MAC)                                      \
  QGROUP(name##_g5, 5, PREP, WORD, MAC)                                      \
  QGROUP(name##_g6, 6, PREP, WORD, MAC)                                      \
  static void name(const uint8_t *a, int64_t lda, int64_t rows,              \
                   const uint8_t *b, int64_t ldb, int64_t k, int64_t n,      \
                   uint32_t *y, int64_t ldy, const uint8_t *next) {          \
    if (rows == 1) {                                                         \
      name##_1(a, b, ldb, k, n, y, next, NULL);                              \
      return;                                                                \
    }                                                                        \
    static void (*const group[QG + 1])(const uint8_t *, int64_t,             \
                                       const __m256i *, int64_t, int64_t,    \
                                       uint32_t *, int64_t) = {              \
        NULL,      name##_g1, name##_g2, name##_g3,                          \
        name##_g4, name##_g5, name##_g6};                                    \
    _Alignas(32) __m256i pb[QK / PER * (QCOLS / 8)];                         \
    int64_t groups = (rows - 1 + QG - 1) / QG;                               \
    for (int64_t t0 = 0; t0 < k; t0 += QK) {                                 \
      int64_t kb = k - t0 < QK ? k - t0 : QK, kw = (kb + PER - 1) / PER;     \
      for (int64_t j0 = 0; j0 < n; j0 += QCOLS) {                            \
        int64_t cb = n - j0 < QCOLS ? n - j0 : QCOLS, c = cb / 16 * 16;      \
        int64_t at = t0 * ldb + j0;                                          \
        name##_keep(a + t0, b + at, ldb, kb, cb, y + j0, next + at, pb);     \
        for (int64_t g = 0, i = 1; g < groups; g++) {                        \
          int64_t r = (rows - i) / (groups - g);                             \
          group[r](a + i * lda + t0, lda, pb, kw, c, y + i * ldy + j0, ldy); \
          i += r;                                                            \
        }                                                                    \
        for (int64_t j = j0 + c; j < j0 + cb; j++)                           \
          for (int64_t i = 1; i < rows; i++)                                 \
            for (int64_t t = t0; t < t0 + kb; t++)                           \
              y[i * ldy + j] += (uint32_t)((T)a[i * lda + t] *               \
                                           (T)b[t * ldb + j]);               \
      }                                                                      \
    }                                                                        \
  }

#endif
