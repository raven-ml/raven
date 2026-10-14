/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Lane order's dot and axpy (cpu.h), written once over a target's vector
   operations of W lanes.

   A dot takes R rows of a against one column of b, each row's
   NX_CPU_LANES lanes in L vectors, lane q in vector q / W. A step of
   NX_CPU_LANES terms loads b's L vectors once and adds them, times each
   row's, into that row's accumulators, fused: R·L chains in flight. The
   accumulators are named variables: gcc keeps an array of vectors in
   memory and stores it at every step. It prefetches b DOT_AHEAD bytes
   ahead, where its column goes on: a core alone reads 28 GB/s of b on
   kimchi through its hardware prefetchers, 33 with these, and 4 rows of a
   on L1 18 GB/s against 27.

   An axpy adds a row's element of a, broadcast, times a row of b into the
   row's accumulators in memory, four vectors a step, and prefetches as
   much of the next b a step: a row of b lies a page or more from the
   next, where no hardware prefetcher follows.

   A tile takes one row of a against W columns of b, a vector, whose
   NX_CPU_LANES lanes are named vectors: term t of each column into
   vector t modulo NX_CPU_LANES, then the vectors summed by the lanes'
   tree, so that a small product finishes W outputs at a time in
   registers, where the dot and the axpy leave a lane at a time in
   memory.

   A target file defines VT (the vector type), LOAD, STORE, BCAST (one
   element into every lane), ADD and FMA (c + a·b, rounded once) before
   it includes this header, then instantiates DOT for its row counts, DOTS
   over them, AXPY and TILE. gcc at -O2 vectorises none of these loops
   written plainly. */

#ifndef NX_CPU_GEMM_LANES_H
#define NX_CPU_GEMM_LANES_H

/* F(r, v, x) for each row r < R and vector v < L. */
#define DOT_V2(F, r, x) F(r, 0, x) F(r, 1, x)
#define DOT_V4(F, r, x) DOT_V2(F, r, x) F(r, 2, x) F(r, 3, x)
#define DOT_V8(F, r, x) \
  DOT_V4(F, r, x) F(r, 4, x) F(r, 5, x) F(r, 6, x) F(r, 7, x)
#define DOT_R1(F, L, x) DOT_V##L(F, 0, x)
#define DOT_R2(F, L, x) DOT_R1(F, L, x) DOT_V##L(F, 1, x)
#define DOT_R4(F, L, x) DOT_R2(F, L, x) DOT_V##L(F, 2, x) DOT_V##L(F, 3, x)
#define DOT_EACH(R, L, F, x) DOT_R##R(F, L, x)

#define DOT_AHEAD 2048

#define DOT_LOAD(r, v, W) VT c##r##_##v = LOAD(l + r * NX_CPU_LANES + v * W);
#define DOT_B(r, v, W) VT b##v = LOAD(b + t + v * W);
#define DOT_ADD(r, v, W) \
  c##r##_##v = FMA(c##r##_##v, LOAD(a + r * lda + t + v * W), b##v);
#define DOT_STORE(r, v, W) STORE(l + r * NX_CPU_LANES + v * W, c##r##_##v);

/* [name] adds the [n] products of [R] rows of [a], [lda] apart, and [b]
   into each row's lanes at [l]: term t into lane t modulo NX_CPU_LANES.
   Terms past the last whole step add one at a time by [SFMA], the scalar
   fused multiply-add. */
#define DOT(name, T, W, L, R, SFMA)                                      \
  static void name(const T *a, int64_t lda, const T *b, int64_t n,       \
                   T *l) {                                               \
    int64_t t = 0;                                                       \
    DOT_EACH(R, L, DOT_LOAD, W)                                          \
    for (; t + NX_CPU_LANES <= n; t += NX_CPU_LANES) {                   \
      for (int q = 0; q < NX_CPU_LANES * (int)sizeof(T); q += 64)        \
        __builtin_prefetch((const char *)(b + t) + DOT_AHEAD + q);       \
      DOT_V##L(DOT_B, 0, W)                                              \
      DOT_EACH(R, L, DOT_ADD, W)                                         \
    }                                                                    \
    DOT_EACH(R, L, DOT_STORE, W)                                         \
    for (int r = 0; r < R; r++)                                          \
      for (int q = 0; t + q < n; q++)                                    \
        l[r * NX_CPU_LANES + q] =                                        \
            SFMA(a[r * lda + t + q], b[t + q], l[r * NX_CPU_LANES + q]); \
  }

/* [name], cpu.h's dot, over [dm] of [M] rows, [d2] of 2 and [d1] of 1: M
   rows at a time, then 2, then 1. */
#define DOTS(name, T, M, dm, d2, d1)                                     \
  static void name(const void *va, int64_t lda, int r, const void *vb,   \
                   int64_t n, void *vl) {                                \
    const T *a = va, *b = vb;                                            \
    T *l = vl;                                                           \
    int i = 0;                                                           \
    for (; i + M <= r; i += M)                                           \
      dm(a + i * lda, lda, b, n, l + i * NX_CPU_LANES);                  \
    if (i + 2 <= r) {                                                    \
      d2(a + i * lda, lda, b, n, l + i * NX_CPU_LANES);                  \
      i += 2;                                                            \
    }                                                                    \
    if (i < r) d1(a + i * lda, lda, b, n, l + i * NX_CPU_LANES);         \
  }

/* [name], cpu.h's axpy, adding the scalars by [SFMA]. */
#define AXPY(name, T, W, SFMA)                                           \
  static void name(const void *va, int64_t lda, int r, const void *vb,   \
                   int64_t n, void *vy, int64_t ldy, const void *next) { \
    const T *a = va, *b = vb;                                            \
    const char *ahead = next;                                            \
    for (int i = 0; i < r; i++) {                                        \
      T s = a[i * lda], *y = (T *)vy + i * ldy;                          \
      VT x = BCAST(s);                                                   \
      int64_t j = 0;                                                     \
      for (; j + 4 * W <= n; j += 4 * W) {                               \
        if (i == 0)                                                      \
          for (int q = 0; q < 4 * W * (int)sizeof(T); q += 64)           \
            __builtin_prefetch(ahead + j * (int64_t)sizeof(T) + q);      \
        VT y0 = FMA(LOAD(y + j), x, LOAD(b + j));                        \
        VT y1 = FMA(LOAD(y + j + W), x, LOAD(b + j + W));                \
        VT y2 = FMA(LOAD(y + j + 2 * W), x, LOAD(b + j + 2 * W));        \
        VT y3 = FMA(LOAD(y + j + 3 * W), x, LOAD(b + j + 3 * W));        \
        STORE(y + j, y0);                                                \
        STORE(y + j + W, y1);                                            \
        STORE(y + j + 2 * W, y2);                                        \
        STORE(y + j + 3 * W, y3);                                        \
      }                                                                  \
      for (; j + W <= n; j += W)                                         \
        STORE(y + j, FMA(LOAD(y + j), x, LOAD(b + j)));                  \
      for (; j < n; j++) y[j] = SFMA(s, b[j], y[j]);                     \
    }                                                                    \
  }

/* F(q) for each lane q, and the lanes' tree over named vectors. */
#define TILE_EACH(F)                                                     \
  F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11) F(12)    \
      F(13) F(14) F(15)
#define TILE_ZERO(q) VT l##q = BCAST(0);
#define TILE_ADD(q)                                                      \
  l##q = FMA(l##q, BCAST(a[(t + q) * a_k]), LOAD(b + (t + q) * b_k));
#define TILE_LAST(q) \
  if (q < k - t) TILE_ADD(q)
#define TILE_TREE                                                         \
  l0 = ADD(l0, l8), l1 = ADD(l1, l9), l2 = ADD(l2, l10), l3 = ADD(l3, l11), \
  l4 = ADD(l4, l12), l5 = ADD(l5, l13), l6 = ADD(l6, l14),                \
  l7 = ADD(l7, l15), l0 = ADD(l0, l4), l1 = ADD(l1, l5), l2 = ADD(l2, l6), \
  l3 = ADD(l3, l7), l0 = ADD(l0, l2), l1 = ADD(l1, l3), l0 = ADD(l0, l1);

/* [name], cpu.h's lane tile: W columns a vector, the last from c - W, so
   that no column past c is read, its lanes overlapping the vector before
   with the same bits; fewer than W columns one at a time, adding by
   [SFMA]. */
#define TILE(name, T, W, SFMA)                                           \
  static void name(const void *va, int64_t a_k, const void *vb,          \
                   int64_t b_k, int64_t k, int64_t c, void *vy) {        \
    const T *a = va;                                                     \
    T *y = vy;                                                           \
    if (c < W) {                                                         \
      for (int64_t j = 0; j < c; j++) {                                  \
        T s[NX_CPU_LANES] = {0};                                         \
        const T *b = (const T *)vb + j;                                  \
        for (int64_t t = 0; t < k; t++)                                  \
          s[t % NX_CPU_LANES] =                                          \
              SFMA(a[t * a_k], b[t * b_k], s[t % NX_CPU_LANES]);         \
        for (int w = NX_CPU_LANES / 2; w > 0; w /= 2)                    \
          for (int q = 0; q < w; q++) s[q] = s[q] + s[q + w];            \
        y[j] = s[0];                                                     \
      }                                                                  \
      return;                                                            \
    }                                                                    \
    for (int64_t j0 = 0; j0 < c; j0 += W) {                              \
      int64_t j = j0 + W <= c ? j0 : c - W, t = 0;                       \
      const T *b = (const T *)vb + j;                                    \
      TILE_EACH(TILE_ZERO)                                               \
      for (; t + NX_CPU_LANES <= k; t += NX_CPU_LANES) {                 \
        TILE_EACH(TILE_ADD)                                              \
      }                                                                  \
      TILE_EACH(TILE_LAST)                                               \
      TILE_TREE                                                          \
      STORE(y + j, l0);                                                  \
    }                                                                    \
  }

#endif
