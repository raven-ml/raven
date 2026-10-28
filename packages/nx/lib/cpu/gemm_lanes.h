/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Lane order's dot, axpy and tile, and the factorisations' gemv
   (cpu.h), written once over a target's vector operations of W lanes.

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
   memory. Its 16 lanes and a broadcast take 17 registers, which a target
   of 32 has. On one of 16 the compiler spills, and TILE_PAIRS runs the
   lanes in four passes instead: lanes q, q + 4, q + 8 and q + 12 of two
   vectors, 8 accumulators sharing each broadcast, summed by the tree's
   first two levels, which add only lanes of one pass. The sums and their
   order are TILE's. On fudge a 4 × 64 × 64 float32 product's tiles took
   0.34 µs so, 0.51 by TILE; float64's 0.63 against 1.07.

   A gemv holds GEMV_VECTORS vectors of y in accumulators while it adds
   their k products, one chain a vector, enough chains in flight to cover
   the fused multiply-add's latency; then half as many, then one vector at
   a time, then the last elements one at a time. It adds what k axpy
   calls would, in the same order; its gain is y held in registers across
   the k columns, where each axpy loads and stores it. On the i9-9900K's
   16 threads, the half block took 0.94 of cholesky-c128-8x32's time and
   0.95 of cholesky-f64-256's (minima of 7 ABBA rounds); k axpy calls in
   its place took 1.18 of lu-f64-256's time, 1.39 of lu-f64-16x32's, 1.13
   of cholesky-f64-256's, 1.27 of cholesky-c128-256's and 1.16 of
   trsm-f64-256x16's.

   A target file defines VT (the vector type), LOAD, STORE, BCAST (one
   element into every lane), ADD and FMA (c + a·b, rounded once) before
   it includes this header, then instantiates DOT for its row counts, DOTS
   over them, AXPY, TILE or TILE_PAIRS, and GEMV. gcc at -O2 vectorises
   none of these loops written plainly. */

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
/* Two levels of the lanes' tree over four vectors: (w + y) + (x + z). */
#define TREE4(w, x, y, z) ADD(ADD(w, y), ADD(x, z))
#define TILE_TREE                                                      \
  l0 = TREE4(TREE4(l0, l4, l8, l12), TREE4(l1, l5, l9, l13),            \
             TREE4(l2, l6, l10, l14), TREE4(l3, l7, l11, l15));

/* Fewer columns than a vector holds, one at a time, adding by [SFMA]. */
#define TILE_FEW(T, SFMA)                                                \
  for (int64_t j = 0; j < c; j++) {                                      \
    T s[NX_CPU_LANES] = {0};                                             \
    const T *b = (const T *)vb + j;                                      \
    for (int64_t t = 0; t < k; t++)                                      \
      s[t % NX_CPU_LANES] =                                              \
          SFMA(a[t * a_k], b[t * b_k], s[t % NX_CPU_LANES]);             \
    NX_CPU_LANE_TREE(q, r, s[q] = s[q] + s[r]);                          \
    y[j] = s[0];                                                         \
  }

/* [name], cpu.h's lane tile: W columns a vector, the last from c - W, so
   that no column past c is read, its lanes overlapping the vector before
   with the same bits; fewer than W columns by TILE_FEW. */
#define TILE(name, T, W, SFMA)                                           \
  static void name(const void *va, int64_t a_k, const void *vb,          \
                   int64_t b_k, int64_t k, int64_t c, void *vy) {        \
    const T *a = va;                                                     \
    T *y = vy;                                                           \
    if (c < W) {                                                         \
      TILE_FEW(T, SFMA)                                                  \
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

/* Lane q of vector u adds term t + q; of u and v, one broadcast. */
#define ONE_ADD(q) \
  u##q = FMA(u##q, BCAST(a[(t + q) * a_k]), LOAD(bu + (t + q) * b_k));
#define ONE_LAST(q) \
  if (q < k - t) ONE_ADD(q)
#define PAIR_ADD(q)                                                      \
  {                                                                      \
    VT x = BCAST(a[(t + q) * a_k]);                                      \
    u##q = FMA(u##q, x, LOAD(bu + (t + q) * b_k));                       \
    v##q = FMA(v##q, x, LOAD(bv + (t + q) * b_k));                       \
  }
#define PAIR_LAST(q) \
  if (q < k - t) PAIR_ADD(q)

/* Lanes q, r = q + 4, s = q + 8 and z = q + 12 of u over every term,
   summed by the tree's first two levels into [su]; the same of u and v
   into [su] and [sv]. */
#define ONE_PASS(q, r, s, z, su)                                         \
  {                                                                      \
    VT u##q = BCAST(0), u##r = BCAST(0), u##s = BCAST(0),                \
       u##z = BCAST(0);                                                  \
    int64_t t = 0;                                                       \
    for (; t + NX_CPU_LANES <= k; t += NX_CPU_LANES) {                   \
      ONE_ADD(q) ONE_ADD(r) ONE_ADD(s) ONE_ADD(z)                        \
    }                                                                    \
    ONE_LAST(q) ONE_LAST(r) ONE_LAST(s) ONE_LAST(z)                      \
    su = TREE4(u##q, u##r, u##s, u##z);                                  \
  }
#define PAIR_PASS(q, r, s, z, su, sv)                                    \
  {                                                                      \
    VT u##q = BCAST(0), u##r = BCAST(0), u##s = BCAST(0),                \
       u##z = BCAST(0), v##q = BCAST(0), v##r = BCAST(0),                \
       v##s = BCAST(0), v##z = BCAST(0);                                 \
    int64_t t = 0;                                                       \
    for (; t + NX_CPU_LANES <= k; t += NX_CPU_LANES) {                   \
      PAIR_ADD(q) PAIR_ADD(r) PAIR_ADD(s) PAIR_ADD(z)                    \
    }                                                                    \
    PAIR_LAST(q) PAIR_LAST(r) PAIR_LAST(s) PAIR_LAST(z)                  \
    su = TREE4(u##q, u##r, u##s, u##z);                                  \
    sv = TREE4(v##q, v##r, v##s, v##z);                                  \
  }

/* [name], cpu.h's lane tile in passes: two vectors at a time, u at j and
   v at j + W, the last pair from c - 2·W and, below 2·W columns, v from
   c - W, so that no column past c is read, overlapping lanes with the
   same bits. W columns take u alone, in passes of 4 accumulators, where a
   pair would sum the same lanes twice: on fudge a 4 × 4 × 64 float64
   product's tiles took 0.051 µs so, 0.085 as a pair and 0.070 by TILE.
   Fewer than W columns by TILE_FEW. */
#define TILE_PAIRS(name, T, W, SFMA)                                     \
  static void name(const void *va, int64_t a_k, const void *vb,          \
                   int64_t b_k, int64_t k, int64_t c, void *vy) {        \
    const T *a = va;                                                     \
    T *y = vy;                                                           \
    if (c < W) {                                                         \
      TILE_FEW(T, SFMA)                                                  \
      return;                                                            \
    }                                                                    \
    if (c == W) {                                                        \
      const T *bu = vb;                                                  \
      VT pu0, pu1, pu2, pu3;                                             \
      ONE_PASS(0, 4, 8, 12, pu0)                                         \
      ONE_PASS(1, 5, 9, 13, pu1)                                         \
      ONE_PASS(2, 6, 10, 14, pu2)                                        \
      ONE_PASS(3, 7, 11, 15, pu3)                                        \
      STORE(y, TREE4(pu0, pu1, pu2, pu3));                               \
      return;                                                            \
    }                                                                    \
    for (int64_t j0 = 0; j0 < c; j0 += 2 * W) {                          \
      int64_t j = j0 + 2 * W <= c ? j0 : c < 2 * W ? 0 : c - 2 * W;      \
      int64_t jv = j + 2 * W <= c ? j + W : c - W;                       \
      const T *bu = (const T *)vb + j, *bv = (const T *)vb + jv;         \
      VT pu0, pu1, pu2, pu3, pv0, pv1, pv2, pv3;                         \
      PAIR_PASS(0, 4, 8, 12, pu0, pv0)                                   \
      PAIR_PASS(1, 5, 9, 13, pu1, pv1)                                   \
      PAIR_PASS(2, 6, 10, 14, pu2, pv2)                                  \
      PAIR_PASS(3, 7, 11, 15, pu3, pv3)                                  \
      STORE(y + j, TREE4(pu0, pu1, pu2, pu3));                           \
      STORE(y + jv, TREE4(pv0, pv1, pv2, pv3));                          \
    }                                                                    \
  }

#define GEMV_VECTORS 8
#define GEMV_EACH(F, W) \
  F(0, W) F(1, W) F(2, W) F(3, W) F(4, W) F(5, W) F(6, W) F(7, W)
#define GEMV_HALF(F, W) F(0, W) F(1, W) F(2, W) F(3, W)
#define GEMV_LOAD(v, W) VT y##v = LOAD(y + i + v * W);
#define GEMV_ADD(v, W) y##v = FMA(y##v, LOAD(ap + v * W), xp);
#define GEMV_STORE(v, W) STORE(y + i + v * W, y##v);

/* [name], cpu.h's gemv, adding the last elements by [SFMA]. */
#define GEMV(name, T, W, SFMA)                                           \
  static void name(int64_t m, int64_t k, const void *va, int64_t lda,    \
                   const void *vx, void *vy) {                           \
    const T *a = va, *x = vx;                                            \
    T *y = vy;                                                           \
    int64_t i = 0;                                                       \
    for (; i + GEMV_VECTORS * W <= m; i += GEMV_VECTORS * W) {           \
      GEMV_EACH(GEMV_LOAD, W)                                            \
      for (int64_t p = 0; p < k; p++) {                                  \
        const T *ap = a + i + p * lda;                                   \
        VT xp = BCAST(x[p]);                                             \
        GEMV_EACH(GEMV_ADD, W)                                           \
      }                                                                  \
      GEMV_EACH(GEMV_STORE, W)                                           \
    }                                                                    \
    for (; i + 4 * W <= m; i += 4 * W) {                                 \
      GEMV_HALF(GEMV_LOAD, W)                                            \
      for (int64_t p = 0; p < k; p++) {                                  \
        const T *ap = a + i + p * lda;                                   \
        VT xp = BCAST(x[p]);                                             \
        GEMV_HALF(GEMV_ADD, W)                                           \
      }                                                                  \
      GEMV_HALF(GEMV_STORE, W)                                           \
    }                                                                    \
    for (; i + W <= m; i += W) {                                         \
      VT y0 = LOAD(y + i);                                               \
      for (int64_t p = 0; p < k; p++)                                    \
        y0 = FMA(y0, LOAD(a + i + p * lda), BCAST(x[p]));                \
      STORE(y + i, y0);                                                  \
    }                                                                    \
    for (; i < m; i++) {                                                 \
      T s = y[i];                                                        \
      for (int64_t p = 0; p < k; p++) s = SFMA(a[i + p * lda], x[p], s); \
      y[i] = s;                                                          \
    }                                                                    \
  }

#endif
