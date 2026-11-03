/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Lane order's dot and tile, chain order's axpy and the
   factorisations' gemv (cpu.h), written once over a target's vector
   operations of W lanes.

   A dot takes R rows of a against one column of b, each row's
   NX_CPU_LANES lanes in L vectors, lane q in vector q / W. A step of
   NX_CPU_LANES terms loads b's L vectors once and adds them, times each
   row's, into that row's accumulators, fused: R·L chains in flight. The
   accumulators are named variables: gcc keeps an array of vectors in
   memory and stores it at every step. Its rows stream and b stays in L1
   (gemm.c's dots). A block of more than DOT_AHEAD bytes a row prefetches
   each row DOT_AHEAD bytes ahead, a line a row each 64 bytes of it: a
   core alone read a streamed column at 28 GB/s on kimchi through its
   hardware prefetchers and 33 with these, and the rows of long blocks
   lie in pages of their own, where the L2's streamer does not cross.
   That costs R instructions a step of float32, 2R of float64, about a
   fifth more for float32's 4 rows. Rows of a shorter block lie adjacent
   in a weight, one stream from call to call, and take no prefetch.
   The lanes' tree then runs in
   registers: its levels across vectors by ADD, those within vector 0 by
   the target's LANE_SUM, so that a dot of a short block, a decoded
   token's scores against 64 elements of each key, finishes each row in a
   few instructions, with no lanes stored and summed one at a time. A
   block of whole steps runs two steps an iteration; one with terms past
   the last whole step adds those one at a time into the lanes stored,
   then the tree runs over those: the same sums.

   An axpy adds a row's element of a, broadcast, times a row of b into
   the row's outputs, or into +0, four vectors a step: a product of a
   depth of one, each output written once.

   A tile takes R rows of a against vectors of b's columns, each lane's
   accumulators of each row named vectors held across the block's rows of
   b, the lanes in passes that fit the target's registers, summed by the
   lanes' tree in registers (Lane tiles, below), so that a token against a
   weight stored [k × n] or attention's values, as the dot does against
   [n × k], reads each element of b once and stores no lane.

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
   element into every lane), ADD, FMA (c + a·b, rounded once) and, where
   it instantiates DOT, LANE_SUM (lane 0's value after the lanes' tree's
   levels within one vector) before it includes this header, then
   instantiates DOT for its row counts, DOTS over them, AXPY,
   LANE_TILES or RT_COLUMNS for its row counts and TILE over them, and
   GEMV. gcc at -O2 vectorises none of these loops
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

/* F(r, x) for each row r < R. */
#define DOT_ROW1(F, x) F(0, x)
#define DOT_ROW2(F, x) F(0, x) F(1, x)
#define DOT_ROW4(F, x) DOT_ROW2(F, x) F(2, x) F(3, x)

#define DOT_ZERO(r, v, W) VT c##r##_##v = BCAST(0);
#define DOT_B(r, v, W) VT b##v = LOAD(b + u + v * W);
#define DOT_ADD(r, v, W) \
  c##r##_##v = FMA(c##r##_##v, LOAD(a + r * lda + u + v * W), b##v);
#define DOT_KEEP(r, v, W) STORE(l[r] + v * W, c##r##_##v);

#define DOT_AHEAD 2048

/* Row r's line DOT_AHEAD bytes past the step at [u], each 64 bytes. */
#define DOT_FETCH(r, T)                                                 \
  for (int q = 0; q < NX_CPU_LANES * (int)sizeof(T); q += 64)           \
    __builtin_prefetch((const char *)(a + r * lda + u) + DOT_AHEAD + q);

/* The step of NX_CPU_LANES terms from [s], each row prefetched where
   [P]. */
#define DOT_STEP(s, T, R, W, L, P)            \
  {                                           \
    int64_t u = (s);                          \
    if (P) {                                  \
      DOT_ROW##R(DOT_FETCH, T)                \
    }                                         \
    DOT_V##L(DOT_B, 0, W)                     \
    DOT_EACH(R, L, DOT_ADD, W)                \
  }

/* The lanes' tree's levels across row r's L vectors, lane i taking lane
   i + h as vector v takes v + h / W, then within vector 0. */
#define DOT_FOLD2(r) c##r##_0 = ADD(c##r##_0, c##r##_1);
#define DOT_FOLD4(r)                  \
  c##r##_0 = ADD(c##r##_0, c##r##_2); \
  c##r##_1 = ADD(c##r##_1, c##r##_3); \
  DOT_FOLD2(r)
#define DOT_FOLD8(r)                  \
  c##r##_0 = ADD(c##r##_0, c##r##_4); \
  c##r##_1 = ADD(c##r##_1, c##r##_5); \
  c##r##_2 = ADD(c##r##_2, c##r##_6); \
  c##r##_3 = ADD(c##r##_3, c##r##_7); \
  DOT_FOLD4(r)
#define DOT_SUM(r, L) DOT_FOLD##L(r) y[r * ldy] = LANE_SUM(c##r##_0);

/* [name] sets y[r·ldy], for each of [R] rows of [a], [lda] apart, to lane
   order's sum of the block of [n] products of the row and [b], its
   accumulators in registers, [n] a multiple of NX_CPU_LANES; the rows
   prefetched where [P]. */
#define DOT_WHOLE(name, T, W, L, R, P)                                   \
  static void name(const T *a, int64_t lda, const T *b, int64_t n, T *y, \
                   int64_t ldy) {                                        \
    int64_t t = 0;                                                       \
    DOT_EACH(R, L, DOT_ZERO, W)                                          \
    for (; t + 2 * NX_CPU_LANES <= n; t += 2 * NX_CPU_LANES) {           \
      DOT_STEP(t, T, R, W, L, P)                                         \
      DOT_STEP(t + NX_CPU_LANES, T, R, W, L, P)                          \
    }                                                                    \
    if (t < n) DOT_STEP(t, T, R, W, L, P)                                \
    DOT_ROW##R(DOT_SUM, L)                                               \
  }

/* [name] sets y[r·ldy] as DOT_WHOLE does, for any [n]: [name]_whole, and
   [name]_ahead which prefetches its rows, where [n] is a multiple of
   NX_CPU_LANES. Otherwise terms past the last whole step add one at a
   time by [SFMA], the scalar fused multiply-add, into the lanes
   stored. */
#define DOT(name, T, W, L, R, SFMA)                                      \
  DOT_WHOLE(name##_whole, T, W, L, R, 0)                                 \
  DOT_WHOLE(name##_ahead, T, W, L, R, 1)                                 \
  static void name(const T *a, int64_t lda, const T *b, int64_t n, T *y, \
                   int64_t ldy) {                                        \
    int64_t t = 0;                                                       \
    DOT_EACH(R, L, DOT_ZERO, W)                                          \
    for (; t + NX_CPU_LANES <= n; t += NX_CPU_LANES)                     \
      DOT_STEP(t, T, R, W, L, 1)                                         \
    T l[R][NX_CPU_LANES];                                                \
    DOT_EACH(R, L, DOT_KEEP, W)                                          \
    for (int r = 0; r < R; r++) {                                        \
      for (int q = 0; t + q < n; q++)                                    \
        l[r][q] = SFMA(a[r * lda + t + q], b[t + q], l[r][q]);           \
      NX_CPU_LANE_TREE(i, j, l[r][i] = l[r][i] + l[r][j]);               \
      y[r * ldy] = l[r][0];                                              \
    }                                                                    \
  }

/* [name], [r] rows over [dm] of [M] rows, [d2] of 2 and [d1] of 1, each
   in its form [F], with the attributes [A]. */
#define DOTS_FORM(name, T, M, dm, d2, d1, F, A)                          \
  static A void name(const T *a, int64_t lda, int r, const T *b,         \
                     int64_t n, T *y, int64_t ldy) {                     \
    int i = 0;                                                           \
    for (; i + M <= r; i += M)                                           \
      dm##F(a + i * lda, lda, b, n, y + i * ldy, ldy);                   \
    if (i + 2 <= r) {                                                    \
      d2##F(a + i * lda, lda, b, n, y + i * ldy, ldy);                   \
      i += 2;                                                            \
    }                                                                    \
    if (i < r) d1##F(a + i * lda, lda, b, n, y + i * ldy, ldy);          \
  }

/* [name], cpu.h's dot, over [dm] of [M] rows, [d2] of 2 and [d1] of 1: M
   rows at a time, then 2, then 1, each its _whole form where n is a
   multiple of NX_CPU_LANES, its _ahead form where also a row of the
   block passes DOT_AHEAD bytes. The other forms, which store their
   lanes, run out of line. */
#define DOTS(name, T, M, dm, d2, d1)                                     \
  DOTS_FORM(name##_whole, T, M, dm, d2, d1, _whole, )                    \
  DOTS_FORM(name##_ahead, T, M, dm, d2, d1, _ahead, )                    \
  DOTS_FORM(name##_any, T, M, dm, d2, d1, , __attribute__((noinline)))   \
  static void name(const void *a, int64_t lda, int r, const void *b,     \
                   int64_t n, void *y, int64_t ldy) {                    \
    if (n % NX_CPU_LANES != 0) name##_any(a, lda, r, b, n, y, ldy);      \
    else if (n * (int64_t)sizeof(T) > DOT_AHEAD)                         \
      name##_ahead(a, lda, r, b, n, y, ldy);                             \
    else name##_whole(a, lda, r, b, n, y, ldy);                          \
  }

/* One row's axpy: [n] elements from [y]'s start [Y](j) of the vector at
   j, [SY](j) of the scalar. */
#define AXPY_ROW(T, W, SFMA, Y, SY)                                      \
  {                                                                      \
    int64_t j = 0;                                                       \
    for (; j + 4 * W <= n; j += 4 * W) {                                 \
      VT y0 = FMA(Y(j), x, LOAD(b + j));                                 \
      VT y1 = FMA(Y(j + W), x, LOAD(b + j + W));                         \
      VT y2 = FMA(Y(j + 2 * W), x, LOAD(b + j + 2 * W));                 \
      VT y3 = FMA(Y(j + 3 * W), x, LOAD(b + j + 3 * W));                 \
      STORE(y + j, y0);                                                  \
      STORE(y + j + W, y1);                                              \
      STORE(y + j + 2 * W, y2);                                          \
      STORE(y + j + 3 * W, y3);                                          \
    }                                                                    \
    for (; j + W <= n; j += W) STORE(y + j, FMA(Y(j), x, LOAD(b + j))); \
    for (; j < n; j++) y[j] = SFMA(s, b[j], SY(j));                      \
  }
#define AXPY_HELD(j) LOAD(y + (j))
#define AXPY_HELD_S(j) y[j]
#define AXPY_ZERO(j) zero
#define AXPY_ZERO_S(j) 0

/* [name], cpu.h's axpy, adding the scalars by [SFMA]. */
#define AXPY(name, T, W, SFMA)                                           \
  static void name(const void *va, int64_t lda, int64_t r,               \
                   const void *vb, int64_t n, void *vy, int64_t ldy,     \
                   nx_cpu_from from) {                                   \
    const T *a = va, *b = vb;                                            \
    VT zero = BCAST(0);                                                  \
    for (int64_t i = 0; i < r; i++) {                                    \
      T s = a[i * lda], *y = (T *)vy + i * ldy;                          \
      VT x = BCAST(s);                                                   \
      if (from == NX_CPU_FROM_ZERO)                                      \
        AXPY_ROW(T, W, SFMA, AXPY_ZERO, AXPY_ZERO_S)                     \
      else AXPY_ROW(T, W, SFMA, AXPY_HELD, AXPY_HELD_S)                  \
    }                                                                    \
  }

/* Lane tiles. A tile's kernel takes R rows of a against V vectors of
   b's columns, its lanes in 16/P passes: pass q takes lanes q + s·16/P
   for s < P, R·P·V accumulators a vector, which share each load of b and
   each broadcast of a's element. A pass folds its lanes by the tree's
   first levels, which add only lanes of one pass. The passes run in the
   bit-reversed order of their first lane, as TILE_REV lists them, so
   that a stack folds the passes' sums by the tree's next levels as the
   blocks' tree folds blocks (NX_CPU_TREE_STEP): lanes 0 and 8, then 4
   and 12, then the two, and so on. The sums and their order are the
   lanes' tree's.

   A wide kernel takes one lane a pass and V·W contiguous columns, a
   broadcast for V fused adds: R·V at most 8 accumulators, so that b's
   vectors and a broadcast fit beside them in 16 registers, and enough
   chains to cover the fused add's latency. Columns past the last wide
   step take a pair of vectors, u at j and v at jl, overlapping the
   columns before with the same bits where fewer remain, or u alone
   where W remain: a pair of one row takes 4 lanes a pass, of 2 rows 2,
   so that its accumulators keep 8 chains in flight.

   Where a lane's rows of b lie more than TILE_REACH bytes apart, two
   pages, beyond the hardware's stride prefetcher and its streamer, as a
   weight's rows of more than 128 float32 do, a step whose rows' columns
   span less than TILE_STREAM bytes prefetches the row each lane reads
   TILE_LINES lines later (AH). A staged block's rows of 512 bytes take
   none: on one thread 4 bfloat16 rows through a 4096 × 4096 weight,
   staged, took 4% more instructions with them. Wider spans the hardware follows: on fudge under load, one thread took a float32
   token through a 2880 × 4096 weight stored [k × n] in 2.3-3.8 ms in
   steps of 256 bytes with prefetches, 1.9-2.7 without; 2 rows through
   a 4096 × 4096 one in 5.1-6.1 ms in steps of 128 with, 4.9-6.3
   without; 4 rows in 5.2-6.3 ms in steps of 64 with, 8.0-11.2
   without. */
#define TILE_LINES 16
#define TILE_STREAM 128
#define TILE_REACH 8192
static const unsigned char TILE_REV[NX_CPU_LANES] = {
    0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

/* F(r, s, v) for each vector v < V of row r's lane s; F(r, v, x) for
   each vector v < V of row r. */
#define RT_V1(F, r, s) F(r, s, 0)
#define RT_V2(F, r, s) RT_V1(F, r, s) F(r, s, 1)
#define RT_V4(F, r, s) RT_V2(F, r, s) F(r, s, 2) F(r, s, 3)
#define RT_V8(F, r, s) \
  RT_V4(F, r, s) F(r, s, 4) F(r, s, 5) F(r, s, 6) F(r, s, 7)
#define RT_W1(F, r, x) F(r, 0, x)
#define RT_W2(F, r, x) RT_W1(F, r, x) F(r, 1, x)
#define RT_W4(F, r, x) RT_W2(F, r, x) F(r, 2, x) F(r, 3, x)
#define RT_W8(F, r, x) \
  RT_W4(F, r, x) F(r, 4, x) F(r, 5, x) F(r, 6, x) F(r, 7, x)
/* F(r, s, x) for each lane s < P of row r; for each row r < R too. */
#define RT_Q1(F, r, x) F(r, 0, x)
#define RT_Q2(F, r, x) RT_Q1(F, r, x) F(r, 1, x)
#define RT_Q4(F, r, x) RT_Q2(F, r, x) F(r, 2, x) F(r, 3, x)
#define RT_RQ1(F, P, x) RT_Q##P(F, 0, x)
#define RT_RQ2(F, P, x) RT_RQ1(F, P, x) RT_Q##P(F, 1, x)
#define RT_RQ3(F, P, x) RT_RQ2(F, P, x) RT_Q##P(F, 2, x)
#define RT_RQ4(F, P, x) RT_RQ3(F, P, x) RT_Q##P(F, 3, x)
/* F(r, v, x) for each row r < R and vector v < V. */
#define RT_RW1(F, V, x) RT_W##V(F, 0, x)
#define RT_RW2(F, V, x) RT_RW1(F, V, x) RT_W##V(F, 1, x)
#define RT_RW3(F, V, x) RT_RW2(F, V, x) RT_W##V(F, 2, x)
#define RT_RW4(F, V, x) RT_RW3(F, V, x) RT_W##V(F, 3, x)
/* F(r, s, x) for each row r < R of lane s. */
#define RT_ROWS1(F, s, x) F(0, s, x)
#define RT_ROWS2(F, s, x) RT_ROWS1(F, s, x) F(1, s, x)
#define RT_ROWS3(F, s, x) RT_ROWS2(F, s, x) F(2, s, x)
#define RT_ROWS4(F, s, x) RT_ROWS3(F, s, x) F(3, s, x)

/* Row r's lane s's accumulator of vector v; lane s's vector v of b at
   the step from [u]; row r's fused adds of lane s's term. */
#define RT_ZERO_V(r, s, v) VT c##r##_##s##_##v = BCAST(0);
#define RT_ZERO(r, s, V) RT_V##V(RT_ZERO_V, r, s)
#define RT_OFF(v) (paired && v == last ? dl : v * width)
#define RT_LOAD_V(r, s, v) \
  VT b##s##_##v = LOAD(p0 + (u + s * G) * b_k + RT_OFF(v));
#define RT_LOAD(r, s, V) RT_V##V(RT_LOAD_V, r, s)
#define RT_FMA_V(r, s, v) \
  c##r##_##s##_##v = FMA(c##r##_##s##_##v, x, b##s##_##v);
#define RT_ADD(r, s, V)                                    \
  {                                                        \
    VT x = BCAST(pa[r * lda + (u + s * G) * a_k]);         \
    RT_V##V(RT_FMA_V, r, s)                                \
  }
#define RT_FETCH(r, s, V)                                              \
  for (int l = 0; l < span; l += 64)                                   \
    __builtin_prefetch((const char *)(p0 + (u + s * G + lead) * b_k) + l);
/* The step of a term of each lane from [u0]. */
#define RT_STEP(u0, R, P, V, AH)                           \
  {                                                        \
    int64_t u = (u0);                                      \
    if (AH && span < TILE_STREAM) {                        \
      RT_Q##P(RT_FETCH, 0, V)                              \
    }                                                      \
    RT_Q##P(RT_LOAD, 0, V)                                 \
    RT_RQ##R(RT_ADD, P, V)                                 \
  }
/* Lane s of the step past the last whole one, where it holds a term. */
#define RT_LAST_R1(s, V) RT_ROWS1(RT_ADD, s, V)
#define RT_LAST_R2(s, V) RT_ROWS2(RT_ADD, s, V)
#define RT_LAST_R3(s, V) RT_ROWS3(RT_ADD, s, V)
#define RT_LAST_R4(s, V) RT_ROWS4(RT_ADD, s, V)
#define RT_LAST(R, s, V)                                   \
  if (q + s * G < k - u) {                                 \
    RT_LOAD(0, s, V)                                       \
    RT_LAST_R##R(s, V)                                     \
  }

/* Row r's pass's lanes of vector v folded by the tree's first levels:
   lanes q and q + 8 (2 a pass), (q + q + 8) + (q + 4 + q + 12) (4). */
#define RT_FOLD1(r, v) c##r##_0_##v
#define RT_FOLD2(r, v) ADD(c##r##_0_##v, c##r##_1_##v)
#define RT_FOLD4(r, v) \
  TREE4(c##r##_0_##v, c##r##_1_##v, c##r##_2_##v, c##r##_3_##v)
#define RT_PUSH(r, v, P) st[top][r][v] = RT_FOLD##P(r, v);
#define RT_STORE(r, v, x) STORE(y + r * ldy + j + RT_OFF(v), st[0][r][v]);

/* Two levels of the lanes' tree over four vectors: (w + y) + (x + z). */
#define TREE4(w, x, y, z) ADD(ADD(w, y), ADD(x, z))

/* [name] sets y[r·ldy + j'] for R rows of a and V vectors of columns, v
   from j + v·W, but where PAIRED the last from jl, to the lane tile's
   sums over the block of [k] terms, prefetching where AH. */
#define RT_KERNEL(name, T, W, R, P, V, AH, PAIRED)                       \
  static void name(const T *a, int64_t lda, int64_t a_k, const T *b,    \
                   int64_t b_k, int64_t k, int64_t j, int64_t jl, T *y, \
                   int64_t ldy) {                                       \
    const int64_t G = NX_CPU_LANES / P;                                 \
    const int64_t lead = NX_CPU_LANES * (TILE_LINES / P);               \
    const int span = V * W * (int)sizeof(T); /* bytes of a row's */     \
    const int paired = PAIRED, last = V - 1;                            \
    const int64_t width = W, dl = jl - j;                               \
    VT st[5][R][V];                                                     \
    int top = 0;                                                        \
    for (int n = 0; n < NX_CPU_LANES / P; n++) {                        \
      int64_t q = TILE_REV[n] / P, t = 0;                               \
      const T *pa = a + q * a_k, *p0 = b + q * b_k + j;                 \
      RT_RQ##R(RT_ZERO, P, V)                                           \
      for (; t + 2 * NX_CPU_LANES <= k; t += 2 * NX_CPU_LANES) {        \
        RT_STEP(t, R, P, V, AH)                                         \
        RT_STEP(t + NX_CPU_LANES, R, P, V, AH)                          \
      }                                                                 \
      if (t + NX_CPU_LANES <= k) {                                      \
        RT_STEP(t, R, P, V, AH)                                         \
        t += NX_CPU_LANES;                                              \
      }                                                                 \
      {                                                                 \
        int64_t u = t;                                                  \
        RT_Q##P(RT_LAST, R, V)                                          \
      }                                                                 \
      RT_RW##R(RT_PUSH, V, P)                                           \
      top++;                                                            \
      NX_CPU_TREE_STEP(n + 1, top, for (int r = 0; r < R; r++) {        \
        for (int v = 0; v < V; v++)                                     \
          st[top - 2][r][v] = ADD(st[top - 2][r][v], st[top - 1][r][v]); \
      });                                                               \
    }                                                                   \
    RT_RW##R(RT_STORE, V, 0)                                            \
  }

/* [name], R rows of a against [c] columns of b, c at least W: V vectors
   at a time, then pairs in passes of P lanes, the last from c - 2·W and,
   below 2·W columns, v from c - W, so that no column past c is read,
   overlapping columns taking the same bits again; W columns take u
   alone, where a pair would sum the same lanes twice: on fudge a 4 × 4 ×
   64 float64 product's tiles took 0.051 µs so, 0.085 as a pair. */
#define RT_COLUMNS(name, T, W, R, V, P)                                  \
  RT_KERNEL(name##_wide, T, W, R, 1, V, 0, 0)                            \
  RT_KERNEL(name##_wide_ahead, T, W, R, 1, V, 1, 0)                      \
  RT_KERNEL(name##_pair, T, W, R, P, 2, 0, 1)                            \
  RT_KERNEL(name##_pair_ahead, T, W, R, P, 2, 1, 1)                      \
  RT_KERNEL(name##_one, T, W, R, P, 1, 0, 0)                             \
  RT_KERNEL(name##_one_ahead, T, W, R, P, 1, 1, 0)                       \
  static void name(const T *a, int64_t lda, int64_t a_k, const T *b,    \
                   int64_t b_k, int64_t k, int64_t c, T *y, int64_t ldy) { \
    int64_t pitch = (b_k < 0 ? -b_k : b_k) * (int64_t)sizeof(T);        \
    int ahead = NX_CPU_LANES * pitch > TILE_REACH, j = 0;               \
    for (; j + V * W <= c; j += V * W)                                  \
      if (ahead)                                                        \
        name##_wide_ahead(a, lda, a_k, b, b_k, k, j, j + (V - 1) * W, y, \
                          ldy);                                         \
      else name##_wide(a, lda, a_k, b, b_k, k, j, j + (V - 1) * W, y, ldy); \
    if (j == c) return;                                                 \
    if (c == W) {                                                       \
      if (ahead) name##_one_ahead(a, lda, a_k, b, b_k, k, 0, 0, y, ldy); \
      else name##_one(a, lda, a_k, b, b_k, k, 0, 0, y, ldy);            \
      return;                                                           \
    }                                                                   \
    for (; j < c; j += 2 * W) {                                         \
      int64_t i = c < 2 * W ? 0 : j + 2 * W <= c ? j : c - 2 * W;      \
      int64_t l = c < 2 * W ? c - W : i + W;                            \
      if (ahead) name##_pair_ahead(a, lda, a_k, b, b_k, k, i, l, y, ldy); \
      else name##_pair(a, lda, a_k, b, b_k, k, i, l, y, ldy);           \
    }                                                                   \
  }

/* [name], cpu.h's lane tile for fewer columns than a vector holds, one
   at a time, adding by [SFMA]. */
#define TILE_FEW(name, T, SFMA)                                          \
  static void name(const void *va, int64_t lda, int64_t a_k, int64_t r, \
                   const void *vb, int64_t b_k, int64_t k, int64_t c,   \
                   void *vy, int64_t ldy) {                             \
    const T *a = va;                                                    \
    T *y = vy;                                                          \
    for (int64_t i = 0; i < r; i++)                                     \
      for (int64_t j = 0; j < c; j++) {                                 \
        T s[NX_CPU_LANES] = {0};                                        \
        const T *b = (const T *)vb + j;                                 \
        for (int64_t t = 0; t < k; t++)                                 \
          s[t % NX_CPU_LANES] = SFMA(a[i * lda + t * a_k], b[t * b_k],  \
                                     s[t % NX_CPU_LANES]);              \
        NX_CPU_LANE_TREE(q, u, s[q] = s[q] + s[u]);                     \
        y[i * ldy + j] = s[0];                                          \
      }                                                                 \
  }

/* [name], cpu.h's lane tile: 4 rows at a time by [r4], the rest by [r3],
   [r2] or [r1], each RT_COLUMNS of W columns a vector; fewer columns by
   [few]. */
#define TILE(name, T, W, r1, r2, r3, r4, few)                            \
  static void name(const void *va, int64_t lda, int64_t a_k, int64_t r, \
                   const void *vb, int64_t b_k, int64_t k, int64_t c,   \
                   void *vy, int64_t ldy) {                             \
    const T *a = va, *b = vb;                                           \
    T *y = vy;                                                          \
    if (c < W) {                                                        \
      few(va, lda, a_k, r, vb, b_k, k, c, vy, ldy);                     \
      return;                                                           \
    }                                                                   \
    int64_t i = 0;                                                      \
    for (; i + 4 <= r; i += 4)                                          \
      r4(a + i * lda, lda, a_k, b, b_k, k, c, y + i * ldy, ldy);        \
    if (r - i == 3)                                                     \
      r3(a + i * lda, lda, a_k, b, b_k, k, c, y + i * ldy, ldy);        \
    else if (r - i == 2)                                                \
      r2(a + i * lda, lda, a_k, b, b_k, k, c, y + i * ldy, ldy);        \
    else if (r - i == 1)                                                \
      r1(a + i * lda, lda, a_k, b, b_k, k, c, y + i * ldy, ldy);        \
  }

/* [name], cpu.h's lane tile over vectors of W elements: RT_COLUMNS of 1,
   2, 3 and 4 rows, fewer columns than W by [few]. */
#define LANE_TILES(name, T, W, few)                                      \
  RT_COLUMNS(name##_r1, T, W, 1, 8, 4)                                   \
  RT_COLUMNS(name##_r2, T, W, 2, 4, 2)                                   \
  RT_COLUMNS(name##_r3, T, W, 3, 2, 1)                                   \
  RT_COLUMNS(name##_r4, T, W, 4, 2, 1)                                   \
  TILE(name, T, W, name##_r1, name##_r2, name##_r3, name##_r4, few)

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
