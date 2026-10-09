/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Thin microkernels, written once over a target's vector operations: a tile
   of M rows (1, 2 or 4) and V vectors of W lanes, as nx_cpu_kernel states
   with MR = M and NR = V·W. A product of few rows, as decoding a token is,
   runs on these: the main kernels' MR rows would waste most of their fused
   adds on padding.

   A step of k broadcasts each row's element of a and adds it times each
   vector of b into that row's accumulator for the vector, fused. The
   accumulators are named variables: gcc keeps an array of vectors in
   memory and stores it at every step. A target file defines LOAD, STORE,
   BCAST (one element into every lane) and FMA (c + a·b, rounded once)
   before it includes this header, then instantiates THIN. */

#ifndef NX_CPU_GEMM_THIN_H
#define NX_CPU_GEMM_THIN_H

/* F(r, x) for each row r < M; F(r, v, x) for each vector v < V of row r. */
#define THIN_R1(F, x) F(0, x)
#define THIN_R2(F, x) THIN_R1(F, x) F(1, x)
#define THIN_R4(F, x) THIN_R2(F, x) F(2, x) F(3, x)
#define THIN_V1(F, r, x) F(r, 0, x)
#define THIN_V2(F, r, x) THIN_V1(F, r, x) F(r, 1, x)
#define THIN_V3(F, r, x) THIN_V2(F, r, x) F(r, 2, x)
#define THIN_V4(F, r, x) THIN_V3(F, r, x) F(r, 3, x)
#define THIN_V6(F, r, x) THIN_V4(F, r, x) F(r, 4, x) F(r, 5, x)
#define THIN_V8(F, r, x) THIN_V6(F, r, x) F(r, 6, x) F(r, 7, x)
#define THIN_V12(F, r, x) \
  THIN_V8(F, r, x) F(r, 8, x) F(r, 9, x) F(r, 10, x) F(r, 11, x)
#define THIN_V16(F, r, x) \
  THIN_V12(F, r, x) F(r, 12, x) F(r, 13, x) F(r, 14, x) F(r, 15, x)
#define THIN_M1(F, V, x) THIN_V##V(F, 0, x)
#define THIN_M2(F, V, x) THIN_M1(F, V, x) THIN_V##V(F, 1, x)
#define THIN_M4(F, V, x) THIN_M2(F, V, x) THIN_V##V(F, 2, x) THIN_V##V(F, 3, x)
#define THIN_EACH(M, V, F, x) THIN_M##M(F, V, x)

#define THIN_DECL(r, v, VT) VT c##r##_##v;
#define THIN_LOAD(r, v, W) c##r##_##v = LOAD(y + r * ldc + v * W);
#define THIN_STORE(r, v, W) STORE(y + r * ldc + v * W, c##r##_##v);
#define THIN_ADD(r, v, W) c##r##_##v = FMA(c##r##_##v, a##r, LOAD(b + v * W));
#define THIN_A(r, VT) VT a##r = BCAST(a + r);

/* [name] for a tile of M rows of V vectors of W lanes of the type [T], [VT]
   being the vector type. */
#define THIN(name, T, VT, W, M, V)                                         \
  static void name(int64_t k, const void *va, const void *vb, void *vc,   \
                   int64_t ldc) {                                         \
    const T *a = va, *b = vb;                                             \
    T *y = vc;                                                            \
    THIN_EACH(M, V, THIN_DECL, VT)                                        \
    THIN_EACH(M, V, THIN_LOAD, W)                                         \
    for (int64_t p = 0; p < k; p++, a += M, b += V * W) {                 \
      THIN_R##M(THIN_A, VT)                                               \
      THIN_EACH(M, V, THIN_ADD, W)                                        \
    }                                                                     \
    THIN_EACH(M, V, THIN_STORE, W)                                        \
  }

#endif
