/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_matmul.c — the backend's owned GEMM.

   The owned path has no external dependency or OpenMP. It is a
   Goto/BLIS-structured blocked kernel over packed panels, with one NEON
   microkernel (f32 + f64) and a portable-C fallback that autovectorizes.

   How the dtype universe is served with a handful of microkernels:

   - Packing is the funnel. Loops 3-5 (below) pack A and B into aligned,
     contiguous, compute-typed panels. The pack READS through arbitrary
     (row,col) element strides — so a transposed view costs nothing extra and is
     never pre-materialized — and CONVERTS storage to the dtype's compute type on the way
     in (nx_c.h's LOAD column). f16/bf16/fp8 therefore run through the f32
     microkernel at near-f32 speed; small ints run through the int64 microkernel;
     the store converts back (nx_c.h's STORE column, modular wrap for integers)
     exactly once per output element.
   - The microkernel is the only compute-type-specific code: f32 (NEON 8x12),
     f64 (NEON 8x4), int64/uint64 and complex32/complex64 (portable 8xNR). Each
     computes a full MR x NR register tile over one k-block in the compute type;
     for k up to MM_KC_FULLK_MAX the block is the whole k (register accumulation,
     one store), and beyond it the k dimension is sub-blocked (KC panels) with the
     partial tiles summed into a compute-typed MC x NC accumulator and truncated
     on store exactly once (see the block-size note and nx_c_gemm_macrotile).

   The five-loop details are documented at nx_c_gemm_macrotile and in the
   block-size note. Threading composes with the engine's single pool
   through nx_c_parallel_for, over one of two disjoint job spaces (nx_c_matmul_run):
   (batch x NC-panel) when there are enough panels to fill the pool — each worker
   packs its own A/B panels from per-worker scratch — or (batch x NC-panel x
   MC-block) when a matrix is too narrow to yield one panel per core, with B
   pre-packed once into a shared buffer and only A packed per worker. Either way
   packing and compute run under the released lock and there is no second pool. A
   single-thread call goes through the panel path (nthreads == 1 still releases
   the lock, since the traffic clears the cutoff). */

#include <stdlib.h>
#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h" /* nx_c_parallel_for + the pool; nx_c.h comes with it */
#include "nx_c_matmul.h" /* pins nx_c_gemm2d_ct's signature against drift */

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* Accelerate is the only platform-specific compute route in this backend.
   It ships with macOS and its AMX-backed sgemm/dgemm
   reach a throughput no portable NEON kernel matches, so the top-level matmul
   driver hands large, cblas-mappable products in the native-compute float dtypes
   (f32/f64/c32/c64) to cblas, keeping the owned GEMM as the fallback and the ONLY
   path off macOS, where this whole section compiles out. Low-precision and
   integer dtypes NEVER route here — Accelerate lacks them and the owned pack
   funnel keeps its large win. The store-once-per-C-element invariant is intact:
   cblas writes C directly, once, in the native storage precision (each of
   f32/f64/c32/c64 is its own compute type). */
#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include <limits.h>
#endif

/* ── Block sizes (Apple M-series L1/L2) ────────────────────────────────────

   Blocking: MC rows of A and NC columns of B per macro-block, and — for large k
   — KC-deep sub-blocks of the contraction. MC/NC need not be multiples of a
   register tile — the pack zero-pads and the macrokernel handles the partial
   trailing MR/NR block — but 256 divides evenly by MR (8) and by NR for f64/int
   (4/8); f32's NR is 12, so the last NC column-block is partial (4 of 12 used),
   a ~2% edge tax on one of 22 blocks. The owned-GEMM benchmark guards these
   choices. */
#define MM_MC 256
#define MM_NC 256

/* KC (contraction) sub-blocking. For k <= MM_KC_FULLK_MAX the whole k is one
   block: the microkernel accumulates the full-k MR x NR tile in registers and
   stores it once — the compute-bound path the single-thread gate measures, which
   must not regress. Above that, a single (MC x k) A-panel or (k x NC) B-panel
   overflows the shared L2, so the micro is re-streamed over MM_KC-deep sub-panels
   and the partial tiles are summed into a compute-typed MC x NC accumulator,
   flushed to storage once after the last KC panel. MM_KC keeps one A sub-panel
   (MC x KC) and one B sub-panel (KC x NC) plus the accumulator resident; the
   FULLK cap keeps the common full-k regime untouched. The owned-GEMM benchmark
   and large-k correctness fixtures guard both regimes. */
#define MM_KC 512
#define MM_KC_FULLK_MAX 2048

/* Below this the pack setup does not pay: run a direct strided triple loop in
   the compute type instead. Also taken when a dimension is smaller than the
   register tile, where blocking would be mostly edge padding. Tuned by the test. */
#define MM_DIRECT_CUTOFF (48 * 48 * 48)

/* Register tile height, shared by every microkernel; NR is per compute type. */
#define MM_MR 8

/* ── Aligned scratch ──────────────────────────────────────────────────────
   Packed panels are allocated ONCE per matmul call (per thread when threaded),
   never per microkernel call, and reused across every block and batch element.
   64-byte alignment keeps the streaming panel loads off cache-line splits. */
static void *mm_alloc(size_t bytes) {
  if (bytes == 0) bytes = 64;
  bytes = (bytes + 63u) & ~(size_t)63u; /* aligned_alloc needs a multiple */
  return aligned_alloc(64, bytes);
}

static int64_t mm_ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

/* ── Microkernels ─────────────────────────────────────────────────────────

   micro(tile, ap, bp, k): tile[r*NR + col] = sum_{p<k} ap[p*MR + r] * bp[p*NR +
   col], for r in [0,MR), col in [0,NR). ap is one MR-row A panel, bp one NR-col
   B panel, both packed compute-typed and p-major (see the pack layout). The tile
   is a caller-owned MR*NR compute buffer. Edge tiles compute full MR x NR over
   zero-padded panels; the store writes only the valid mr x nr corner. */

/* MR=8, NR=12: 24 accumulator registers (8 rows x 3 vectors of 4) plus 3 B and
   2 A vectors = 29 of the 32 NEON registers. The wide tile amortizes the B/A
   loads (24 FMA : 5 loads per k-step) and gives enough independent accumulator
   chains to hide the ~4-cycle FMA latency across the four FP pipes — measurably
   faster than 8x8 single-thread on Firestorm. */
static void nx_c_micro_f32(void *vtile, const void *vap, const void *vbp,
                          int64_t k) {
  const float *ap = (const float *)vap;
  const float *bp = (const float *)vbp;
  float *tile = (float *)vtile;
#if defined(__ARM_NEON)
  float32x4_t c0a = vdupq_n_f32(0), c0b = vdupq_n_f32(0), c0c = vdupq_n_f32(0);
  float32x4_t c1a = vdupq_n_f32(0), c1b = vdupq_n_f32(0), c1c = vdupq_n_f32(0);
  float32x4_t c2a = vdupq_n_f32(0), c2b = vdupq_n_f32(0), c2c = vdupq_n_f32(0);
  float32x4_t c3a = vdupq_n_f32(0), c3b = vdupq_n_f32(0), c3c = vdupq_n_f32(0);
  float32x4_t c4a = vdupq_n_f32(0), c4b = vdupq_n_f32(0), c4c = vdupq_n_f32(0);
  float32x4_t c5a = vdupq_n_f32(0), c5b = vdupq_n_f32(0), c5c = vdupq_n_f32(0);
  float32x4_t c6a = vdupq_n_f32(0), c6b = vdupq_n_f32(0), c6c = vdupq_n_f32(0);
  float32x4_t c7a = vdupq_n_f32(0), c7b = vdupq_n_f32(0), c7c = vdupq_n_f32(0);
  for (int64_t p = 0; p < k; p++) {
    float32x4_t b0 = vld1q_f32(bp + p * 12);
    float32x4_t b1 = vld1q_f32(bp + p * 12 + 4);
    float32x4_t b2 = vld1q_f32(bp + p * 12 + 8);
    float32x4_t a0 = vld1q_f32(ap + p * 8);
    float32x4_t a1 = vld1q_f32(ap + p * 8 + 4);
    c0a = vfmaq_laneq_f32(c0a, b0, a0, 0);
    c0b = vfmaq_laneq_f32(c0b, b1, a0, 0);
    c0c = vfmaq_laneq_f32(c0c, b2, a0, 0);
    c1a = vfmaq_laneq_f32(c1a, b0, a0, 1);
    c1b = vfmaq_laneq_f32(c1b, b1, a0, 1);
    c1c = vfmaq_laneq_f32(c1c, b2, a0, 1);
    c2a = vfmaq_laneq_f32(c2a, b0, a0, 2);
    c2b = vfmaq_laneq_f32(c2b, b1, a0, 2);
    c2c = vfmaq_laneq_f32(c2c, b2, a0, 2);
    c3a = vfmaq_laneq_f32(c3a, b0, a0, 3);
    c3b = vfmaq_laneq_f32(c3b, b1, a0, 3);
    c3c = vfmaq_laneq_f32(c3c, b2, a0, 3);
    c4a = vfmaq_laneq_f32(c4a, b0, a1, 0);
    c4b = vfmaq_laneq_f32(c4b, b1, a1, 0);
    c4c = vfmaq_laneq_f32(c4c, b2, a1, 0);
    c5a = vfmaq_laneq_f32(c5a, b0, a1, 1);
    c5b = vfmaq_laneq_f32(c5b, b1, a1, 1);
    c5c = vfmaq_laneq_f32(c5c, b2, a1, 1);
    c6a = vfmaq_laneq_f32(c6a, b0, a1, 2);
    c6b = vfmaq_laneq_f32(c6b, b1, a1, 2);
    c6c = vfmaq_laneq_f32(c6c, b2, a1, 2);
    c7a = vfmaq_laneq_f32(c7a, b0, a1, 3);
    c7b = vfmaq_laneq_f32(c7b, b1, a1, 3);
    c7c = vfmaq_laneq_f32(c7c, b2, a1, 3);
  }
  vst1q_f32(tile + 0, c0a);
  vst1q_f32(tile + 4, c0b);
  vst1q_f32(tile + 8, c0c);
  vst1q_f32(tile + 12, c1a);
  vst1q_f32(tile + 16, c1b);
  vst1q_f32(tile + 20, c1c);
  vst1q_f32(tile + 24, c2a);
  vst1q_f32(tile + 28, c2b);
  vst1q_f32(tile + 32, c2c);
  vst1q_f32(tile + 36, c3a);
  vst1q_f32(tile + 40, c3b);
  vst1q_f32(tile + 44, c3c);
  vst1q_f32(tile + 48, c4a);
  vst1q_f32(tile + 52, c4b);
  vst1q_f32(tile + 56, c4c);
  vst1q_f32(tile + 60, c5a);
  vst1q_f32(tile + 64, c5b);
  vst1q_f32(tile + 68, c5c);
  vst1q_f32(tile + 72, c6a);
  vst1q_f32(tile + 76, c6b);
  vst1q_f32(tile + 80, c6c);
  vst1q_f32(tile + 84, c7a);
  vst1q_f32(tile + 88, c7b);
  vst1q_f32(tile + 92, c7c);
#else
  float acc[96];
  for (int i = 0; i < 96; i++) acc[i] = 0.0f;
  for (int64_t p = 0; p < k; p++) {
    const float *a = ap + p * 8;
    const float *b = bp + p * 12;
    for (int r = 0; r < 8; r++) {
      float av = a[r];
      for (int col = 0; col < 12; col++) acc[r * 12 + col] += av * b[col];
    }
  }
  for (int i = 0; i < 96; i++) tile[i] = acc[i];
#endif
}

static void nx_c_micro_f64(void *vtile, const void *vap, const void *vbp,
                          int64_t k) {
  const double *ap = (const double *)vap;
  const double *bp = (const double *)vbp;
  double *tile = (double *)vtile;
#if defined(__ARM_NEON)
  float64x2_t c0l = vdupq_n_f64(0), c0h = vdupq_n_f64(0);
  float64x2_t c1l = vdupq_n_f64(0), c1h = vdupq_n_f64(0);
  float64x2_t c2l = vdupq_n_f64(0), c2h = vdupq_n_f64(0);
  float64x2_t c3l = vdupq_n_f64(0), c3h = vdupq_n_f64(0);
  float64x2_t c4l = vdupq_n_f64(0), c4h = vdupq_n_f64(0);
  float64x2_t c5l = vdupq_n_f64(0), c5h = vdupq_n_f64(0);
  float64x2_t c6l = vdupq_n_f64(0), c6h = vdupq_n_f64(0);
  float64x2_t c7l = vdupq_n_f64(0), c7h = vdupq_n_f64(0);
  for (int64_t p = 0; p < k; p++) {
    float64x2_t a0 = vld1q_f64(ap + p * 8);
    float64x2_t a1 = vld1q_f64(ap + p * 8 + 2);
    float64x2_t a2 = vld1q_f64(ap + p * 8 + 4);
    float64x2_t a3 = vld1q_f64(ap + p * 8 + 6);
    float64x2_t b0 = vld1q_f64(bp + p * 4);
    float64x2_t b1 = vld1q_f64(bp + p * 4 + 2);
    c0l = vfmaq_laneq_f64(c0l, b0, a0, 0);
    c0h = vfmaq_laneq_f64(c0h, b1, a0, 0);
    c1l = vfmaq_laneq_f64(c1l, b0, a0, 1);
    c1h = vfmaq_laneq_f64(c1h, b1, a0, 1);
    c2l = vfmaq_laneq_f64(c2l, b0, a1, 0);
    c2h = vfmaq_laneq_f64(c2h, b1, a1, 0);
    c3l = vfmaq_laneq_f64(c3l, b0, a1, 1);
    c3h = vfmaq_laneq_f64(c3h, b1, a1, 1);
    c4l = vfmaq_laneq_f64(c4l, b0, a2, 0);
    c4h = vfmaq_laneq_f64(c4h, b1, a2, 0);
    c5l = vfmaq_laneq_f64(c5l, b0, a2, 1);
    c5h = vfmaq_laneq_f64(c5h, b1, a2, 1);
    c6l = vfmaq_laneq_f64(c6l, b0, a3, 0);
    c6h = vfmaq_laneq_f64(c6h, b1, a3, 0);
    c7l = vfmaq_laneq_f64(c7l, b0, a3, 1);
    c7h = vfmaq_laneq_f64(c7h, b1, a3, 1);
  }
  vst1q_f64(tile + 0, c0l);
  vst1q_f64(tile + 2, c0h);
  vst1q_f64(tile + 4, c1l);
  vst1q_f64(tile + 6, c1h);
  vst1q_f64(tile + 8, c2l);
  vst1q_f64(tile + 10, c2h);
  vst1q_f64(tile + 12, c3l);
  vst1q_f64(tile + 14, c3h);
  vst1q_f64(tile + 16, c4l);
  vst1q_f64(tile + 18, c4h);
  vst1q_f64(tile + 20, c5l);
  vst1q_f64(tile + 22, c5h);
  vst1q_f64(tile + 24, c6l);
  vst1q_f64(tile + 26, c6h);
  vst1q_f64(tile + 28, c7l);
  vst1q_f64(tile + 30, c7h);
#else
  double acc[32];
  for (int i = 0; i < 32; i++) acc[i] = 0.0;
  for (int64_t p = 0; p < k; p++) {
    const double *a = ap + p * 8;
    const double *b = bp + p * 4;
    for (int r = 0; r < 8; r++) {
      double av = a[r];
      for (int col = 0; col < 4; col++) acc[r * 4 + col] += av * b[col];
    }
  }
  for (int i = 0; i < 32; i++) tile[i] = acc[i];
#endif
}

/* Portable outer-product microkernel for the compute types with no hand-vector
   kernel: integer (accumulate in the dtype's wide compute type — modular,
   matching numpy and nx_c.h's modular store; the SIGNED kernel is instantiated
   over uint64_t so the wrap is defined without -fwrapv, which stays in the
   flags as belt and suspenders — the int64_t panels/tile are read and written
   through the corresponding unsigned type, a sanctioned aliasing pair, and the
   bits are identical) and complex (native C _Complex multiply-accumulate;
   chosen over the 3M method so the generic packing is reused unchanged and
   there is one code path — complex GEMM is not the hot path). The inner col
   loop autovectorizes under -O3. */
#define MM_GEN_MICRO(name, T, NR)                                              \
  static void name(void *vtile, const void *vap, const void *vbp, int64_t k) { \
    const T *ap = (const T *)vap;                                             \
    const T *bp = (const T *)vbp;                                             \
    T *tile = (T *)vtile;                                                     \
    T acc[MM_MR * (NR)];                                                      \
    for (int i = 0; i < MM_MR * (NR); i++) acc[i] = (T)0;                     \
    for (int64_t p = 0; p < k; p++) {                                         \
      const T *a = ap + p * MM_MR;                                            \
      const T *b = bp + p * (NR);                                             \
      for (int r = 0; r < MM_MR; r++) {                                       \
        T av = a[r];                                                          \
        for (int col = 0; col < (NR); col++) acc[r * (NR) + col] += av * b[col]; \
      }                                                                        \
    }                                                                          \
    for (int i = 0; i < MM_MR * (NR); i++) tile[i] = acc[i];                  \
  }
MM_GEN_MICRO(nx_c_micro_u64, uint64_t, 8)
MM_GEN_MICRO(nx_c_micro_c32, nx_c_complex32, 8)
MM_GEN_MICRO(nx_c_micro_c64, nx_c_complex64, 8)
#undef MM_GEN_MICRO

/* Per-compute-type traits, keyed by the dtype table's `compute` C type token so
   the descriptor rows below stay one line each. bool's compute type (uint8_t)
   maps to a NULL microkernel: matmul is arithmetic, and the frontend promotes
   bool away before it reaches here, so bool matmul is unsupported. */
#define MM_MICRO_float nx_c_micro_f32
#define MM_MICRO_double nx_c_micro_f64
/* i64 shares the u64 microkernel: the modular product/sum in the unsigned
   width is bit-identical to the signed wrap (see the microkernel note). */
#define MM_MICRO_int64_t nx_c_micro_u64
#define MM_MICRO_uint64_t nx_c_micro_u64
#define MM_MICRO_nx_c_complex32 nx_c_micro_c32
#define MM_MICRO_nx_c_complex64 nx_c_micro_c64
#define MM_MICRO_uint8_t NULL
#define MM_NR_float 12
#define MM_NR_double 4
#define MM_NR_int64_t 8
#define MM_NR_uint64_t 8
#define MM_NR_nx_c_complex32 8
#define MM_NR_nx_c_complex64 8
#define MM_NR_uint8_t 0

/* ── Pack / store / direct, generated per dtype ───────────────────────────

   One set per compute dtype (packed int4/uint4 never reach here). Each reads or
   writes storage through nx_c.h's per-dtype LOAD/STORE, so dtype conversion and
   modular integer semantics live in exactly one place. Element (row,col) of the
   source is at byte (row*rs + col*cs) * sizeof(storage) from the operand base —
   arbitrary strides, transposition resolved for free.

   pack_a A panel layout: ceil(mc/MR) row-blocks; block b at dst + b*MR*k, and
   within it position [p][r] = A(row0 + b*MR + r, p) at offset p*MR + r, with
   rows past mc zero-padded. pack_b B panel layout is the NR-column mirror:
   block b at dst + b*NR*k, [p][col] = B(p, col0 + b*NR + col) at p*NR + col. */

/* Compute-typed multiply-accumulate and add for mm_direct / mm_acc. The SINT
   forms run in the unsigned width: the contract is modular wrap (SINT compute
   is always int64_t — dtype table), defined without -fwrapv, which stays in
   the flags as belt and suspenders. */
#define MM_MAC_NX_C_CAT_SINT(acc, x, y)                                         \
  (acc) = (int64_t)((uint64_t)(acc) + (uint64_t)(x) * (uint64_t)(y))
#define MM_MAC_NX_C_CAT_UINT(acc, x, y) (acc) += (x) * (y)
#define MM_MAC_NX_C_CAT_FLOAT(acc, x, y) (acc) += (x) * (y)
#define MM_MAC_NX_C_CAT_COMPLEX(acc, x, y) (acc) += (x) * (y)
#define MM_MAC_NX_C_CAT_BOOL(acc, x, y) (acc) += (x) * (y)
#define MM_ADD_NX_C_CAT_SINT(a, b) ((int64_t)((uint64_t)(a) + (uint64_t)(b)))
#define MM_ADD_NX_C_CAT_UINT(a, b) ((a) + (b))
#define MM_ADD_NX_C_CAT_FLOAT(a, b) ((a) + (b))
#define MM_ADD_NX_C_CAT_COMPLEX(a, b) ((a) + (b))
#define MM_ADD_NX_C_CAT_BOOL(a, b) ((a) + (b))

#define MM_GEN(sfx, kind, storage, compute, ld, st, cat)                       \
  static void mm_pack_a_##sfx(void *vdst, const void *vsrc, int64_t rs,        \
                              int64_t cs, int64_t row0, int64_t mc, int64_t k, \
                              int MR) {                                        \
    compute *dst = (compute *)vdst;                                           \
    const char *src = (const char *)vsrc;                                     \
    int64_t esz = (int64_t)sizeof(storage);                                   \
    int64_t nblk = mm_ceil_div(mc, MR);                                       \
    for (int64_t b = 0; b < nblk; b++) {                                      \
      int64_t r0 = b * MR;                                                    \
      int64_t rows = mc - r0;                                                 \
      if (rows > MR) rows = MR;                                               \
      compute *db = dst + b * (int64_t)MR * k;                                \
      for (int64_t p = 0; p < k; p++) {                                       \
        compute *dc = db + p * MR;                                            \
        for (int r = 0; r < MR; r++) {                                        \
          if (r < rows) {                                                     \
            int64_t idx = (row0 + r0 + r) * rs + p * cs;                      \
            dc[r] = nx_c_ld_##sfx(src + idx * esz);                            \
          } else                                                             \
            dc[r] = (compute)0;                                               \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  static void mm_pack_b_##sfx(void *vdst, const void *vsrc, int64_t rs,        \
                              int64_t cs, int64_t col0, int64_t nc, int64_t k, \
                              int NR) {                                        \
    compute *dst = (compute *)vdst;                                           \
    const char *src = (const char *)vsrc;                                     \
    int64_t esz = (int64_t)sizeof(storage);                                   \
    int64_t nblk = mm_ceil_div(nc, NR);                                       \
    for (int64_t b = 0; b < nblk; b++) {                                      \
      int64_t c0 = b * NR;                                                    \
      int64_t cols = nc - c0;                                                 \
      if (cols > NR) cols = NR;                                               \
      compute *db = dst + b * (int64_t)NR * k;                                \
      for (int64_t p = 0; p < k; p++) {                                       \
        compute *dr = db + p * NR;                                            \
        for (int c = 0; c < NR; c++) {                                        \
          if (c < cols) {                                                     \
            int64_t idx = p * rs + (col0 + c0 + c) * cs;                      \
            dr[c] = nx_c_ld_##sfx(src + idx * esz);                            \
          } else                                                             \
            dr[c] = (compute)0;                                               \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  static void mm_store_##sfx(const void *vtile, void *vc, int64_t crs,         \
                             int64_t ccs, int64_t row0, int64_t col0, int mr,  \
                             int nr, int NR) {                                 \
    const compute *tile = (const compute *)vtile;                            \
    char *c = (char *)vc;                                                     \
    int64_t esz = (int64_t)sizeof(storage);                                   \
    for (int i = 0; i < mr; i++)                                              \
      for (int j = 0; j < nr; j++) {                                          \
        int64_t idx = (row0 + i) * crs + (col0 + j) * ccs;                    \
        nx_c_st_##sfx(c + idx * esz, tile[i * NR + j]);                        \
      }                                                                        \
  }                                                                            \
  static void mm_direct_##sfx(const void *va, int64_t ars, int64_t acs,        \
                              const void *vb, int64_t brs, int64_t bcs,        \
                              void *vc, int64_t crs, int64_t ccs, int64_t m,   \
                              int64_t n, int64_t k) {                          \
    const char *a = (const char *)va;                                        \
    const char *b = (const char *)vb;                                        \
    char *c = (char *)vc;                                                     \
    int64_t esz = (int64_t)sizeof(storage);                                   \
    for (int64_t i = 0; i < m; i++)                                           \
      for (int64_t j = 0; j < n; j++) {                                       \
        compute acc = (compute)0;                                            \
        for (int64_t p = 0; p < k; p++)                                       \
          MM_MAC_##cat(acc, nx_c_ld_##sfx(a + (i * ars + p * acs) * esz),      \
                       nx_c_ld_##sfx(b + (p * brs + j * bcs) * esz));          \
        nx_c_st_##sfx(c + (i * crs + j * ccs) * esz, acc);                    \
      }                                                                        \
  }                                                                            \
  /* KC-panel accumulate: dst[i] += src[i] over `count` compute elements, the \
     one place a partial MR x NR tile folds into the compute-typed C tile. In  \
     compute precision (int64 wraps modularly BY CONSTRUCTION — MM_ADD's SINT  \
     form adds in the unsigned width, not via -fwrapv; float sums, _Complex    \
     adds), so the KC-blocked sum stays exact-to-full-k up to the same         \
     wrap/rounding the full-k path already has — the store still truncates    \
     once. */                                                                  \
  static void mm_acc_##sfx(void *vdst, const void *vsrc, int count) {          \
    compute *dst = (compute *)vdst;                                           \
    const compute *src = (const compute *)vsrc;                               \
    for (int i = 0; i < count; i++)                                           \
      dst[i] = (compute)MM_ADD_##cat(dst[i], src[i]);                         \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(MM_GEN)
#undef MM_GEN

/* ── Dtype descriptor ─────────────────────────────────────────────────────
   One row per dtype, generated from the same table: the generated pack/store/
   direct for the storage dtype, plus the microkernel / tile shape / compute
   element size for its compute type. Packed dtypes (int4/uint4) are absent from
   the compute iterator, so their slots stay zero (micro == NULL). */
typedef void (*nx_c_mm_pack)(void *, const void *, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int);
typedef void (*nx_c_mm_store)(const void *, void *, int64_t, int64_t, int64_t,
                             int64_t, int, int, int);
typedef void (*nx_c_mm_direct)(const void *, int64_t, int64_t, const void *,
                              int64_t, int64_t, void *, int64_t, int64_t,
                              int64_t, int64_t, int64_t);
typedef void (*nx_c_mm_micro)(void *, const void *, const void *, int64_t);
typedef void (*nx_c_mm_acc)(void *, const void *, int);

typedef struct {
  nx_c_mm_pack pack_a, pack_b;
  nx_c_mm_store store;
  nx_c_mm_direct direct;
  nx_c_mm_micro micro; /* NULL: dtype unsupported for matmul */
  nx_c_mm_acc acc;     /* KC-panel tile accumulate (compute-typed) */
  int MR, NR;
  int64_t csize; /* bytes of one compute element */
} nx_c_mm_desc;

#define MM_DESC_ROW(sfx, kind, storage, compute, ld, st, cat)                  \
  [NX_C_DTYPE_##sfx] = {mm_pack_a_##sfx,                                        \
                       mm_pack_b_##sfx,                                        \
                       mm_store_##sfx,                                         \
                       mm_direct_##sfx,                                        \
                       MM_MICRO_##compute,                                     \
                       mm_acc_##sfx,                                           \
                       MM_MR,                                                  \
                       MM_NR_##compute,                                        \
                       (int64_t)sizeof(compute)},
static const nx_c_mm_desc mm_desc[NX_C_DTYPE_COUNT] = {
    NX_C_FOR_EACH_COMPUTE_DTYPE(MM_DESC_ROW)};
#undef MM_DESC_ROW

#undef MM_MICRO_float
#undef MM_MICRO_double
#undef MM_MICRO_int64_t
#undef MM_MICRO_uint64_t
#undef MM_MICRO_nx_c_complex32
#undef MM_MICRO_nx_c_complex64
#undef MM_MICRO_uint8_t
#undef MM_NR_float
#undef MM_NR_double
#undef MM_NR_int64_t
#undef MM_NR_uint64_t
#undef MM_NR_nx_c_complex32
#undef MM_NR_nx_c_complex64
#undef MM_NR_uint8_t

/* ── Blocked 2-D GEMM ─────────────────────────────────────────────────────

   Deviation from §4.8's five-loop: MC/NC macro-blocking with the contraction
   sub-blocked (KC) only above MM_KC_FULLK_MAX. Either way each output element is
   stored exactly once in the compute type — that single store is what gives
   f16/bf16/fp8 float accumulation and small ints int64 accumulation. For
   k <= MM_KC_FULLK_MAX the microkernel sums the whole k in registers and mm_store
   truncates once (the compute-bound path the single-thread gate measures). Above
   it, a single (MC x k) A-panel / (k x NC) B-panel overflows the shared L2 and
   the micro re-streams the same panels for each column block; KC blocking caps
   the resident A/B sub-panel to (MC x KC)/(KC x NC) and folds each partial
   MR x NR tile into a compute-typed MC x NC accumulator, flushed once. A
   KC-blocked in-place accumulation in storage precision would instead round or
   wrap the partial sums — hence the compute-typed Cacc. */

/* One MC x NC output block from packed panels Ap (mc x k, MR-blocked) and Bp
   (nc x k, NR-blocked), stored at C[row0.., col0..]. On the KC path Cacc is a
   per-worker compute-typed accumulator laid out as nnb*nmb contiguous MR*NR tile
   blocks; it is NULL (and unused) for k <= MM_KC_FULLK_MAX. `tile` and Cacc are
   per-worker scratch, so disjoint output blocks never contend. */
static void nx_c_gemm_macrotile(const nx_c_mm_desc *d, const char *Ap,
                               const char *Bp, int64_t k, int64_t mc, int64_t nc,
                               char *c, int64_t crs, int64_t ccs, int64_t row0,
                               int64_t col0, char *Cacc) {
  int MR = d->MR, NR = d->NR;
  int64_t cs_ = d->csize;
  int64_t nmb = mm_ceil_div(mc, MR);
  int64_t nnb = mm_ceil_div(nc, NR);
  _Alignas(16) char tile[MM_MR * 8 * 16]; /* MR * max(NR) * max(compute size) */

  if (k <= MM_KC_FULLK_MAX || Cacc == NULL) {
    for (int64_t jr = 0; jr < nnb; jr++) {
      int64_t cj = jr * NR;
      int nr = (int)(nc - cj);
      if (nr > NR) nr = NR;
      const char *bp = Bp + jr * (int64_t)NR * k * cs_;
      for (int64_t ir = 0; ir < nmb; ir++) {
        int64_t ri = ir * MR;
        int mr = (int)(mc - ri);
        if (mr > MR) mr = MR;
        const char *ap = Ap + ir * (int64_t)MR * k * cs_;
        d->micro(tile, ap, bp, k);
        d->store(tile, c, crs, ccs, row0 + ri, col0 + cj, mr, nr, NR);
      }
    }
    return;
  }

  /* KC-blocked: sum MM_KC-deep partials into Cacc (compute type), flush once. */
  int tilecnt = MR * NR;
  memset(Cacc, 0, (size_t)(nnb * nmb) * (size_t)tilecnt * (size_t)cs_);
  for (int64_t pc = 0; pc < k; pc += MM_KC) {
    int64_t kc = k - pc;
    if (kc > MM_KC) kc = MM_KC;
    for (int64_t jr = 0; jr < nnb; jr++) {
      const char *bp = Bp + (jr * (int64_t)NR * k + pc * NR) * cs_;
      for (int64_t ir = 0; ir < nmb; ir++) {
        const char *ap = Ap + (ir * (int64_t)MR * k + pc * MR) * cs_;
        d->micro(tile, ap, bp, kc);
        char *acc = Cacc + (jr * nmb + ir) * (int64_t)tilecnt * cs_;
        d->acc(acc, tile, tilecnt);
      }
    }
  }
  for (int64_t jr = 0; jr < nnb; jr++) {
    int64_t cj = jr * NR;
    int nr = (int)(nc - cj);
    if (nr > NR) nr = NR;
    for (int64_t ir = 0; ir < nmb; ir++) {
      int64_t ri = ir * MR;
      int mr = (int)(mc - ri);
      if (mr > MR) mr = MR;
      const char *acc = Cacc + (jr * nmb + ir) * (int64_t)tilecnt * cs_;
      d->store(acc, c, crs, ccs, row0 + ri, col0 + cj, mr, nr, NR);
    }
  }
}

/* One NC-wide output panel [., jc:jc+nc): pack B once (reused across all ic),
   then for each MC row-block pack A and compute the MC x NC block. Ap/Bp/Cacc are
   the caller's per-thread scratch (Cacc NULL unless the KC path is active). */
static void nx_c_gemm_panel(const nx_c_mm_desc *d, const char *a, int64_t ars,
                           int64_t acs, const char *b, int64_t brs, int64_t bcs,
                           char *c, int64_t crs, int64_t ccs, int64_t m,
                           int64_t k, int64_t jc, int64_t nc, char *Ap, char *Bp,
                           char *Cacc) {
  int MR = d->MR, NR = d->NR;
  d->pack_b(Bp, b, brs, bcs, jc, nc, k, NR);
  for (int64_t ic = 0; ic < m; ic += MM_MC) {
    int64_t mc = m - ic;
    if (mc > MM_MC) mc = MM_MC;
    d->pack_a(Ap, a, ars, acs, ic, mc, k, MR);
    nx_c_gemm_macrotile(d, Ap, Bp, k, mc, nc, c, crs, ccs, ic, jc, Cacc);
  }
}

/* Shared read-only plan for the parallel bodies. Pointers reference the driver's
   stack, which outlives nx_c_parallel_for (it returns only after every worker
   finishes). Ap/Bp/Cacc are nthreads contiguous 64-byte-aligned slots; a body
   takes its own by `worker`, so no locking. */
typedef struct {
  const nx_c_mm_desc *d;
  const nx_c_ndarray *A;
  const nx_c_ndarray *B;
  const nx_c_ndarray *C;
  int64_t m, n, k, esz;
  int64_t a_rs, a_cs, b_rs, b_cs, c_rs, c_cs;
  int batch_nd;
  const int64_t *bshape;
  const int64_t *as_;
  const int64_t *bs_;
  const int64_t *cs_;
  int64_t n_jc;             /* NC-panels per batch element */
  int64_t n_ic;            /* MC-blocks per matrix (M-parallel path) */
  char *Ap, *Bp;            /* per-thread pack scratch bases */
  char *Cacc;               /* per-thread KC accumulator base, NULL if unused */
  int64_t ap_slot, bp_slot; /* per-thread slot bytes */
  int64_t cacc_slot;        /* per-thread KC accumulator slot bytes (0 if unused) */
  /* M-parallel path only: B pre-packed once into Bshared, indexed per
     (batch, NC-panel) by bp_panel; the body packs A per (batch, panel, MC-block)
     and reads its shared B panel. */
  char *Bshared;
  int64_t bp_panel; /* bytes of one shared packed B panel */
} mm_ctx;

/* Byte base of matrix operands for batch element bt (broadcast strides). */
static void mm_batch_base(const mm_ctx *x, int64_t bt, const char **ab,
                          const char **bb, char **cb) {
  int64_t ao = x->A->offset, bo = x->B->offset, co = x->C->offset;
  int64_t rem = bt;
  for (int i = x->batch_nd - 1; i >= 0; i--) {
    int64_t q = rem % x->bshape[i];
    rem /= x->bshape[i];
    ao += q * x->as_[i];
    bo += q * x->bs_[i];
    co += q * x->cs_[i];
  }
  *ab = (const char *)x->A->data + ao * x->esz;
  *bb = (const char *)x->B->data + bo * x->esz;
  *cb = (char *)x->C->data + co * x->esz;
}

/* Per-worker KC accumulator slot, or NULL when the KC path is inactive. */
static char *mm_worker_cacc(const mm_ctx *x, int worker) {
  return x->Cacc ? x->Cacc + (int64_t)worker * x->cacc_slot : NULL;
}

/* Panel path: one job is one (batch, NC-panel) pair. The worker packs both A and
   B and drives every MC-block of the panel (best locality: B packed once per
   panel, reused across the MC-blocks). Used when there are >= pool panels. */
static void mm_blocked_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  const mm_ctx *x = (const mm_ctx *)vctx;
  char *Ap = x->Ap + (int64_t)worker * x->ap_slot;
  char *Bp = x->Bp + (int64_t)worker * x->bp_slot;
  char *Cacc = mm_worker_cacc(x, worker);
  for (int64_t job = lo; job < hi; job++) {
    int64_t bt = job / x->n_jc;
    int64_t jc = (job % x->n_jc) * MM_NC;
    int64_t nc = x->n - jc;
    if (nc > MM_NC) nc = MM_NC;
    const char *ab, *bb;
    char *cb;
    mm_batch_base(x, bt, &ab, &bb, &cb);
    nx_c_gemm_panel(x->d, ab, x->a_rs, x->a_cs, bb, x->b_rs, x->b_cs, cb, x->c_rs,
                   x->c_cs, x->m, x->k, jc, nc, Ap, Bp, Cacc);
  }
}

/* M-parallel path: one job is one (batch, NC-panel, MC-block) triple — an
   MC x NC output tile. B is pre-packed once per (batch, panel) into Bshared, so a
   worker only packs its A row-block; disjoint (ic, jc) tiles never collide, so no
   locking. Used when a matrix is too narrow to yield one panel per core. */
static void mm_mpar_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  const mm_ctx *x = (const mm_ctx *)vctx;
  char *Ap = x->Ap + (int64_t)worker * x->ap_slot;
  char *Cacc = mm_worker_cacc(x, worker);
  int MR = x->d->MR;
  for (int64_t job = lo; job < hi; job++) {
    int64_t ip = job % x->n_ic;
    int64_t t = job / x->n_ic;
    int64_t jp = t % x->n_jc;
    int64_t bt = t / x->n_jc;
    int64_t jc = jp * MM_NC;
    int64_t nc = x->n - jc;
    if (nc > MM_NC) nc = MM_NC;
    int64_t ic = ip * MM_MC;
    int64_t mc = x->m - ic;
    if (mc > MM_MC) mc = MM_MC;
    const char *ab, *bb;
    char *cb;
    mm_batch_base(x, bt, &ab, &bb, &cb);
    (void)bb; /* B comes pre-packed from the shared buffer, not the raw base */
    char *Bp = x->Bshared + (bt * x->n_jc + jp) * x->bp_panel;
    x->d->pack_a(Ap, ab, x->a_rs, x->a_cs, ic, mc, x->k, MR);
    nx_c_gemm_macrotile(x->d, Ap, Bp, x->k, mc, nc, cb, x->c_rs, x->c_cs, ic, jc,
                       Cacc);
  }
}

/* Direct path: one job is one whole batch matrix (small, or force_direct). */
static void mm_direct_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const mm_ctx *x = (const mm_ctx *)vctx;
  for (int64_t bt = lo; bt < hi; bt++) {
    const char *ab, *bb;
    char *cb;
    mm_batch_base(x, bt, &ab, &bb, &cb);
    x->d->direct(ab, x->a_rs, x->a_cs, bb, x->b_rs, x->b_cs, cb, x->c_rs,
                 x->c_cs, x->m, x->n, x->k);
  }
}

/* ── Accelerate hook (macOS only) ──────────────────────────────────────────

   For f32/f64/c32/c64, the top-level driver routes large cblas-mappable products
   to Accelerate (see the file-header note). Everything here is under __APPLE__;
   the owned path above is untouched and is the only path off macOS. */
#if defined(__APPLE__)

/* Per-matrix flop count below which the cblas call overhead outweighs its
   throughput edge and the owned direct/blocked path wins; above it Accelerate
   dominates. The public matmul benchmark guards this crossover. Batched-small
   (per-matrix under the cutoff) therefore
   stays on the owned batch-parallel path, which beats serial cblas calls. */
#define MM_ACCEL_CUTOFF (64 * 64 * 64)

/* Test-only routing override: -1 use the automatic platform policy (default),
   0 force owned, 1 force Accelerate. Lets the backend-local differential
   exercise both routes in one process. */
static int g_accel_override = -1;

/* Hook enabled? The test override wins; normal macOS builds always use
   Accelerate for eligible products. */
static int mm_accel_enabled(void) {
  if (g_accel_override >= 0) return g_accel_override;
  return 1;
}

/* Map a 2-D operand's (row,col) element strides to a cblas transpose + leading
   dimension — the owned pack's stride insight applied to cblas's ld model: a unit
   column stride is row-major (NoTrans, ld = row stride); a unit row stride is its
   transpose (Trans, ld = col stride); any other layout is not cblas-mappable and
   the caller falls back to the owned pack, which resolves arbitrary strides.
   rows/cols bound ld so a degenerate stride cannot alias distinct elements onto
   one cell. Returns 1 on success. */
static int mm_accel_map(int64_t rows, int64_t cols, int64_t rs, int64_t cs,
                        enum CBLAS_TRANSPOSE *trans, int *ld) {
  if (rs <= 0 || cs <= 0) return 0;
  if (cs == 1) {
    if (rs < cols || rs > INT_MAX) return 0;
    *trans = CblasNoTrans;
    *ld = (int)rs;
    return 1;
  }
  if (rs == 1) {
    if (cs < rows || cs > INT_MAX) return 0;
    *trans = CblasTrans;
    *ld = (int)cs;
    return 1;
  }
  return 0;
}

/* cblas parameters resolved once — batch matrices share the 2-D strides, so the
   transpose/ld triple is computed in the driver and the body only walks the batch
   base pointers (mm_batch_base, via the shared mm_ctx). */
typedef struct {
  const mm_ctx *x;
  nx_c_dtype dt;
  enum CBLAS_TRANSPOSE ta, tb;
  int lda, ldb, ldc;
  int m, n, k;
} mm_accel_ctx;

/* Eligible for Accelerate? A native-compute float dtype, above the cutoff, with
   all three operands cblas-mappable and the output row-major (cblas writes C in
   row-major only). Fills `ac` on success; leaves it untouched otherwise. */
static int mm_accel_eligible(nx_c_dtype dt, int64_t m, int64_t n, int64_t k,
                             int64_t a_rs, int64_t a_cs, int64_t b_rs,
                             int64_t b_cs, int64_t c_rs, int64_t c_cs,
                             mm_accel_ctx *ac) {
  if (dt != NX_C_DTYPE_f32 && dt != NX_C_DTYPE_f64 && dt != NX_C_DTYPE_c32 &&
      dt != NX_C_DTYPE_c64)
    return 0;
  if (m > INT_MAX || n > INT_MAX || k > INT_MAX) return 0;
  if (m * n * k < MM_ACCEL_CUTOFF) return 0;
  enum CBLAS_TRANSPOSE ta, tb, tc;
  int lda, ldb, ldc;
  if (!mm_accel_map(m, k, a_rs, a_cs, &ta, &lda)) return 0;
  if (!mm_accel_map(k, n, b_rs, b_cs, &tb, &ldb)) return 0;
  if (!mm_accel_map(m, n, c_rs, c_cs, &tc, &ldc)) return 0;
  if (tc != CblasNoTrans) return 0;
  ac->dt = dt;
  ac->ta = ta;
  ac->tb = tb;
  ac->lda = lda;
  ac->ldb = ldb;
  ac->ldc = ldc;
  ac->m = (int)m;
  ac->n = (int)n;
  ac->k = (int)k;
  return 1;
}

/* One cblas GEMM per batch matrix, on the calling thread with the runtime lock
   released by nx_c_parallel_for (nthreads == 1, so this runs inline and NEVER on a
   pool worker — Accelerate owns its own internal threads, and calling it from N
   workers would oversubscribe). Accelerate has no batched API, so batched-large
   is serial per-matrix cblas. */
static void mm_accel_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const mm_accel_ctx *ac = (const mm_accel_ctx *)vctx;
  const mm_ctx *x = ac->x;
  for (int64_t bt = lo; bt < hi; bt++) {
    const char *ab, *bb;
    char *cb;
    mm_batch_base(x, bt, &ab, &bb, &cb);
    switch (ac->dt) {
      case NX_C_DTYPE_f32:
        cblas_sgemm(CblasRowMajor, ac->ta, ac->tb, ac->m, ac->n, ac->k, 1.0f,
                    (const float *)ab, ac->lda, (const float *)bb, ac->ldb, 0.0f,
                    (float *)cb, ac->ldc);
        break;
      case NX_C_DTYPE_f64:
        cblas_dgemm(CblasRowMajor, ac->ta, ac->tb, ac->m, ac->n, ac->k, 1.0,
                    (const double *)ab, ac->lda, (const double *)bb, ac->ldb, 0.0,
                    (double *)cb, ac->ldc);
        break;
      case NX_C_DTYPE_c32: {
        const nx_c_complex32 alpha = 1, beta = 0;
        cblas_cgemm(CblasRowMajor, ac->ta, ac->tb, ac->m, ac->n, ac->k, &alpha,
                    (const nx_c_complex32 *)ab, ac->lda,
                    (const nx_c_complex32 *)bb, ac->ldb, &beta,
                    (nx_c_complex32 *)cb, ac->ldc);
        break;
      }
      case NX_C_DTYPE_c64: {
        const nx_c_complex64 alpha = 1, beta = 0;
        cblas_zgemm(CblasRowMajor, ac->ta, ac->tb, ac->m, ac->n, ac->k, &alpha,
                    (const nx_c_complex64 *)ab, ac->lda,
                    (const nx_c_complex64 *)bb, ac->ldb, &beta,
                    (nx_c_complex64 *)cb, ac->ldc);
        break;
      }
      default:
        break; /* unreachable: mm_accel_eligible gated the dtype */
    }
  }
}
#endif /* __APPLE__ */

/* ── Driver: validate, resolve batch broadcast, iterate ───────────────────

   All three operands share dtype dt (checked in the funnel). Batch dims broadcast
   via strides exactly as the frontend supplies them (size-1 dims carry stride 0).
   Returns a status; the funnel raises. */
static const char MM_ERR_DTYPE_MISMATCH[] =
    "matmul operands must share one dtype";

/* nthreads: 0 = engine policy (nx_c_threads_for), > 0 = forced by maintenance
   tests. allow_accel: 1 lets the
   macOS Accelerate hook claim eligible f32/f64/c32/c64 products (the frontend
   path); 0 forces the owned kernel for maintenance tests and benchmarks. */
static nx_c_status nx_c_matmul_run(const nx_c_ndarray *A, const nx_c_ndarray *B,
                                 const nx_c_ndarray *C, nx_c_dtype dt,
                                 int force_direct, int nthreads, int allow_accel) {
  const nx_c_mm_desc *d = &mm_desc[dt];
  if (d->micro == NULL)
    return nx_c_dtype_is_packed(dt) ? NX_C_ERR_PACKED : NX_C_ERR_UNSUPPORTED_DTYPE;
  if (A->ndim < 2 || B->ndim < 2) return NX_C_ERR_SHAPE;
  int nd = A->ndim > B->ndim ? A->ndim : B->ndim;
  if (C->ndim != nd) return NX_C_ERR_SHAPE;

  int64_t m = A->shape[A->ndim - 2];
  int64_t k = A->shape[A->ndim - 1];
  int64_t kk = B->shape[B->ndim - 2];
  int64_t n = B->shape[B->ndim - 1];
  if (k != kk) return NX_C_ERR_SHAPE;
  if (C->shape[C->ndim - 2] != m || C->shape[C->ndim - 1] != n)
    return NX_C_ERR_SHAPE;

  int64_t esz = nx_c_elem_size(dt);
  int MR = d->MR, NR = d->NR;

  /* Batch broadcast strides (element units). */
  int batch_nd = nd - 2;
  int64_t bshape[NX_C_MAX_NDIM];
  int64_t as_[NX_C_MAX_NDIM], bs_[NX_C_MAX_NDIM], cs_[NX_C_MAX_NDIM];
  int a_bo = nd - A->ndim, b_bo = nd - B->ndim;
  int64_t nbatch = 1;
  for (int i = 0; i < batch_nd; i++) {
    int64_t sa = 1, sb = 1, sta = 0, stb = 0;
    if (i >= a_bo) {
      int ai = i - a_bo;
      sa = A->shape[ai];
      sta = A->strides[ai];
    }
    if (i >= b_bo) {
      int bi = i - b_bo;
      sb = B->shape[bi];
      stb = B->strides[bi];
    }
    if (sa != sb && sa != 1 && sb != 1) return NX_C_ERR_SHAPE;
    int64_t s = sa > sb ? sa : sb;
    if (C->shape[i] != s) return NX_C_ERR_SHAPE;
    bshape[i] = s;
    as_[i] = (sa == 1) ? 0 : sta;
    bs_[i] = (sb == 1) ? 0 : stb;
    cs_[i] = C->strides[i];
    /* A/B batch dims may broadcast (stride 0); the OUTPUT never can — a 0-stride
       over an extent > 1 aliases distinct batch matrices onto one cell. The
       binding always allocates a contiguous out, so this is future-proofing the
       driver the way nx_c_map_run guards its output. */
    if (s > 1 && cs_[i] == 0) return NX_C_ERR_OUT_ALIASED;
    nbatch *= s;
  }

  /* Empty output (or empty batch) writes nothing. k == 0 is a real zero-fill
     handled by the normal path (the microkernel sums an empty k to 0). */
  if (m == 0 || n == 0 || nbatch == 0) return NX_C_OK;

  int64_t a_rs = A->strides[A->ndim - 2], a_cs = A->strides[A->ndim - 1];
  int64_t b_rs = B->strides[B->ndim - 2], b_cs = B->strides[B->ndim - 1];
  int64_t c_rs = C->strides[C->ndim - 2], c_cs = C->strides[C->ndim - 1];
  if ((m > 1 && c_rs == 0) || (n > 1 && c_cs == 0))
    return NX_C_ERR_OUT_ALIASED; /* distinct C rows/cols would collide on one cell */

  int use_direct = force_direct || (m < MR) || (n < NR) ||
                   ((int64_t)m * n * k < MM_DIRECT_CUTOFF);

  /* Rough total traffic, for the pool's lock-release decision (HEAVY threads
     off run count, not this). A large GEMM clears the cutoff and releases. */
  int64_t bytes = (m * k + k * n + m * n) * esz * nbatch;

  mm_ctx x;
  x.d = d;
  x.A = A;
  x.B = B;
  x.C = C;
  x.m = m;
  x.n = n;
  x.k = k;
  x.esz = esz;
  x.a_rs = a_rs;
  x.a_cs = a_cs;
  x.b_rs = b_rs;
  x.b_cs = b_cs;
  x.c_rs = c_rs;
  x.c_cs = c_cs;
  x.batch_nd = batch_nd;
  x.bshape = bshape;
  x.as_ = as_;
  x.bs_ = bs_;
  x.cs_ = cs_;
  x.Ap = NULL;
  x.Bp = NULL;
  x.Cacc = NULL;
  x.Bshared = NULL;
  x.ap_slot = 0;
  x.bp_slot = 0;
  x.cacc_slot = 0;
  x.bp_panel = 0;
  x.n_jc = 0;
  x.n_ic = 0;

  /* Accelerate hook: for eligible f32/f64/c32/c64 the driver hands each batch
     matrix to cblas, one call at a time, on the calling thread with the runtime
     lock released by nx_c_parallel_for (nthreads == 1 keeps it off the pool — the
     bytes cutoff still releases for a large product). Ineligible operands
     (low-precision/int, non-mappable layout, sub-cutoff, off macOS) fall through
     to the owned path below. */
#if defined(__APPLE__)
  if (allow_accel && mm_accel_enabled()) {
    mm_accel_ctx ac;
    if (mm_accel_eligible(dt, m, n, k, a_rs, a_cs, b_rs, b_cs, c_rs, c_cs, &ac)) {
      ac.x = &x;
      nx_c_parallel_for(1, nbatch, bytes, mm_accel_body, &ac, NULL);
      return NX_C_OK;
    }
  }
#else
  (void)allow_accel;
#endif

  if (use_direct) {
    /* One job per batch matrix; a lone matrix runs on one thread (the fair naive
       baseline the test measures against). */
    int nth = nthreads > 0
                  ? nthreads
                  : nx_c_threads_for(NX_C_COST_HEAVY, nbatch, m * n * k, bytes);
    if (nth > nbatch) nth = (int)nbatch;
    if (nth < 1) nth = 1;
    nx_c_parallel_for(nth, nbatch, bytes, mm_direct_body, &x, NULL);
    return NX_C_OK;
  }

  /* Blocked path. Two disjoint job spaces over the engine's one pool:
       panel:     (batch x NC-panel), a worker owns whole panels — best locality.
       M-parallel:(batch x NC-panel x MC-block), used only when panels < the
                  threads the work wants, so a narrow matrix (the extreme n <= NC
                  gives one panel, otherwise serial) still fills the pool. B is
                  pre-packed once into a shared buffer; workers pack only A.
     The KC accumulator (Cacc) is per-worker and shared by both, allocated only
     when k forces contraction sub-blocking. */
  int64_t n_jc = mm_ceil_div(n, MM_NC);
  int64_t n_ic = mm_ceil_div(m, MM_MC);
  int64_t panels = nbatch * n_jc;
  int use_kc = (k > MM_KC_FULLK_MAX);
  int64_t mcm = m < MM_MC ? m : MM_MC;
  int64_t ncm = n < MM_NC ? n : MM_NC;

  int use_mpar = 0;
  int nth;
  int64_t total_jobs;
  if (nthreads > 0) {
    nth = nthreads; /* forced (the ST gate) — always the panel path */
    total_jobs = panels;
  } else {
    int nth_panel = nx_c_threads_for(NX_C_COST_HEAVY, panels, m * ncm * k, bytes);
    if (nth_panel > panels) nth_panel = (int)panels;
    int64_t fine = panels * n_ic;
    int nth_fine =
        n_ic > 1 ? nx_c_threads_for(NX_C_COST_HEAVY, fine, mcm * ncm * k, bytes)
                 : nth_panel;
    if (nth_fine > fine) nth_fine = (int)fine;
    if (n_ic > 1 && nth_fine > nth_panel) {
      use_mpar = 1;
      nth = nth_fine;
      total_jobs = fine;
    } else {
      nth = nth_panel;
      total_jobs = panels;
    }
  }
  if (nth < 1) nth = 1;
  if (nth > total_jobs) nth = (int)total_jobs;

  int64_t ap_slot = mm_ceil_div(mcm, MR) * MR * k * d->csize;
  int64_t bp_slot = mm_ceil_div(ncm, NR) * NR * k * d->csize;
  int64_t cacc_slot =
      use_kc ? mm_ceil_div(mcm, MR) * mm_ceil_div(ncm, NR) * (int64_t)MR * NR *
                   d->csize
             : 0;
  if (ap_slot < 64) ap_slot = 64; /* nonzero (k==0) and 64-aligned per slot */
  if (bp_slot < 64) bp_slot = 64;
  ap_slot = (ap_slot + 63) & ~(int64_t)63;
  bp_slot = (bp_slot + 63) & ~(int64_t)63;
  if (cacc_slot) cacc_slot = (cacc_slot + 63) & ~(int64_t)63;
  x.ap_slot = ap_slot;
  x.bp_slot = bp_slot;
  x.cacc_slot = cacc_slot;
  x.n_jc = n_jc;
  x.n_ic = n_ic;

  /* One allocation for every scratch region: the primitive frees it leak-safely
     across the lock re-acquire (nx_c_engine.h). Cacc trails the pack scratch. */
  if (use_mpar) {
    /* [Bshared: panels packed B panels][Ap: nth][Cacc: nth]. B lives once,
       indexed per (batch, panel); A and the KC accumulator are per-worker. */
    size_t sz = (size_t)panels * (size_t)bp_slot +
                (size_t)nth * (size_t)(ap_slot + cacc_slot);
    char *scratch = mm_alloc(sz);
    if (!scratch) return NX_C_ERR_ALLOC;
    x.Bshared = scratch;
    x.bp_panel = bp_slot;
    x.Ap = scratch + (size_t)panels * (size_t)bp_slot;
    x.Cacc = cacc_slot ? x.Ap + (size_t)nth * (size_t)ap_slot : NULL;
    /* Pre-pack every B panel once, serially, under the runtime lock. This path
       runs only when n_ic > 1 (m > MC), and the pack touches O(nbatch*n*k)
       elements against the GEMM's O(nbatch*m*n*k) flops — O(1/m) of the work,
       hence under 1/MC = 1/256 whenever it runs. That bound (not merely "panels
       is small") is what keeps the serial pack and its lock hold negligible. */
    for (int64_t bt = 0; bt < nbatch; bt++) {
      const char *ab, *bb;
      char *cb;
      mm_batch_base(&x, bt, &ab, &bb, &cb);
      for (int64_t jp = 0; jp < n_jc; jp++) {
        int64_t jc = jp * MM_NC;
        int64_t nc = n - jc;
        if (nc > MM_NC) nc = MM_NC;
        char *Bp = x.Bshared + (bt * n_jc + jp) * bp_slot;
        d->pack_b(Bp, bb, b_rs, b_cs, jc, nc, k, NR);
      }
    }
    nx_c_parallel_for(nth, total_jobs, bytes, mm_mpar_body, &x, scratch);
    return NX_C_OK;
  }

  /* Panel path: [Ap: nth][Bp: nth][Cacc: nth]. */
  size_t sz = (size_t)nth * (size_t)(ap_slot + bp_slot + cacc_slot);
  char *scratch = mm_alloc(sz);
  if (!scratch) return NX_C_ERR_ALLOC;
  x.Ap = scratch;
  x.Bp = scratch + (size_t)ap_slot * (size_t)nth;
  x.Cacc = cacc_slot ? x.Bp + (size_t)bp_slot * (size_t)nth : NULL;
  nx_c_parallel_for(nth, total_jobs, bytes, mm_blocked_body, &x, scratch);
  return NX_C_OK;
}

/* ── Compute-typed GEMM entry for the linalg family (nx_c_matmul.h) ─────────

   One 2-D C = A·B over COMPUTE-typed strided buffers, dt in {f32,f64,c32,c64}
   (nx_c_linalg upcasts the low-precision floats to f32 before calling). Transpose
   / conjugate are expressed by the caller through strides on already-materialized
   panels — the pack resolves any (row,col) stride, so no transpose flag is
   needed; a caller wanting A·Bᵀ passes B with its strides swapped. Serial: the
   linalg driver owns batch parallelism above this, so this reuses the GEMM's
   blocking/microkernels (KC sub-blocking and all) without touching its pool.
   Scratch is allocated per call (linalg calls this O(n/nb) times per
   factorization; the alloc is amortized). */
nx_c_status nx_c_gemm2d_ct(nx_c_dtype dt, int64_t m, int64_t n, int64_t k,
                         const char *A, int64_t a_rs, int64_t a_cs,
                         const char *B, int64_t b_rs, int64_t b_cs, char *C,
                         int64_t c_rs, int64_t c_cs) {
  const nx_c_mm_desc *d = &mm_desc[dt];
  if (d->micro == NULL) return NX_C_ERR_UNSUPPORTED_DTYPE;
  if (m <= 0 || n <= 0) return NX_C_OK;
  int MR = d->MR, NR = d->NR;
  if (k <= 0 || m < MR || n < NR || (int64_t)m * n * k < MM_DIRECT_CUTOFF) {
    d->direct(A, a_rs, a_cs, B, b_rs, b_cs, C, c_rs, c_cs, m, n, k);
    return NX_C_OK;
  }
  int64_t mcm = m < MM_MC ? m : MM_MC;
  int64_t ncm = n < MM_NC ? n : MM_NC;
  char *Ap = mm_alloc((size_t)(mm_ceil_div(mcm, MR) * MR * k) * (size_t)d->csize);
  char *Bp = mm_alloc((size_t)(mm_ceil_div(ncm, NR) * NR * k) * (size_t)d->csize);
  char *Cacc = NULL;
  if (k > MM_KC_FULLK_MAX)
    Cacc = mm_alloc((size_t)(mm_ceil_div(mcm, MR) * mm_ceil_div(ncm, NR) *
                             (int64_t)MR * NR) *
                    (size_t)d->csize);
  if (!Ap || !Bp || (k > MM_KC_FULLK_MAX && !Cacc)) {
    free(Ap);
    free(Bp);
    free(Cacc);
    return NX_C_ERR_ALLOC;
  }
  for (int64_t jc = 0; jc < n; jc += MM_NC) {
    int64_t nc = n - jc;
    if (nc > MM_NC) nc = MM_NC;
    nx_c_gemm_panel(d, A, a_rs, a_cs, B, b_rs, b_cs, C, c_rs, c_cs, m, k, jc, nc,
                   Ap, Bp, Cacc);
  }
  free(Ap);
  free(Bp);
  free(Cacc);
  return NX_C_OK;
}

/* ── Caller-workspace GEMM for pooled linalg bodies (nx_c_matmul.h) ─────────

   The DEVIATION note in nx_c_matmul.h is now redeemed: nx_c_gemm2d_ct_ws is
   nx_c_gemm2d_ct with its packing panels (and the KC accumulator, when the
   contraction sub-blocks) taken from caller-provided `scratch` instead of
   mm_alloc. A linalg factorization calling GEMM from inside a pooled worker body
   then allocates nothing — nx_c_engine.h's "bodies never allocate" holds to the
   letter, not just its purposes. The driver sizes one per-worker slot with
   nx_c_gemm2d_ct_scratch and sub-slots it by worker index. `scratch` must be
   64-byte aligned and at least the size query returns; it may be NULL exactly
   when the query is 0 (the direct path packs nothing). The size formula, the
   layout, and the direct-path cutoff mirror nx_c_gemm2d_ct one-for-one — the two
   MUST move together if the blocking changes. nx_c_gemm2d_ct itself is unchanged
   (still the frontend/non-pooled entry). */
int64_t nx_c_gemm2d_ct_scratch(nx_c_dtype dt, int64_t m, int64_t n, int64_t k) {
  const nx_c_mm_desc *d = &mm_desc[dt];
  if (d->micro == NULL) return 0;
  if (m <= 0 || n <= 0) return 0;
  int MR = d->MR, NR = d->NR;
  if (k <= 0 || m < MR || n < NR || (int64_t)m * n * k < MM_DIRECT_CUTOFF)
    return 0;
  int64_t mcm = m < MM_MC ? m : MM_MC;
  int64_t ncm = n < MM_NC ? n : MM_NC;
  int64_t ap = mm_ceil_div(mcm, MR) * MR * k * d->csize;
  int64_t bp = mm_ceil_div(ncm, NR) * NR * k * d->csize;
  int64_t cacc = (k > MM_KC_FULLK_MAX)
                     ? mm_ceil_div(mcm, MR) * mm_ceil_div(ncm, NR) *
                           (int64_t)MR * NR * d->csize
                     : 0;
  ap = (ap + 63) & ~(int64_t)63;
  bp = (bp + 63) & ~(int64_t)63;
  if (cacc) cacc = (cacc + 63) & ~(int64_t)63;
  return ap + bp + cacc;
}

nx_c_status nx_c_gemm2d_ct_ws(nx_c_dtype dt, int64_t m, int64_t n, int64_t k,
                            const char *A, int64_t a_rs, int64_t a_cs,
                            const char *B, int64_t b_rs, int64_t b_cs, char *C,
                            int64_t c_rs, int64_t c_cs, char *scratch) {
  const nx_c_mm_desc *d = &mm_desc[dt];
  if (d->micro == NULL) return NX_C_ERR_UNSUPPORTED_DTYPE;
  if (m <= 0 || n <= 0) return NX_C_OK;
  int MR = d->MR, NR = d->NR;
  if (k <= 0 || m < MR || n < NR || (int64_t)m * n * k < MM_DIRECT_CUTOFF) {
    d->direct(A, a_rs, a_cs, B, b_rs, b_cs, C, c_rs, c_cs, m, n, k);
    return NX_C_OK;
  }
  int64_t mcm = m < MM_MC ? m : MM_MC;
  int64_t ncm = n < MM_NC ? n : MM_NC;
  int64_t ap = mm_ceil_div(mcm, MR) * MR * k * d->csize;
  int64_t bp = mm_ceil_div(ncm, NR) * NR * k * d->csize;
  ap = (ap + 63) & ~(int64_t)63;
  bp = (bp + 63) & ~(int64_t)63;
  char *Ap = scratch;
  char *Bp = scratch + ap;
  char *Cacc = (k > MM_KC_FULLK_MAX) ? scratch + ap + bp : NULL;
  for (int64_t jc = 0; jc < n; jc += MM_NC) {
    int64_t nc = n - jc;
    if (nc > MM_NC) nc = MM_NC;
    nx_c_gemm_panel(d, A, a_rs, a_cs, B, b_rs, b_cs, C, c_rs, c_cs, m, k, jc, nc,
                   Ap, Bp, Cacc);
  }
  return NX_C_OK;
}

/* ── FFI stub ─────────────────────────────────────────────────────────────
   Output first (nx_c map-family convention); the OCaml binding allocates the
   contiguous result and passes it in. Extraction and dtype coherence run with
   the lock held; the driver runs the GEMM. */
static NX_C_NORETURN void mm_raise(const char *op, nx_c_status s) {
  /* Shape/contraction and aliased-output violations are the caller's bad
     argument; everything else (unsupported dtype, packed, dtype mismatch,
     allocation) is a Failure. Cold path, so strcmp is free — and a status must be
     compared by content, never by pointer (nx_c_selftest.c). */
  if (strcmp(s, NX_C_ERR_SHAPE) == 0 || strcmp(s, NX_C_ERR_OUT_ALIASED) == 0)
    nx_c_raise_invalid(op, s);
  nx_c_raise(op, s);
}

static nx_c_status nx_c_matmul_extract(value va, value vb, value vc,
                                     nx_c_ndarray *A, nx_c_ndarray *B,
                                     nx_c_ndarray *C, nx_c_dtype *dt) {
  nx_c_status s;
  if ((s = nx_c_ndarray_of_value(va, A)) != NX_C_OK) return s;
  if ((s = nx_c_ndarray_of_value(vb, B)) != NX_C_OK) return s;
  if ((s = nx_c_ndarray_of_value(vc, C)) != NX_C_OK) return s;
  nx_c_dtype da = nx_c_dtype_of_value(va);
  nx_c_dtype db = nx_c_dtype_of_value(vb);
  nx_c_dtype dc = nx_c_dtype_of_value(vc);
  if (da == NX_C_DTYPE_COUNT || db == NX_C_DTYPE_COUNT || dc == NX_C_DTYPE_COUNT)
    return NX_C_ERR_BAD_KIND;
  if (da != db || da != dc) return MM_ERR_DTYPE_MISMATCH;
  *dt = da;
  return NX_C_OK;
}

static void mm_stub(const char *op, value vout, value va, value vb,
                    int force_direct, int nthreads, int allow_accel) {
  nx_c_ndarray A, B, C;
  nx_c_dtype dt;
  nx_c_status s = nx_c_matmul_extract(va, vb, vout, &A, &B, &C, &dt);
  if (s != NX_C_OK) mm_raise(op, s);
  /* The run releases and re-acquires the runtime lock internally, via
     nx_c_parallel_for, and fans (batch x panel) over the shared engine pool
     (workers touch only C scratch, never the runtime) — or, on the Accelerate
     path, hands each batch matrix to cblas under the released lock. Called with
     the lock held, as any CAMLprim is. */
  s = nx_c_matmul_run(&A, &B, &C, dt, force_direct, nthreads, allow_accel);
  if (s != NX_C_OK) mm_raise(op, s);
}

CAMLprim value caml_nx_c_matmul(value vout, value va, value vb) {
  CAMLparam3(vout, va, vb);
  /* Frontend path: engine thread policy, Accelerate hook allowed (default-on on
     macOS). */
  mm_stub("matmul", vout, va, vb, 0, 0, 1);
  CAMLreturn(Val_unit);
}

/* Internal maintenance hook. Test/benchmark-only stubs bind this C function;
   the installed OCaml library exposes no path-selection API. Modes 0-2 force the
   owned kernel so it stays covered on macOS where eligible products use cblas:
     0 = owned blocked, engine thread policy   (owned multi-thread path)
     1 = owned blocked, forced single thread
     2 = owned direct naive triple loop
     3 = automatic Accelerate route if eligible, else owned policy. */
void nx_c_matmul_maintenance(value vout, value va, value vb, int mode) {
  int force_direct = (mode == 2);
  int nthreads = (mode == 0 || mode == 3) ? 0 : 1;
  int allow_accel = (mode == 3);
  mm_stub("matmul", vout, va, vb, force_direct, nthreads, allow_accel);
}

int nx_c_matmul_accelerate_available(void) {
#if defined(__APPLE__)
  return 1;
#else
  return 0;
#endif
}

int nx_c_matmul_accelerate_enabled(void) {
#if defined(__APPLE__)
  return mm_accel_enabled();
#else
  return 0;
#endif
}

void nx_c_matmul_accelerate_override(int mode) {
#if defined(__APPLE__)
  g_accel_override = mode;
#else
  (void)mode;
#endif
}
