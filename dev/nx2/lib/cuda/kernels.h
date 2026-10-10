/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.cuda's kernels as the host and the device both see them: the
   instances, the tiles and each family's parameters.

   A kernel reads one parameter struct, the parameters of its launch. The
   host writes it field by field at the offsets kernels.ml states, and rig
   turns each address field into the address of a buffer of the submit.
   kernels.ml holds every fact of this header the host plans with;
   test/cuda/test_cuda_kernels.ml checks that the two agree. */

#ifndef NX_CUDA_KERNELS_H
#define NX_CUDA_KERNELS_H

#include <stdint.h>

/* Kernels */

/* X(name, FAMILY, ...) for every kernel of the cubin, in enum order; the
   arguments after the family are its instance's:
   - PACK: none.
   - MMA (contract.cu): the operands' kind (bf16, f16, s8, or any for
     each of them, by a's dtype at the kernel's entry), the contiguous
     axis of a (k or m) and of b (k or n), and the tile.
   - SIMT (contract.cu): the accumulator's type, a lane's TM x TN outputs
     and the WARPS_M x WARPS_N warps of the block.
   - SKINNY (contract.cu): the accumulator's type.
   - FOLD (fold.cu): none; the monoid and dtype are parameters.
   Instances are a budget: each names the rows that keep it. The plan
   packs an operand into a layout and dtype an instance reads. */
#define NX_CUDA_KERNELS(X)                                                    \
  /* f8 operands, rows not of vectors, layouts with no instance */          \
  X(pack, PACK)                                                             \
  /* bf16 4096 to 8192, 4096x14336x4096, 512x201088x2880 */                 \
  X(contract_bf16_kk_t128x256, MMA, bf16, k, k, t128x256)                   \
  /* the 4096 rows in layouts kf, fk and ff */                              \
  X(contract_bf16_kn_t128x256, MMA, bf16, k, n, t128x256)                   \
  X(contract_bf16_mk_t128x256, MMA, bf16, m, k, t128x256)                   \
  X(contract_bf16_mn_t128x256, MMA, bf16, m, n, t128x256)                   \
  /* bf16 1024, 2048, 512x5120x2880, 512x2880x4096, 64x512x512x512 */       \
  X(contract_bf16_kk_t128x128, MMA, bf16, k, k, t128x128)                   \
  /* bf16 512 */                                                            \
  X(contract_bf16_kk_t64x64, MMA, bf16, k, k, t64x64)                       \
  /* f16 4096, 8192 */                                                      \
  X(contract_f16_kk_t128x256, MMA, f16, k, k, t128x256)                     \
  /* the 4096 rows in layouts kf, fk and ff */                              \
  X(contract_f16_kn_t128x256, MMA, f16, k, n, t128x256)                     \
  X(contract_f16_mk_t128x256, MMA, f16, m, k, t128x256)                     \
  X(contract_f16_mn_t128x256, MMA, f16, m, n, t128x256)                     \
  /* f16 1024, 2048 */                                                      \
  X(contract_f16_kk_t128x128, MMA, f16, k, k, t128x128)                     \
  /* f16 512 */                                                             \
  X(contract_f16_kk_t64x64, MMA, f16, k, k, t64x64)                         \
  /* bf16 and f16 256, and decode: bf16 1x5120x2880 and 1x201088x2880,      \
     f16, float8 and int8 1x5120x2880 */                                    \
  X(contract_any_kk_t16x64, MMA, any, k, k, t16x64)                         \
  /* int8 4096 */                                                           \
  X(contract_s8_kk_t128x256, MMA, s8, k, k, t128x256)                       \
  /* f32 2048 to 8192 and the 4096 layouts */                               \
  X(contract_simt_f32_128x256, SIMT, f32, 16, 8, 2, 4)                      \
  /* f32 256 to 1024 */                                                     \
  X(contract_simt_f32_64x64, SIMT, f32, 8, 4, 2, 2)                         \
  /* f32 decode 1x5120x2880 */                                              \
  X(contract_skinny_f32, SKINNY, f32)                                       \
  /* float64 and integer sums, which every library computes: float64-       \
     1024x1024x1024 and -1x5120x2880, int16-1024x1024x1024 and              \
     -1x5120x2880 (int64 sums) */                                           \
  X(contract_simt_f64_64x64, SIMT, f64, 4, 4, 4, 2)                         \
  X(contract_simt_i64_64x64, SIMT, i64, 4, 4, 4, 2)                         \
  X(contract_skinny_f64, SKINNY, f64)                                       \
  X(contract_skinny_i64, SKINNY, i64)                                       \
  /* reductions whose terms run along the operand: sum-rows-*, sum-all-* */ \
  X(fold_rows, FOLD)                                                        \
  /* reductions across outputs, and few terms: sum-cols-*, sum-rows-4-4M */ \
  X(fold_cols, FOLD)                                                        \
  /* an output's ranges, where several fold it: sum-all-* */                 \
  X(fold_tree, FOLD)                                                        \
  /* scans: cumsum-*, the totals where a slice has several chunks */        \
  X(scan_totals, FOLD)                                                      \
  X(scan_rescan, FOLD)

#define NX_CUDA_ENUM(name, ...) NX_CUDA_##name,
enum nx_cuda_kernel { NX_CUDA_KERNELS(NX_CUDA_ENUM) NX_CUDA_KERNEL_COUNT };
#undef NX_CUDA_ENUM

/* X(name, BM, BN, BKB, WM, WN, STAGES): the mma kernels' tiles. A block
   computes BM x BN outputs, (BM / WM) x (BN / WN) warps of WM x WN each,
   over k-tiles of BKB bytes of each operand row, STAGES of them in flight
   in its BM x BKB and BN x BKB shared buffers. */
#define NX_CUDA_TILES(X)                     \
  X(t128x128, 128, 128, 64, 64, 32, 4)       \
  X(t128x256, 128, 256, 64, 64, 64, 4)       \
  X(t64x64, 64, 64, 128, 32, 32, 4)          \
  X(t16x64, 16, 64, 128, 16, 16, 4)

/* Contract */

/* y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k], rounded once
   to y's dtype, for z < batch, i < m, j < n: each operand's element
   (z, i, k) at its address plus z, i and k times its strides, in elements.
   [init] is NULL for none. When [splits] > 1 the sum over k is cut into
   [splits] ranges, each summed by its own block into [partials], and the
   last block to take its tile's ticket adds them in range order
   (combine.cuh): [tickets] holds a zero word a tile, which that block
   returns to zero. a and b are of the dtypes [a_dtype] and [b_dtype], the
   kernel's own: an mma kernel of kind any sums the kind of [a_dtype], and
   the integer skinny kernel's loop reads [b_dtype]. The sum, of
   [acc_dtype], reaches y as a cast from it does. [aligned] holds the
   NX_CONTRACT_ bits below. */
typedef struct {
  const void *a, *b, *init;
  void *y, *partials;
  uint32_t *tickets;
  int64_t sa[3], sb[3], si[3], sy[3];
  int32_t batch, m, n, k;
  int32_t splits, a_dtype, b_dtype, init_dtype;
  int32_t y_dtype, acc_dtype, aligned, unused;
} contract_params;

/* contract_params' bits of [aligned]. */
enum {
  NX_CONTRACT_A_VECTORS = 1, /* a's rows are 16-byte vectors */
  NX_CONTRACT_B_VECTORS = 2, /* b's rows are 16-byte vectors */
  NX_CONTRACT_B_ACROSS = 4,  /* skinny: b's n axis is contiguous */
  NX_CONTRACT_Y_WHOLE = 8    /* y's outputs store 16 bytes at once */
};

/* The SIMT kernels' k-tile, in elements, and the k-tiles in flight. */
#define NX_SIMT_BK 16
#define NX_SIMT_STAGES 3

/* The skinny kernels' block: NX_SKINNY_ROWS rows of 32 columns. */
#define NX_SKINNY_ROWS 4

/* pack: an operand of [batch] x [rows] x [k] elements of [dtype], element
   (z, r, q) at [src] plus z, r and q times [s], copied to [dst] as
   elements of [out], [bytes] bytes each, with k contiguous, [lead]
   elements a row (a whole number of 16-byte vectors, the padding zero),
   rows * lead a batch element: the layout and dtype the kernels load as
   vectors. Each value converts exactly: [out] is a float at least as wide
   (bfloat16 for a float8), or an integer as wide or wider. */
typedef struct {
  const void *src;
  void *dst;
  int64_t s[3], lead;
  int32_t batch, rows, k, dtype, out, bytes;
} pack_params;

/* Fold */

/* Reductions and scans of one operand by a monoid (fold.cu, whose header
   states the association): terms in blocks of NX_FOLD_BLOCK, NX_FOLD_LANES
   lanes a block; a scan's chunks of NX_SCAN_CHUNK. */
#define NX_FOLD_BLOCK 1024
#define NX_FOLD_LANES 16
#define NX_SCAN_CHUNK 4096
#define NX_FOLD_RANK 32 /* Layout.max_rank */

/* y[o] = the fold by [monoid] (nx_spec.h's NX_SUM to NX_MIN) of the
   [terms] terms of output o < [outputs] of x, both of the dtype [dtype].
   Output o lies along [nkept] kept axes, term t along [nred] reduced ones,
   each in C order: kept[d] and red[d] hold an axis's extent, then x's
   stride and y's, in elements. y holds the outputs in order. An output's
   [blocks] blocks fall into [groups] ranges of [span] blocks, [full] of
   them whole, folded into [partials] where [groups] > 1 (fold_tree's
   [span] is values a thread).

   A scan's [outputs] slices run along the one axis red[0], [terms] long,
   in [blocks] chunks; slice s of y lies at its kept offset. */
typedef struct {
  const void *x;
  void *y, *partials;
  int64_t outputs, terms, blocks;
  int64_t groups, full, span;
  int32_t monoid, dtype, nkept, nred;
  int64_t kept[NX_FOLD_RANK][3];
  int64_t red[NX_FOLD_RANK][3];
} fold_params;

#endif
