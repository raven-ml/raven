/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.cuda's kernels as the host and the device both see them: the
   instances, the tiles and each family's parameters.

   A kernel reads one parameter struct, the parameters of its launch. The
   host writes it field by field at the offsets kernels.ml states, and rig
   turns each address field into the address of a buffer of the submit.
   kernels/gen.py writes kernels.ml from this header: its X-macros' rows,
   its macros of an integer, its enumerations' constants and its structs'
   offsets. The host calls its functions and NX_GEMV_STAGE through
   nx_cuda_stubs.c. */

#ifndef NX_CUDA_KERNELS_H
#define NX_CUDA_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* Kernels */

/* A step's arithmetic format, where one kernel serves every format. */
#define NX_FMT_F32 0
#define NX_FMT_F64 1
#define NX_FMT_C64 2
#define NX_FMT_C128 3

/* X(name, FAMILY, ...) for every kernel of the cubin, in enum order; the
   arguments after the family are its instance's:
   - PACK, ROWS, KS (contract.cu): none. A pack copies an operand into the
     workspace; ROWS and KS lay out a segmented call's groups.
   - GEMM (contract.cu): the compute type (f32, f64, u32, u64), the block's
     BM x BN outputs over k-tiles of BK, a lane's TM x TN outputs, its
     warp's LM rows of lanes, the WM x WN warps, the blocks an SM must
     hold (launch bounds) and the blocks it holds (OCC, from ptxas's
     registers and the tile's shared memory, which the plan's model
     reads), and the load: native (operands of the compute type) or
     convert (any real dtype).
   - GEMV (contract.cu): the compute type; at most NX_GEMV_ROWS rows.
   - MMA (mma.cuh): the operands' kind (bf16, f16, s8, or any for each of
     them, by a's dtype at the kernel's entry), the contiguous axis of a
     (k or m) and of b (k or n), and the tile.
   - LOOP (loop.cu): none; the program is its parameters.
   - LINALG (factor.cu): the step, its parameters' struct (la_<params>_params)
     and its arithmetic format (f32, f64, c64, c128, or any); LINALG_ANY
     the step and its struct, whose [fmt] field is the format's NX_FMT_.
   - SPECTRAL (spectral.cu): the step, whose parameters are an sp_args,
     and its arithmetic format; SPECTRAL_ANY and SPECTRAL_REAL the step,
     of any format or of a real one, the sp_args' word NX_SP_FMT its
     NX_FMT_.
   - MOVE, ASSEMBLE, FOLD, SCATTER (moves.cu): none; RUNS and
     SCATTER_SORT the step; SCATTER_ADD the step and its format, f32 or
     f64.
   - CLEAR, SORT, UNIQUE (sort.cu): none.
   - FFT, FFT_REAL (fft.cu): the precision, f32 or f64.
   - ATTENTION, ATTENTION_MERGE (attention.cu): none; the path is a
     parameter (NX_ATT_KERNEL).
   Instances are a budget: the plan packs an operand into a layout and
   dtype an instance reads. */
#define NX_CUDA_KERNELS(X)                                                    \
  X(contract_pack, PACK)                                                    \
  X(contract_rows, ROWS)                                                    \
  X(contract_ks, KS)                                                        \
  /* float32: large, mid, small and batched, 8 rows or fewer */             \
  X(contract_f32_128x128, GEMM, f32, 128, 128, 16, 8, 8, 4, 4, 2, 2, 2, native) \
  X(contract_f32_64x64, GEMM, f32, 64, 64, 16, 4, 4, 4, 4, 2, 3, 3, native) \
  X(contract_f32_32x32, GEMM, f32, 32, 32, 16, 4, 4, 4, 2, 1, 6, 8, native) \
  X(contract_f32_r8, GEMM, f32, 8, 256, 16, 8, 2, 1, 1, 4, 4, 2, native)    \
  /* float64 */                                                             \
  X(contract_f64_64x64, GEMM, f64, 64, 64, 16, 4, 4, 4, 4, 2, 2, 2, native) \
  X(contract_f64_32x32, GEMM, f64, 32, 32, 16, 4, 4, 4, 2, 1, 4, 5, native) \
  X(contract_f64_r8, GEMM, f64, 8, 128, 16, 8, 1, 1, 1, 4, 4, 2, native)    \
  /* integers, wrapping */                                                  \
  X(contract_u32_128x128, GEMM, u32, 128, 128, 16, 8, 8, 4, 4, 2, 2, 2, native) \
  X(contract_u64_64x64, GEMM, u64, 64, 64, 16, 4, 4, 4, 4, 2, 2, 2, native) \
  /* every other dtype pair */                                              \
  X(contract_cv_f32_128x128, GEMM, f32, 128, 128, 16, 8, 8, 4, 4, 2, 1, 1, convert) \
  X(contract_cv_f64_64x64, GEMM, f64, 64, 64, 16, 4, 4, 4, 4, 2, 1, 1, convert) \
  X(contract_cv_u32_64x64, GEMM, u32, 64, 64, 16, 4, 4, 4, 4, 2, 2, 2, convert) \
  X(contract_cv_u64_64x64, GEMM, u64, 64, 64, 16, 4, 4, 4, 4, 2, 2, 2, convert) \
  /* few rows by a b of any real dtype */                                   \
  X(contract_gemv_f32, GEMV, f32)                                           \
  X(contract_gemv_f64, GEMV, f64)                                           \
  X(contract_gemv_u32, GEMV, u32)                                           \
  X(contract_gemv_u64, GEMV, u64)                                           \
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
  /* maps, reductions and scans: every program, dtype and load */        \
  X(loop, LOOP)                                                             \
  /* the factorisations' steps (factor.cu), by format */                   \
  X(la_trsm_f32, LINALG, trsm, trsm, f32)                                  \
  X(la_trsm_f64, LINALG, trsm, trsm, f64)                                  \
  X(la_trsm_c64, LINALG, trsm, trsm, c64)                                  \
  X(la_trsm_c128, LINALG, trsm, trsm, c128)                                \
  X(la_lu_panel_f32, LINALG, lu_panel, panel, f32)                         \
  X(la_lu_panel_f64, LINALG, lu_panel, panel, f64)                         \
  X(la_lu_panel_c64, LINALG, lu_panel, panel, c64)                         \
  X(la_lu_panel_c128, LINALG, lu_panel, panel, c128)                       \
  X(la_gather, LINALG, gather, gather, any)                                \
  X(la_chol_f32, LINALG, chol, chol, f32)                                  \
  X(la_chol_f64, LINALG, chol, chol, f64)                                  \
  X(la_chol_c64, LINALG, chol, chol, c64)                                  \
  X(la_chol_c128, LINALG, chol, chol, c128)                                \
  X(la_chol_finish, LINALG_ANY, chol_finish, chol_finish)                  \
  X(la_diag_zero, LINALG_ANY, diag_zero, diag)                             \
  X(la_nan_where, LINALG_ANY, nan_where, diag)                             \
  X(la_qr_panel_f32, LINALG, qr_panel, qr, f32)                            \
  X(la_qr_panel_f64, LINALG, qr_panel, qr, f64)                            \
  X(la_qr_panel_c64, LINALG, qr_panel, qr, c64)                            \
  X(la_qr_panel_c128, LINALG, qr_panel, qr, c128)                          \
  X(la_qr_out, LINALG_ANY, qr_out, qr_out)                                 \
  X(la_dense_v, LINALG_ANY, dense_v, dense)                                \
  X(la_iota, LINALG, iota, iota, any)                                      \
  X(la_fill, LINALG, fill, fill, any)                                      \
  /* the spectral routines' steps (spectral.cu), by format */              \
  X(sp_eigh_jacobi_f32, SPECTRAL, eigh_jacobi, f32)                        \
  X(sp_eigh_jacobi_f64, SPECTRAL, eigh_jacobi, f64)                        \
  X(sp_eigh_jacobi_c64, SPECTRAL, eigh_jacobi, c64)                        \
  X(sp_eigh_jacobi_c128, SPECTRAL, eigh_jacobi, c128)                      \
  X(sp_svd_jacobi_f32, SPECTRAL, svd_jacobi, f32)                          \
  X(sp_svd_jacobi_f64, SPECTRAL, svd_jacobi, f64)                          \
  X(sp_svd_jacobi_c64, SPECTRAL, svd_jacobi, c64)                          \
  X(sp_svd_jacobi_c128, SPECTRAL, svd_jacobi, c128)                        \
  X(sp_td_stage, SPECTRAL_ANY, td_stage)                                   \
  X(sp_td_step_f32, SPECTRAL, td_step, f32)                                \
  X(sp_td_step_f64, SPECTRAL, td_step, f64)                                \
  X(sp_td_step_c64, SPECTRAL, td_step, c64)                                \
  X(sp_td_step_c128, SPECTRAL, td_step, c128)                              \
  X(sp_td_hemv, SPECTRAL_ANY, td_hemv)                                     \
  X(sp_td_diag, SPECTRAL_ANY, td_diag)                                     \
  X(sp_td_vcopy, SPECTRAL_ANY, td_vcopy)                                   \
  X(sp_td_larft, SPECTRAL_ANY, td_larft)                                   \
  X(sp_td_zinit, SPECTRAL_ANY, td_zinit)                                   \
  X(sp_td_out, SPECTRAL_ANY, td_out)                                       \
  X(sp_dc_leaves, SPECTRAL_REAL, dc_leaves)                                \
  X(sp_dc_scatter, SPECTRAL_REAL, dc_scatter)                              \
  X(sp_dc_deflate_f32, SPECTRAL, dc_deflate, f32)                          \
  X(sp_dc_deflate_f64, SPECTRAL, dc_deflate, f64)                          \
  X(sp_dc_secular_f32, SPECTRAL, dc_secular, f32)                          \
  X(sp_dc_secular_f64, SPECTRAL, dc_secular, f64)                          \
  X(sp_dc_product, SPECTRAL_REAL, dc_product)                              \
  X(sp_dc_sort, SPECTRAL_REAL, dc_sort)                                    \
  X(sp_tri_bisect, SPECTRAL_REAL, tri_bisect)                              \
  X(sp_bd_stage, SPECTRAL_ANY, bd_stage)                                   \
  X(sp_bd_left_f32, SPECTRAL, bd_left, f32)                                \
  X(sp_bd_left_f64, SPECTRAL, bd_left, f64)                                \
  X(sp_bd_left_c64, SPECTRAL, bd_left, c64)                                \
  X(sp_bd_left_c128, SPECTRAL, bd_left, c128)                              \
  X(sp_bd_wh, SPECTRAL_ANY, bd_wh)                                         \
  X(sp_bd_updl, SPECTRAL_ANY, bd_updl)                                     \
  X(sp_bd_right_f32, SPECTRAL, bd_right, f32)                              \
  X(sp_bd_right_f64, SPECTRAL, bd_right, f64)                              \
  X(sp_bd_right_c64, SPECTRAL, bd_right, c64)                              \
  X(sp_bd_right_c128, SPECTRAL, bd_right, c128)                            \
  X(sp_bd_wn, SPECTRAL_ANY, bd_wn)                                         \
  X(sp_bd_updr, SPECTRAL_ANY, bd_updr)                                     \
  X(sp_bd_tgk, SPECTRAL_ANY, bd_tgk)                                       \
  X(sp_bd_values, SPECTRAL_ANY, bd_values)                                 \
  X(sp_bd_ginit, SPECTRAL_ANY, bd_ginit)                                   \
  X(sp_bd_gen_f32, SPECTRAL, bd_gen, f32)                                  \
  X(sp_bd_gen_f64, SPECTRAL, bd_gen, f64)                                  \
  X(sp_bd_gen_c64, SPECTRAL, bd_gen, c64)                                  \
  X(sp_bd_gen_c128, SPECTRAL, bd_gen, c128)                                \
  X(sp_bd_apply, SPECTRAL_ANY, bd_apply)                                   \
  X(sp_bd_xinit, SPECTRAL_ANY, bd_xinit)                                   \
  X(sp_bd_vcopy, SPECTRAL_ANY, bd_vcopy)                                   \
  X(sp_bd_larft, SPECTRAL_ANY, bd_larft)                                   \
  X(sp_bd_rank, SPECTRAL_ANY, bd_rank)                                     \
  X(sp_bd_out, SPECTRAL_ANY, bd_out)                                       \
  X(sp_eig_schur_f32, SPECTRAL, eig_schur, f32)                            \
  X(sp_eig_schur_f64, SPECTRAL, eig_schur, f64)                            \
  X(sp_eig_schur_c64, SPECTRAL, eig_schur, c64)                            \
  X(sp_eig_schur_c128, SPECTRAL, eig_schur, c128)                          \
  X(sp_eig_norms, SPECTRAL_ANY, eig_norms)                                 \
  X(sp_eig_vectors_f32, SPECTRAL, eig_vectors, f32)                        \
  X(sp_eig_vectors_f64, SPECTRAL, eig_vectors, f64)                        \
  X(sp_eig_vectors_c64, SPECTRAL, eig_vectors, c64)                        \
  X(sp_eig_vectors_c128, SPECTRAL, eig_vectors, c128)                      \
  X(sp_eig_finish, SPECTRAL_ANY, eig_finish)                               \
  X(sp_eye, SPECTRAL_ANY, eye)                                             \
  /* gathers, range gathers, copies and fills of every dtype */             \
  X(move, MOVE)                                                             \
  /* a range gather's runs: their ends' running maximum */                  \
  X(runs_chunk, RUNS, chunk)                                                \
  X(runs_carry, RUNS, carry)                                                \
  X(runs_merge, RUNS, merge)                                                \
  /* assemblies, their fill and their pieces */                             \
  X(assemble, ASSEMBLE)                                                     \
  /* folds of every dtype Prog.Add takes */                                 \
  X(fold, FOLD)                                                             \
  /* scatters of one pass: stores, owners, atomics, extremes */             \
  X(scatter, SCATTER)                                                       \
  /* a float Add's position tree, in float32 or float64 */                  \
  X(scatter_tile_f32, SCATTER_ADD, tile, f32)                               \
  X(scatter_tile_f64, SCATTER_ADD, tile, f64)                               \
  X(scatter_rows_f32, SCATTER_ADD, rows, f32)                               \
  X(scatter_rows_f64, SCATTER_ADD, rows, f64)                               \
  X(scatter_spill_f32, SCATTER_ADD, spill, f32)                             \
  X(scatter_spill_f64, SCATTER_ADD, spill, f64)                             \
  X(scatter_tree1_f32, SCATTER_ADD, tree1, f32)                             \
  X(scatter_tree1_f64, SCATTER_ADD, tree1, f64)                             \
  X(scatter_tree2_f32, SCATTER_ADD, tree2, f32)                             \
  X(scatter_tree2_f64, SCATTER_ADD, tree2, f64)                             \
  /* the spilled entries' radix sort, a pass in three launches */           \
  X(scatter_hist, SCATTER_SORT, hist)                                       \
  X(scatter_scan, SCATTER_SORT, scan)                                       \
  X(scatter_place, SCATTER_SORT, place)                                     \
  /* sorts and numberings: the workspace's zero words, a block's slices,   \
     one-sweep passes of 8-, 12- and 24-byte items, top-k by select */     \
  X(sort_clear, CLEAR)                                                      \
  X(sort_block, SORT)                                                       \
  X(sort_hist, SORT)                                                        \
  X(sort_pass8, SORT)                                                       \
  X(sort_pass12, SORT)                                                      \
  X(sort_pass24, SORT)                                                      \
  X(select_round, SORT)                                                     \
  X(select_gather, SORT)                                                    \
  X(unique_heads, UNIQUE)                                                   \
  X(unique_rank, UNIQUE)                                                    \
  X(unique_ids, UNIQUE)                                                     \
  /* Fourier transforms, by precision */                                    \
  X(fft_f32, FFT, f32)                                                      \
  X(fft_f64, FFT, f64)                                                      \
  X(fft_real_f32, FFT_REAL, f32)                                            \
  X(fft_real_f64, FFT_REAL, f64)                                            \
  /* attention, one entry for every path, which share their code for        \
     units, visibility and tapes; and the merge of split keys' parts */     \
  X(attention, ATTENTION)                                                   \
  X(attention_merge, ATTENTION_MERGE)

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

/* An operand a or b as a contraction kernel reads it: its first element
   [p]; its strides along batch, its rows (a's) or columns (b's), and k, in
   elements; its rows, past which loads read the last; its dtype's code,
   a sub-byte dtype's elements one a byte; and NX_OP_ bits. */
typedef struct {
  const void *p;
  int64_t s[3];
  int32_t rows, dtype, flags, unused;
} contract_operand;

/* contract_operand's bits of [flags]. */
enum {
  NX_OP_FREE = 1 /* its free axis is contiguous, not its k */
};

/* y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k] in the
   accumulator [acc_dtype], rounded once to y's dtype, for z < batch,
   i < m, j < n, over real dtypes: a complex call is the real contraction
   of its parts, which the plan lays out, n = 2 n' columns and k = 2 k'
   terms. [init] is NULL for none; init and y have one element per
   output, strides [si] and [sy] along batch, rows and columns.

   The FMA kernels (GEMM, GEMV) share k out as contract.cu states: tiles
   [0, main) of the call's [tiles_m] x [tiles_n] tiles a batch element sum
   in [s1] ranges of k, the others in [s2], each range a block; the split
   tiles' ranges' sums go to [partials] and [tickets] holds a zero word a
   tile. The mma kernels cut k into [s1] ranges, blocks (tile, range,
   batch element). [partials] holds a tile's sums for each block, in
   block order, where a call splits or a float range passes
   NX_CONTRACT_BLOCK terms. NX_CONTRACT_ bits in [flags].

   Rows segments: a's rows grouped by their ids (ROWS); [desc] a row tile
   each, four words (batch element, group, first, count), [ntiles] of
   them; the tile's rows are perm[first ..] and multiply b's group's
   slice, [sbg] elements apart in b; group [groups] is the rows in no
   group, which have no products. Contracting segments (KS): output slice
   e of batch element z sums the k kperm[koff[z * (kgroups + 1) + e] ..]
   before the next group's, in increasing k; the slices lie [syg] (y) and
   [sig] (init) elements apart; the call's batch, for its tiles, is
   kgroups * batch, slice e of z at e * batch + z; with
   NX_CONTRACT_K_PAIRS, a complex call's, its terms 2q and 2q + 1 those of
   the index kperm takes as q. koff's words are int64, kperm's int64 with
   NX_CONTRACT_K_WIDE and int32 otherwise. Each NULL for a call without
   them. */
typedef struct {
  contract_operand a, b;
  const void *init;
  void *y;
  int64_t si[3], sy[3];
  int32_t init_dtype, y_dtype, acc_dtype, flags;
  int32_t batch, m, n, kgroups;
  int64_t k;
  int32_t tiles_m, tiles_n, main, s1;
  int32_t s2, groups;
  void *partials;
  uint32_t *tickets;
  const int32_t *perm, *desc, *ntiles;
  const void *kperm;
  const int64_t *koff;
  int64_t sbg, syg, sig;
} contract_params;

/* contract_params' bits of [flags]. */
enum {
  /* y's outputs store whole vectors: y of the accumulator's own dtype (of
     float32, bfloat16 or float16 from an mma kernel's float32), its
     columns contiguous, its rows on 16-byte boundaries */
  NX_CONTRACT_Y_VECTORS = 1,
  NX_CONTRACT_K_PAIRS = 2,
  /* An unsegmented FMA call's first range of k sums from init, the others
     from +0, each range one chain, in no blocks, and init is not added
     after: with one range, each output is init then its products in
     increasing k, one fused multiply-add each */
  NX_CONTRACT_INIT_FIRST = 4,
  /* An unsegmented FMA call computes no tile whose outputs (i, j) all have
     i < j (LOWER) or all i > j (UPPER), j a complex column on a complex
     accumulator; the others' outputs are all stored */
  NX_CONTRACT_LOWER = 8,
  NX_CONTRACT_UPPER = 16,
  /* A Contracting call's kperm holds int64 words: its contracted extent
     passes 2^31 - 1 */
  NX_CONTRACT_K_WIDE = 32
};

/* The terms a range of an FMA kernel's split sum takes a whole number of,
   and an mma kernel's. */
#define NX_CONTRACT_RUN 32
#define NX_MMA_RUN 128

/* The most terms a float range's running sum takes from one chain: a
   longer range sums in blocks of this many from its start, each from +0,
   added in order into a running sum in the block's slot of [partials]
   (combine.cuh's fold). */
#define NX_CONTRACT_BLOCK 256

/* gemv's rows at most, its threads, and the k of a's rows a block holds
   at once. Its dynamic shared memory: a's rows, m x NX_GEMV_STRETCH
   elements of the compute type, rounded up to 16 bytes, then
   NX_GEMV_STAGE(t) bytes a warp for b's values in a compute type of t
   bytes: a row of 16 values and 16 bytes a lane. */
#define NX_GEMV_ROWS 8
#define NX_GEMV_THREADS 256
#define NX_GEMV_STRETCH 256
#define NX_GEMV_STAGE(t) (32 * (16 * (t) + 16))

/* pack: [numel] elements in C order of the [rank] extents [n], element i
   of the walk at [src_first] plus i's indices times [ss] of [src], of
   [src_dtype], to [dst_first] plus its indices times [ds] of [dst], of
   [dst_dtype], positions in elements. A sub-byte element lies at bit
   position times its bits of its buffer, or one a byte (NX_PACK_SPREAD_).
   A raw pack copies an element's bits; NX_PACK_VALUE stores its value as a cast into [acc_dtype] holds it,
   times its scale where [scale] is not NULL, rounded once into [acc_dtype]:
   [scale_dtype]'s element at [scale_first] plus the indices divided by
   [ratio] times [cs], or with NX_PACK_EXPONENT 2^(e - 127) of the code e,
   NaN of 255. A complex value into a complex dtype stores both parts;
   with NX_PACK_BLOCK into a real one, the 2 x 2 block [[re, -im],
   [im, re]] from its position, the second row [block] elements on. */
#define NX_PACK_AXES 32
typedef struct {
  const void *src;
  void *dst;
  const void *scale;
  int64_t src_first, dst_first, scale_first, numel;
  int32_t rank, src_dtype, dst_dtype, scale_dtype;
  int32_t mode, acc_dtype;
  int64_t n[NX_PACK_AXES], ss[NX_PACK_AXES], ds[NX_PACK_AXES];
  int64_t cs[NX_PACK_AXES];
  int32_t ratio[NX_PACK_AXES];
  int64_t block;
} pack_params;

/* pack_params' bits of [mode]: NX_PACK_VALUE, else a raw copy; an
   Exponent scale; a sub-byte source or destination one element a byte, as
   in the workspace; NX_PACK_CAST, a sum of the workspace's dtype (the
   accumulator's, or its compute type's) into [dst_dtype] as a cast from
   [acc_dtype] writes it; with NX_PACK_BLOCK, NX_PACK_CONJ blocks the
   value's conjugate and NX_PACK_FLIP negates the block's second row, the
   one a's imaginary parts multiply, so that the sums take a's conjugate;
   NX_PACK_NEG, a value negated, which is exact. */
enum {
  NX_PACK_VALUE = 1,
  NX_PACK_EXPONENT = 2,
  NX_PACK_SPREAD_SRC = 4,
  NX_PACK_SPREAD_DST = 8,
  NX_PACK_CAST = 16,
  NX_PACK_BLOCK = 32,
  NX_PACK_CONJ = 64,
  NX_PACK_FLIP = 128,
  NX_PACK_NEG = 256
};

/* ROWS and KS: the ids of a segmented call, element (z, x) at [ids] plus
   z * sz + x * sx, for z < batch and x < extent, index q (a row from
   [first], or a contracted index) at x = q / inner % extent, of [count]
   indices a batch element, into [groups] groups; each a few launches, by
   [phase] (contract.cu). ROWS fills [counts] (batch * (groups + 1) + 1
   words, twice), [cursor] (batch * (groups + 1)), [perm] (batch * count),
   [desc] (row tiles of [bm] rows) and [ntiles], int32 words; KS fills
   [counts] as contract_params' koff, [perm] as its kperm, int64 words
   where [wide] and int32 ones otherwise, [cursor] (batch * groups) and
   [desc] (batch * bm * groups, [bm] parts of a batch element's k), int64
   words but for [perm]. */
typedef struct {
  const int64_t *ids;
  void *counts, *cursor, *perm, *desc, *ntiles;
  int64_t sz, sx, inner, count, extent, first;
  int32_t batch, groups, bm, phase, wide, unused;
} segment_params;

/* Factorisations (factor.cu)

   Every field is 8 bytes. A matrix's element (i, j) of batch element z
   lies at its pointer plus z, i and j times its strides, in elements of
   its format; a routine's own matrices are C-contiguous. */

/* x <- op(l)^-1 x for the lower triangular n x n view op(l), n at most
   NX_LA_LEAF (half at c128), and the n x r view x of each of [matrices],
   a thread a column of x: op(l) conjugated under [conj], its diagonal
   taken as ones under [unit]. [src] (or NULL) is b, read in place of x
   with x's strides; with [whole], a zero on the diagonal makes the
   column NaN. */
typedef struct {
  const void *l;
  void *x;
  const void *src;
  int64_t sl, lrs, lcs, sx, xrs, xcs;
  int64_t n, r, matrices, conj, unit, whole;
} la_trsm_params;

/* LU of the panel of columns [c0, c0 + w) of rows [c0, m) of each matrix
   of [a] (rows lda apart, matrices sbat apart), in place, pivots at
   [piv] (spiv a matrix). G threads a matrix (16, or the block) and P
   blocks a matrix, block q its rows [q per, (q + 1) per); with P > 1, one
   launch of min(m - c0, w) + 1 a step, [k] the launch's, the blocks
   passing candidates through [slots] (2 P of slot_bytes a matrix).
   [src_of] (m - c0 a matrix) receives each position's row for gather;
   with [src], the operand read in place of a, [perm] (sperm a matrix)
   receives it instead. The tile is in shared memory where [in_smem], else
   at [spill], spill_bytes a block. */
typedef struct {
  void *a;
  int64_t *piv, *src_of;
  const void *src;
  int64_t *perm;
  void *spill;
  void *slots;
  int64_t sbat, lda, spiv, sperm, m, c0, w, matrices, per;
  int64_t P, G, ld, in_smem, spill_bytes, slot_bytes, k;
} la_panel_params;

/* A panel's row interchanges on each matrix's other columns and on perm
   (sperm a matrix). */
typedef struct {
  void *a;
  const int64_t *piv, *src_of;
  int64_t *perm;
  int64_t sbat, lda, n, spiv, sperm, m, c0, w, steps, matrices, fmt;
} la_gather_params;

/* Cholesky of the diagonal block of order nb at (k, k) of each matrix, in
   place, element (i, j) at i lda + j cs, G threads a matrix, [fail] a
   word a matrix; with [src], the whole matrix read from it and stored
   whole, Lower or [upper]. */
typedef struct {
  void *a;
  int32_t *fail;
  const void *src;
  int64_t sbat, lda, k, nb, matrices, G, ld, upper, cs;
} la_chol_params;

/* Each n x n matrix: NaN where [fail], else its other triangle +0, under
   [upper] the conjugate transpose of its lower one. With [tr] the factor
   lies transposed, above the diagonal: L or, under [upper], Lᴴ, from
   it. */
typedef struct {
  void *a;
  const int32_t *fail;
  int64_t n, matrices, upper, fmt, tr;
} la_chol_finish_params;

/* A triangular solve's diagonal, [sd] apart: [flag] a word a matrix,
   set where a diagonal element of a is zero; then x NaN where it is
   set. */
typedef struct {
  const void *a;
  void *x;
  int32_t *flag;
  int64_t sa, sd, n, sx, xn, matrices, fmt;
} la_diag_params;

/* Householder QR of the panel of columns [c0, c0 + w) of rows [c0, m) of
   each matrix of [a], in place (factor.cu states its sums' order), tau
   at [tau] (stau a matrix); dense V at [v] (sv a matrix) and T at [t] (st
   a matrix, rows ldto apart) where not NULL. Where the panel is the whole
   matrix, [q] (m x qc) and [r] (qc x n) receive Q's first qc columns and
   R. With P > 1, launches of [stage] (factor.cu's stages) and [k];
   [parts] holds a matrix's partials (parts_stride elements) where not in
   shared memory. */
typedef struct {
  void *a, *tau, *v, *t, *q, *r, *spill;
  void *parts;
  int64_t sbat, lda, stau, sv, st, ldto, qc, n, m, c0, w, matrices, per;
  int64_t P, G, ld, in_smem, parts_in_smem, most, spill_bytes, parts_stride;
  int64_t stage, k;
} la_qr_params;

/* q, the first qc columns of the m x m identity; r, the upper triangle of
   f's first rows, rows past m zero. */
typedef struct {
  const void *f;
  void *q, *r;
  int64_t m, n, qc, matrices, fmt;
} la_qr_out_params;

/* The dense V of f's columns [c0, c0 + w), rows [c0, m). */
typedef struct {
  const void *f;
  void *v;
  int64_t m, n, c0, w, matrices, fmt;
} la_dense_params;

/* perm[g] = g mod m for g < total; x's [count] words of [value]. */
typedef struct {
  int64_t *perm;
  int64_t m, total;
} la_iota_params;

typedef struct {
  uint64_t *x;
  int64_t count;
  uint64_t value;
} la_fill_params;

/* The most rows of a triangular solve's leaf at 4 and 8 bytes an element
   (half at 16), its threads, and a meeting slot's bytes before its
   row. */
#define NX_LA_LEAF 64
#define NX_LA_TRSM_THREADS 128
#define NX_LA_SLOT_HEAD 32

/* QR's sums over rows: blocks of NX_LA_QBLK rows from row 0. */
#define NX_LA_QBLK 256

/* la_qr_params' stages with P > 1: a step's first and second launch, and
   the Gram pairs' launches. */
#define NX_LA_QR_STEP_A 1
#define NX_LA_QR_STEP_B 2
#define NX_LA_QR_STEP_G 3

/* Spectral routines (spectral.cu)

   An entry reads its step's parameters from [w] in order, each an
   address, an integer or a size, as spectral.cu's sp_ functions call
   them. Its matrices' workspace is laid out by the functions below, which
   the device code and spectral.ml both follow, [t] and [r] the bytes of an
   element and of its real part. */
#define NX_SP_ARGS 13
#define NX_SP_FMT 12
typedef struct {
  uint64_t w[NX_SP_ARGS];
} sp_args;

/* A group's threads; the tridiagonal reduction's panel; divide and
   conquer's leaves; the bidiagonal reflectors' blocks. */
#define NX_SP_THREADS 256
#define NX_SP_TNB 32
#define NX_SP_LEAF 32
#define NX_SP_BD_NB 32

#ifdef __CUDACC__
#define NX_SP_FN __host__ __device__ __forceinline__
#else
#define NX_SP_FN static inline
#endif

/* Jacobi's matrices a group and their shared bytes: eigh's of order n,
   svd's of m x n, g its M x K orientation. A group's dynamic shared
   memory is at most NX_SP_JACOBI_SMEM: sm_89's 99 KiB less the 6 KiB of
   its static slots. */
#define NX_SP_JACOBI_SMEM (93 * 1024)
NX_SP_FN int nx_sp_eigh_per(int n) {
  return n * n >= NX_SP_THREADS ? 1 : NX_SP_THREADS / (n * n);
}
NX_SP_FN size_t nx_sp_eigh_smem(int n, int t, int r) {
  size_t per = (size_t)nx_sp_eigh_per(n);
  return per * (2 * (size_t)n * n + 2 * (size_t)n) * t + (per * n + 1) * 4 +
         per * n * r;
}
NX_SP_FN size_t nx_sp_svd_span(int M, int K) {
  return (size_t)M * K + (size_t)K * K + (size_t)M * M + 2 * (size_t)K + M;
}
NX_SP_FN int nx_sp_svd_per(int m, int n, int t, int r) {
  int M = m > n ? m : n, K = m > n ? n : m;
  int per = m * n >= NX_SP_THREADS ? 1 : NX_SP_THREADS / (m * n);
  size_t slot = nx_sp_svd_span(M, K) * t + 2 * (size_t)K * 4 + (size_t)(K + M) * r;
  while (per > 1 && per * slot > NX_SP_JACOBI_SMEM) per--;
  return per;
}
NX_SP_FN size_t nx_sp_svd_smem(int m, int n, int t, int r) {
  int M = m > n ? m : n, K = m > n ? n : m;
  size_t per = (size_t)nx_sp_svd_per(m, n, t, r);
  return per * nx_sp_svd_span(M, K) * t + per * 2 * K * 4 + per * (K + M) * r;
}

/* A matrix's workspace, on 256 bytes: the tridiagonal reduction's (A, W,
   Vc, T, G, tau; d, e; two words), divide and conquer's (Q, Qt, S; ten
   vectors; six of ints and four words) and eig's (H, Z, X; X's complex
   copy; tau; four vectors of R; a word). */
NX_SP_FN size_t nx_sp_round(size_t x) { return (x + 255) & ~(size_t)255; }
NX_SP_FN size_t nx_sp_td_per(int n, int t, int r) {
  size_t nn = (size_t)n * n;
  return nx_sp_round((nn + 2 * (size_t)n * NX_SP_TNB + 2 * NX_SP_TNB * NX_SP_TNB + n) * t +
                     2 * (size_t)n * r + 16);
}
NX_SP_FN size_t nx_sp_dc_per(int n, int r) {
  size_t nn = (size_t)n * n;
  return nx_sp_round((3 * nn + 10 * (size_t)n) * r + (6 * (size_t)n + 4) * 4);
}
NX_SP_FN size_t nx_sp_eig_per(int n, int t, int r) {
  size_t nn = (size_t)n * n;
  return nx_sp_round(3 * nn * t + nn * 2 * r + (size_t)n * t + 4 * (size_t)n * r + 16);
}

/* A large svd's workspace a matrix, as byte offsets: its arrays of
   elements (g, x, y, vc, tm, gg, w1, w2, wv, tauq, taup), of reals (d, e,
   U_B, V_B, 4 a rotation record, scalars) and of ints (sweeps, state);
   [cap] the records, and the sweeps, a pass holds. g is M x K, M >= K,
   gc the columns of its left factor the call stores. */
typedef struct {
  size_t g, x, y, vc, tm, gg, w1, w2, wv, tauq, taup;
  size_t d, e, ub, vb, rec, sc;
  size_t sw, st;
  size_t per;
  int cap;
} nx_sp_bd;

/* The state words. */
enum {
  NX_SP_ST_M, NX_SP_ST_ITER, NX_SP_ST_SWEEPS, NX_SP_ST_OLDLL, NX_SP_ST_OLDM,
  NX_SP_ST_DONE, NX_SP_ST_FAIL, NX_SP_ST_OK, NX_SP_ST_EX, NX_SP_ST_NSW,
  NX_SP_ST_NREC, NX_SP_ST_WORDS
};

NX_SP_FN nx_sp_bd nx_sp_bd_layout(int64_t M, int K, int gc, int t, int r) {
  nx_sp_bd l;
  size_t o = 0, mk = (size_t)(M > K ? M : K), wc = (size_t)(gc > K ? gc : K);
#define NX_SP_T(f, n) (l.f = o, o += (size_t)(n) * t)
#define NX_SP_R(f, n) (l.f = o, o += (size_t)(n) * r)
  NX_SP_T(g, (size_t)M * K);
  NX_SP_T(x, (size_t)M * gc);
  NX_SP_T(y, (size_t)K * K);
  NX_SP_T(vc, mk * NX_SP_BD_NB);
  NX_SP_T(tm, NX_SP_BD_NB * NX_SP_BD_NB);
  NX_SP_T(gg, NX_SP_BD_NB * NX_SP_BD_NB);
  NX_SP_T(w1, NX_SP_BD_NB * wc);
  NX_SP_T(w2, NX_SP_BD_NB * wc);
  NX_SP_T(wv, mk);
  NX_SP_T(tauq, K);
  NX_SP_T(taup, K);
  l.cap = 64 * K + 64;
  NX_SP_R(d, K);
  NX_SP_R(e, K);
  NX_SP_R(ub, (size_t)K * K);
  NX_SP_R(vb, (size_t)K * K);
  NX_SP_R(rec, 4 * (size_t)l.cap);
  NX_SP_R(sc, 4);
#undef NX_SP_T
#undef NX_SP_R
  l.sw = o;
  o += 3 * (size_t)l.cap * 4;
  l.st = o;
  o += NX_SP_ST_WORDS * 4;
  l.per = nx_sp_round(o);
  return l;
}

/* Loops */

/* A loop (loop.cu, whose header states how it runs) reads one parameter
   struct of NX_LOOP_WORDS 64-bit words: a header of NX_LOOP_HEADER words,
   the address of each buffer of the submit from word NX_LOOP_HEADER, then
   the loop's extents, its operands' descriptors and its tape, which the
   header points at by word. A group has NX_LOOP_THREADS threads, each of
   [v] elements of a tile (NX_LOOP_MODE's bits 8 to 15), at most
   NX_LOOP_MOST. A group copies the parameters it uses, up to the tape's
   end, into the shared memory a launch asks for, from its byte
   NX_LOOP_COPY, after the planes, or reads them where it is -1; its own
   holds a flag, 16 bytes at most.
   A reduction keeps
   nx.cpu's lane order: blocks of NX_LOOP_BLOCK terms, NX_LOOP_LANES lanes
   a block; a scan its chunks of NX_LOOP_CHUNK terms. A state in the
   workspace takes NX_LOOP_STATE 32-bit words. */
#define NX_LOOP_WORDS 512
#define NX_LOOP_THREADS 256
#define NX_LOOP_MOST 8
#define NX_LOOP_BLOCK 1024
#define NX_LOOP_LANES 16
#define NX_LOOP_CHUNK 4096
#define NX_LOOP_STATE 8

/* The header's words. */
enum {
  NX_LOOP_MODE,      /* NX_LOOP_MAP to NX_LOOP_NETWORK, then v << 8, then a
                        reduction's threads a block << 16 */
  NX_LOOP_RANK,      /* the loop's axes */
  NX_LOOP_SHAPE,     /* the word of its first extent */
  NX_LOOP_NUMEL,     /* its indices */
  NX_LOOP_TAPE,      /* the word of the tape's first instruction */
  NX_LOOP_PROLOGUE,  /* the words of the tape run once a group: constants */
  NX_LOOP_BODY,      /* the words run on each tile, after the prologue */
  NX_LOOP_EPILOGUE,  /* the words run after a scan's results, after the body */
  NX_LOOP_LOADS,     /* the word of the loads' table: each load's word */
  NX_LOOP_DSTS,      /* the word of the destinations' table */
  NX_LOOP_TERM,      /* the plane of a reduction's or a scan's terms */
  NX_LOOP_STATE_AT,  /* the first of a reduction's NX_LOOP_STATE planes */
  NX_LOOP_RESULT,    /* the first of a scan's result planes */
  NX_LOOP_CENTER,    /* the plane of each term's center */
  NX_LOOP_REDUCTION, /* reduction, then compute type << 8, carrier << 16,
                        finish << 24 */
  NX_LOOP_TERMS,     /* a row's terms */
  NX_LOOP_SEGMENT,   /* an output's terms at most: a segment of a row */
  NX_LOOP_SEGMENTS,  /* a row's segments */
  NX_LOOP_OUTPUTS,   /* the outputs: rows times segments */
  NX_LOOP_GROUP,     /* the blocks of a group, a power of two */
  NX_LOOP_GROUPS,    /* an output's groups */
  NX_LOOP_RUNS,      /* a block's runs, of its threads times v terms; a
                        scan's of v */
  NX_LOOP_WORK,      /* the workspace's slot */
  NX_LOOP_TICKETS,   /* the tickets' slot, a zero word an output */
  NX_LOOP_PARTIALS,  /* the byte of the groups' states, or a network's levels */
  NX_LOOP_TOTALS,    /* the byte of a scan's chunk totals */
  NX_LOOP_CENTERS,   /* the byte of the outputs' centers */
  NX_LOOP_CARRY,     /* a scan's carry: its load, or -1 */
  NX_LOOP_COPY,      /* the shared byte of the parameters' copy, or -1 */
  NX_LOOP_STAGE,     /* the shared byte of a staged scan's chunks */
  NX_LOOP_HEADER
};

/* Modes: a map stores the body's values; a reduction folds each output's
   terms; a scan folds each chunk's in order; a network scans each slice by
   nx_kernel.mli's odd/even network; carries turn each slice's chunk
   totals into the states its chunks start from; a plain reduction folds a
   Sum, Max or Min of one plain load at its carrier, of a loop of one
   output axis and one term axis at most, without a tape, into its
   destination or, its outputs the chunks of a scan's slices, into the
   totals; a plain scan and its carries scan so too; a staged scan is a
   Sum, Max or Min scan into a destination at its carrier whose results
   step by one, in one pass, NX_LOOP_GROUP chunks a group at a time
   through the shared memory the launch asks for from its byte
   NX_LOOP_STAGE, its terms one plain load that steps by one, or, where
   the tape has a body, the body's term plane, its chunks' totals from
   byte NX_LOOP_TOTALS of the workspace, their inclusive carries from byte
   NX_LOOP_PARTIALS, a flag a chunk, then a ticket and a count, in the
   tickets; a look-back folds NX_LOOP_LOOK totals at a time through the
   shared memory after the chunks. A plain
   reduction's group merges an output's groups NX_LOOP_PLAIN_CHUNK at a
   time; its columns take NX_LOOP_PLAIN_TALL blocks at most. */
enum {
  NX_LOOP_MAP, NX_LOOP_REDUCE, NX_LOOP_SCAN, NX_LOOP_NETWORK, NX_LOOP_CARRIES,
  NX_LOOP_PLAIN, NX_LOOP_PLAIN_SCAN, NX_LOOP_PLAIN_CARRIES, NX_LOOP_STAGED_SCAN
};
#define NX_LOOP_PLAIN_CHUNK 2048
#define NX_LOOP_PLAIN_TALL 16
#define NX_LOOP_LOOK 512

/* A descriptor's words: its dtype, then whether it is padded << 8 and its
   slot << 16; its first element's position; then a plain operand's
   stride along each axis of the loop, or a padded one's fill (two words),
   its rank r and, per axis of the operand, NX_LOOP_PAD words: its lo, its
   interior, its extent, its stride, the loop axes it reads (the second <<
   16, 0xffff for none), and their coefficients. */
#define NX_LOOP_PAD 7

/* An instruction: its op, then its planes d, a, b and c of 12 bits from
   bits 16, 28, 40 and 52. A kind's op is its compute type << 7 and its
   code: an op1's, NX_LOOP_BINARY plus an op2's, NX_LOOP_TERNARY plus an
   op3's (nx_kernel.h). A constant's two words follow it. */
#define NX_LOOP_BINARY 32
#define NX_LOOP_TERNARY 64

/* Fma at a narrow float, a kind at NX_LOOP_F32 over its float32 carriers
   past the op3s' codes: nx_kinds.h's nx_fma_narrow, or nx_fma_bf16 for
   bfloat16, the exact value rounded to odd, so that the encode after it
   rounds once. */
#define NX_LOOP_FMA_NARROW (NX_LOOP_TERNARY + 2)
#define NX_LOOP_FMA_BF16 (NX_LOOP_TERNARY + 3)
enum {
  NX_LOOP_F32, NX_LOOP_F64, NX_LOOP_C64, NX_LOOP_C128,
  NX_LOOP_I32, NX_LOOP_U32, NX_LOOP_I64, NX_LOOP_U64
};
enum {
  NX_LOOP_LOAD = 1024, /* d: load a's element */
  NX_LOOP_COORD,       /* d: the index along loop axis a */
  NX_LOOP_CONST,       /* d: the two words after */
  NX_LOOP_CAST,        /* d: a of dtype b cast to dtype c */
  NX_LOOP_BITCAST,     /* d: a of dtype b read in dtype c */
  NX_LOOP_WHERE1,      /* d: b where a, else c: of one word */
  NX_LOOP_WHERE2,      /* of two */
  NX_LOOP_WHERE4,      /* of four */
  NX_LOOP_DECODE,      /* d: the float32 of a, a code of dtype b */
  NX_LOOP_ENCODE,      /* d: the code of dtype b of a, a float32 */
  NX_LOOP_WRAP,        /* d: a wrapped to the integer dtype b */
  NX_LOOP_TRUTH,       /* d: whether a, a boolean's byte, is not 0 */
  NX_LOOP_STORE,       /* destination b: a */
  NX_LOOP_WAIT         /* the loads issued before have landed */
};

/* Reductions: a monoid's fold, Arg's, Logsumexp's second pass, a sum of
   exp (x - c), and Moments', sums of (x - c)^2 and x - c. A network
   scans by Logsumexp's or Moments' states. */
enum {
  NX_LOOP_SUM, NX_LOOP_PROD, NX_LOOP_MAX, NX_LOOP_MIN, NX_LOOP_ARG_MAX,
  NX_LOOP_ARG_MIN, NX_LOOP_EXPS, NX_LOOP_SQUARES, NX_LOOP_LOGSUMEXP,
  NX_LOOP_MOMENTS
};

/* What a reduction does with an output's state: stores it, as a total of
   a scan's chunk, as Logsumexp's center, as Moments' mean, as Logsumexp,
   as Moments. */
enum {
  NX_LOOP_PUT, NX_LOOP_PUT_TOTAL, NX_LOOP_PUT_CENTER, NX_LOOP_PUT_MEAN,
  NX_LOOP_PUT_LOGSUMEXP, NX_LOOP_PUT_MOMENTS
};

typedef struct {
  uint64_t w[NX_LOOP_WORDS];
} loop_params;

/* Moves (moves.cu, whose header states how they run)

   A move's parameters address each array by its buffer's first byte and
   its first element's position, in elements as a layout states them: a
   sub-byte element at position p holds bits p b to p b + b - 1. An element
   is [count] units of [unit] bits, its dtype's alignment: 1, 4, 8, 16, 32
   or 64, and 2 units for a complex number. Arrays have up to NX_MOVE_RANK
   axes, a launch NX_MOVE_THREADS threads a group. */
#define NX_MOVE_RANK 32
#define NX_MOVE_THREADS 256

/* move_params' modes: dst at each index c of the space gets src at c; src
   at c with axis [axis] replaced by the position idx holds at c, or zero
   bits for one outside [0, d); src at the position a range gather's runs
   give; [fill]. */
enum { NX_MOVE_COPY, NX_MOVE_GATHER, NX_MOVE_RANGES, NX_MOVE_FILL };

/* An axis of a move's space: its extent and the strides of dst (C order),
   src and the positions (idx; for NX_MOVE_RANGES, the rows of the runs). */
typedef struct {
  int64_t n, sd, ss, si;
} move_axis;

/* dst, C-contiguous over the space of [numel] elements, gets each element
   as [mode] says. [sa] is src's stride along [axis], [d] its extent there;
   [k] a range gather's runs a row, of [starts], [ends] and [maxima], their
   ends' running maximum. [small]: [numel] < 2^32. */
typedef struct {
  void *dst;
  const void *src;
  const int64_t *idx, *starts, *ends, *maxima;
  int64_t numel, dst_off, src_off, idx_off;
  int64_t d, sa, k;
  uint64_t fill[2];
  int32_t rank, axis, mode, small, unit, count;
  move_axis ax[NX_MOVE_RANK];
} move_params;

/* A range gather's runs: [rows] lead rows of [k] runs, copied from [in_s]
   and [in_e] (strides [s_k] and [e_k] along the runs, [ax] along the lead
   axes) to [starts] and [ends], and their ends' running maximum to
   [maxima], in chunks of NX_RUNS_CHUNK runs, each chunk's maximum in
   [carry]. */
#define NX_RUNS_CHUNK 4096

typedef struct {
  int64_t n, ss, se;
} runs_axis;

typedef struct {
  const int64_t *in_s, *in_e;
  int64_t *starts, *ends, *maxima, *carry;
  int64_t rows, k, chunks, s_k, e_k, s_off, e_off;
  int32_t lead, small;
  runs_axis ax[NX_MOVE_RANK];
} runs_params;

/* Assemblies: up to NX_ASSEMBLE_PIECES pieces a launch, piece i at [src[i]]
   and its record at tab[i * (NX_ASSEMBLE_HEAD + rank NX_ASSEMBLE_AXIS)]:
   its first element, its first element's place in dst from [dst_off], the
   elements of the launch's pieces before it and the mask of the launch's
   later pieces whose regions meet its own; then per axis its
   extent, its stride, its region's start and step along dst's axis, and
   that step times dst's stride. dst has the extents [dn]. A launch of
   [mode] NX_ASSEMBLE_FILL stores [fill], skipping with [check] what the
   launch's pieces cover, which are then every piece; one of
   NX_ASSEMBLE_COPY copies its pieces. */
#define NX_ASSEMBLE_PIECES 16
#define NX_ASSEMBLE_WORDS 448
#define NX_ASSEMBLE_HEAD 4
#define NX_ASSEMBLE_AXIS 5
enum { NX_ASSEMBLE_FILL, NX_ASSEMBLE_COPY };

typedef struct {
  void *dst;
  const void *src[NX_ASSEMBLE_PIECES];
  int64_t total, dst_off;
  int64_t dn[NX_MOVE_RANK];
  uint64_t fill[2];
  int32_t rank, pieces, small, check, unit, count, mode, unused;
  int64_t tab[NX_ASSEMBLE_WORDS];
} assemble_params;

/* Folds, the adjoint of a Padded load of dst's shape: per axis of dst, its
   extent, the padding before it and between its elements, the padded
   extent and x's stride; a windowed axis ([win] >= 0) its windows' taps,
   step, dilation and count and x's strides along the windows and the
   taps. */
typedef struct {
  int64_t n, lo, inter, pext, sx;
  int64_t taps, step, dil, nwin, sw, st;
  int32_t win, unused;
} fold_axis;

typedef struct {
  void *dst;
  const void *x;
  int64_t numel, dst_off, x_off;
  int32_t rank, dtype, small, unit, count, unused;
  fold_axis ax[NX_MOVE_RANK];
} fold_params;

/* Scatters of one pass (scatter_<width>): stores under ~unique; each
   target's last update, its position plus one by atomic maximum into
   [owner], then the owner's store; an integer Add by atomics; Max and Min;
   a float Add under ~unique. */
enum {
  NX_SCATTER_SET,
  NX_SCATTER_OWN,
  NX_SCATTER_OWNED,
  NX_SCATTER_ADD_INT,
  NX_SCATTER_MAX,
  NX_SCATTER_MIN,
  NX_SCATTER_ADD_UNIQUE
};

/* A float Add's tiles: NX_SCATTER_TILE positions of one slice, or of
   several short ones; a row tile's NX_SCATTER_ROWS positions by up to
   [cols] columns, NX_SCATTER_ROW_BYTES bytes of values; the sort's chunks
   of NX_SCATTER_CHUNK entries. */
#define NX_SCATTER_TILE 1024
#define NX_SCATTER_TILE_BITS 10
#define NX_SCATTER_ROWS 256
#define NX_SCATTER_ROWS_BITS 8
#define NX_SCATTER_ROW_BYTES 32768
#define NX_SCATTER_CHUNK 2048

/* An axis of the updates: its extent and the strides of dst (C order of
   into's shape), idx and the updates. */
typedef struct {
  int64_t n, sd, si, su;
} scatter_axis;

/* A float Add past one tile keeps in the workspace: [tiles] (2 where a
   target's entries lie in several tiles, else 1, a row block a slot in row
   tiles), each tile's [est] entries of a target and a row of [cols] values
   ([ent_g], [ent_v]) and their count ([ent_n]), the spilled entries' keys
   and indices into [ent_v] ([keys], [index], ping-pong), the pieces' sums
   ([piece]), the sort's histograms ([hist]) and [ctl]: the spilled count,
   then each digit's total. [ent_g] holds the pieces' ends once spilled. */
typedef struct {
  void *dst;
  const int64_t *idx;
  const void *upd;
  void *owner;
  uint32_t *locks, *tiles_of;
  uint64_t *ent_g;
  void *ent_v;
  uint32_t *ent_n;
  uint64_t *keys[2], *index[2];
  void *piece;
  uint64_t *hist, *ctl;
  int64_t numel_u, numel_d, dst_off, idx_off, upd_off;
  int64_t d, m, sda;
  int64_t slices, run, group, nb, nsg, tiles, est;
  int64_t last, nc;
  int32_t rank, axis, op, dtype, small, wide, kb_bits, key_bits;
  int32_t shift, lcols, unit, count;
  scatter_axis ax[NX_MOVE_RANK];
} scatter_params;

/* Sorts and numberings (sort.cu, whose header states the paths) */

/* A group's threads; the most kept axes, after the sorted one; the
   elements a sort_hist, select_round or select_gather group reads; the
   longest slice sort_block sorts, and for 16-byte keys; the rows a
   unique_heads group and the words a unique_rank group take. */
#define NX_SORT_THREADS 256
#define NX_SORT_RANK 31
#define NX_SORT_CHUNK 16384
#define NX_SORT_CAP 4096
#define NX_SORT_CAP16 2048
#define NX_UNIQUE_TILE 4096

/* The items a sort_pass thread ranks: 8-, 12- and 24-byte items. */
#define NX_SORT_ITEMS8 16
#define NX_SORT_ITEMS12 8
#define NX_SORT_ITEMS24 5

/* Key formats: how an element's bits become its key (sort.cu, Keys). */
enum {
  NX_KEY_UNSIGNED, NX_KEY_SIGNED, NX_KEY_FLOAT, NX_KEY_BOOL, NX_KEY_COMPLEX
};

/* sort_params' bits of [flags]. */
enum {
  NX_SORT_VALUES = 1,     /* store the values */
  NX_SORT_ORDER = 2,      /* the first pass reads x through [order] */
  NX_SORT_CANDIDATES = 4, /* sort_block sorts each slice's candidates */
  NX_SORT_REUSE = 8       /* buf[0] is the positions' destination */
};

/* clear: [words] zero words from [p]. */
typedef struct {
  uint64_t *p;
  int64_t words;
} clear_params;

/* A slice's select, zero before its first round: the digits of (key,
   position) resolved in [prefix], the kept elements still to find among
   those that match them in [need], the candidates gathered in [cnt], the
   rounds resolved, whether it is done, and the round's counts. */
typedef struct {
  uint64_t need, cnt;
  uint32_t rounds, done;
  uint8_t prefix[32];
  uint64_t hist[256];
} sort_select;

/* A sort of each slice of x along one axis into [values] (where
   NX_SORT_VALUES) and [positions], [keep] of each slice's [n] in order.
   Offsets and strides count elements: x's along the axis [sx], the
   destinations' [sv] and [sp], their first elements [off], and each of
   the [nkept] kept axes as [kept]: its extent, then x's, values' and
   positions' strides. A key is [kbytes] bytes of an element of [ebits]
   bits by the format [kind]: [nan_abs] is a float's greatest magnitude
   not NaN, [kmask] a real element's NaN key, [flip] the bits xored into a
   key by sign, [dmask] the bits xored into every key, all ones
   descending. [cap] and [per_block] are sort_block's longest slice and
   slices a group; [tiles] a pass's tiles a slice; [hblocks] sort_hist's
   and a select's groups a slice; [pbytes] a select's bytes of position;
   [pass] the key byte of a pass or the digit of a select's round. The
   workspace holds the counts [hist], each slice's passes [meta], the
   look-back words [status], the tiles each pass took [counter], each
   slice's [tickets], the selects [sel] and candidates [cand], and the
   passes' items [buf]. */
typedef struct {
  const void *x;
  void *values;
  int64_t *positions;
  const int64_t *order;
  uint64_t *hist;
  uint32_t *meta;
  uint64_t *status[2];
  uint32_t *counter;
  uint32_t *tickets;
  sort_select *sel;
  uint64_t *cand;
  void *buf[2];
  int64_t n, keep, slices;
  int64_t sx, sv, sp;
  int64_t off[3];
  int64_t tiles, hblocks;
  uint64_t nan_abs, kmask, flip[2], dmask[2];
  int32_t nkept, kind, ebits, kbytes;
  int32_t cap, per_block, pbytes, pass;
  int32_t flags, unused;
  int64_t kept[NX_SORT_RANK][4];
} sort_params;

/* unique_params' bits of [flags]. */
enum {
  NX_UNIQUE_ORDER = 1, /* the rows are sorted into [order] */
  NX_UNIQUE_SKEYS = 2  /* their one word sorted into [skeys] */
};

/* A numbering of [n] rows of [w] words, word j of row i at keys[i s0 + j
   s1], after a stable sort of the rows: [ids], [first], [counts] of [m]
   entries, and [count]. The workspace holds each sorted row's key in
   sorted order [gid], each key's first row [gfirst] and first sorted row
   [gstart], a bit at each first row [bits] and the bits set before each
   word [below], the look-back words [status], the tiles each kernel took
   [counter] and the keys counted [k]. */
typedef struct {
  const uint64_t *keys;
  int64_t *ids, *first, *counts, *count;
  const int64_t *order;
  const uint64_t *skeys;
  int64_t *gid, *gfirst, *gstart;
  uint64_t *bits, *below, *status;
  uint32_t *counter;
  int64_t *k;
  int64_t n, w, s0, s1, m;
  int32_t flags, unused;
} unique_params;

/* Fourier transforms

   fft.cu runs a transform of N points as Stockham steps, a launch of fft
   each (fft.ml plans them). A step's virtual line v is batch line
   l = v / m and butterfly b = v mod m, m = N / l: it loads the l points
   b + r m of line l, from [nsrc] on +0, twiddles point r by
   w_N^((b mod ns) r gts) (gts = N / (ns l), none where 0), runs its [ops]
   on them in shared memory, the inner FFT of [np] passes among them, and
   stores output k at (b / ns) ns l + b mod ns + k ns, below [limit]. A
   line that fits a group is one step, m = ns = 1. A group holds [g] lines
   of [tl] threads each, [lp] slots apart, side by side in memory where
   [lf_ld] (loads) and [lf_st] (stores).

   A plan is one buffer of tables, which a launch reaches through [plan],
   each table at a byte offset from its first: the inner FFT's w_l^k [tw]
   (k < l / 2 where [half]), a long line's w_N^m = glo[m mod 2^g_bits]
   ghi[m >> g_bits], a real line's w_2h^k by [rlo], [rhi] and [r_bits],
   Bluestein's chirp and its kernel's transform [bspec], or Rader's
   kernel [bspec] and discrete logarithms [rlog]. fft_real splits a long
   real line's bins from its half-length transform in place, or makes the
   half-length points from the bins. */

#define NX_FFT_RANK 32     /* a side's batch axes: NX_MAX_RANK */
#define NX_FFT_PASSES 16   /* the inner FFT's most passes */
#define NX_FFT_OPS 8       /* a step's most ops */
#define NX_FFT_THREADS 512 /* a group's most threads */

/* A side's lines: line l's first element is at off + Σ idx_d str[d], l in
   C order over [dim], and its point j [ps] further, in elements of the
   side's type, a complex or a real. */
typedef struct {
  void *ptr;
  int64_t off, ps;
  int32_t rank, pad;
  int64_t dim[NX_FFT_RANK], str[NX_FFT_RANK];
} fft_side;

/* A pass of the inner FFT: radix r after ns points, q = l / r
   butterflies, twiddle step ts = l / (ns r), nsm the multiplier that
   divides by ns. */
typedef struct {
  int32_t r, ns, q, ts;
  uint32_t nsm, pad;
} fft_pass;

/* Loads: complex points, real points, real pairs as complex points,
   bins completed to n = nh points by conjugation. Stores: complex points,
   real parts, complex points as real pairs. */
enum { NX_FFT_LD_C, NX_FFT_LD_R, NX_FFT_LD_RP, NX_FFT_LD_HC };
enum { NX_FFT_ST_C, NX_FFT_ST_RE, NX_FFT_ST_RP };

/* Ops in shared memory: the inner FFT; Bluestein's chirp, kernel product
   and last chirp; a real line's split and pre-pass (h = nh); Rader's
   kernel product. Stores' pointwise steps: Bluestein's kernel product or
   last chirp. */
enum {
  NX_FFT_OP_FFT,
  NX_FFT_OP_CHIRP,
  NX_FFT_OP_BMUL,
  NX_FFT_OP_BFIN,
  NX_FFT_OP_SPLIT,
  NX_FFT_OP_HPRE,
  NX_FFT_OP_RMUL
};
enum { NX_FFT_POST_NONE, NX_FFT_POST_BMUL, NX_FFT_POST_BFIN };

/* fft: [nv] virtual lines; [lld] points loaded and [lst] stored a line;
   conj_in and conj_out conjugate loads and stores; chirp_in multiplies
   loads by the chirp; [rader] permutes a prime line's points. */
typedef struct {
  fft_side in, out;
  const void *plan;
  int64_t nv, m, ns, nsrc, limit, gts, nh;
  int64_t tw, glo, ghi, rlo, rhi, chirp, bspec, rlog;
  int32_t l, lld, lst, lp;
  int32_t g, tl, lf_ld, lf_st;
  int32_t ld, st, conj_in, conj_out;
  int32_t chirp_in, post, np, half;
  int32_t nops, gshift, g_bits, r_bits;
  int32_t rader, pad;
  int32_t ops[NX_FFT_OPS];
  fft_pass pass[NX_FFT_PASSES];
} fft_params;

/* fft_real: [nl] lines of [h] complex points, split in place on [out]
   ([split]) or made from [in]'s bins into [out]. */
typedef struct {
  fft_side in, out;
  const void *plan;
  int64_t nl, h, rlo, rhi;
  int32_t r_bits, split;
} fft_real_params;

/* Attention (attention.cu, whose header states how it runs)

   A call reads one struct of NX_ATT_WORDS 64-bit words: a header of
   NX_ATT_HEADER words, the address of each buffer of the submit from word
   NX_ATT_HEADER, then the lead extents, the arrays' descriptors, the pair
   program's tapes (loop.cu's instructions) and the tables of the planes the
   kernel fills for each tape. An array's descriptor is its slot | its
   dtype's code << 16, its first element, then its strides: q's, o's and a
   load's along the lead axes, the heads, the rows and the last axis
   (dimensions, columns or keys); lse's along the lead axes, the heads and
   the rows; dense keys' and values' along the lead axes, the key heads, the
   keys and the dimensions; pools' along the pages, the key heads, a page's
   keys and the dimensions; the table's along the lead axes and its pages;
   the lengths' along the lead axes. A group has NX_ATT_THREADS threads and
   reads keys in blocks of NX_ATT_KEYS; a tape of pairs runs NX_ATT_THREADS
   times NX_ATT_V elements, at most NX_ATT_MOST_V, and the ranges' tape
   NX_ATT_CLASS_RUN, a block an element, their planes first in shared memory. A
   box of a load holding more than NX_ATT_BOX elements stands at its dtype's
   ends. */
#define NX_ATT_WORDS 512
#define NX_ATT_THREADS 256
#define NX_ATT_KEYS 64
#define NX_ATT_COLUMNS 128
#define NX_ATT_MMA_DK 128
#define NX_ATT_BOX 256
#define NX_ATT_MOST_V 4
#define NX_ATT_CLASS_RUN 64

/* The kinds attention's tapes run, each X(S, T, A, U, n) as loop.cu names
   a kind: S its compute type and T its C type, A its arity, 1, 2 or C for
   a comparison, and U and n its op: in a float accumulator, arithmetic,
   exp, log and tanh; in int64, the integer kinds but recip, pow and fma;
   on booleans as uint32 words, their kinds. Beside them a tape holds
   constants, casts among float32, float64, int64 and bool, selections of
   one or two words and truths, and nothing else: the host computes a
   program of another instruction in part through map first. A kind's op
   is NX_ATT_OP(S, A, U). */
#define NX_ATT_OP(S, A, U) ((NX_ATT_CT_##S << 7) + NX_ATT_CODE_##A(U))
#define NX_ATT_CT_f32 NX_LOOP_F32
#define NX_ATT_CT_f64 NX_LOOP_F64
#define NX_ATT_CT_i64 NX_LOOP_I64
#define NX_ATT_CT_u32 NX_LOOP_U32
#define NX_ATT_CODE_1(U) NX_OP1_##U
#define NX_ATT_CODE_2(U) (NX_LOOP_BINARY + NX_OP2_##U)
#define NX_ATT_CODE_C(U) NX_ATT_CODE_2(U)
#define NX_ATT_COMPARE(X, S, T)                                                \
  X(S, T, C, EQUAL, equal) X(S, T, C, NOT_EQUAL, not_equal)                    \
  X(S, T, C, LESS, less) X(S, T, C, LESS_EQUAL, less_equal)
#define NX_ATT_REAL(X, S, T)                                                   \
  X(S, T, 1, NEG, neg) X(S, T, 1, EXP, exp) X(S, T, 1, LOG, log)               \
  X(S, T, 1, TANH, tanh) X(S, T, 2, ADD, add) X(S, T, 2, SUB, sub)             \
  X(S, T, 2, MUL, mul) X(S, T, 2, FDIV, fdiv) X(S, T, 2, MAXIMUM, maximum)     \
  X(S, T, 2, MINIMUM, minimum) NX_ATT_COMPARE(X, S, T)
#define NX_ATT_INT(X, S, T)                                                    \
  X(S, T, 1, NEG, neg) X(S, T, 1, ABS, abs) X(S, T, 1, SIGN, sign)             \
  X(S, T, 2, ADD, add) X(S, T, 2, SUB, sub) X(S, T, 2, MUL, mul)               \
  X(S, T, 2, IDIV, idiv) X(S, T, 2, MOD, mod) X(S, T, 2, MAXIMUM, maximum)     \
  X(S, T, 2, MINIMUM, minimum) X(S, T, 2, AND, and) X(S, T, 2, OR, or)         \
  X(S, T, 2, XOR, xor) NX_ATT_COMPARE(X, S, T)
#define NX_ATT_BOOL(X, S, T)                                                   \
  X(S, T, 2, AND, and) X(S, T, 2, OR, or) X(S, T, 2, XOR, xor)                 \
  X(S, T, 2, MAXIMUM, maximum) X(S, T, 2, MINIMUM, minimum)                    \
  NX_ATT_COMPARE(X, S, T)
#define NX_ATT_KINDS(X)                                                        \
  NX_ATT_REAL(X, f32, float) NX_ATT_REAL(X, f64, double)                       \
  NX_ATT_INT(X, i64, int64_t) NX_ATT_BOOL(X, u32, uint32_t)

/* The header's words. */
enum {
  NX_ATT_KERNEL,  /* the path: NX_ATT_F32 to NX_ATT_MMA */
  NX_ATT_NLEAD,   /* the lead axes */
  NX_ATT_LEAD,    /* the word of the first lead extent */
  NX_ATT_B,       /* the lead indices */
  NX_ATT_HQ,      /* query heads */
  NX_ATT_HK,      /* key heads */
  NX_ATT_TQ,      /* rows */
  NX_ATT_TK,      /* keys */
  NX_ATT_DK,      /* q's and k's dimensions */
  NX_ATT_DV,      /* v's and o's */
  NX_ATT_DT,      /* q's, k's and v's dtype's code, then o's << 8 */
  NX_ATT_SCALE,   /* the scale rounded to the accumulator, a double's bits */
  NX_ATT_PAGE,    /* a page's keys, 0 for dense keys */
  NX_ATT_PAGES,   /* the pools' pages */
  NX_ATT_Q,       /* descriptors' words, 0 for none */
  NX_ATT_K,
  NX_ATT_VALUES,
  NX_ATT_TABLE,
  NX_ATT_LENGTHS,
  NX_ATT_O,
  NX_ATT_LSE,
  NX_ATT_WORK,    /* the workspace's slot: the parts' states, or the scores */
  NX_ATT_HG,      /* a unit's heads of a group at most */
  NX_ATT_TR,      /* a unit's rows of a head at most */
  NX_ATT_CHUNKS,  /* a group's chunks of heads */
  NX_ATT_TILES,   /* a head's tiles of rows */
  NX_ATT_UNITS,   /* the units: lead indices x key heads x chunks x tiles */
  NX_ATT_PARTS,   /* the keys' parts, 1 for none */
  NX_ATT_PBLOCKS, /* a part's blocks */
  NX_ATT_V,       /* a thread's elements of a tape's run */
  NX_ATT_SHARED,  /* the byte of the kernel's shared memory past the tapes'
                     planes; an FFMA path's staging overlaps them */
  NX_ATT_FLAGS,   /* NX_ATT_MASK to NX_ATT_SCORES bits */
  NX_ATT_MASK_AT, /* output 0 on pairs: its tape's first word, its
                     constants first */
  NX_ATT_MASK_N,  /* its words */
  NX_ATT_MASK_GIVEN, /* the word of its table of given planes */
  NX_ATT_MASK_OUT,   /* the plane of its result */
  NX_ATT_SCORE_AT,   /* output 1 on a pair's score */
  NX_ATT_SCORE_N,
  NX_ATT_SCORE_GIVEN,
  NX_ATT_SCORE_OUT,
  NX_ATT_RANGES_AT,  /* output 0's least and greatest over a block's box */
  NX_ATT_RANGES_N,
  NX_ATT_RANGES_GIVEN,
  NX_ATT_RANGES_LO,  /* the planes of the least and the greatest */
  NX_ATT_RANGES_HI,
  NX_ATT_LOADS,      /* the word of the loads' table: each one's descriptor */
  NX_ATT_HEADER
};

/* Paths: fused multiply-adds on tiles of 128 rows into float32, of 16
   rows (a key head's rows are few) and of 64 rows into float64, and the
   matrix units. */
enum { NX_ATT_F32, NX_ATT_F32_R16, NX_ATT_F64, NX_ATT_MMA };

/* NX_ATT_FLAGS' bits: output 0 is not true everywhere and runs on a
   block's pairs; output 1 is not the score and runs on it; output 0's
   ranges class the blocks; the launch stores each pair's score, in the
   array at the workspace's slot ([lead...; hq; tq; tk], C-contiguous),
   and nothing else, its keys in one part. */
enum { NX_ATT_MASK = 1, NX_ATT_SCORE = 2, NX_ATT_RANGES = 4, NX_ATT_SCORES = 8 };

/* A table of given planes is its count, then an entry a word: the plane's
   kind, an index << 8 (a loop axis or a load) and the first plane << 32.
   The kernel fills each before the tape runs: the score, a coordinate, a
   load's element, or a coordinate's or a load's least or greatest over a
   block's box. */
enum {
  NX_ATT_GIVE_SCORE,
  NX_ATT_GIVE_COORD,
  NX_ATT_GIVE_LOAD,
  NX_ATT_GIVE_COORD_LO,
  NX_ATT_GIVE_COORD_HI,
  NX_ATT_GIVE_LOAD_LO,
  NX_ATT_GIVE_LOAD_HI
};

typedef struct {
  uint64_t w[NX_ATT_WORDS];
} attention_params;

#endif
