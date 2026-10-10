/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.amd's kernels as the host and the device both see them: the
   instances, the tiles and each family's parameters.

   A kernel reads one parameter struct, the parameters of its launch. The
   host writes it field by field at the offsets kernels.ml states, and rig
   turns each address field into the address of a buffer of the submit.
   Every struct's size is a multiple of 8. A kernel declares its local data
   share at its own size: a launch adds none. */

#ifndef NX_AMD_KERNELS_H
#define NX_AMD_KERNELS_H

#include <stdint.h>

/* Kernels */

/* X(name, FAMILY, ...) for every kernel of the code object, in enum order;
   the arguments after the family are its instance's:
   - ZERO, PACK: none.
   - WMMA (contract.hip): the operands' kind (bf16, f16, s8), a's and b's
     contiguous axis (k, or their free axis m and n) and the tile.
   - SIMT (contract.hip): the accumulator's type and the tile's side.
   - SKINNY (contract.hip): the accumulator's type and the form: column,
     b's k axis contiguous, or across, its n axis.
   Instances are a budget: each names the rows that keep it. A layout or
   tile with no instance is computed by one that has it, its operand
   packed first; an int32 accumulator sums in int64 and wraps. */
#define NX_AMD_KERNELS(X)                                                     \
  /* every split sum */                                                       \
  X(zero_u32, ZERO)                                                          \
  /* f8 operands, rows not of vectors, layouts with no instance */          \
  X(pack, PACK)                                                              \
  /* bf16 1024 to 8192, 4096x14336x4096, the gpt-oss prefill rows,          \
     64x512x512x512 */                                                       \
  X(contract_bf16_kk_t128x128, WMMA, bf16, k, k, t128x128)                  \
  /* the 4096 rows in layouts kf, fk and ff */                               \
  X(contract_bf16_kn_t128x128, WMMA, bf16, k, n, t128x128)                  \
  X(contract_bf16_mk_t128x128, WMMA, bf16, m, k, t128x128)                  \
  X(contract_bf16_mn_t128x128, WMMA, bf16, m, n, t128x128)                  \
  /* bf16 256, 512 */                                                        \
  X(contract_bf16_kk_t64x64, WMMA, bf16, k, k, t64x64)                      \
  /* bf16 decode: 1x5120x2880, 1x201088x2880 */                              \
  X(contract_bf16_kk_t16x64, WMMA, bf16, k, k, t16x64)                      \
  /* f16 1024 to 8192 */                                                     \
  X(contract_f16_kk_t128x128, WMMA, f16, k, k, t128x128)                    \
  /* the 4096 rows in layouts kf, fk and ff */                               \
  X(contract_f16_kn_t128x128, WMMA, f16, k, n, t128x128)                    \
  X(contract_f16_mk_t128x128, WMMA, f16, m, k, t128x128)                    \
  X(contract_f16_mn_t128x128, WMMA, f16, m, n, t128x128)                    \
  /* f16 256, 512 */                                                         \
  X(contract_f16_kk_t64x64, WMMA, f16, k, k, t64x64)                        \
  /* int8 4096 */                                                            \
  X(contract_s8_kk_t128x128, WMMA, s8, k, k, t128x128)                      \
  /* f32 2048 to 8192 and the 4096 layouts */                                \
  X(contract_simt_f32_128, SIMT, f32, 128)                                  \
  /* f32 256 to 1024 */                                                      \
  X(contract_simt_f32_64, SIMT, f32, 64)                                    \
  /* f32 decode 1x5120x2880, 1x20480x2880, 8x20480x2880 */                   \
  X(contract_skinny_f32, SKINNY, f32, column)                               \
  /* f32 decode 1x5120x2880, 1x20480x2880, 8x20480x2880, b stored k by n */ \
  X(contract_skinny_across_f32, SKINNY, f32, across)                        \
  /* float64 and integer sums: no row; they make the family total */         \
  X(contract_simt_f64_64, SIMT, f64, 64)                                    \
  X(contract_simt_i64_64, SIMT, i64, 64)                                    \
  X(contract_skinny_f64, SKINNY, f64, column)                               \
  X(contract_skinny_i64, SKINNY, i64, column)                               \
  X(contract_skinny_across_f64, SKINNY, f64, across)                        \
  X(contract_skinny_across_i64, SKINNY, i64, across)

#define NX_AMD_ENUM(name, ...) NX_AMD_##name,
enum nx_amd_kernel { NX_AMD_KERNELS(NX_AMD_ENUM) NX_AMD_KERNEL_COUNT };
#undef NX_AMD_ENUM

/* X(name, BM, BN, BKB, WM, WN): the WMMA kernels' tiles. A workgroup
   computes BM x BN outputs, (BM / WM) x (BN / WN) waves of WM x WN each,
   over k-tiles of BKB bytes of each operand row, two in flight: one in its
   local data share, the next in registers. */
#define NX_AMD_TILES(X)                \
  X(t128x128, 128, 128, 64, 64, 32)    \
  X(t64x64, 64, 64, 64, 32, 32)        \
  X(t16x64, 16, 64, 128, 16, 16)

/* Contract */

/* y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k], rounded once
   to y's dtype, for z < batch, i < m, j < n: each operand's element
   (z, i, k) at its address plus z, i and k times its strides, in elements.
   [init] is NULL for none. [partials] and [tickets] are the workspace's when
   [splits] > 1: the sum over k is cut into [splits] ranges, each summed by
   its own workgroup, and the last to arrive adds them in range order
   (combine.h). A SIMT kernel reads a and b of [a_dtype] and [b_dtype],
   converting to its accumulator. The sum, of [acc_dtype], reaches y as a
   cast from it does. [aligned] holds the NX_CONTRACT_ bits below. */
typedef struct {
  const void *a, *b, *init;
  void *y, *partials;
  uint32_t *tickets;
  int64_t sa[3], sb[3], si[3], sy[3];
  int32_t batch, m, n, k;
  int32_t splits, a_dtype, b_dtype, init_dtype;
  int32_t y_dtype, acc_dtype, aligned;
  int32_t zero; /* 0: the last word, which would be padding, so that the
                   parameters' bytes are their fields' alone */
} contract_params;

/* contract_params' bits of [aligned]. */
enum {
  NX_CONTRACT_A_VECTORS = 1, /* a's rows are 16-byte vectors */
  NX_CONTRACT_B_VECTORS = 2, /* b's rows are 16-byte vectors */
  NX_CONTRACT_Y_WHOLE = 4    /* y's outputs store 16 bytes at once */
};

/* The work-items of a workgroup of pack, zero_u32 and the SIMT and skinny
   kernels: a work-item per element for the first two. */
#define NX_CONTRACT_THREADS 256

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

/* zero_u32: [n] 32-bit words at [p] set to 0. */
typedef struct {
  uint32_t *p;
  uint64_t n;
} zero_params;

#endif
