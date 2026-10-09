/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.metal's kernels as device and host see them: the launch record, one
   parameter structure per family, and the enum of every kernel.

   A launch is a record followed in memory by its parameter bytes. The
   kernel reads them as [constant P &p [[buffer(0)]]]. The first [addrs]
   words of the parameters are GPU addresses (an MTLBuffer's gpuAddress plus
   an offset), which kernels read as device pointers; bit i of [scratch]
   says address i is an offset into the call's scratch, to which whoever
   allocates the scratch adds its base before the fill runs. A run of
   records holds no pointer, so it can be kept, moved and submitted after
   the call that planned it.

   Float arithmetic is the GPU's: on Apple GPUs a float32 subnormal operand
   reads as zero and a float32 subnormal result is written as a zero of its
   sign, while half arithmetic and the conversions keep subnormals.

   Results are bitwise the same on every run of one GPU family: plan.c
   reads no core count, occupancy or clock, kernels add no float
   atomically, and every sum's order is fixed by the shape.

   This header compiles as C on the host and as the Metal Shading Language
   on the device, with nothing but metal_stdlib. */

#ifndef NX_METAL_KERNELS_H
#define NX_METAL_KERNELS_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

/* X(name) for every kernel of the library, the function [name] of its
   metallib. Dense float instances are named by operand dtype, then tile
   and b's order: large tiles (_n, _t), small ones (_s), wide ones (_wn,
   _wt); n: b stored [k][n], t: b stored [n][k]. Every dense instance
   reads a stored [m][k].

   Instances are a budget: each names the rows that keep it. An operand
   in a layout with no instance is packed first, and a product past whole
   tiles runs on small ones. */
#define NX_METAL_KERNELS(X)                                                \
  /* squares 1024 to 4096, 4096-tn, 64 x 512 batches */                   \
  X(contract_f32_n) X(contract_f16_n) X(contract_bf16_n)                   \
  /* 4096-nt and -tt, 512-row prefills, Llama's up projection */          \
  X(contract_f32_t) X(contract_f16_t) X(contract_bf16_t)                   \
  /* squares 256 and 512, and every product past whole tiles */           \
  X(contract_f32_s) X(contract_f16_s) X(contract_bf16_s)                   \
  /* 8 rows nn; 3, 8 and 16 rows nt */                                    \
  X(contract_f32_wn) X(contract_f16_wn) X(contract_bf16_wn)                \
  X(contract_f32_wt) X(contract_f16_wt) X(contract_bf16_wt)                \
  /* int8-4096 */                                                         \
  X(contract_i8)                                                           \
  /* decode, b stored [k][n] and [n][k]: float16 and bfloat16 reading     \
     their dtype at run time run 4-5% slower at 1x2880x5120 */            \
  X(skinny_f32_n) X(skinny_f32_t)                                          \
  X(skinny_f16_n) X(skinny_f16_t)                                          \
  X(skinny_bf16_n) X(skinny_bf16_t)                                        \
  /* every split sum; every other integer contraction; every operand in a \
     layout with no instance */                                           \
  X(contract_combine) X(contract_int) X(pack)

#define NX_METAL_ENUM(name) NX_METAL_##name,
enum nx_metal_kernel { NX_METAL_KERNELS(NX_METAL_ENUM) NX_METAL_KERNEL_COUNT };
#undef NX_METAL_ENUM

/* A launch: [entry] indexes the run's pipelines; [bytes] is a multiple of
   8 and at most 4,096, what Metal's setBytes takes; [addrs]·8 <= bytes. */
typedef struct {
  uint32_t entry;      /* an nx_metal_kernel */
  uint32_t groups[3];  /* threadgroups per grid */
  uint32_t threads[3]; /* threads per threadgroup */
  uint32_t bytes;      /* parameter bytes that follow */
  uint32_t addrs;      /* the parameters' first addrs words are addresses */
  uint32_t scratch;    /* bit i: address i is an offset into the scratch */
} nx_metal_launch;

/* Contract */

/* The contraction kernels' geometry, which plan.c and contract.metal
   share. Dense tiles of NX_METAL_THREADS threads, rows × columns × steps
   of k: large ones, small ones for products of few tiles, wide ones for
   products of few rows; NX_METAL_BK_HALF steps for half and bytes in the
   square tiles, NX_METAL_BK otherwise, NX_METAL_BK_WIDE in the wide ones.
   Skinny products of one row: columns of out a threadgroup computes, b
   stored [n][k] (T) or [k][n] (N). Integers on the SIMD units: tile and
   threads. Packs: a threadgroup's tile side, and its threads, PACK / 4
   × PACK_ROWS. */
#define NX_METAL_THREADS 128
#define NX_METAL_LARGE 64
#define NX_METAL_SMALL 32
#define NX_METAL_WIDE_M 16
#define NX_METAL_WIDE_N 64
#define NX_METAL_BK 16
#define NX_METAL_BK_HALF 32
#define NX_METAL_BK_WIDE 32
#define NX_METAL_SKINNY_T 4
#define NX_METAL_SKINNY_N 32
#define NX_METAL_INT_TILE 64
#define NX_METAL_INT_THREADS 256
#define NX_METAL_PACK 64
#define NX_METAL_PACK_ROWS 16

/* out[p][i][j] = round_out(init[p][i][j] + Σ_l a[p][i][l] · b[p][l][j]),
   for p < batch, i < m, j < n, l < k: floats sum in float32, integers in
   the kernel's accumulator, wrapping, and reach out's width by wrapping.
   Element (p, i, l) of a lies at p·a_batch + i·a_m + l·a_k, b's and
   init's likewise; for floats one of a_m and a_k is 1, and one of b_k and
   b_n. out is C-contiguous. Without init, the sum starts at 0. Dtypes are
   nx_dtype.h codes. */
typedef struct {
  uint64_t a, b, init, out;
  int64_t a_batch, b_batch, init_batch;
  uint32_t a_m, a_k, b_k, b_n, init_m, init_n;
  uint32_t batch, m, n, k;
  uint32_t init_dtype, out_dtype; /* init_dtype NX_DTYPE_COUNT: no init */
  uint32_t swizzle;               /* tiles of a column, as a power of two */
  uint32_t dtype;                 /* a's and b's */
  uint32_t acc;                   /* the accumulator's dtype */
} nx_metal_contract;

/* out[p][i][j] = round_out(init[p][i][j] + Σ_q parts[q][p][i][j]) for
   q < split, the sum in increasing q in float32, init as
   nx_metal_contract reads it: the parts of a contraction split along k,
   each a float32 contraction of its terms. */
typedef struct {
  uint64_t out, parts, init;
  int64_t init_batch;
  uint32_t init_m, init_n, batch, m, n, split;
  uint32_t init_dtype, out_dtype;
} nx_metal_combine;

/* dst[p][r][c] = src[p·src_batch + r·row + c·col] for p < batch (grid
   z), r < rows, c < cols, elements of [bytes] bytes; dst[p][r][c] = 0 for
   cols <= c < ld: an operand copied with cols contiguous, rows ld
   elements apart, batch elements dst_batch apart. */
typedef struct {
  uint64_t src, dst;
  int64_t src_batch, dst_batch;
  uint32_t row, col, rows, cols, ld, bytes;
} nx_metal_pack;

#endif /* NX_METAL_KERNELS_H */
