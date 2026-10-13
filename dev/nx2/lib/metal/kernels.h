/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.metal's kernels as the device sees them: every kernel, one parameter
   structure per family, and the geometry the plan shares with them.
   kernels.ml states the same facts for the host's plan, and
   test_metal_kernels.ml checks the two agree.

   A kernel reads its parameters as [constant P &p [[buffer(0)]]]. Their
   first words are GPU addresses (a buffer's gpuAddress plus an offset),
   which kernels read as device pointers.

   Float arithmetic is the GPU's: on Apple GPUs a float32 subnormal operand
   reads as zero and a float32 subnormal result is written as a zero of its
   sign, while half arithmetic and the conversions keep subnormals.

   Results are bitwise the same on every run of one GPU family: the plan
   reads no core count, occupancy or clock, kernels add no float
   atomically, and every sum's order is fixed by its number of terms, the
   dtypes and the order of summation the call names.

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
   metallib. Dense instances are named by operand dtype, then tile and the
   orders of a and b, n as named (a stored [m][k], b stored [k][n]) and t
   transposed (a stored [k][m], b stored [n][k]): large tiles (_nn to _tt),
   small ones (_s, either order, nx_metal_contract's order says which),
   wide ones (_wnn to _wtt), narrow ones (_vnn to _vtt), short ones (_rnn
   to _rtt), large ones
   reading past the matrix (_l, either order), and small ones reading any
   strides in 64-bit offsets (_g). Every operand is read where it lies.

   An instance reads its order at run time where doing so costs no row
   measurably, and is compiled per order elsewhere: each instance below
   names the rows that measured it. */
#define NX_METAL_KERNELS(X)                                                \
  /* nn: squares 1024 to 4096, 64 x 512 batches; nt: 4096-nt, 512-row     \
     prefills, Llama's up projection; tn, tt: 4096-tn and -tt */           \
  X(contract_f32_nn) X(contract_f32_nt) X(contract_f32_tn)                 \
  X(contract_f32_tt) X(contract_f16_nn) X(contract_f16_nt)                 \
  X(contract_f16_tn) X(contract_f16_tt) X(contract_bf16_nn)                \
  X(contract_bf16_nt) X(contract_bf16_tn) X(contract_bf16_tt)              \
  /* squares 256 and 512, and products past whole tiles */                \
  X(contract_f32_s) X(contract_f16_s) X(contract_bf16_s)                   \
  /* few rows: 8 rows nn, 3 to 16 rows nt, 48 rows of the half types */   \
  X(contract_f32_wnn) X(contract_f32_wnt) X(contract_f32_wtn)              \
  X(contract_f32_wtt) X(contract_f16_wnn) X(contract_f16_wnt)              \
  X(contract_f16_wtn) X(contract_f16_wtt) X(contract_bf16_wnn)             \
  X(contract_bf16_wnt) X(contract_bf16_wtn) X(contract_bf16_wtt)           \
  /* 9 to 16 rows in few tiles, and 48 rows of the half types: 16 rows by \
     2880 by 5120, 48 by 5120 by 2880, which wide tiles ran up to 1.3     \
     times slower */                                                       \
  X(contract_f32_vnn) X(contract_f32_vnt) X(contract_f32_vtn)              \
  X(contract_f32_vtt) X(contract_f16_vnn) X(contract_f16_vnt)              \
  X(contract_f16_vtn) X(contract_f16_vtt) X(contract_bf16_vnn)             \
  X(contract_bf16_vnt) X(contract_bf16_vtn) X(contract_bf16_vtt)           \
  /* 8 rows or fewer in few tiles: 3 and 8 rows by 2880 by 5120, which     \
     narrow tiles ran 1.1-1.9 times slower */                              \
  X(contract_f32_rnn) X(contract_f32_rnt) X(contract_f32_rtn)              \
  X(contract_f32_rtt) X(contract_f16_rnn) X(contract_f16_rnt)              \
  X(contract_f16_rtn) X(contract_f16_rtt) X(contract_bf16_rnn)             \
  X(contract_bf16_rnt) X(contract_bf16_rtn) X(contract_bf16_rtt)           \
  /* past whole tiles, 64 rows or more: the 1000 cubes */                 \
  X(contract_f32_l) X(contract_bf16_l)                                     \
  /* operands with no axis of unit stride, or offsets past 2^31 */        \
  X(contract_f32_g) X(contract_f16_g) X(contract_bf16_g)                   \
  /* int8-4096 */                                                         \
  X(contract_i8_nn) X(contract_i8_nt) X(contract_i8_tn) X(contract_i8_tt)  \
  /* lane order, b read along n and along k: float16 and bfloat16       \
     reading their dtype at run time run 4-5% slower at 1x2880x5120, b's  \
     order 1.6 times slower (-nt) */                                      \
  X(skinny_f32_n) X(skinny_f32_t)                                          \
  X(skinny_f16_n) X(skinny_f16_t)                                          \
  X(skinny_bf16_n) X(skinny_bf16_t)                                        \
  /* lane order past 1 row, which only a direct caller gives, b read once  \
     for 4 rows: -2x2880x5120 to -4x2880x5120 and their -nn rows; summing  \
     1 row, they slowed decode 1.1-2.5 times */                            \
  X(skinny4_f32_n) X(skinny4_f32_t)                                        \
  X(skinny4_f16_n) X(skinny4_f16_t)                                        \
  X(skinny4_bf16_n) X(skinny4_bf16_t)                                      \
  /* every other integer contraction */                                  \
  X(contract_int)


/* Contract */

/* The contraction kernels' geometry, which the plan and contract.metal
   share. */

/* Dense tiles of THREADS threads, rows × columns: large ones for products
   of many whole tiles, small ones for products of few tiles or past whole
   tiles, wide ones for products of few rows, and narrow ones, of the
   wide ones' rows, and short ones, of the narrow ones' columns, for
   products of few rows in few tiles. */
#define NX_METAL_THREADS 128
#define NX_METAL_LARGE 64
#define NX_METAL_SMALL 32
#define NX_METAL_WIDE_M 16
#define NX_METAL_WIDE_N 64
#define NX_METAL_NARROW_N 32
#define NX_METAL_SHORT_M 8

/* The steps of k a dense tile stages: float16 and bytes in the square
   tiles, the other dtypes there, every dtype in the wide and narrow ones,
   and in the short ones, whose 8 rows stage 4 elements a thread. */
#define NX_METAL_BK_HALF 32
#define NX_METAL_BK 16
#define NX_METAL_BK_WIDE 32
#define NX_METAL_BK_SHORT 64

/* The columns of out a skinny threadgroup computes, b read along k (T) or
   along n (N). */
#define NX_METAL_SKINNY_T 4
#define NX_METAL_SKINNY_N 32

/* The rows a skinny4 threadgroup sums, reading b once for them. */
#define NX_METAL_SKINNY_ROWS 4

/* The SIMD integer kernel's tile side and threads. */
#define NX_METAL_INT_TILE 64
#define NX_METAL_INT_THREADS 256

/* out[p][i][j] = round_out(init[p][i][j] + Σ_l a[p][i][l] · b[p][l][j]),
   for p < batch, i < m, j < n, l < k: floats sum in float32, integers in
   the kernel's accumulator, wrapping, and reach out's width by wrapping.
   Element (p, i, l) of a lies at p·a_batch + i·a_m + l·a_k elements from
   a, b's, init's and out's likewise, out's columns adjacent; for the dense
   kernels but the general ones, one of a_m and a_k is 1, and one of b_k
   and b_n, and an operand of R × C with strides s1 and s2 keeps
   |s1|·R' + |s2|·C' below 2^31, R' and C' its extents rounded up to
   multiples of 64, so that every offset they form fits in 32 bits; for
   the skinny kernels reading b along n, b_n is 1. Without init, the sum
   starts at 0. Dtypes are nx_dtype.h codes. */
typedef struct {
  uint64_t a, b, init, out;
  int64_t a_batch, a_m, a_k;
  int64_t b_batch, b_k, b_n;
  int64_t init_batch, init_m, init_n;
  int64_t out_batch, out_m;
  uint32_t batch, m, n, k;
  uint32_t init_dtype, out_dtype; /* init_dtype NX_DTYPE_COUNT: no init */
  uint32_t swizzle;               /* tiles of a column, as a power of two */
  uint32_t dtype;                 /* a's and b's */
  uint32_t acc;                   /* the accumulator's dtype */
  uint32_t order;                 /* NX_METAL_A_T | NX_METAL_B_T bits */
} nx_metal_contract;

/* nx_metal_contract's order: a stored [k][m], b stored [n][k]. An
   instance compiled for one order ignores it. */
#define NX_METAL_A_T 1u
#define NX_METAL_B_T 2u

#endif /* NX_METAL_KERNELS_H */
