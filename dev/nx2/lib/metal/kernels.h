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
   metallib. Dense instances are named by operand dtype, then tile and the
   orders of a and b, n as named (a stored [m][k], b stored [k][n]) and t
   transposed (a stored [k][m], b stored [n][k]): large tiles (_nn to _tt),
   small ones (_s, either order, nx_metal_contract's order says which),
   wide ones (_wnn to _wtt), large ones reading past the matrix (_l,
   either order). Every operand is read where it lies.

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
  /* past whole tiles, 64 rows or more: the 1000 cubes */                 \
  X(contract_f32_l) X(contract_bf16_l)                                     \
  /* int8-4096 */                                                         \
  X(contract_i8_nn) X(contract_i8_nt) X(contract_i8_tn) X(contract_i8_tt)  \
  /* decode, b stored [k][n] and [n][k]: float16 and bfloat16 reading     \
     their dtype at run time run 4-5% slower at 1x2880x5120, b's order    \
     1.6 times slower (-nt) */                                            \
  X(skinny_f32_n) X(skinny_f32_t)                                          \
  X(skinny_f16_n) X(skinny_f16_t)                                          \
  X(skinny_bf16_n) X(skinny_bf16_t)                                        \
  /* every split sum; every other integer contraction */                  \
  X(contract_combine) X(contract_int)

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
   share. */

/* Dense tiles of THREADS threads, rows × columns: large ones for products
   of many whole tiles, small ones for products of few tiles or past whole
   tiles, wide ones for products of few rows. */
#define NX_METAL_THREADS 128
#define NX_METAL_LARGE 64
#define NX_METAL_SMALL 32
#define NX_METAL_WIDE_M 16
#define NX_METAL_WIDE_N 64

/* The steps of k a dense tile stages: float16 and bytes in the square
   tiles, the other dtypes there, and every dtype in the wide ones. */
#define NX_METAL_BK_HALF 32
#define NX_METAL_BK 16
#define NX_METAL_BK_WIDE 32

/* The columns of out a skinny threadgroup computes, b stored [n][k] (T) or
   [k][n] (N). */
#define NX_METAL_SKINNY_T 4
#define NX_METAL_SKINNY_N 32

/* The SIMD integer kernel's tile side and threads. */
#define NX_METAL_INT_TILE 64
#define NX_METAL_INT_THREADS 256

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
  uint32_t order;                 /* NX_METAL_A_T | NX_METAL_B_T bits */
} nx_metal_contract;

/* nx_metal_contract's order: a stored [k][m], b stored [n][k]. An
   instance compiled for one order ignores it. */
#define NX_METAL_A_T 1u
#define NX_METAL_B_T 2u

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

#endif /* NX_METAL_KERNELS_H */
