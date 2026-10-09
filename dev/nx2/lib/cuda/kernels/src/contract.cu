/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contract: y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k]
   (kernels.h's contract_params), by three bodies, each its own
   association, fixed by the dtypes and the shape and never by a layout:
   mma (mma.cuh), SIMT (simt.cuh) and skinny (skinny.cuh), over the
   elements of elements.cuh. In each, [splits] ranges of k (whole k-tiles
   for mma and SIMT) combine in range order (combine.cuh); init then enters
   the sum, and the result rounds once to y's dtype. Each body reads its
   operands in the layout and dtype the plan packs them into (pack.cuh):
   the same values in the same tiles, so the same bits. */

#include "combine.cuh"
#include "kernels.h"
#include "nx_dtype.h"

typedef unsigned long long u64;

/* No device function here or in the headers is static: nvcc names an
   out-of-line function of internal linkage after a hash of the directory
   it compiles in, so the cubin's bytes would change with the checkout. */

#include "elements.cuh"
#include "mma.cuh"
#include "simt.cuh"
#include "skinny.cuh"
#include "pack.cuh"

/* Instances */

extern "C" __global__ void zero_u32(const __grid_constant__ zero_params p) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < p.n;
       i += (u64)gridDim.x * blockDim.x)
    p.p[i] = 0;
}

#define TRANSPOSED_k false
#define TRANSPOSED_m true
#define TRANSPOSED_n true
#define ACC_f32 float
#define ACC_f64 double
#define ACC_i64 u64

#define TILE(name, bm, bn, bkb, wm, wn, s)                                     \
  struct name {                                                                \
    enum { BM = bm, BN = bn, BKB = bkb, WM = wm, WN = wn, STAGES = s };        \
    enum { THREADS = (bm / wm) * (bn / wn) * 32 };                             \
  };
NX_CUDA_TILES(TILE)
#undef TILE

/* The mma kernel of [KIND]; for KIND_any, each kind's by a's dtype, each
   loop compiled apart so that none keeps another's registers live. */
template <int KIND, bool A_T, bool B_T, typename Tile>
__device__ void mma_kernel(const contract_params &p) {
  if constexpr (KIND != KIND_any)
    mma_contract<KIND, A_T, B_T, Tile>(p);
  else if (p.a_dtype == NX_INT8)
    mma_contract<KIND_s8, A_T, B_T, Tile>(p);
  else if (p.a_dtype == NX_FLOAT16)
    mma_contract<KIND_f16, A_T, B_T, Tile>(p);
  else
    mma_contract<KIND_bf16, A_T, B_T, Tile>(p);
}

#define DEFINE(name, FAMILY, ...) FAMILY(name, __VA_ARGS__)
#define ZERO(name, ...)
#define PACK(name, ...)                                                        \
  extern "C" __global__ void name(const __grid_constant__ pack_params p) {     \
    pack_rows(p);                                                              \
  }
#define MMA(name, kind, a, b, tile)                                            \
  extern "C" __global__ void __launch_bounds__(tile::THREADS)                  \
      name(const __grid_constant__ contract_params p) {                        \
    mma_kernel<KIND_##kind, TRANSPOSED_##a, TRANSPOSED_##b, tile>(p);          \
  }
#define SIMT(name, acc, bm)                                                    \
  extern "C" __global__ void __launch_bounds__(256)                            \
      name(const __grid_constant__ contract_params p) {                        \
    simt_contract<ACC_##acc, bm>(p);                                           \
  }
#define SKINNY(name, acc)                                                      \
  extern "C" __global__ void __launch_bounds__(256)                            \
      name(const __grid_constant__ contract_params p) {                        \
    skinny_kernel<ACC_##acc>(p);                                               \
  }
NX_CUDA_KERNELS(DEFINE)
