/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_kinds.h's kinds by compute type, as overloads: k_<kind> of a float,
   double, int32_t, uint32_t, int64_t or uint64_t is nx_<kind>_<type>. A
   comparison gives 0 or 1. Part of kernels.cu's translation unit, after
   nx_kinds.h. */

#ifndef NX_CUDA_KINDS_CUH
#define NX_CUDA_KINDS_CUH

#define K1(T, S, k) \
  __device__ __forceinline__ T k_##k(T a) { return nx_##k##_##S(a); }
#define K2(T, S, k) \
  __device__ __forceinline__ T k_##k(T a, T b) { return nx_##k##_##S(a, b); }
#define KC(T, S, k) \
  __device__ __forceinline__ uint32_t k_##k(T a, T b) { return nx_##k##_##S(a, b) != 0; }
#define KINDS_ANY(T, S)                                                        \
  K1(T, S, neg) K1(T, S, recip) K1(T, S, abs) K1(T, S, sign)                   \
  K2(T, S, add) K2(T, S, sub) K2(T, S, mul) K2(T, S, mod) K2(T, S, pow)        \
  K2(T, S, maximum) K2(T, S, minimum)                                          \
  KC(T, S, equal) KC(T, S, not_equal) KC(T, S, less) KC(T, S, less_equal)      \
  __device__ __forceinline__ T k_fma(T a, T b, T c) { return nx_fma_##S(a, b, c); }
#define KINDS_FLOAT(T, S)                                                      \
  KINDS_ANY(T, S)                                                              \
  K1(T, S, sqrt) K1(T, S, exp) K1(T, S, exp2) K1(T, S, log) K1(T, S, log2)     \
  K1(T, S, log1p) K1(T, S, expm1) K1(T, S, sin) K1(T, S, cos) K1(T, S, tan)    \
  K1(T, S, asin) K1(T, S, acos) K1(T, S, atan) K1(T, S, sinh) K1(T, S, cosh)   \
  K1(T, S, tanh) K1(T, S, erf) K1(T, S, floor) K1(T, S, ceil) K1(T, S, round)  \
  K1(T, S, trunc) K2(T, S, fdiv) K2(T, S, atan2)
#define KINDS_INT(T, S)                                                        \
  KINDS_ANY(T, S)                                                              \
  K2(T, S, idiv) K2(T, S, and) K2(T, S, or) K2(T, S, xor)

KINDS_FLOAT(float, f32)
KINDS_FLOAT(double, f64)
KINDS_INT(int32_t, i32)
KINDS_INT(uint32_t, u32)
KINDS_INT(int64_t, i64)
KINDS_INT(uint64_t, u64)

#undef K1
#undef K2
#undef KC
#undef KINDS_ANY
#undef KINDS_FLOAT
#undef KINDS_INT

/* Whether T is a float type. */
template <typename T> constexpr bool real = T(0.5) != T(0);

#endif
