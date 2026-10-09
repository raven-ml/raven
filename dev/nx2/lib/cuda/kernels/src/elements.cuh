/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What every contraction body reads and writes: an operand's elements as
   the accumulator's type, and y's outputs, init added and rounded once, as
   a cast from the accumulator writes them. Part of contract.cu's translation unit, after
   kernels.h and nx_dtype.h. */

#ifndef NX_CUDA_ELEMENTS_CUH
#define NX_CUDA_ELEMENTS_CUH

/* Element [i] of [p], of the dtype [dt], as a double or an int64. */
__device__ __forceinline__ double float_at(const void *p, int64_t i, int dt) {
  switch (dt) {
  case NX_FLOAT64: return ((const double *)p)[i];
  case NX_FLOAT32: return ((const float *)p)[i];
  case NX_FLOAT16: return nx_f16_to_float(((const uint16_t *)p)[i]);
  case NX_BFLOAT16: return nx_bf16_to_float(((const uint16_t *)p)[i]);
  case NX_FLOAT8_E4M3FN: return nx_e4m3fn_to_float(((const uint8_t *)p)[i]);
  case NX_FLOAT8_E5M2: return nx_e5m2_to_float(((const uint8_t *)p)[i]);
  }
  return 0;
}

__device__ __forceinline__ u64 int_at(const void *p, int64_t i, int dt) {
  switch (dt) {
  case NX_INT64:
  case NX_UINT64: return ((const u64 *)p)[i];
  case NX_INT32: return (u64)(int64_t)((const int32_t *)p)[i];
  case NX_UINT32: return ((const uint32_t *)p)[i];
  case NX_INT16: return (u64)(int64_t)((const int16_t *)p)[i];
  case NX_UINT16: return ((const uint16_t *)p)[i];
  case NX_INT8: return (u64)(int64_t)((const int8_t *)p)[i];
  case NX_UINT8:
  case NX_BOOL: return ((const uint8_t *)p)[i];
  }
  return 0;
}

/* The same out of line, for init's element, which a store reads at every
   element of an unrolled tile. */
__device__ __noinline__ double read_float(const void *p, int64_t i, int dt) {
  return float_at(p, i, dt);
}

__device__ __noinline__ u64 read_int(const void *p, int64_t i, int dt) {
  return int_at(p, i, dt);
}

/* Whether the accumulator type T is a float: integers wrap, so they sum
   unsigned. */
template <typename T> constexpr bool is_float = T(0.5) != T(0);

/* [s] as the accumulator T: an integer widened by its own sign. */
template <typename T, typename S> __device__ T widen(S s) {
  if constexpr (T(-1) > T(0) && S(-1) < S(0))
    return (T)(int64_t)s;
  else
    return (T)s;
}

/* Element [i] of [p], of the dtype [dt], as the accumulator T. */
template <typename T> __device__ T read(const void *p, int64_t i, int dt) {
  if constexpr (is_float<T>)
    return (T)read_float(p, i, dt);
  else
    return (T)read_int(p, i, dt);
}

/* Stores the bits [c] of an element of the integer, bool or narrow float
   dtype [dt] at element [i] of [p]: their low bits, as wide as [dt]. */
__device__ void put(void *p, int64_t i, int dt, int64_t c) {
  switch (dt) {
  case NX_INT64:
  case NX_UINT64: ((int64_t *)p)[i] = c; break;
  case NX_INT32:
  case NX_UINT32: ((uint32_t *)p)[i] = (uint32_t)c; break;
  case NX_INT16:
  case NX_UINT16:
  case NX_FLOAT16:
  case NX_BFLOAT16: ((uint16_t *)p)[i] = (uint16_t)c; break;
  default: ((uint8_t *)p)[i] = (uint8_t)c;
  }
}

/* The sum [v] at element [i] of y, as a cast from the accumulator writes
   it. A float sum rounds once to y's dtype, nx_dtype.h's conversions
   reaching the narrow floats, the integers and bool. */
__device__ void write_float(const contract_params &p, int64_t i, double v) {
  switch (p.y_dtype) {
  case NX_FLOAT64: ((double *)p.y)[i] = v; return;
  case NX_FLOAT32: ((float *)p.y)[i] = (float)v; return;
  }
  put(p.y, i, p.y_dtype, nx_double_to_bits(p.y_dtype, v));
}

/* An integer sum [v], wrapped to the accumulator and widened by its sign:
   wrapped to an integer y, rounded once to a float y, nonzero for bool. A
   32-bit accumulator's sum, summed in 64 bits, wraps to 32 here: the same
   sum, modulo 2^32. */
__device__ void write_int(const contract_params &p, int64_t i, u64 v) {
  if (p.acc_dtype == NX_INT32) v = (u64)(int64_t)(int32_t)v;
  if (p.acc_dtype == NX_UINT32) v = (uint32_t)v;
  const bool s = p.acc_dtype == NX_INT32 || p.acc_dtype == NX_INT64;
  switch (p.y_dtype) {
  case NX_FLOAT64:
    ((double *)p.y)[i] = s ? (double)(int64_t)v : (double)v;
    return;
  case NX_FLOAT32:
    ((float *)p.y)[i] = s ? (float)(int64_t)v : (float)v;
    return;
  case NX_FLOAT16:
  case NX_BFLOAT16:
  case NX_FLOAT8_E4M3FN:
  case NX_FLOAT8_E5M2: {
    const double odd = s ? nx_i64_odd((int64_t)v) : nx_u64_odd(v);
    put(p.y, i, p.y_dtype, nx_double_to_bits(p.y_dtype, odd));
    return;
  }
  case NX_BOOL: put(p.y, i, NX_BOOL, v != 0); return;
  }
  put(p.y, i, p.y_dtype, (int64_t)v);
}

/* The sum [v], of the accumulator type T, at element [i] of y. */
template <typename T>
__device__ void write(const contract_params &p, int64_t i, T v) {
  if constexpr (is_float<T>)
    write_float(p, i, (double)v);
  else
    write_int(p, i, widen<u64>(v));
}

/* y's element (z, i, j), if inside: init added, then rounded once. Out of
   line: kernels call it at every element of an unrolled tile. */
template <typename T>
__device__ __noinline__ void store(const contract_params &p, int z,
                                          int i, int j, T v) {
  if (i >= p.m || j >= p.n) return;
  if (p.init)
    v += read<T>(p.init, z * p.si[0] + i * p.si[1] + j * p.si[2],
                 p.init_dtype);
  write(p, z * p.sy[0] + i * p.sy[1] + j * p.sy[2], v);
}

/* Two outputs (z, i, j) and (z, i, j + 1). With no init, y's pairs whole
   and aligned along a contiguous j, and y of the accumulator's dtype, or
   bfloat16 or float16 from float32 (NX_CONTRACT_Y_WHOLE), one
   store of both; through store otherwise. The conversions round to
   nearest even, as nx_dtype.h's: only a NaN's payload may differ. */
__device__ void store_pair(const contract_params &p, int z, int i,
                                  int j, float v0, float v1) {
  if ((p.aligned & NX_CONTRACT_Y_WHOLE) && i < p.m && j + 1 < p.n) {
    const int64_t o = z * p.sy[0] + i * p.sy[1] + j;
    uint32_t h;
    switch (p.y_dtype) {
    case NX_FLOAT32:
      *(float2 *)((float *)p.y + o) = make_float2(v0, v1);
      return;
    case NX_BFLOAT16:
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(v1), "f"(v0));
      *(uint32_t *)((uint16_t *)p.y + o) = h;
      return;
    case NX_FLOAT16:
      asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(v1), "f"(v0));
      *(uint32_t *)((uint16_t *)p.y + o) = h;
      return;
    }
  }
  store(p, z, i, j, v0);
  store(p, z, i, j + 1, v1);
}

template <typename T>
__device__ void store_pair(const contract_params &p, int z, int i,
                                  int j, T v0, T v1) {
  if ((p.aligned & NX_CONTRACT_Y_WHOLE) && i < p.m && j + 1 < p.n) {
    T *y = (T *)p.y + z * p.sy[0] + i * p.sy[1] + j;
    y[0] = v0, y[1] = v1;
    return;
  }
  store(p, z, i, j, v0);
  store(p, z, i, j + 1, v1);
}

/* Eight outputs (z, i, j + e), e < 8, of [v]. With no init, y's j
   contiguous, its rows aligned on 16 bytes, and y of the accumulator's
   dtype, or bfloat16 or float16 from float32 (NX_CONTRACT_Y_WHOLE): one
   16-byte store, rounding to nearest even as nx_dtype.h's encoders do
   (only a NaN's payload may differ). Through store otherwise. */
template <typename T>
__device__ void store8(const contract_params &p, int z, int i, int j,
                              const T *v) {
  if ((p.aligned & NX_CONTRACT_Y_WHOLE) && i < p.m && j + 8 <= p.n) {
    const int64_t o = z * p.sy[0] + i * p.sy[1] + j;
    if (sizeof(T) == 4 && (p.y_dtype == NX_BFLOAT16 || p.y_dtype == NX_FLOAT16)) {
      uint32_t h[4];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const float lo = (float)v[2 * e], hi = (float)v[2 * e + 1];
        if (p.y_dtype == NX_BFLOAT16)
          asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(h[e]) : "f"(hi), "f"(lo));
        else
          asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h[e]) : "f"(hi), "f"(lo));
      }
      *(uint4 *)((uint16_t *)p.y + o) = make_uint4(h[0], h[1], h[2], h[3]);
      return;
    }
#pragma unroll
    for (int e = 0; e < 8 * (int)sizeof(T) / 16; e++)
      ((uint4 *)((T *)p.y + o))[e] = ((const uint4 *)v)[e];
    return;
  }
  for (int e = 0; e < 8; e++) store(p, z, i, j + e, v[e]);
}

/* a * b + c in the accumulator type: one rounding for floats, wrapping for
   integers. */
__device__ float fma_(float a, float b, float c) { return fmaf(a, b, c); }
__device__ double fma_(double a, double b, double c) { return fma(a, b, c); }
__device__ uint32_t fma_(uint32_t a, uint32_t b, uint32_t c) { return a * b + c; }
__device__ u64 fma_(u64 a, u64 b, u64 c) { return a * b + c; }

#endif
