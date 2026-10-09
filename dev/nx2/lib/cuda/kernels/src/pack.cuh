/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The pack: an operand copied into the layout and dtype a body reads, each
   value converted exactly. Part of contract.cu's translation unit, after elements.cuh. */

#ifndef NX_CUDA_PACK_CUH
#define NX_CUDA_PACK_CUH

/* An element of a packed operand: [dt]'s value as [out]'s, exactly, as
   [out]'s bits. */
__device__ __forceinline__ u64 packed(const void *src, int64_t x, int dt,
                                      int out) {
  switch (out) {
  case NX_BFLOAT16: return nx_float_to_bf16((float)float_at(src, x, dt));
  case NX_FLOAT16: return nx_float_to_f16((float)float_at(src, x, dt));
  case NX_FLOAT32: return nx_float_bits((float)float_at(src, x, dt));
  case NX_FLOAT64: return nx_double_bits(float_at(src, x, dt));
  }
  return int_at(src, x, dt);
}

/* The bytes of an element of the dtype [dt], at compile time. */
__host__ __device__ constexpr int dt_bytes(int dt) {
  return dt == NX_FLOAT64 || dt == NX_INT64 || dt == NX_UINT64   ? 8
         : dt == NX_FLOAT32 || dt == NX_INT32 || dt == NX_UINT32 ? 4
         : dt == NX_FLOAT16 || dt == NX_BFLOAT16 || dt == NX_INT16 ||
                 dt == NX_UINT16
             ? 2
             : 1;
}

/* N bytes as one load: 2, 4, 8 or 16. */
template <int N> struct Raw;
template <> struct Raw<2> { typedef uint16_t type; };
template <> struct Raw<4> { typedef uint32_t type; };
template <> struct Raw<8> { typedef uint2 type; };
template <> struct Raw<16> { typedef uint4 type; };

/* The unsigned type of N bytes. */
template <int N> struct Bits;
template <> struct Bits<1> { typedef uint8_t type; };
template <> struct Bits<2> { typedef uint16_t type; };
template <> struct Bits<4> { typedef uint32_t type; };
template <> struct Bits<8> { typedef u64 type; };

/* The packed operand's 16-byte vectors, one a thread, counted in [I], of
   [OB]-byte elements: from the dtype [DT] to [OUT], each compiled for its
   pair, or from [p.dtype] to [p.out] where they are -1. A vector is a run
   of one row: its indices are divided once for its elements, and a run of
   the source whose k is contiguous and aligned is one load. */
template <int DT, int OUT, int OB, typename I>
__device__ void pack_vectors(const pack_params &p) {
  typedef typename Bits<OB>::type Out;
  constexpr int SB = DT < 0 ? 0 : dt_bytes(DT);
  const int dt = DT < 0 ? p.dtype : DT, out = OUT < 0 ? p.out : OUT;
  constexpr int PER = 16 / OB;
  const char *src = (const char *)p.src;
  const I per_row = (I)(p.lead / PER), n = (I)p.batch * p.rows * per_row;
  /* A float8 source converts through a table of its 256 codes' values,
     made by the block from the same conversion. */
  constexpr bool TABLE = DT == NX_FLOAT8_E4M3FN || DT == NX_FLOAT8_E5M2;
  __shared__ Out table[TABLE ? 256 : 1];
  if (TABLE) {
    for (int c = threadIdx.x; c < 256; c += blockDim.x) {
      const uint8_t code = (uint8_t)c;
      table[c] = (Out)packed(&code, 0, dt, out);
    }
    __syncthreads();
  }
  auto conv = [&](const void *at, int64_t x) {
    return TABLE ? table[((const uint8_t *)at)[x]] : (Out)packed(at, x, dt, out);
  };
  for (I v = blockIdx.x * (I)blockDim.x + threadIdx.x; v < n;
       v += (I)gridDim.x * blockDim.x) {
    const I row = v / per_row, z = row / (I)p.rows;
    const int64_t q0 = (int64_t)(v - row * per_row) * PER;
    const int64_t at = (int64_t)z * p.s[0] +
                       (int64_t)(row - z * p.rows) * p.s[1] + q0 * p.s[2];
    union {
      Out e[PER];
      uint4 all;
    } o;
    if (SB > 0 && p.s[2] == 1 && q0 + PER <= p.k &&
        (uintptr_t)(src + at * SB) % (PER * SB) == 0) {
      typedef typename Raw<(SB > 0 ? PER * SB : 16)>::type R;
      union {
        R all;
        uint8_t b[sizeof(R)];
      } in = {*(const R *)(src + at * SB)};
#pragma unroll
      for (int e = 0; e < PER; e++) o.e[e] = conv(in.b, e);
    } else
#pragma unroll
      for (int e = 0; e < PER; e++)
        o.e[e] = q0 + e < p.k ? conv(src, at + e * p.s[2]) : Out(0);
    ((uint4 *)p.dst)[v] = o.all;
  }
}

/* X(dtype, out): the packs the plan makes, each compiled for its pair; any
   other runs the general loop. */
#define PACKS(X)                                                               \
  X(NX_INT8, NX_INT8)                                                          \
  X(NX_BFLOAT16, NX_BFLOAT16)                                                  \
  X(NX_FLOAT8_E4M3FN, NX_BFLOAT16)                                             \
  X(NX_FLOAT8_E5M2, NX_BFLOAT16)                                               \
  X(NX_FLOAT16, NX_FLOAT16)                                                    \
  X(NX_FLOAT16, NX_FLOAT32)                                                    \
  X(NX_BFLOAT16, NX_FLOAT32)                                                   \
  X(NX_FLOAT8_E4M3FN, NX_FLOAT32)                                              \
  X(NX_FLOAT8_E5M2, NX_FLOAT32)                                                \
  X(NX_FLOAT32, NX_FLOAT32)                                                    \
  X(NX_FLOAT16, NX_FLOAT64)                                                    \
  X(NX_BFLOAT16, NX_FLOAT64)                                                   \
  X(NX_FLOAT8_E4M3FN, NX_FLOAT64)                                              \
  X(NX_FLOAT8_E5M2, NX_FLOAT64)                                                \
  X(NX_FLOAT32, NX_FLOAT64)                                                    \
  X(NX_FLOAT64, NX_FLOAT64)                                                    \
  X(NX_BOOL, NX_INT64)                                                         \
  X(NX_INT8, NX_INT64)                                                         \
  X(NX_UINT8, NX_INT64)                                                        \
  X(NX_INT16, NX_INT64)                                                        \
  X(NX_UINT16, NX_INT64)                                                       \
  X(NX_INT32, NX_INT64)                                                        \
  X(NX_UINT32, NX_INT64)                                                       \
  X(NX_INT64, NX_INT64)                                                        \
  X(NX_UINT64, NX_INT64)

__device__ void pack_rows(const pack_params &p) {
  const u64 n = (u64)p.batch * p.rows * p.lead * p.bytes / 16;
  if (n <= UINT32_MAX) {
#define PACK_CASE(d, o)                                                        \
  if (p.dtype == d && p.out == o)                                              \
    return pack_vectors<d, o, dt_bytes(o), uint32_t>(p);
    PACKS(PACK_CASE)
#undef PACK_CASE
  }
  switch (p.bytes) {
  case 1: return pack_vectors<-1, -1, 1, u64>(p);
  case 2: return pack_vectors<-1, -1, 2, u64>(p);
  case 4: return pack_vectors<-1, -1, 4, u64>(p);
  default: return pack_vectors<-1, -1, 8, u64>(p);
  }
}

#endif
