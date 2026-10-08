/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The harness's kernels, as harness.h states them. The host sizes their
   grids, and the blocks of all but the floors. */

#include "harness.h"
#include "nx_dtype.h"

typedef unsigned long long u64;

static __device__ u64 now(void) {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

static __device__ uint32_t sm_index(void) {
  uint32_t sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  return sm;
}

static __device__ u64 thread_index(void) {
  return (u64)blockIdx.x * blockDim.x + threadIdx.x;
}

/* Timing */

extern "C" __global__ void empty(void) {}

extern "C" __global__ void stamp(const __grid_constant__ stamp_params p) {
  *p.at = now();
}

extern "C" __global__ void sm_clock(const __grid_constant__ sm_clock_params p) {
  u64 t0 = now();
  long long c0 = clock64();
  while (clock64() - c0 < (long long)p.cycles) {
  }
  *p.ns = now() - t0;
}

extern "C" __global__ void delay(const __grid_constant__ delay_params p) {
  u64 t0 = now();
  while (*p.flag < p.want)
    if (now() - t0 > p.ns) {
      *p.late = 1;
      return;
    }
}

extern "C" __global__ void __launch_bounds__(1024)
    hog(const __grid_constant__ hog_params p) {
  extern __shared__ uint32_t held[];
  u64 t0 = now();
  if (threadIdx.x == 0) {
    held[0] = 0;
    p.sm[blockIdx.x] = sm_index();
    atomicAdd(p.started, 1);
  }
  while (now() - t0 < p.ns) {
  }
}

extern "C" __global__ void where(const __grid_constant__ where_params p) {
  if (threadIdx.x == 0) p.sm[blockIdx.x] = sm_index();
}

/* Operands */

/* splitmix64's finaliser. */
static __device__ u64 mix(u64 x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static __device__ double draw_float(u64 h, uint32_t draw, int spread) {
  if (draw != NX_DRAW_WIDE)
    return (double)((int64_t)(h >> 40) - (1 << 23)) / (1 << 23);
  int e = (int)((h >> 32) % (u64)(2 * spread + 1)) - spread;
  double m = 1.0 + (double)(h & 0x7FFFFFFFull) / 2147483648.0;
  return (h >> 31) & 1 ? -ldexp(m, e) : ldexp(m, e);
}

extern "C" __global__ void generate(const __grid_constant__ generate_params p) {
  for (u64 i = thread_index(); i < p.n; i += (u64)gridDim.x * blockDim.x) {
    u64 h = mix(p.seed ^ mix(i));
    double f = draw_float(h, p.draw, p.spread);
    /* NX_DRAW_SMALL's integers: [-8, 8) as signed, [0, 16) as unsigned. */
    u64 s = p.draw == NX_DRAW_SMALL ? (h & 15) - 8 : h;
    u64 u = p.draw == NX_DRAW_SMALL ? h & 15 : h;
    switch (p.dtype) {
    case NX_FLOAT64: ((double *)p.out)[i] = f; break;
    case NX_FLOAT32: ((float *)p.out)[i] = (float)f; break;
    case NX_FLOAT16: ((uint16_t *)p.out)[i] = nx_double_to_f16(f); break;
    case NX_BFLOAT16: ((uint16_t *)p.out)[i] = nx_double_to_bf16(f); break;
    case NX_FLOAT8_E4M3FN:
      ((uint8_t *)p.out)[i] = nx_double_to_e4m3fn(f);
      break;
    case NX_FLOAT8_E5M2: ((uint8_t *)p.out)[i] = nx_double_to_e5m2(f); break;
    case NX_INT64: ((u64 *)p.out)[i] = s; break;
    case NX_UINT64: ((u64 *)p.out)[i] = u; break;
    case NX_INT32: ((uint32_t *)p.out)[i] = (uint32_t)s; break;
    case NX_UINT32: ((uint32_t *)p.out)[i] = (uint32_t)u; break;
    case NX_INT16: ((uint16_t *)p.out)[i] = (uint16_t)s; break;
    case NX_UINT16: ((uint16_t *)p.out)[i] = (uint16_t)u; break;
    case NX_INT8: ((uint8_t *)p.out)[i] = (uint8_t)s; break;
    case NX_UINT8: ((uint8_t *)p.out)[i] = (uint8_t)u; break;
    case NX_BOOL: ((uint8_t *)p.out)[i] = (uint8_t)(h & 1); break;
    }
  }
}

/* Floors */

extern "C" __global__ void __launch_bounds__(NX_COPY_THREADS)
    floor_copy(const __grid_constant__ copy_params p) {
  u64 i = (u64)blockIdx.x * NX_COPY_THREADS + threadIdx.x;
  if (i >= p.vecs) return;
  uint4 v = make_uint4(0, 0, 0, 0);
#pragma unroll
  for (int j = 0; j < 3; j++)
    if (i < p.lens[j]) {
      uint4 x = ((const uint4 *)p.in[j])[i];
      v.x ^= x.x, v.y ^= x.y, v.z ^= x.z, v.w ^= x.w;
    }
  ((uint4 *)p.out)[i] = v;
}

/* A thread's loads are issued before it reduces them. */
extern "C" __global__ void __launch_bounds__(NX_READ_THREADS)
    floor_read(const __grid_constant__ read_params p) {
  __shared__ uint32_t warps[NX_READ_THREADS / 32];
  u64 first = (u64)blockIdx.x * NX_READ_THREADS * NX_READ_VECS + threadIdx.x;
  uint32_t x = 0;
#pragma unroll
  for (int k = 0; k < NX_READ_VECS; k++) {
    u64 i = first + k * NX_READ_THREADS;
    if (i < p.vecs) {
      uint4 v = ((const uint4 *)p.in)[i];
      x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  for (int d = 16; d > 0; d /= 2) x ^= __shfl_xor_sync(0xFFFFFFFFu, x, d);
  if (threadIdx.x % 32 == 0) warps[threadIdx.x / 32] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < NX_READ_THREADS / 32; w++) x ^= warps[w];
    p.out[blockIdx.x] = x;
  }
}

/* Each peak kernel's round: 16 independent instructions. */
#define CHAINS 16

#define MMA_OP(name, inst, T, K)                                          \
  static __device__ void name(T *c, const uint32_t *a, const uint32_t *b) { \
    asm volatile(inst " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "          \
                      "{%0,%1,%2,%3};\n"                                  \
                 : "+" K(c[0]), "+" K(c[1]), "+" K(c[2]), "+" K(c[3])      \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),  \
                   "r"(b[1]));                                             \
  }

MMA_OP(mma_bf16_op, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
       float, "f")
MMA_OP(mma_f16_op, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
       float, "f")
MMA_OP(mma_s8_op, "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", int,
       "r")

/* Operands of small magnitude, so that no sum overflows: 2^-7 in each
   16-bit half (bfloat16 0x3C00, float16 0x2000), 1 in each byte. */
template <typename T, typename Op>
static __device__ void mma_peak(const peak_params &p, uint32_t half, Op op) {
  uint32_t a[4] = {half, half, half, half}, b[2] = {half, half};
  T c[CHAINS][4] = {};
  for (u64 r = 0; r < p.rounds; r++)
#pragma unroll
    for (int j = 0; j < CHAINS; j++) op(c[j], a, b);
  T s = 0;
#pragma unroll
  for (int j = 0; j < CHAINS; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (threadIdx.x % 32 == 0) ((T *)p.out)[thread_index() / 32] = s;
}

extern "C" __global__ void mma_bf16(const __grid_constant__ peak_params p) {
  mma_peak<float>(p, 0x3C003C00u, mma_bf16_op);
}

extern "C" __global__ void mma_f16(const __grid_constant__ peak_params p) {
  mma_peak<float>(p, 0x20002000u, mma_f16_op);
}

extern "C" __global__ void mma_s8(const __grid_constant__ peak_params p) {
  mma_peak<int>(p, 0x01010101u, mma_s8_op);
}

template <typename T> static __device__ void fma_peak(const peak_params &p) {
  T x[CHAINS];
#pragma unroll
  for (int j = 0; j < CHAINS; j++) x[j] = (T)(threadIdx.x + j);
  const T m = (T)0.999, a = (T)0.001;
  /* Unrolled so that the loop's own instructions take few issue slots. */
#pragma unroll 16
  for (u64 r = 0; r < p.rounds; r++)
#pragma unroll
    for (int j = 0; j < CHAINS; j++) x[j] = fma(x[j], m, a);
  T s = 0;
#pragma unroll
  for (int j = 0; j < CHAINS; j++) s += x[j];
  if (threadIdx.x % 32 == 0) ((T *)p.out)[thread_index() / 32] = s;
}

extern "C" __global__ void fma_f32(const __grid_constant__ peak_params p) {
  fma_peak<float>(p);
}

extern "C" __global__ void fma_f64(const __grid_constant__ peak_params p) {
  fma_peak<double>(p);
}

/* Probes */

template <typename T>
static __device__ void div_sqrt(const div_sqrt_params &p) {
  u64 i = thread_index();
  if (i >= p.n) return;
  T a = ((const T *)p.a)[i], b = ((const T *)p.b)[i];
  ((T *)p.q)[i] = a / b;
  ((T *)p.r)[i] = sqrt(a);
}

extern "C" __global__ void div_sqrt_f32(const __grid_constant__ div_sqrt_params p) {
  div_sqrt<float>(p);
}

extern "C" __global__ void div_sqrt_f64(const __grid_constant__ div_sqrt_params p) {
  div_sqrt<double>(p);
}

extern "C" __global__ void codecs(const __grid_constant__ codecs_params p) {
  u64 i = thread_index();
  int wide = p.dtype == NX_FLOAT16 || p.dtype == NX_BFLOAT16;
  if (i < (wide ? 1u << 16 : 1u << 8))
    p.dec[i] = nx_bits_to_float(p.dtype, (uint32_t)i);
  if (i >= p.n) return;
  int64_t c = nx_double_to_bits(p.dtype, p.x[i]);
  if (wide) ((uint16_t *)p.enc)[i] = (uint16_t)c;
  else ((uint8_t *)p.enc)[i] = (uint8_t)c;
}
