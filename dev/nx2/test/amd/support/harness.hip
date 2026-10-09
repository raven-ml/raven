/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The harness's kernels, as harness.h states them. The host sizes their
   grids. */

#include "device.h"
#include "harness.h"
#include "nx_dtype.h"

/* The GPU's real-time clock, counted at [clock_hz]: the delay's and the
   hog's spans need it only roughly. */
DEVICE u64 now(void) { return __builtin_readsteadycounter(); }

/* HW_ID1's shader engine (bits 18-20), shader array (16) and work-group
   processor (10-13): hwreg(HW_REG_HW_ID1), whose id is 23, read whole. Its
   SIMD bits do not tell a processor's two compute units apart on gfx1201:
   the processor is the unit a wave can name. */
#define HW_ID1 ((31 << 11) | 23)
#define HW_ID1_WGP 0x001D3C00u

DEVICE uint32_t wgp_index(void) {
  return __builtin_amdgcn_s_getreg(HW_ID1) & HW_ID1_WGP;
}

DEVICE uint32_t load_agent(const uint32_t *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

/* Timing */

KERNEL void empty(void) {}

KERNEL void cu_clock(cu_clock_params p) {
  u64 c0 = __builtin_readcyclecounter(), c = c0;
  while (c - c0 < p.cycles) c = __builtin_readcyclecounter();
  *p.spun = c - c0;
}

KERNEL void delay(delay_params p) {
  u64 t0 = now();
  while (load_agent(p.flag) < p.want)
    if (now() - t0 > p.ticks) {
      *p.late = 1;
      return;
    }
}

KERNEL SIZED(NX_HOG_THREADS) void hog(hog_params p) {
  /* The most local data share a workgroup takes on gfx12: 64 KiB. */
  static __attribute__((shared)) uint32_t held[65536 / 4];
  /* 192 VGPRs a wave: a SIMD's 1,536 shared by the 8 waves of the
     workgroup's 32 that each of the processor's 4 SIMDs runs. */
  asm volatile("" ::: "v191");
  u64 t0 = now();
  if (item() == 0) {
    ((volatile uint32_t *)held)[0] = 0;
    p.wgp[group()] = wgp_index();
    __hip_atomic_fetch_add(p.started, 1u, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
  }
  while (now() - t0 < p.ticks) {
  }
}

KERNEL SIZED(32) void where(where_params p) {
  if (item() == 0) p.wgp[group()] = wgp_index();
}

/* Operands */

/* splitmix64's finaliser. */
DEVICE u64 mix(u64 x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

DEVICE double draw_float(u64 h, uint32_t draw, int spread) {
  if (draw != NX_DRAW_WIDE)
    return (double)((int64_t)(h >> 40) - (1 << 23)) / (1 << 23);
  int e = (int)((h >> 32) % (u64)(2 * spread + 1)) - spread;
  double m = 1.0 + (double)(h & 0x7FFFFFFFull) / 2147483648.0;
  return (h >> 31) & 1 ? -ldexp(m, e) : ldexp(m, e);
}

KERNEL SIZED(NX_THREADS) void generate(generate_params p) {
  u64 i = index_of(NX_THREADS);
  if (i >= p.n) return;
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
  case NX_FLOAT8_E4M3FN: ((uint8_t *)p.out)[i] = nx_double_to_e4m3fn(f); break;
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

/* Floors */

KERNEL SIZED(NX_COPY_THREADS) void floor_copy(copy_params p) {
  u64 i = index_of(NX_COPY_THREADS);
  if (i >= p.vecs) return;
  u32x4 v = 0;
#pragma unroll
  for (int j = 0; j < 3; j++)
    if (i < p.lens[j]) v ^= ((const u32x4 *)p.in[j])[i];
  ((u32x4 *)p.out)[i] = v;
}

/* A work-item's loads are issued before it reduces them; a wave of 32
   reduces through ds_bpermute, then the workgroup through its local data
   share. */
KERNEL SIZED(NX_READ_THREADS) void floor_read(read_params p) {
  static __attribute__((shared)) uint32_t waves[NX_READ_THREADS / 32];
  u64 first = (u64)group() * NX_READ_THREADS * NX_READ_VECS + item();
  uint32_t x = 0;
#pragma unroll
  for (int k = 0; k < NX_READ_VECS; k++) {
    u64 i = first + k * NX_READ_THREADS;
    if (i < p.vecs) {
      u32x4 v = ((const u32x4 *)p.in)[i];
      x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  uint32_t lane = item() % 32;
  for (uint32_t d = 16; d > 0; d /= 2)
    x ^= (uint32_t)__builtin_amdgcn_ds_bpermute((int)((lane ^ d) * 4), (int)x);
  if (lane == 0) waves[item() / 32] = x;
  barrier();
  if (item() == 0) {
    for (int w = 1; w < NX_READ_THREADS / 32; w++) x ^= waves[w];
    p.out[group()] = x;
  }
}

/* Each peak kernel's round: 16 independent instructions. */
#define CHAINS 16

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

/* Operands of small magnitude, so that no sum overflows: 2^-7. */
KERNEL SIZED(NX_THREADS) void wmma_bf16(peak_params p) {
  bf16x8 a = (__bf16)0.0078125f;
  f32x8 c[CHAINS] = {};
  for (u64 r = 0; r < p.rounds; r++)
#pragma unroll
    for (int j = 0; j < CHAINS; j++)
      c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, a, c[j]);
  float s = 0;
#pragma unroll
  for (int j = 0; j < CHAINS; j++)
#pragma unroll
    for (int k = 0; k < 8; k++) s += c[j][k];
  if (item() % 32 == 0) ((float *)p.out)[index_of(NX_THREADS) / 32] = s;
}

KERNEL SIZED(NX_THREADS) void fma_f32(peak_params p) {
  float x[CHAINS];
#pragma unroll
  for (int j = 0; j < CHAINS; j++) x[j] = (float)(item() + j);
  const float m = 0.999f, a = 0.001f;
  /* Unrolled so that the loop's own instructions take few issue slots. */
#pragma unroll 16
  for (u64 r = 0; r < p.rounds; r++)
#pragma unroll
    for (int j = 0; j < CHAINS; j++) x[j] = __builtin_fmaf(x[j], m, a);
  float s = 0;
#pragma unroll
  for (int j = 0; j < CHAINS; j++) s += x[j];
  if (item() % 32 == 0) ((float *)p.out)[index_of(NX_THREADS) / 32] = s;
}

/* Probes */

KERNEL SIZED(NX_THREADS) void div_sqrt_f32(div_sqrt_params p) {
  u64 i = index_of(NX_THREADS);
  if (i >= p.n) return;
  float a = ((const float *)p.a)[i], b = ((const float *)p.b)[i];
  ((float *)p.q)[i] = a / b;
  ((float *)p.r)[i] = sqrtf(a);
}

KERNEL SIZED(NX_THREADS) void div_sqrt_f64(div_sqrt_params p) {
  u64 i = index_of(NX_THREADS);
  if (i >= p.n) return;
  double a = ((const double *)p.a)[i], b = ((const double *)p.b)[i];
  ((double *)p.q)[i] = a / b;
  ((double *)p.r)[i] = sqrt(a);
}

/* One WMMA: lane l loads k 8 (l / 16) + e of a's and b's row l % 16, and
   holds rows 8 (l / 16) + e of column l % 16 of c and d. */
KERNEL SIZED(32) void wmma_probe(wmma_probe_params p) {
  typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
  const int l = item(), r = l % 16, k = l / 16 * 8;
  const u32x4 a = *(const u32x4 *)((const uint16_t *)p.a + r * 16 + k);
  const u32x4 b = *(const u32x4 *)((const uint16_t *)p.b + r * 16 + k);
  f32x8 c;
  for (int e = 0; e < 8; e++) c[e] = p.c[(k + e) * 16 + r];
  c = p.f16 ? __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                  __builtin_bit_cast(f16x8, a), __builtin_bit_cast(f16x8, b), c)
            : __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                  __builtin_bit_cast(bf16x8, a), __builtin_bit_cast(bf16x8, b),
                  c);
  for (int e = 0; e < 8; e++) p.d[(k + e) * 16 + r] = c[e];
}

KERNEL SIZED(NX_THREADS) void codecs(codecs_params p) {
  u64 i = index_of(NX_THREADS);
  int wide = p.dtype == NX_FLOAT16 || p.dtype == NX_BFLOAT16;
  if (i < (wide ? 1u << 16 : 1u << 8))
    p.dec[i] = nx_bits_to_float(p.dtype, (uint32_t)i);
  if (i >= p.n) return;
  int64_t c = nx_double_to_bits(p.dtype, p.x[i]);
  if (wide) ((uint16_t *)p.enc)[i] = (uint16_t)c;
  else ((uint8_t *)p.enc)[i] = (uint8_t)c;
}
