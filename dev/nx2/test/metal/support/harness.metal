/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The harness's kernels, as harness.h states them: the floors each row is
   judged against, the operand generator, the peaks, and the probes of what
   the compiler and the GPU do to float arithmetic under the build's
   options. */

#include "harness.h"
#include "nx_dtype.h"

using namespace metal;

#define THREADS NX_HARNESS_THREADS

kernel void empty() {}

/* Floors: one vector a thread, so an L2-sized array spreads over every
   core; from DRAM it moves as fast as wider threads do. */

template <uint INS> static void stream(constant move_params &p, uint i) {
  if (i >= p.n) return;
  uint4 v = ((device const uint4 *)p.in[0])[i];
  for (uint j = 1; j < INS; j++) v ^= ((device const uint4 *)p.in[j])[i];
  ((device uint4 *)p.out)[i] = v;
}

kernel void move(constant move_params &p [[buffer(0)]],
                 uint i [[thread_position_in_grid]]) {
  switch (p.ins) {
  case 1: stream<1>(p, i); break;
  case 2: stream<2>(p, i); break;
  default: stream<3>(p, i);
  }
}

kernel void read(constant move_params &p [[buffer(0)]],
                 uint g [[threadgroup_position_in_grid]],
                 uint t [[thread_position_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint4 partial[THREADS / 32];
  uint i = g * THREADS + t;
  uint4 x = simd_xor(i < p.n ? ((device const uint4 *)p.in[0])[i] : uint4(0));
  if (lane == 0) partial[sg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  for (uint s = 1; s < THREADS / 32; s++) x ^= partial[s];
  ((device uint4 *)p.out)[g] = x;
}

/* Operands */

/* A 32-bit integer hash (Wellons' lowbias32): consecutive inputs give
   independent-looking outputs. */
static uint hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

kernel void generate(constant generate_params &p [[buffer(0)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint r = hash(p.seed ^ hash(2 * i)), s = hash(p.seed ^ hash(2 * i + 1));
  int e = p.spread == 0 ? 0 : int(s % (2 * p.spread + 1)) - int(p.spread);
  float f = ldexp(as_type<float>(0x3f800000u | (r & 0x7fffffu)), e);
  f = (s >> 31) ? -f : f;
  uint b = as_type<uint>(f);
  switch (p.dtype) {
  case NX_FLOAT64: {
    /* f is normal: its exponent rebiases and its significand widens. */
    ulong x = ((b >> 23) & 0xffu) - 127 + 1023;
    ((device ulong *)p.out)[i] =
        ulong(b >> 31) << 63 | x << 52 | ulong(b & 0x7fffffu) << 29;
    break;
  }
  case NX_FLOAT32: ((device float *)p.out)[i] = f; break;
  case NX_FLOAT16: ((device ushort *)p.out)[i] = nx_float_to_f16(f); break;
  case NX_BFLOAT16: ((device ushort *)p.out)[i] = nx_float_to_bf16(f); break;
  case NX_INT64:
  case NX_UINT64: ((device ulong *)p.out)[i] = ulong(r) << 32 | s; break;
  case NX_INT32:
  case NX_UINT32: ((device uint *)p.out)[i] = r; break;
  case NX_INT16:
  case NX_UINT16: ((device ushort *)p.out)[i] = ushort(r); break;
  case NX_INT8:
  case NX_UINT8: ((device uchar *)p.out)[i] = uchar(r); break;
  }
}

/* Peaks: [iters] rounds over 32 chains a thread, as eight 4-vectors, the
   chains' sum stored so that no round is dead. The fma peak is the SIMD
   units' float32 issue rate, which saturates at about 4.7 TF/s on the M1
   Max whatever the chains; the matrix units reach 10. */

template <typename T> static void fmas(constant spin_params &p, uint i) {
  typedef vec<T, 4> V;
  device T *out = (device T *)p.out;
  /* |x| < 1, so the chains stay finite and normal. */
  T x = fract(out[i]) * T(0.5) + T(0.25), y = out[i];
  V a0 = V(x, x + T(1), x + T(2), x + T(3)), a1 = a0 + T(4), a2 = a0 + T(8),
    a3 = a0 + T(12), a4 = a0 + T(16), a5 = a0 + T(20), a6 = a0 + T(24),
    a7 = a0 + T(28);
  for (uint k = 0; k < p.iters; k++) {
    a0 = fma(a0, x, y), a1 = fma(a1, x, y), a2 = fma(a2, x, y);
    a3 = fma(a3, x, y), a4 = fma(a4, x, y), a5 = fma(a5, x, y);
    a6 = fma(a6, x, y), a7 = fma(a7, x, y);
  }
  V s = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
  out[i] = (s.x + s.y) + (s.z + s.w);
}

kernel void fma_f32(constant spin_params &p [[buffer(0)]],
                    uint i [[thread_position_in_grid]]) {
  fmas<float>(p, i);
}

kernel void fma_f16(constant spin_params &p [[buffer(0)]],
                    uint i [[thread_position_in_grid]]) {
  fmas<half>(p, i);
}

/* An 8x8x8 product per simdgroup, chain and round: 1,024 flops. Each
   simdgroup reads and writes the 64 values at its own index. */

#define CHAINS 8

template <typename T> static void mma(constant spin_params &p, uint sg) {
  device T *out = (device T *)p.out + 64 * sg;
  simdgroup_matrix<T, 8, 8> a, b, c[CHAINS];
  simdgroup_load(a, out, 8);
  simdgroup_load(b, out, 8, ulong2(0), true);
  for (uint j = 0; j < CHAINS; j++) c[j] = a;
  for (uint k = 0; k < p.iters; k++)
    for (uint j = 0; j < CHAINS; j++)
      simdgroup_multiply_accumulate(c[j], a, b, c[j]);
  for (uint j = 1; j < CHAINS; j++)
    simdgroup_multiply_accumulate(c[0], c[j], b, c[0]);
  simdgroup_store(c[0], out, 8);
}

kernel void mma_f32(constant spin_params &p [[buffer(0)]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint g [[threadgroup_position_in_grid]]) {
  mma<float>(p, g * (THREADS / 32) + sg);
}

kernel void mma_f16(constant spin_params &p [[buffer(0)]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint g [[threadgroup_position_in_grid]]) {
  mma<half>(p, g * (THREADS / 32) + sg);
}

/* Probes: each reads its operand tuples and writes one 32-bit result per
   tuple, which the host compares with its own. */

/* which 0: a·b + c written as one expression, which rounds twice unless
   the compiler contracts it; 1: fma(a, b, c), which rounds once. */
kernel void probe_contract(constant probe_params &p [[buffer(0)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  device const float *t = (device const float *)p.in + 3 * i;
  float a = t[0], b = t[1], c = t[2];
  ((device float *)p.out)[i] = p.which == 0 ? a * b + c : fma(a, b, c);
}

/* which 0: x / y; 1: sqrt(x). */
kernel void probe_div_sqrt(constant probe_params &p [[buffer(0)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  device const float *t = (device const float *)p.in + 2 * i;
  float x = t[0], y = t[1];
  ((device float *)p.out)[i] = p.which == 0 ? x / y : sqrt(x);
}

/* On the word x: which 0, the bits of half(x); 1, the bits of the float
   of the half bits x holds in its low 16 bits; 2, the bits of h + h, h
   those half bits, a sum in half arithmetic. */
kernel void probe_half(constant probe_params &p [[buffer(0)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint x = ((device const uint *)p.in)[i];
  half h = as_type<half>(ushort(x));
  ((device uint *)p.out)[i] =
      p.which == 0   ? uint(as_type<ushort>(half(as_type<float>(x))))
      : p.which == 1 ? as_type<uint>(float(h))
                     : uint(as_type<ushort>(half(h + h)));
}

/* nx_dtype.h's codecs on the GPU, for the narrow format [dtype]: which 0,
   the bits of the float the code x decodes to; 1, the code x's bits, read
   as a float, encode to. */
kernel void probe_codec(constant probe_params &p [[buffer(0)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint x = ((device const uint *)p.in)[i];
  float f = as_type<float>(x);
  uint r = 0;
  switch (p.dtype) {
  case NX_FLOAT16:
    r = p.which == 0 ? nx_float_bits(nx_f16_to_float(x)) : nx_float_to_f16(f);
    break;
  case NX_BFLOAT16:
    r = p.which == 0 ? nx_float_bits(nx_bf16_to_float(x)) : nx_float_to_bf16(f);
    break;
  case NX_FLOAT8_E4M3FN:
    r = p.which == 0 ? nx_float_bits(nx_e4m3fn_to_float(x)) : nx_float_to_e4m3fn(f);
    break;
  case NX_FLOAT8_E5M2:
    r = p.which == 0 ? nx_float_bits(nx_e5m2_to_float(x)) : nx_float_to_e5m2(f);
    break;
  case NX_FLOAT4_E2M1FN:
    r = p.which == 0 ? nx_float_bits(nx_e2m1fn_to_float(x)) : nx_float_to_e2m1fn(f);
    break;
  }
  ((device uint *)p.out)[i] = r;
}
