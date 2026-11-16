/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The AMD suite's harness: kernels that probe what nx.amd's launch path
   relies on, compiled by nx.amd's gen.py with nx.amd's flags. A kernel
   takes its parameters as 64-bit words read through the kernarg pointer,
   its addresses first, as nx.amd's kernels do. */

#include <stdint.h>

#define KERNEL extern "C" __attribute__((global))
#define SIZED(n) __attribute__((amdgpu_flat_work_group_size(n, n)))
#define DEVICE static inline __attribute__((device, always_inline))
#define SHARED __attribute__((shared))

template <int n> struct Words {
  int64_t w[n];
};

DEVICE const int64_t *words(void) {
  return (const int64_t *)__builtin_amdgcn_kernarg_segment_ptr();
}

template <typename T> DEVICE T *at(int i) { return (T *)(uintptr_t)words()[i]; }

DEVICE uint32_t item(void) { return __builtin_amdgcn_workitem_id_x(); }

/* Kernel arguments: word 0 the address of a 64-bit result, the sum of
   words 1 to n - 1 wrapping. Of 256 bytes, 1 KiB, 4 KiB and 8 KiB. */

#define ARGS(name, n)                                                         \
  KERNEL SIZED(32) void name(Words<n>) {                                      \
    if (item() != 0) return;                                                  \
    int64_t s = 0;                                                            \
    for (int i = 1; i < n; i++) s += words()[i];                              \
    *at<int64_t>(0) = s;                                                      \
  }

ARGS(args_256, 32)
ARGS(args_1k, 128)
ARGS(args_4k, 512)
ARGS(args_8k, 1024)

/* Tickets: each of the grid's groups adds its group's value, g + 1 for
   group g, into [partials] at g, then takes the ticket at word 1's address;
   the last stores the ticket back to zero, then the sum of the partials
   into word 2's address, and adds 1 to the count of last groups at word
   3's. Words 4 and 5 are the groups and how many times each group spins
   on its own clock before it stores, so that groups finish out of order.
   It probes, under that disorder, the hardware's order of the protocol
   nx_amd.mli's sums count on: agent-scope fences around an acq_rel
   fetch_add, with workgroup barriers between. */

#define TICKET_THREADS 256

DEVICE void barrier(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

KERNEL SIZED(TICKET_THREADS) void tickets(Words<6>) {
  static SHARED bool is_last;
  const int64_t *w = words();
  uint64_t *partials = at<uint64_t>(0);
  uint32_t *ticket = at<uint32_t>(1);
  const uint32_t g = __builtin_amdgcn_workgroup_id_x();
  const uint32_t n = (uint32_t)w[4];
  for (int64_t i = 0; i < (w[5] * (g % 7)); i++)
    __builtin_amdgcn_s_sleep(1);
  if (item() == 0) partials[g] = g + 1;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (item() == 0) {
    is_last = __hip_atomic_fetch_add(ticket, 1u, __ATOMIC_ACQ_REL,
                                     __HIP_MEMORY_SCOPE_AGENT) == n - 1;
    if (is_last)
      __hip_atomic_store(ticket, 0u, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
  }
  barrier();
  if (!is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (item() != 0) return;
  uint64_t s = 0;
  for (uint32_t i = 0; i < n; i++) s += partials[i];
  *at<uint64_t>(2) = s;
  __hip_atomic_fetch_add(at<uint32_t>(3), 1u, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
}

/* Grids: group (x, y, z) of a grid of words 1 to 3 groups adds 1 to the
   32-bit word x + gx (y + gy z) from word 0's address. */

KERNEL SIZED(32) void grid(Words<4>) {
  if (item() != 0) return;
  const int64_t *w = words();
  const uint64_t x = __builtin_amdgcn_workgroup_id_x();
  const uint64_t y = __builtin_amdgcn_workgroup_id_y();
  const uint64_t z = __builtin_amdgcn_workgroup_id_z();
  at<uint32_t>(0)[x + w[1] * (y + w[2] * z)] += 1;
}

/* Timing, operands and floors: the kernels the AMD bench runs beside
   nx.amd's. Times are counts of the GPU's clock, which runs at the
   capability's clock_hz (rig_amd_abi's Capability): the counter a queue
   stamps, which s_sendmsg_rtn reads as REALTIME. */

DEVICE uint64_t now(void) { return __builtin_readsteadycounter(); }

/* A word of host memory the GPU maps, read and written past the GPU's
   caches. */
DEVICE uint64_t load_host(const uint64_t *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

DEVICE void store_host(uint64_t *p, uint64_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

/* Nothing: a launch's own cost. */
KERNEL SIZED(32) void empty(void) {}

/* Writes the clock to word 0's address. One work-item. Word 1 addresses a
   buffer it does not touch, which orders it against the parts that write
   that buffer. */
KERNEL SIZED(32) void stamp(Words<2>) {
  if (item() == 0) store_host(at<uint64_t>(0), now());
}

/* Spins one work-item for word 1's count of its shader's cycles, then
   writes the cycles it spun and the clock's count meanwhile to the two
   64-bit words at word 0's address. */
KERNEL SIZED(32) void shader_clock(Words<2>) {
  if (item() != 0) return;
  const uint64_t cycles = (uint64_t)words()[1];
  const uint64_t t0 = now(), c0 = __builtin_readcyclecounter();
  uint64_t c;
  do c = __builtin_readcyclecounter(); while (c - c0 < cycles);
  const uint64_t t1 = now();
  at<uint64_t>(0)[0] = c - c0;
  at<uint64_t>(0)[1] = t1 - t0;
}

/* Returns once the host word at word 0's address is at least word 2, or
   once word 3's count of the clock has passed, then setting the host word
   at word 1's address to 1. One work-item. The work behind it is queued
   meanwhile. */
KERNEL SIZED(32) void delay(Words<4>) {
  if (item() != 0) return;
  const uint64_t want = (uint64_t)words()[2], bound = (uint64_t)words()[3];
  const uint64_t t0 = now();
  while (load_host(at<uint64_t>(0)) < want) {
    if (now() - t0 > bound) {
      store_host(at<uint64_t>(1), 1);
      return;
    }
    __builtin_amdgcn_s_sleep(2);
  }
}

/* Operands. nx_dtype.h's encoders inlined: no kernel calls a function. */

#pragma clang attribute push(__attribute__((always_inline)), apply_to = function)
#include "nx_dtype.h"
#pragma clang attribute pop

/* How generate draws a value: floats uniform in [-1, 1) to 24 bits, then
   rounded to the dtype, integers uniform over the dtype's range
   (UNIFORM); floats of a random sign, an exponent uniform in [-spread,
   spread] and a random significand, integers as UNIFORM (WIDE); integers
   in [lo, lo + 16), floats as UNIFORM (SMALL). */
enum { DRAW_UNIFORM, DRAW_WIDE, DRAW_SMALL };

/* splitmix64's finaliser. */
DEVICE uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

DEVICE double draw_float(uint64_t h, int64_t draw, int64_t spread) {
  if (draw != DRAW_WIDE)
    return (double)((int64_t)(h >> 40) - (1 << 23)) / (1 << 23);
  int e = (int)((h >> 32) % (uint64_t)(2 * spread + 1)) - (int)spread;
  double m = 1.0 + (double)(h & 0x7FFFFFFFull) / 2147483648.0;
  return (h >> 31) & 1 ? -__builtin_ldexp(m, e) : __builtin_ldexp(m, e);
}

/* The value of the bits [v] of a float dtype's element, and the smallest
   normal value of the dtype. */
DEVICE double float_value(int64_t dtype, uint64_t v) {
  if (dtype == NX_FLOAT64) return nx_bits_double(v);
  if (dtype == NX_FLOAT32) return nx_bits_float((uint32_t)v);
  return nx_bits_to_float((int)dtype, (uint32_t)v);
}

DEVICE double smallest_normal(int64_t dtype) {
  switch (dtype) {
  case NX_FLOAT64: return 0x1p-1022;
  case NX_FLOAT32: case NX_BFLOAT16: return 0x1p-126;
  case NX_FLOAT16: case NX_FLOAT8_E5M2: return 0x1p-14;
  default: return 0x1p-6; /* float8_e4m3fn */
  }
}

/* Element i of the elements at word 0's address, of the dtype of code
   word 3 and word 6 bytes, is a draw of word 4 (word 5 its spread) from
   the hash of seed word 2 and i, for i below word 1: word 7 the low end of
   a small integer, word 8 nonzero for a float narrower than float32. Over
   word 9 groups. Dtypes of a byte or wider. A float that rounds to a
   subnormal is zero, and a uniform or small one that rounds to a magnitude
   of 1
   takes the dtype's next value toward zero, which a float's magnitude
   bits below its sign give. */
KERNEL SIZED(256) void generate(Words<10>) {
  const int64_t *w = words();
  uint8_t *out = at<uint8_t>(0);
  const uint64_t n = w[1], seed = w[2], groups = w[9];
  const int64_t dtype = w[3], draw = w[4], spread = w[5], bytes = w[6];
  for (uint64_t i = __builtin_amdgcn_workgroup_id_x() * 256ull + item();
       i < n; i += groups * 256) {
    uint64_t h = mix(seed ^ mix(i));
    double f = draw_float(h, draw, spread);
    uint64_t v = draw == DRAW_SMALL ? (uint64_t)((int64_t)(h & 15) + w[7]) : h;
    if (dtype == NX_FLOAT64) v = nx_double_bits(f);
    else if (dtype == NX_FLOAT32) v = nx_float_bits((float)f);
    else if (dtype == NX_BOOL) v = h & 1;
    else if (w[8]) v = (uint64_t)nx_double_to_bits((int)dtype, f);
    const int is_float = dtype == NX_FLOAT64 || dtype == NX_FLOAT32 || w[8];
    if (is_float) {
      const double x = __builtin_fabs(float_value(dtype, v));
      if (x != 0 && x < smallest_normal(dtype)) v = 0;
      else if (draw != DRAW_WIDE && x >= 1) v -= 1;
    }
    switch (bytes) {
    case 1: out[i] = (uint8_t)v; break;
    case 2: ((uint16_t *)out)[i] = (uint16_t)v; break;
    case 4: ((uint32_t *)out)[i] = (uint32_t)v; break;
    default: ((uint64_t *)out)[i] = v;
    }
  }
}

/* Floors: the least time a row's bytes take to move. Each work-item moves
   16-byte vectors, over a grid of groups that strides. */

typedef uint32_t vec __attribute__((ext_vector_type(4)));

/* Word [i]'s address as global memory, which global loads and stores
   reach: a generic address would take flat ones. */
#define GLOBAL __attribute__((address_space(1)))
template <typename T> DEVICE GLOBAL T *global(int i) {
  return (GLOBAL T *)(uintptr_t)words()[i];
}

/* Vector i of word 3's address, for i below word 7, is the XOR of vector i
   of the inputs at words 0 to 2's addresses, input j read only below word
   4 + j vectors: the floor of a row that reads them and writes it. Over
   word 8 groups. */
KERNEL SIZED(256) void floor_copy(Words<9>) {
  const int64_t *w = words();
  const uint64_t vecs = w[7], step = w[8] * 256;
  for (uint64_t i = __builtin_amdgcn_workgroup_id_x() * 256ull + item();
       i < vecs; i += step) {
    vec v = {0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < 3; j++)
      if (i < (uint64_t)w[4 + j]) v ^= global<const vec>(j)[i];
    global<vec>(3)[i] = v;
  }
}

/* Reads word 2 vectors at word 0's address, four a work-item a step, their
   loads issued before it combines them; group g writes the XOR of the
   vectors it read to the 32-bit word g at word 1's address: the floor of a
   reduction. Over word 3 groups. */
KERNEL SIZED(256) void floor_read(Words<4>) {
  static SHARED uint32_t parts[256];
  const int64_t *w = words();
  GLOBAL const vec *in = global<const vec>(0);
  const uint64_t vecs = w[2], threads = w[3] * 256;
  const uint32_t g = __builtin_amdgcn_workgroup_id_x();
  vec x = {0, 0, 0, 0};
  for (uint64_t i = g * 256ull + item(); i < vecs; i += 4 * threads) {
    vec v[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      uint64_t j = i + k * threads;
      v[k] = j < vecs ? in[j] : (vec){0, 0, 0, 0};
    }
#pragma unroll
    for (int k = 0; k < 4; k++) x ^= v[k];
  }
  parts[item()] = x.x ^ x.y ^ x.z ^ x.w;
  barrier();
  for (uint32_t s = 128; s > 0; s /= 2) {
    if (item() < s) parts[item()] ^= parts[item() + s];
    barrier();
  }
  if (item() == 0) global<uint32_t>(1)[g] = parts[0];
}

/* Peaks: each wave runs word 1 rounds of 16 steps, 8 on each of two
   chains that depend on nothing in memory, and its work-items store their
   sums to the 32-byte slots at word 0's address: the GPU's rate at one
   kind of instruction. A step is a 16x16x16 matrix instruction, 8,192
   flops a wave, or an fma of each element of a 32-byte vector: 512 flops
   a wave in float32, 256 in float64. */

typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef __bf16 b8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int32_t i2 __attribute__((ext_vector_type(2)));
typedef int32_t i8 __attribute__((ext_vector_type(8)));
typedef double d4 __attribute__((ext_vector_type(4)));

DEVICE uint64_t thread(void) {
  return __builtin_amdgcn_workgroup_id_x() * 256ull + item();
}

#define PEAK(name, T, operand, C, step)                                       \
  KERNEL SIZED(256) void name(Words<2>) {                                     \
    const int64_t rounds = words()[1];                                        \
    const T a = operand;                                                      \
    C c0 = {}, c1 = {};                                                       \
    for (int64_t r = 0; r < rounds; r++) {                                    \
      _Pragma("unroll") for (int k = 0; k < 8; k++) {                         \
        c0 = step(c0);                                                        \
        c1 = step(c1);                                                        \
      }                                                                       \
    }                                                                         \
    at<C>(0)[thread()] = c0 + c1;                                             \
  }

#define WMMA_F16(c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, a, c)
#define WMMA_BF16(c) __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, a, c)
#define WMMA_FP8(c) __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, a, c)
#define WMMA_IU8(c)                                                           \
  __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, a, c, false)
#define FMA(c) __builtin_elementwise_fma(a, c, a)

PEAK(peak_wmma_f16, h8, (h8)(_Float16)(item() * 0x1p-8f), f8, WMMA_F16)
PEAK(peak_wmma_bf16, b8, (b8)(__bf16)(item() * 0x1p-8f), f8, WMMA_BF16)
/* 8 bytes hashed from the work-item's index, none of them a NaN of
   float8_e4m3fn (0x7F or 0xFF): operands that switch as data does. */
DEVICE i2 bytes8(void) {
  uint64_t h = mix(thread());
  for (int k = 0; k < 8; k++)
    if (((h >> (8 * k)) & 0x7F) == 0x7F) h ^= 1ull << (8 * k);
  return (i2){(int32_t)h, (int32_t)(h >> 32)};
}

PEAK(peak_wmma_fp8, i2, bytes8(), f8, WMMA_FP8)
PEAK(peak_wmma_iu8, i2, bytes8(), i8, WMMA_IU8)
PEAK(peak_fma_f32, f8, (f8)(item() * 0x1p-8f), f8, FMA)
PEAK(peak_fma_f64, d4, (d4)(item() * 0x1p-8), d4, FMA)
/* One matrix instruction (WMMA, 16 x 16 outputs) a group, of the kind word
   0's 32-bit word at the group's index names, on raw register images, so
   that a test states the lane layout rather than this kernel. Group g's
   lane l reads its four A words from word 1's address at 128 g + 4 l, its
   four B words from word 2's at the same index, its eight C words from
   word 3's at 256 g + 8 l, and stores its eight D words at word 4's at the
   same index. A kind takes as many of its A and B words as it reads,
   first ones first:

   0 float16, 1 bfloat16 (four words, two elements a word, low half
   first), 2 to 5 float8 pairs e4m3·e4m3, e4m3·e5m2, e5m2·e4m3, e5m2·e5m2
   (two words, four elements a word, low byte first), into float32; 6 to 9
   8-bit integers (two words) and 10 to 13 4-bit integers (one word, eight
   a word, low nibble first) of k 16, 14 to 17 4-bit integers of k 32 (two
   words), each four with A then B signed, signed, unsigned, unsigned,
   alternating, into int32, wrapping. */

typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

#define IU(f, sa, sb, a, b) f(sa, a, sb, b, ci, false)

KERNEL SIZED(32) void wmma(Words<5>) {
  const uint32_t g = __builtin_amdgcn_workgroup_id_x(), l = item();
  const uint32_t kind = at<uint32_t>(0)[g];
  const u32x4 a = at<u32x4>(1)[32 * g + l];
  const u32x4 b = at<u32x4>(2)[32 * g + l];
  const f32x8 c = at<f32x8>(3)[32 * g + l];
  const i32x8 ci = __builtin_bit_cast(i32x8, c);
  const i32x2 a2 = {(int)a[0], (int)a[1]}, b2 = {(int)b[0], (int)b[1]};
  f32x8 d;
  switch (kind) {
  case 0:
    d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        __builtin_bit_cast(f16x8, a), __builtin_bit_cast(f16x8, b), c);
    break;
  case 1:
    d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
        __builtin_bit_cast(bf16x8, a), __builtin_bit_cast(bf16x8, b), c);
    break;
  case 2: d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a2, b2, c); break;
  case 3: d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(a2, b2, c); break;
  case 4: d = __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(a2, b2, c); break;
  case 5: d = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a2, b2, c); break;
#define IU8(n, sa, sb)                                                        \
  case n:                                                                     \
    d = __builtin_bit_cast(                                                   \
        f32x8, IU(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12, sa, sb,   \
                  a2, b2));                                                   \
    break;
#define IU4(n, sa, sb)                                                        \
  case n:                                                                     \
    d = __builtin_bit_cast(                                                   \
        f32x8, IU(__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12, sa, sb,   \
                  (int)a[0], (int)b[0]));                                     \
    break;
#define IU4K32(n, sa, sb)                                                     \
  case n:                                                                     \
    d = __builtin_bit_cast(                                                   \
        f32x8, IU(__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12, sa, sb,   \
                  a2, b2));                                                   \
    break;
    IU8(6, true, true) IU8(7, true, false) IU8(8, false, true)
    IU8(9, false, false)
    IU4(10, true, true) IU4(11, true, false) IU4(12, false, true)
    IU4(13, false, false)
    IU4K32(14, true, true) IU4K32(15, true, false) IU4K32(16, false, true)
    IU4K32(17, false, false)
  default: return;
  }
  at<f32x8>(4)[32 * g + l] = d;
}
