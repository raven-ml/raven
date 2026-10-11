/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The kernels the CUDA suite and bench run beside nx.cuda's: the timing
   kernels, the operand generator, the floors, and the probes. harness.cu
   defines them; the host launches them as rig's launches, as nx.cuda
   launches its own.

   Each kernel reads one parameter struct of 8-byte words, its addresses
   first. */

#ifndef NX_CUDA_HARNESS_H
#define NX_CUDA_HARNESS_H

#include <stdint.h>

/* X(name) for every kernel of the harness's cubin, in enum order. */
#define NX_HARNESS_KERNELS(X)                                          \
  X(empty) X(stamp) X(sm_clock) X(delay) X(generate)                   \
  X(floor_copy) X(floor_read)                                          \
  X(mma_bf16) X(mma_f16) X(mma_s8) X(fma_f32) X(fma_f64)               \
  X(div_sqrt_f32) X(div_sqrt_f64) X(codecs)

#define NX_HARNESS_ENUM(name) NX_HARNESS_##name,
enum nx_harness_kernel { NX_HARNESS_KERNELS(NX_HARNESS_ENUM) NX_HARNESS_COUNT };
#undef NX_HARNESS_ENUM

/* Timing */

/* Writes the GPU's timer, in nanoseconds, to [at]. One thread. */
typedef struct {
  uint64_t *at;
} stamp_params;

/* Spins one thread for [cycles] cycles of its SM's clock and writes the
   nanoseconds they took to [*ns]. */
typedef struct {
  uint64_t *ns;
  uint64_t cycles;
} sm_clock_params;

/* Returns once the 32-bit word at [flag] is at least [want], or after
   [ns] nanoseconds, then setting the 32-bit word at [late]. One thread.
   The work behind it in its stream is queued meanwhile. */
typedef struct {
  volatile uint32_t *flag;
  uint32_t *late;
  uint32_t want, unused;
  uint64_t ns;
} delay_params;

/* Operands */

/* How generate draws a value. */
enum nx_harness_draw {
  /* Floats uniform in [-1, 1) to 24 bits, then rounded to the dtype;
     integers uniform over the dtype's range. */
  NX_DRAW_UNIFORM,
  /* Floats of a random sign, exponent uniform in [-spread, spread] and a
     random significand: wide spread, and cancellation in sums. Integers as
     NX_DRAW_UNIFORM. */
  NX_DRAW_WIDE,
  /* Integers uniform in [-8, 8) (unsigned: [0, 16)); floats as
     NX_DRAW_UNIFORM. Small integers keep integer contractions from
     wrapping, or let a test choose to. */
  NX_DRAW_SMALL
};

/* Element i of [out], of the dtype [dtype] (nx_dtype.h's code), [bytes]
   bytes, is a draw of [draw] from the hash of [seed] and i, for i < n:
   [narrow] for a float narrower than float32, [lo] -8 for a signed integer
   and 0 for an unsigned one. Byte-wide or wider dtypes. */
typedef struct {
  void *out;
  uint64_t n, seed;
  uint32_t dtype, draw;
  int32_t spread, bytes, lo, narrow;
} generate_params;

/* Floors */

/* floor_copy: out[i] = in[0][i] ^ in[1][i] ^ in[2][i] over [vecs] 16-byte
   vectors, input j read only below its [lens[j]] vectors: the copy floor
   of a row that reads and writes those bytes. Blocks of NX_COPY_THREADS
   threads, a vector each: an L2-sized row spreads over every SM. */
#define NX_COPY_THREADS 128

typedef struct {
  const void *in[3];
  void *out;
  uint64_t lens[3], vecs;
} copy_params;

/* floor_read: reads [vecs] 16-byte vectors at [in]; each block writes the
   XOR of its vectors to out[block], the floor of a reduction. Blocks of
   NX_READ_THREADS threads, NX_READ_VECS vectors each. */
#define NX_READ_THREADS 256
#define NX_READ_VECS 4

typedef struct {
  const void *in;
  uint32_t *out;
  uint64_t vecs;
} read_params;

/* Each warp runs [rounds] rounds of the kernel's instruction on registers,
   then writes a value of its accumulators to out[warp], so that nothing
   is dead. A round is 16 independent instructions: mma.sync.m16n8k16 with
   float32 accumulators (mma_bf16, mma_f16), mma.sync.m16n8k32 with int32
   ones (mma_s8), or 32 lanes' fma (fma_f32, fma_f64). */
typedef struct {
  void *out;
  uint64_t rounds;
} peak_params;

/* Probes */

/* q[i] = a[i] / b[i], r[i] = sqrt(a[i]), i < n. */
typedef struct {
  const void *a, *b;
  void *q, *r;
  uint64_t n;
} div_sqrt_params;

/* For the narrow float dtype [dtype] (float16, bfloat16, float8 e4m3fn or
   e5m2): dec[c] is the value of the code c, for every code of the dtype,
   and enc[i] the code x[i] stores as, for i < n: nx_dtype.h's codecs as
   the device compiles them. */
typedef struct {
  const double *x;
  double *dec;
  void *enc;
  uint64_t n;
  uint32_t dtype, unused;
} codecs_params;

#endif
