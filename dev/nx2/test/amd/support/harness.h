/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The kernels the AMD suite and bench run beside nx.amd's: the timing
   kernels, the operand generator, the floors, and the probes. harness.hip
   defines them; the host launches them through launch records, as it
   launches the library's.

   Each kernel reads one parameter struct, its addresses first. A kernel
   reads no implicit argument, so its workgroup size is fixed: those whose
   indices depend on it say how many work-items a workgroup has. Times are
   ticks of the GPU's clock, which runs at the device's clock_hz
   (Rig_amd_abi.Capability). */

#ifndef NX_AMD_HARNESS_H
#define NX_AMD_HARNESS_H

#include <stdint.h>

/* X(name) for every kernel of the harness's code object, in enum order. */
#define NX_HARNESS_KERNELS(X)                                     \
  X(empty) X(cu_clock) X(delay) X(hog) X(where) X(generate)       \
  X(floor_copy) X(floor_read) X(wmma_bf16) X(fma_f32)             \
  X(div_sqrt_f32) X(div_sqrt_f64) X(codecs)

#define NX_HARNESS_ENUM(name) NX_HARNESS_##name,
enum nx_harness_kernel { NX_HARNESS_KERNELS(NX_HARNESS_ENUM) NX_HARNESS_COUNT };
#undef NX_HARNESS_ENUM

/* The work-items of a workgroup of generate, the peaks and the probes. */
#define NX_THREADS 256

/* Timing */

/* Spins one work-item for at least [cycles] cycles of its compute unit's
   clock and writes the cycles it spun to [*spun]. */
typedef struct {
  uint64_t *spun;
  uint64_t cycles;
} cu_clock_params;

/* Returns once the 32-bit word at [flag] is at least [want], or after
   [ticks] ticks, then setting the 32-bit word at [late]. One work-item.
   The work behind it on its queue waits meanwhile. */
typedef struct {
  uint32_t *flag, *late;
  uint32_t want, unused;
  uint64_t ticks;
} delay_params;

/* Each workgroup of NX_HOG_THREADS work-items holds its compute unit for
   [ticks] ticks from its start, when it writes its unit (where's) to
   cu[group] and adds 1 to [*started]. It runs in CU mode with the largest
   local data share a workgroup takes, so that it fills one compute unit's
   waves and its half of the processor's local data share. Work queued
   behind a delay until [*started] counts every workgroup runs only on the
   compute units the hog left free. */
#define NX_HOG_THREADS 1024

typedef struct {
  uint32_t *started, *cu;
  uint64_t ticks;
} hog_params;

/* Each workgroup of one wave writes its compute unit to cu[group]: the
   shader engine, shader array, work-group processor and its compute unit,
   as bits of the wave's HW_ID1 register. */
typedef struct {
  uint32_t *cu;
} where_params;

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

/* Element i of [out], of the dtype [dtype] (nx_dtype.h's code), is a draw
   of [draw] from the hash of [seed] and i, for i < n: a work-item per
   element. Byte-wide or wider dtypes. */
typedef struct {
  void *out;
  uint64_t n, seed;
  uint32_t dtype, draw;
  int32_t spread, unused;
} generate_params;

/* Floors */

/* floor_copy: out[i] = in[0][i] ^ in[1][i] ^ in[2][i] over [vecs] 16-byte
   vectors, input j read only below its [lens[j]] vectors: the copy floor
   of a row that reads and writes those bytes. Workgroups of
   NX_COPY_THREADS work-items, a vector each: an L2-sized row spreads over
   every compute unit. */
#define NX_COPY_THREADS 128

typedef struct {
  const void *in[3];
  void *out;
  uint64_t lens[3], vecs;
} copy_params;

/* floor_read: reads [vecs] 16-byte vectors at [in]; each workgroup writes
   the XOR of its vectors to out[group], the floor of a reduction.
   Workgroups of NX_READ_THREADS work-items, NX_READ_VECS vectors each. */
#define NX_READ_THREADS 256
#define NX_READ_VECS 4

typedef struct {
  const void *in;
  uint32_t *out;
  uint64_t vecs;
} read_params;

/* Each wave runs [rounds] rounds of the kernel's instruction on registers,
   then writes a value of its accumulators to out[wave], so that nothing
   is dead. A round is 16 independent instructions: v_wmma_f32_16x16x16_bf16
   (wmma_bf16), or 32 lanes' fma (fma_f32). */
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
