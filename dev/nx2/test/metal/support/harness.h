/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The kernels the Metal suite and bench run beside nx.metal's: the floors,
   the operand generator, the peaks and the probes. harness.metal defines
   them; the host launches them as rig's Launch parts, as nx.metal launches
   its own. Each reads one parameter struct, its addresses first.

   This header compiles as C on the host and as the Metal Shading Language
   on the device. */

#ifndef NX_METAL_HARNESS_H
#define NX_METAL_HARNESS_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

/* X(name) for every kernel of the harness's metallib. */
#define NX_HARNESS_KERNELS(X)                                          \
  X(empty) X(move) X(read) X(generate) X(fma_f32) X(fma_f16)           \
  X(mma_f32) X(mma_f16) X(probe_contract) X(probe_div_sqrt)            \
  X(probe_half) X(probe_codec)

/* Threads per threadgroup of every harness kernel. */
#define NX_HARNESS_THREADS 256

/* move: out = in[0] ^ … ^ in[ins - 1] over n 16-byte vectors, one a
   thread: the copy floor for one input, the floor of an operation reading
   [ins] arrays and writing one for more. read: each threadgroup reads one
   vector a thread and writes their xor at out[group]. */
typedef struct {
  uint64_t out, in[3];
  uint32_t ins, n;
} move_params;

/* out[i], i < n, is the element i of a deterministic sequence of the
   dtype (an nx_dtype.h code) from [seed]: a float has a random sign and
   significand and an exponent drawn in [-spread, spread]; an integer has
   random bits. float64, float32, float16, bfloat16 and the integers of 8
   to 64 bits. */
typedef struct {
  uint64_t out;
  uint32_t n, dtype, seed, spread;
} generate_params;

/* [iters] rounds over independent chains per thread, from out's values,
   the result written back: the peaks. */
typedef struct {
  uint64_t out;
  uint32_t iters, unused;
} spin_params;

/* out[i] is the probe's operation, [which], on the i-th of n operand
   tuples at [in]; probe_codec's [dtype] names the narrow format. */
typedef struct {
  uint64_t out, in;
  uint32_t n, which, dtype, unused;
} probe_params;

#endif /* NX_METAL_HARNESS_H */
