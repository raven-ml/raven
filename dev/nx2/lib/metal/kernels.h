/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.metal's kernels as device and host see them: the launch record, one
   parameter structure per family, and the enum of every kernel.

   A launch is a record followed in memory by its parameter bytes. The
   kernel reads them as [constant P &p [[buffer(0)]]]. The first [addrs]
   words of the parameters are GPU addresses (an MTLBuffer's gpuAddress plus
   an offset), which kernels read as device pointers; bit i of [scratch]
   says address i is an offset into the call's scratch, to which whoever
   allocates the scratch adds its base before the fill runs. A run of
   records holds no pointer, so it can be kept, moved and submitted after
   the call that planned it.

   Float arithmetic is the GPU's: on Apple GPUs a float32 subnormal operand
   reads as zero and a float32 subnormal result is written as a zero of its
   sign, while half arithmetic and the conversions keep subnormals.

   Results are bitwise the same on every run of one GPU family: plan.c
   reads no core count, occupancy or clock, kernels add no float
   atomically, and every sum's order is fixed by the shape.

   This header compiles as C on the host and as the Metal Shading Language
   on the device, with nothing but metal_stdlib. */

#ifndef NX_METAL_KERNELS_H
#define NX_METAL_KERNELS_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

/* X(name) for every kernel of the library, the function [name] of its
   metallib. */
#define NX_METAL_KERNELS(X)

#define NX_METAL_ENUM(name) NX_METAL_##name,
enum nx_metal_kernel { NX_METAL_KERNELS(NX_METAL_ENUM) NX_METAL_KERNEL_COUNT };
#undef NX_METAL_ENUM

/* A launch: [entry] indexes the run's pipelines; [bytes] is a multiple of
   8 and at most 4,096, what Metal's setBytes takes; [addrs]·8 <= bytes. */
typedef struct {
  uint32_t entry;      /* an nx_metal_kernel */
  uint32_t groups[3];  /* threadgroups per grid */
  uint32_t threads[3]; /* threads per threadgroup */
  uint32_t bytes;      /* parameter bytes that follow */
  uint32_t addrs;      /* the parameters' first addrs words are addresses */
  uint32_t scratch;    /* bit i: address i is an offset into the scratch */
} nx_metal_launch;

#endif /* NX_METAL_KERNELS_H */
