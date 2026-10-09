/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.cpu's private interface: targets, jobs, walks and the stage.

   A kernel reads its operands through the door (nx_array.h) and coalesces
   them into a loop. The walk cuts the loop into 2-D blocks and hands them
   to a job, which runs them on the pool's threads. Per block the kernel
   reads each operand in place where the target's runs can, and otherwise
   stages it into a buffer in L1; it writes its result through the unstage,
   which converts as a cast does.

     kernel ─► walk ─► job ─► rig.pool threads ─► block
                                                   │
                     operand ─► stage ─► buffer ─► unstage ─► result
                                     (target's runs)
*/

#ifndef NX_CPU_H
#define NX_CPU_H

#include <stdint.h>

#include "nx_array.h"
#include "rig_pool.h"

/* Targets

   A target is the instruction set the kernels' runs are compiled for: one
   table per target, chosen once at initialisation. Every table computes the
   same bits. */

/* A run converts [n] contiguous elements at [src] into [dst]. */
typedef void (*nx_cpu_run)(const void *src, void *dst, int64_t n);

typedef struct {
  const char *name;
  /* convert[s][d] converts elements of the dtype s into the dtype d, as a
     cast does, where s is a carrier, or where s is a narrow float and d
     float32. A sub-byte dtype has one element per byte, in its low bits:
     int4 and uint4 their value modulo 16, float4 its code, bit 0 or 1. */
  nx_cpu_run convert[NX_DTYPE_COUNT][NX_DTYPE_COUNT];
} nx_cpu_target;

/* The tables, each filled when the program starts on a host that runs it,
   by convert.c compiled for its target. */
extern nx_cpu_target nx_cpu_base;
void nx_cpu_fill_base(nx_cpu_target *t);
#if defined(__x86_64__)
extern nx_cpu_target nx_cpu_v3;
void nx_cpu_fill_v3(nx_cpu_target *t);
#endif

/* The table the kernels run: the best one whose instructions the host
   has. */
extern const nx_cpu_target *nx_cpu_runs;

/* The dtype a block of [dt] is staged in: float32 for a narrow float, int8
   for int4, uint8 for uint4, bool for bit, and [dt] itself otherwise. */
static inline int nx_cpu_carrier(int dt) {
  switch (dt) {
    case NX_FLOAT16:
    case NX_BFLOAT16:
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2:
    case NX_FLOAT4_E2M1FN: return NX_FLOAT32;
    case NX_INT4: return NX_INT8;
    case NX_UINT4: return NX_UINT8;
    case NX_BIT: return NX_BOOL;
    default: return dt;
  }
}

/* The bytes a staged element of [dt] takes: a sub-byte one takes a byte. */
static inline int nx_cpu_width(int dt) {
  int bits = nx_dtype_row_of(dt).bits;
  return bits < 8 ? 1 : bits / 8;
}

/* Jobs */

/* Runs [body] over the units [0, total) of work that touches [bytes] bytes
   and takes as long as memcpy of [cost] bytes, on as many of the pool's
   threads as the cost pays for. A job of more than one thread, or a long
   one, runs with the runtime released: [body] reads no OCaml value. */
void nx_cpu_job(int64_t total, int64_t bytes, int64_t cost, rig_pool_body body,
                void *ctx);

/* Walks */

/* A block of a loop: [n1] rows of [n0] elements. Operand k's first element
   is at position at[k] from its base, and it steps s0[k] elements along a
   row and s1[k] from a row to the next. */
typedef struct {
  int64_t n0, n1;
  int64_t at[NX_MAX_OPERANDS];
  int64_t s0[NX_MAX_OPERANDS];
  int64_t s1[NX_MAX_OPERANDS];
} nx_cpu_block;

typedef void (*nx_cpu_block_fn)(const nx_cpu_block *b, void *ctx);

/* Calls [f] on blocks of at most [most] elements that cover the loop [l]
   over the [n] operands [a], operand 0 the written one, each element once,
   from a job. Order and threads decide no bit: [f] computes each element
   from operands' elements at its index alone. */
void nx_cpu_walk(int n, const nx_array *a, const nx_loop *l, int64_t most,
                 nx_cpu_block_fn f, void *ctx);

/* The stage */

/* The bytes of a staged block: a walk for the stage takes blocks of at most
   NX_CPU_SLOT / w elements, w the widest form its elements take, so that a
   block's buffers stay in L1. */
#define NX_CPU_SLOT (16 * 1024)

/* Copies operand [k] of block [b], the array [a], into [dst] in its carrier:
   row j's element i at [dst + j·row + i·w], w the carrier's width. */
void nx_cpu_stage(const nx_array *a, const nx_cpu_block *b, int k, uint8_t *dst,
                  int64_t row);

/* Writes [src], elements of the dtype [c] with row j's element i at
   [src + j·row + i·w], w [c]'s width, into operand [k] of block [b], the
   array [a], converted as a cast does. [c] is a carrier. */
void nx_cpu_unstage(const nx_array *a, const nx_cpu_block *b, int k,
                    const uint8_t *src, int64_t row, int c);

#endif /* NX_CPU_H */
