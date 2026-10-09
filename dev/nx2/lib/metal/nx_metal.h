/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.metal's host side: runs of launch records, and the fill that encodes
   them into rig.metal's encoder (rig_metal_abi.mli).

     records --(the binding resolves each kernel to a pipeline: entries)
             --> nx_metal_fill, called by rig.metal's submit
             --> a pipeline, setBytes and a dispatch per record

   The fill resolves nothing and allocates nothing: what it reads was made
   before the submission. */

#ifndef NX_METAL_H
#define NX_METAL_H

#include <stddef.h>
#include <stdint.h>

#include "kernels.h"

/* A run of launch records: [len] bytes of records at [bytes], in a buffer
   of [cap] bytes that nx_metal_add grows. A zeroed run is empty. */
typedef struct {
  unsigned char *bytes;
  size_t len, cap;
} nx_metal_records;

/* Appends a launch of [entry] over [groups] threadgroups of [threads]
   threads, whose parameters are the [bytes] bytes at [params], their first
   [addrs] 8-byte words addresses and [scratch] the mask of those that are
   scratch offsets, as nx_metal_launch states them. Returns 0, or -1 if
   memory runs out, the run unchanged. */
int nx_metal_add(nx_metal_records *r, uint32_t entry, const uint32_t groups[3],
                 const uint32_t threads[3], const void *params, uint32_t bytes,
                 uint32_t addrs, uint32_t scratch);

/* Adds [base] to each address of the [len] bytes of records at [r] that
   its record's scratch mask names: the call's scratch, allocated after
   planning, at [base]. */
void nx_metal_rebase(unsigned char *r, size_t len, uint64_t base);

/* Plans */

/* What a plan answers when it does not compute a call: nx's expansion of
   the operation runs instead. */
#define NX_NOT_COMPUTED (-1)

/* An operand of a call as a plan reads it: its GPU address, its dtype (an
   nx_dtype.h code), and its strides in elements over the call's three
   axes, such as (batch, m, k) for a contraction's a. */
typedef struct {
  uint64_t address;
  int dtype;
  int64_t strides[3];
} nx_metal_operand;

/* A contraction: out[p][i][j] = Σ_l a[p][i][l] · b[p][l][j] + init, the
   sum in [acc], rounded once to out's dtype, or wrapped for integers. */
typedef struct {
  uint32_t batch, m, n, k;
  int acc; /* the accumulator's dtype */
} nx_metal_contract_in;

/* Appends the launches of the contraction [c] to [r] and is their count,
   NX_NOT_COMPUTED, or -2 if memory runs out. The operands are a (batch, m,
   k), b (batch, k, n), out, C-contiguous (batch, m, n), and init (batch,
   m, n) unless [init] is NULL. [*scratch] is set to the scratch bytes the
   launches address, a multiple of 16: nx_metal_rebase places it. */
int nx_metal_plan_contract(const nx_metal_contract_in *c,
                           const nx_metal_operand *a,
                           const nx_metal_operand *b,
                           const nx_metal_operand *out,
                           const nx_metal_operand *init,
                           nx_metal_records *r, size_t *scratch);

/* The metallib of nx.metal's kernels, [*len] bytes: one for every Apple
   GPU. */
const char *nx_metal_metallib(size_t *len);

/* The name of each kernel, by its nx_metal_kernel. */
extern const char *const nx_metal_kernel_names[NX_METAL_KERNEL_COUNT];

/* A fill's argument: [count] pipelines, pipelines[k] the
   MTLComputePipelineState of the run's kernel k (Rig_metal.entry), and a
   run of [bytes] bytes of records, which follows this header in memory. */
typedef struct {
  const uint64_t *pipelines;
  uint64_t count, bytes;
} nx_metal_run;

/* The fill of rig.metal's calling convention (rig_metal_abi.mli) over the
   nx_metal_run [arg]: it encodes each launch in order into the encoder,
   whose dispatches run one after the other, and returns 0, or 1 having
   encoded nothing if a launch names a kernel past [count] or one with no
   pipeline. It ends no encoder and makes no command buffer. */
int nx_metal_fill(void *queue, void *arg, uint64_t v);

#endif /* NX_METAL_H */
