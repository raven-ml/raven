/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.cuda's host side: runs of launch records, and the fill that enqueues
   them on a stream of rig.cuda's (rig_cuda_abi.mli).

     records --(the binding resolves each kernel to a CUfunction: entries)
             --> nx_cuda_fill, called by rig.cuda's submit on its stream
             --> cuLaunchKernel per record

   The fill resolves nothing and allocates nothing: what it reads was made
   before the submission.

   The kernels need a driver of CUDA 13 (R580) or later: nvcc 13.4 builds
   their cubins, which a driver of the same major version runs, newer
   minor versions included, since they carry no PTX to compile. */

#ifndef NX_CUDA_H
#define NX_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "kernels.h"
#include "nx_array.h"

/* What a plan answers when it does not compute a call: nx's expansion of
   the operation runs instead. */
#define NX_NOT_COMPUTED (-1)

/* What a plan answers when the host's memory runs out. */
#define NX_OUT_OF_MEMORY (-2)

/* The cubin of the architecture [arch] (89 for sm_89), [*len] bytes, or
   NULL if the library has none: it computes on the GPUs of the
   architectures it has a cubin for. */
const char *nx_cuda_cubin(int arch, size_t *len);

/* The name of each kernel, by its nx_cuda_kernel. */
extern const char *const nx_cuda_kernel_names[NX_CUDA_KERNEL_COUNT];

/* Runs */

/* A run of launch records: [len] bytes of records at [bytes], in a buffer
   of [cap] bytes that nx_cuda_add grows. A zeroed run is empty. */
typedef struct {
  unsigned char *bytes;
  size_t len, cap;
} nx_cuda_records;

/* Appends a launch of [kernel] over [grid] blocks of [block] threads, with
   [shared] dynamic shared bytes, whose parameters are the [bytes] bytes at
   [params], their first [addrs] 8-byte words addresses and [scratch] the
   mask of those that are scratch offsets. Returns 0; -1 if [bytes] is not
   a multiple of 8, [addrs] words outgrow [bytes] or [scratch] marks a word
   past the addresses; or NX_OUT_OF_MEMORY if memory runs out. The run is
   unchanged unless it returns 0. */
int nx_cuda_add(nx_cuda_records *r, uint32_t kernel, const uint32_t grid[3],
                const uint32_t block[3], uint32_t shared, const void *params,
                uint32_t bytes, uint32_t addrs, uint32_t scratch);

/* Adds [base] to each address of the [len] bytes of records at [r] that
   its record's scratch mask names: the call's scratch, allocated after
   planning, at [base]. */
void nx_cuda_rebase(unsigned char *r, size_t len, uint64_t base);

/* The fill */

/* cuLaunchKernel's type. */
typedef int (*nx_cuda_launch_fn)(void *f, unsigned gx, unsigned gy,
                                 unsigned gz, unsigned bx, unsigned by,
                                 unsigned bz, unsigned shared, void *stream,
                                 void **params, void **extra);

/* What a fill launches with: cuLaunchKernel, from the device's capability,
   and the CUfunction of each of [count] kernels, by index. */
typedef struct {
  nx_cuda_launch_fn launch;
  uint32_t count;
  void *funcs[];
} nx_cuda_entries;

/* The argument of nx_cuda_fill: [len] bytes of records at [records],
   launched with [entries]. */
typedef struct {
  const nx_cuda_entries *entries;
  const unsigned char *records;
  size_t len;
} nx_cuda_run;

/* The fill of rig_cuda_abi.mli over the nx_cuda_run [arg]: launches each
   record in order on [stream]. Returns 0 (CUDA_SUCCESS), the CUresult of
   the first launch CUDA refused, or 1 (CUDA_ERROR_INVALID_VALUE) for a
   record whose kernel has no entry, launching none after it. */
int nx_cuda_fill(void *stream, void *arg, uint64_t v);

/* Plans */

/* An operand on the GPU: its element at index (i0, …, ik-1) is at
   [address] plus the Σ ij·dim[rank + j] elements, ij < dim[j]. */
typedef struct {
  uint64_t address;
  int dtype, rank;
  int64_t dim[2 * NX_MAX_RANK];
} nx_cuda_operand;

/* A contraction with plain loads and no scales or segments: the
   batch pairs and contracting pairs of a's and b's axes, the accumulator's
   dtype, and whether an init operand is given. */
typedef struct {
  int batch[NX_MAX_RANK][2], nbatch;
  int contracting[NX_MAX_RANK][2], ncontracting;
  int acc, init;
} nx_cuda_contract_in;

/* Appends the launches of one contraction to [out] and returns their
   count; or NX_NOT_COMPUTED, or NX_OUT_OF_MEMORY if the records cannot
   grow, having appended nothing. [ops] are a, b, init
   if [in->init], and y. [arch] is the cubin's architecture, as 89 for
   sm_89. [*scratch] is set to the scratch bytes the launches address. */
int nx_cuda_plan_contract(const nx_cuda_contract_in *in,
                          const nx_cuda_operand *ops, int arch,
                          nx_cuda_records *out, size_t *scratch);

#endif
