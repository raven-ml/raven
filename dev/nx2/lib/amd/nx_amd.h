/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.amd's host side: runs of launch records, and the fill that places
   them on the compute queue of a device of rig.amd's (Rig_amd_abi's
   Capability).

     records --(the binding resolves each kernel to its dispatch's PM4
                words on the device: entries)
             --> nx_amd_fill, called by rig.amd's submit on its queue
             --> per record, its parameters in the argument segment and
                 its dispatch's words on the ring

   The fill resolves nothing and allocates nothing: what it reads was made
   before the submission. */

#ifndef NX_AMD_H
#define NX_AMD_H

#include <stddef.h>
#include <stdint.h>

#include "kernels.h"

/* Runs */

/* A run of launch records: [len] bytes of records at [bytes], in a buffer
   of [cap] bytes that nx_amd_add grows. A zeroed run is empty. */
typedef struct {
  unsigned char *bytes;
  size_t len, cap;
} nx_amd_records;

/* Appends a launch of [kernel] over [groups] workgroups of [threads]
   work-items, whose parameters are the [bytes] bytes at [params], their
   first [addrs] 8-byte words addresses and [scratch] the mask of those
   that are scratch offsets. Returns 0; -1 if [bytes] is not a multiple of
   8, [addrs] words outgrow [bytes] or [scratch] marks a word past the
   addresses; or -2 if memory runs out. The run is unchanged unless it
   returns 0. */
int nx_amd_add(nx_amd_records *r, uint32_t kernel, const uint32_t groups[3],
               const uint32_t threads[3], const void *params, uint32_t bytes,
               uint32_t addrs, uint32_t scratch);

/* The fill */

/* The most words of a dispatch. */
#define NX_AMD_DISPATCH_WORDS 128

/* A kernel's dispatch on one device: the [n] PM4 words at [words] that
   run it after the work before it and before the work after it
   (Rig_amd_abi.Pm4.run of Pm4.dispatch), with its arguments' address,
   work-items and workgroups left for the fill. [args] indexes the two
   words of the address, low first; threads[d] and groups[d] index the
   word of dimension d. [kernarg] is the bytes of arguments the kernel
   reads. */
typedef struct {
  const uint32_t *words;
  uint32_t n, args, threads[3], groups[3], kernarg;
} nx_amd_dispatch;

/* What a fill places with: the queue writer's place and segment, from the
   device's capability, and the dispatch of each of [count] kernels, by
   index. */
typedef struct {
  int (*place)(void *queue, const uint32_t *words, size_t n);
  int (*segment)(void *queue, size_t n, void **host, uint64_t *address);
  uint32_t count;
  nx_amd_dispatch kernels[];
} nx_amd_entries;

/* The argument of nx_amd_fill: [len] bytes of records at [records],
   placed with [entries]. */
typedef struct {
  const nx_amd_entries *entries;
  const unsigned char *records;
  size_t len;
} nx_amd_run;

/* What a part that fills [run] declares: the ring words it places at
   [*words], and the bytes it takes of the argument segment at [*bytes],
   each launch's taken whole in multiples of 64. Returns 0, or -1 for a
   record whose kernel has no entry or one longer than
   NX_AMD_DISPATCH_WORDS, or whose parameters are not the [kernarg] bytes
   its kernel reads. */
int nx_amd_size(const nx_amd_run *run, uint64_t *words, uint64_t *bytes);

/* The fill of Rig_amd_abi's Capability over the nx_amd_run [arg]: places
   each record in order on [queue], its parameters copied to the argument
   segment. Returns 0, the failure of the first place or segment that
   failed, or -1 for a record nx_amd_size refuses, placing none after
   it. */
int nx_amd_fill(void *queue, void *arg, uint64_t v);

#endif
