/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.amd's kernels as the host and the device both see them: the launch
   record, and each family's parameters.

   A kernel reads one parameter struct, whose bytes a launch record
   carries. Its addresses come first, as 8-byte words, so that whoever
   moves or records a launch finds every address it names without knowing
   the kernel. Every struct's size is a multiple of 8. A kernel declares
   its local data share at its own size: a launch adds none. */

#ifndef NX_AMD_KERNELS_H
#define NX_AMD_KERNELS_H

#include <stdint.h>

/* One launch, followed in memory by its [bytes] parameter bytes; the next
   record follows them. [kernel] indexes the table of dispatches the fill
   reads. The parameters' first [addrs] 8-byte words are addresses; bit i
   of [scratch] marks address i as an offset into the call's scratch,
   which whoever allocates the scratch turns into an address, adding the
   scratch's base, before the fill runs: scratch addresses come among the
   first 32. A record holds no pointer, so a run of them can be kept,
   moved and submitted after the call that planned it. */
typedef struct __attribute__((aligned(8))) {
  uint32_t kernel;
  uint32_t groups[3];  /* workgroups per grid */
  uint32_t threads[3]; /* work-items per workgroup */
  uint32_t bytes;      /* a multiple of 8 */
  uint32_t addrs;
  uint32_t scratch;
} nx_amd_launch;

#endif
