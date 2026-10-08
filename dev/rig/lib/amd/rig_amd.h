/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to an AMD device from C.

   Rig_amd.room and Rig_amd.submit, for a caller that holds its
   submissions in C, over the structures and codes of rig_edge.h. [self] is
   Rig_amd.self. Both are called without the OCaml runtime: they call no
   function of it and read no OCaml value, and neither blocks.

   Queue 0 is the compute queue "COMPUTE:0", queue 1 the copy queue
   "COPY:0". A part is words placed on its queue, a fill called with the
   queue's writer (rig_amd_abi's Capability), or, on queue 1, a copy
   between handles, which are GPU addresses. A wait is RIG_WORD on a 64-bit
   word the device maps, which the compute queue waits on.
   [handles] is ignored: the GPU's work names its memory by address. */

#ifndef RIG_AMD_H
#define RIG_AMD_H

#include <rig_edge.h>

/* RIG_FITS if [parts] fit the device's rings and argument segment now,
   RIG_LATER if they fit once a value the device was given completes, as
   the timeline word reads now, and
   RIG_NEVER if they exceed an empty ring or the segment, are more than 512,
   or a part is one the device does not run: a copy on queue 0, words on an
   AQL queue that are not whole packets, an [after] index not below its own
   part's. */
int rig_amd_room(void *self, const struct rig_part *parts, int n);

/* Places [parts], which rig_amd_room answered RIG_FITS for, as the work
   of [v], the value after the last one it received, and rings the queues'
   doorbells: RIG_OK, or RIG_FAILED with [*failure] set when a fill failed.
   The queues then run none of [parts], and the timeline word still reaches
   [v]. After a failure every call answers RIG_FAILED with the first
   failure's message, which lives as long as the process. */
int rig_amd_submit(void *self, uint64_t v, const struct rig_wait *waits,
                      int nwaits, const struct rig_part *parts, int nparts,
                      const uint64_t *handles, int nhandles,
                      const char **failure);

#endif
