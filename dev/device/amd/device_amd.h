/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to an AMD device from C.

   Device_amd.room and Device_amd.submit, for a caller that holds its
   submissions in C, over the structures and codes of nx_edge.h. [self] is
   Device_amd.self. Both are called without the OCaml runtime: they call no
   function of it and read no OCaml value, and neither blocks.

   Queue 0 is the compute queue "COMPUTE:0", queue 1 the copy queue
   "COPY:0". A part is words placed on its queue, a fill called with the
   queue's writer (device_amd_abi's Capability), or, on queue 1, a copy
   between handles, which are GPU addresses. A wait is NX_WORD on a 64-bit
   word the device maps, which the compute queue waits on.
   [handles] is ignored: the GPU's work names its memory by address. */

#ifndef DEVICE_AMD_H
#define DEVICE_AMD_H

#include <nx_edge.h>

/* NX_FITS if [parts] fit the device's rings and argument segment now,
   NX_LATER if they fit once a value the device was given completes, as
   the timeline word reads now, and
   NX_NEVER if they exceed an empty ring or the segment, are more than 512,
   or a part is one the device does not run: a copy on queue 0, words on an
   AQL queue that are not whole packets, an [after] index not below its own
   part's. */
int device_amd_room(void *self, const struct nx_part *parts, int n);

/* Places [parts], which device_amd_room answered NX_FITS for, as the work
   of [v], the value after the last one it received, and rings the queues'
   doorbells: NX_OK, or NX_FAILED with [*failure] set when a fill failed.
   The queues then run none of [parts], and the timeline word still reaches
   [v]. After a failure every call answers NX_FAILED with the first
   failure's message, which lives as long as the process. */
int device_amd_submit(void *self, uint64_t v, const struct nx_wait *waits,
                      int nwaits, const struct nx_part *parts, int nparts,
                      const uint64_t *handles, int nhandles,
                      const char **failure);

#endif
