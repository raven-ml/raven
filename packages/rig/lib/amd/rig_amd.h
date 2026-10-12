/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to an AMD device from C.

   A device's room check, submit and settle, for the caller that holds its
   submissions in C, over the structures and codes of rig_edge.h. [self] is
   the edge of Rig_amd's facts, whose struct rig_driver holds them. They are
   called without the OCaml runtime: they call no function of it and read no
   OCaml value, and none blocks, but the submit may wait out the
   publication of a new AQL scratch, a few stores.

   Queue 0 is the logical compute queue "COMPUTE:0", queue 1 the copy queue
   "COPY:0". A part is words placed on its queue, on queue 0 a launch of what
   the device's entry made, or, on queue 1, a copy between handles,
   which are GPU addresses. Set copy.device_memory to 1 only when both
   original allocations are Device memory owned by this device, otherwise
   0. Nonempty eligible copies may run on the compute engine; all parts,
   including empty ones, retain their logical queue order. A wait is RIG_WORD
   on a 64-bit word the device
   maps, which the compute queue waits on, at most 255 per submission.
   Rig_amd's interface states the rules in full. */

#ifndef RIG_AMD_H
#define RIG_AMD_H

#include <rig_edge.h>

/* RIG_FITS if [parts] fit the device's rings and argument segment now,
   RIG_LATER if they fit once a value the device was given completes, as
   the timeline word reads now, and
   RIG_NEVER if they exceed an empty ring or the segment, are more than 512,
   or a part is one the device does not run: a copy on queue 0, words on an
   AQL queue that are not whole packets, a launch with fewer parameter
   bytes than its function's explicit arguments, or whose block in [args]
   has an empty axis or more threads per group or shared memory than its
   function takes, or, on an AQL queue, 2^32 work-items or more along an
   axis, an [after] index not below its own part's. */
int rig_amd_room(void *self, const struct rig_part *parts, int n,
                 void *routes, const uint8_t *args);

/* Places [parts], which rig_amd_room answered RIG_FITS for, as the work
   of [v], the value after the last one it received, and rings the queues'
   doorbells: RIG_COMMITTED, or RIG_FAILED with [*failure] set when the
   waits are more than the device holds or ones it cannot
   make. The queues then run none of [parts], and the timeline word still
   reaches [v]. After a failure every call answers RIG_FAILED with the first
   failure's message, which lives as long as the process. With [times], the
   queues stamp [v]'s work with the GPU's clock, which rig_amd_settle
   converts to the host's clock. */
int rig_amd_submit(void *self, uint64_t v, const struct rig_wait *waits,
                   int nwaits, const struct rig_part *parts, int nparts,
                   void *routes, const uint8_t *args,
                   const uint64_t *slots, int nslots, uint64_t *times,
                   const char **failure);

/* Writes the times of [v], whose work has completed, from its stamps,
   unless a later rig_amd_submit wrote them (rig_settle_fn), and those of
   every earlier timed value first. Answers RIG_FAILED if a stamp of a
   completed value was never written, or the GPU's clock cannot be read. */
int rig_amd_settle(void *self, uint64_t v, const char **failure);

/* A path's reader of the GPU's clock against the host's clock (Rig_amd's
   [Source] clock), which the path keeps while a device may read it,
   usually as the first member of a structure of its own. [read] stores the
   clock's count into [*ticks] and the host's clock (rig_now_ns) at the
   same instant into [*host], and answers 0, or -1 if it could not read
   them. It never waits for the GPU's work, takes a few microseconds at
   most, and calls nothing of the OCaml runtime. */
struct rig_amd_clock {
  int (*read)(struct rig_amd_clock *c, uint64_t *ticks, uint64_t *host);
};

#endif
