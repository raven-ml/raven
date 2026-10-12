/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to an NVIDIA device from C.

   The device's room check, submission and settle, over the structures and
   codes of rig_edge.h, for rig and any caller that holds its submissions
   in C. [self] is the device's [edge] fact, whose struct rig_driver holds
   them. They are called without the OCaml runtime: they call no function
   of it and read no OCaml value.

   Queue 0 is "COMPUTE:0", queue 1 "COPY:0". A part
   is ring entries, two words each, low first (Rig_nv_abi.Gpfifo), a copy
   between handles on queue 1, or a launch on queue 0 of a function the
   driver's [entry] set up. Eligible copies with copy.device_memory use
   the compute channel; all others use their declared queue's channel.
   Set copy.device_memory to 1 only when both endpoints are Device memory
   owned by this device, and to 0 otherwise.
   The parts on one queue run in array order across channel changes. A
   wait is RIG_WORD on a 64-bit word the device maps, compared circularly;
   a submission has at most 256, for which rig_nv_room, which does not see
   them, keeps room. */

#ifndef RIG_NV_H
#define RIG_NV_H

#include <rig_edge.h>

/* RIG_NEVER if a part is no words, copy or launch, has an odd number of
   words, is a copy on queue 0 or a launch on queue 1, is on no queue of the
   device, has an [after] index not below its own part's, is a launch whose
   block in [args] its function does not run (Rig_nv's facts), or if the
   parts exceed the device's empty rings or number more than 65,535;
   RIG_LATER if they fit once a value the device was given is reached, as
   its timeline word reads now; RIG_FITS otherwise. */
int rig_nv_room(void *self, const struct rig_part *parts, int n,
                void *scratch, const uint8_t *args);

/* Writes [parts], which rig_nv_room answered RIG_FITS for with nothing
   submitted since, into the device's rings as the work of [v], the value
   after the last one it received, and wakes the channels. It answers
   RIG_COMMITTED: its stores are to this machine's memory and cannot fail.
   It never blocks. With [times], the channels stamp [v]'s work with the
   GPU's timer, which rig_nv_settle converts to the host's clock. */
int rig_nv_submit(void *self, uint64_t v, const struct rig_wait *waits,
                  int nwaits, const struct rig_part *parts, int nparts,
                  void *scratch, const uint8_t *args,
                  const uint64_t *slots, int nslots, uint64_t *times,
                  const char **failure);

/* Writes the times of [v], whose work has completed, from its stamps,
   unless a later rig_nv_submit wrote them (rig_settle_fn), and those of
   every earlier timed value first. Answers RIG_FAILED if a stamp of a
   completed value was never written, or the GPU's timer cannot be read. */
int rig_nv_settle(void *self, uint64_t v, const char **failure);

/* A path's reader of the GPU's timer against the host's clock (Rig_nv's
   [Source] clock), which the path keeps while a device may read it,
   usually as the first member of a structure of its own. [read] stores the
   timer's nanoseconds into [*gpu] and the host's clock (rig_now_ns) at the
   same instant into [*host], and answers 0, or -1 if it could not read
   them. It never waits for the GPU's work, takes a few microseconds at
   most, and calls nothing of the OCaml runtime. */
struct rig_nv_clock {
  int (*read)(struct rig_nv_clock *c, uint64_t *gpu, uint64_t *host);
};

#endif
