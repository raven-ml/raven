/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to an NVIDIA device from C.

   The device's room check and submission, over the structures and codes of
   rig_edge.h, for rig and any caller that holds its submissions in C.
   [self] is Rig_nv.self. Both are called without the OCaml runtime: they
   call no function of it and read no OCaml value.

   Queue 0 is the channel "COMPUTE:0", queue 1 the channel "COPY:0". A part
   is ring entries, two words each, low first (Rig_nv_abi.Gpfifo), or a
   copy between handles on queue 1; it is no fill and declares no ring units
   or segment bytes. The parts on one queue run in array order. A wait is
   RIG_WORD on a 64-bit word the device maps, compared circularly; a
   submission has at most 256, for which rig_nv_room, which does not see
   them, keeps room. [handles] is ignored: the device's work names its memory
   by address. */

#ifndef RIG_NV_H
#define RIG_NV_H

#include <rig_edge.h>

/* RIG_NEVER if a part is a fill, declares ring units or segment bytes, has an
   odd number of words, is a copy on queue 0, is on no queue of the device,
   has an [after] index not below its own part's, or if the parts exceed the
   device's empty rings or number more than 65,535; RIG_LATER if they fit
   once a value the device was given is reached, as its timeline word reads
   now; RIG_FITS otherwise. */
int rig_nv_room(void *self, const struct rig_part *parts, int n);

/* Writes [parts], which rig_nv_room answered RIG_FITS for with nothing
   submitted since, into the device's rings as the work of [v], the value
   after the last one it received, and wakes the channels. It answers
   RIG_COMMITTED: its stores are to this machine's memory and cannot fail.
   It never blocks. */
int rig_nv_submit(void *self, uint64_t v, const struct rig_wait *waits,
                  int nwaits, const struct rig_part *parts, int nparts,
                  const uint64_t *handles, int nhandles, const char **failure);

#endif
