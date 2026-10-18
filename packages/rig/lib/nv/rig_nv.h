/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to an NVIDIA device from C.

   The device's room check, submission and settle: this driver's
   rig_room_fn, rig_submit_fn and rig_settle_fn of rig_edge.h, which the
   struct rig_driver of [self], the device's [edge] fact, holds. Rig_nv's
   Facts and Submitting sections (rig_nv.mli) state the device's queues,
   the parts each runs, its waits, and what each function answers. A
   caller other than rig must call rig_nv_room before each rig_nv_submit,
   with no other call of either on the device between them. */

#ifndef RIG_NV_H
#define RIG_NV_H

#include <rig_edge.h>

/* The room check of Rig_nv's Submitting section (rig_room_fn). */
int rig_nv_room(void *self, const struct rig_part *parts, int n,
                void *scratch, const uint8_t *args);

/* The submission of Rig_nv's Submitting section (rig_submit_fn), of
   [parts], which rig_nv_room answered RIG_FITS for with nothing submitted
   since. */
int rig_nv_submit(void *self, uint64_t v, const struct rig_wait *waits,
                  int nwaits, const struct rig_part *parts, int nparts,
                  void *scratch, uint8_t *args,
                  const uint64_t *slots, int nslots, uint64_t *times,
                  const char **failure);

/* The settle of Rig_nv's Submitting section (rig_settle_fn). */
int rig_nv_settle(void *self, uint64_t v, const char **failure);

/* A path's reader of the GPU's timer against the host's clock (Rig_nv's
   [Source] clock), which the path keeps while a device may read it,
   usually as the first member of a structure of its own. [read] stores the
   timer's nanoseconds into [*gpu], the host's clock (rig_now_ns) at the
   same instant into [*host], and into [*width] the interval between the
   host clock's readings that bound that instant, and answers 0, or -1 if
   it could not read them. It never waits for the GPU's work and calls
   nothing of the OCaml runtime. Devices of one GPU may call it
   concurrently. */
struct rig_nv_clock {
  int (*read)(struct rig_nv_clock *c, uint64_t *gpu, uint64_t *host,
              uint64_t *width);
};

#endif
