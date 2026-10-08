/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to a Metal device from C.

   The device's room check and submit, over the structures and codes of
   rig_edge.h: the only way work reaches the device. [self] is
   Rig_metal.self. Both are called without the OCaml runtime: they call
   no function of it and read no OCaml value.

   The device runs fills only: a part with a [fill] and its [arg], on queue
   0, with no words, copy, ring units or segment bytes; a fill may start any
   number of command buffers (rig_metal_abi.mli). [nwaits] is 0: a Metal
   device waits on no word. */

#ifndef RIG_METAL_H
#define RIG_METAL_H

#include <rig_edge.h>

/* RIG_NEVER if a part is no fill on queue 0: it has words, a copy, no fill,
   or ring units or segment bytes other than 0. RIG_FITS otherwise. */
int rig_metal_room(void *self, const struct rig_part *parts, int n);

/* Runs [parts], which rig_metal_room answered RIG_FITS for, as the work
   of [v], the value after the last one it received: RIG_OK, or RIG_FAILED with
   [*failure] set to the device's message. Once the device recorded a
   failure every call answers RIG_FAILED with the first failure's message,
   which lives while the process runs, and runs nothing. One call at a time;
   it waits while the device's 1,024 command buffers are in flight. */
int rig_metal_submit(void *self, uint64_t v, const struct rig_wait *waits,
                        int nwaits, const struct rig_part *parts, int nparts,
                        const uint64_t *handles, int nhandles,
                        const char **failure);

#endif
