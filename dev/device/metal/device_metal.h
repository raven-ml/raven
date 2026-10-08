/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to a Metal device from C.

   Device_metal.room and Device_metal.submit, for a caller that holds its
   submissions in C, over the structures and codes of nx_edge.h. [self] is
   Device_metal.self. Both are called without the OCaml runtime: they call no
   function of it and read no OCaml value.

   The device runs fills only: a part with a [fill] and its [arg], on queue
   0, with no words, copy, ring units or segment bytes; a fill may start any
   number of command buffers (device_metal_abi.mli). [nwaits] is 0: a Metal
   device waits on no word. */

#ifndef DEVICE_METAL_H
#define DEVICE_METAL_H

#include <nx_edge.h>

/* NX_NEVER if a part is no fill on queue 0: it has words, a copy, no fill,
   or ring units or segment bytes other than 0. NX_FITS otherwise. */
int device_metal_room(void *self, const struct nx_part *parts, int n);

/* Runs [parts], which device_metal_room answered NX_FITS for, as the work
   of [v], the value after the last one it received: NX_OK, or NX_FAILED with
   [*failure] set to the device's message. Once the device recorded a
   failure every call answers NX_FAILED with the first failure's message,
   which lives while the process runs, and runs nothing. One call at a time;
   it waits while the device's 1,024 command buffers are in flight. */
int device_metal_submit(void *self, uint64_t v, const struct nx_wait *waits,
                        int nwaits, const struct nx_part *parts, int nparts,
                        const uint64_t *handles, int nhandles,
                        const char **failure);

#endif
