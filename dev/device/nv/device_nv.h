/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to an NVIDIA device from C.

   Device_nv.room and Device_nv.submit, for a caller that holds its
   submissions in C, over the structures and codes of nx_edge.h. [self] is
   Device_nv.self. Both are called without the OCaml runtime: they call no
   function of it and read no OCaml value.

   Queue 0 is the channel "COMPUTE:0", queue 1 the channel "COPY:0". A part
   is ring entries, two words each, low first (Device_nv_abi.Gpfifo), or a
   copy between handles on queue 1; it is no fill and declares no ring units
   or segment bytes. A wait is NX_WORD on a 64-bit word the device maps,
   compared circularly. [handles] is ignored: the device's work names its
   memory by address. */

#ifndef DEVICE_NV_H
#define DEVICE_NV_H

#include <nx_edge.h>

/* NX_NEVER if a part is a fill, declares ring units or segment bytes, has an
   odd number of words, is a copy on queue 0, is on no queue of the device,
   has an [after] index not below its own part's, or if the parts exceed the
   device's empty rings or number more than 65,535; NX_LATER if they fit once a value the device was
   given is reached, as its timeline word reads now; NX_FITS otherwise. */
int device_nv_room(void *self, const struct nx_part *parts, int n);

/* Writes [parts], which device_nv_room answered NX_FITS for with nothing
   submitted since, into the device's rings as the work of [v], the value
   after the last one it received, and wakes the channels. It answers NX_OK:
   its stores are to this machine's memory and cannot fail. It never
   blocks. */
int device_nv_submit(void *self, uint64_t v, const struct nx_wait *waits,
                     int nwaits, const struct nx_part *parts, int nparts,
                     const uint64_t *handles, int nhandles,
                     const char **failure);

#endif
