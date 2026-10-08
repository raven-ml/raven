/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to a CUDA device from C.

   Device_cuda.room and Device_cuda.submit, for a caller that holds its
   submissions in C, over the structures and codes of nx_edge.h. [self] is
   Device_cuda.self. Both are called without the OCaml runtime: they call no
   function of it and read no OCaml value. They run one call at a time, in
   value order, with Device_cuda.room and Device_cuda.submit among them;
   Device_cuda.sleep may run meanwhile.

   Queue 0 is the stream "COMPUTE:0", queue 1 the stream "COPY:0". A part is
   a fill, called with the queue's CUstream and the device's context
   current (device_cuda_abi.mli), or a copy between handles; it has no
   words, ring units or segment bytes. A wait is NX_WORD on a 64-bit word
   the device maps. [handles] is ignored: CUDA's work names its
   memory by address. */

#ifndef DEVICE_CUDA_H
#define DEVICE_CUDA_H

#include <nx_edge.h>

/* NX_NEVER if a part has words, ring units or segment bytes, or is on no
   queue of the device; NX_FITS otherwise. */
int device_cuda_room(void *self, const struct nx_part *parts, int n);

/* Runs [parts], which device_cuda_room answered NX_FITS for and whose
   [after] name only earlier parts, as the work of [v], the value after the
   last one it received: NX_OK, or NX_FAILED with [*failure] set to the
   failing step and CUDA's error. After a failure every call enqueues none of
   its parts and answers NX_FAILED with the first failure's message, which
   lives as long as the process. A call that answers NX_FAILED still writes
   [v] after its waits, every earlier value and the work it queued, unless
   the context failed or CUDA refuses a call that orders the write. It may
   block while a stream is full. */
int device_cuda_submit(void *self, uint64_t v, const struct nx_wait *waits,
                       int nwaits, const struct nx_part *parts, int nparts,
                       const uint64_t *handles, int nhandles,
                       const char **failure);

#endif
