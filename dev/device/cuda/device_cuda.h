/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to a CUDA device from C.

   Device_cuda.room and Device_cuda.submit, for a caller that holds its
   submissions in C, over the structures and codes of nx_edge.h. [self] is
   Device_cuda.self. Both are called without the OCaml runtime: they call no
   function of it and read no OCaml value.

   Queue 0 is the stream "COMPUTE:0", queue 1 the stream "COPY:0". A part is
   a fill, called with the queue's CUstream and the device's context
   current (device_cuda_abi.mli), or a copy between handles; it has no
   words, ring units or segment bytes. A wait is NX_WORD or NX_EQUAL on a
   64-bit word the device maps. [handles] is ignored: CUDA's work names its
   memory by address. */

#ifndef DEVICE_CUDA_H
#define DEVICE_CUDA_H

#include <nx_edge.h>

/* NX_NEVER if a part has words, ring units or segment bytes, is on no
   queue of the device, or has an [after] index not below its own part's;
   NX_FITS otherwise. */
int device_cuda_room(void *self, const struct nx_part *parts, int n);

/* Runs [parts], which device_cuda_room answered NX_FITS for, as the work
   of [v], the value after the last one it received: NX_OK, or NX_FAILED with
   [*failure] set to CUDA's error. After a failure every call answers
   NX_FAILED with the first failure's message, which lives as long as the
   process. It may block while a stream is full. */
int device_cuda_submit(void *self, uint64_t v, const struct nx_wait *waits,
                       int nwaits, const struct nx_part *parts, int nparts,
                       const uint64_t *handles, int nhandles,
                       const char **failure);

#endif
