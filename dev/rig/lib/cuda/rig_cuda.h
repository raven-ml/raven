/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to a CUDA device from C.

   The device's room check, submit and commit, over the structures and
   codes of rig_edge.h: the only way work reaches the device. [self] is
   Rig_cuda.self. They are called without the OCaml runtime: they call no
   function of it and read no OCaml value. They run one call at a time, in
   value order; Rig_cuda.sleep may run meanwhile.

   Queue 0 is the stream "COMPUTE:0", queue 1 the stream "COPY:0". A part is
   a fill, called with the queue's CUstream and the device's context
   current (rig_cuda_abi.mli), or a copy between handles; it has no
   words, ring units or segment bytes. A wait is RIG_WORD on a 64-bit word
   the device maps. [handles] is ignored: CUDA's work names its
   memory by address. */

#ifndef RIG_CUDA_H
#define RIG_CUDA_H

#include <rig_edge.h>

/* RIG_NEVER if a part has words, ring units or segment bytes, or is on no
   queue of the device; RIG_FITS otherwise. */
int rig_cuda_room(void *self, const struct rig_part *parts, int n);

/* Runs [parts], which rig_cuda_room answered RIG_FITS for and whose [after]
   name only earlier parts, as the work of [v], the value after the last one
   it received. The word shows [v] once a commit wrote it: the call commits
   [v] itself once 64 values passed the last commit, or if [v] has a copy,
   and answers RIG_COMMITTED then, RIG_OK otherwise, or RIG_FAILED with
   [*failure] set to the failing step and CUDA's error. After a failure
   every call enqueues none of its parts and answers RIG_FAILED with the
   first failure's message, which lives as long as the process. A call that
   answers RIG_FAILED still writes [v] after its waits, every earlier value
   and the work it queued, unless the context failed or CUDA refuses a call
   that orders the write. It may block while a stream is full. */
int rig_cuda_submit(void *self, uint64_t v, const struct rig_wait *waits,
                       int nwaits, const struct rig_part *parts, int nparts,
                       const uint64_t *handles, int nhandles,
                       const char **failure);

/* Writes [v], at most the last value rig_cuda_submit received, into the
   word after the work of every value up to it, unless a commit wrote [v]
   or a later value: RIG_OK, or RIG_FAILED with [*failure] set as
   rig_cuda_submit's, after which every call fails. It may block while a
   stream is full. */
int rig_cuda_commit(void *self, uint64_t v, const char **failure);

#endif
