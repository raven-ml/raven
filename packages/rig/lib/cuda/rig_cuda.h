/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A CUDA device's C functions of rig_edge.h. rig_cuda.mli's Work and
   Failure sections state what they do. */

#ifndef RIG_CUDA_H
#define RIG_CUDA_H

#include <rig_edge.h>

int rig_cuda_room(void *self, const struct rig_part *parts, int n,
                  void *scratch, const uint8_t *args);

int rig_cuda_submit(void *self, uint64_t v, const struct rig_wait *waits,
                    int nwaits, const struct rig_part *parts, int nparts,
                    void *scratch, const uint8_t *args,
                    const uint64_t *slots, int nslots, uint64_t *times,
                    const char **failure);

int rig_cuda_commit(void *self, uint64_t v, const char **failure);

int rig_cuda_settle(void *self, uint64_t v, const char **failure);

#endif
