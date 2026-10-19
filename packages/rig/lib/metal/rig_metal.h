/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A Metal device's C functions of rig_edge.h. rig_metal.mli's Work and
   Failure sections state what they do. */

#ifndef RIG_METAL_H
#define RIG_METAL_H

#include <rig_edge.h>

int rig_metal_room(void *self, int nwaits, const struct rig_part *parts,
                   int n, void *scratch, const uint8_t *args);

int rig_metal_submit(void *self, uint64_t v, const struct rig_wait *waits,
                     int nwaits, const struct rig_part *parts, int nparts,
                     void *scratch, uint8_t *args,
                     const uint64_t *slots, int nslots, uint64_t *times,
                     const char **failure);

int rig_metal_commit(void *self, uint64_t v, const char **failure);

#endif
