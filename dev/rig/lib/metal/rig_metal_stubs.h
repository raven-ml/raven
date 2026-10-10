/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* An open device, as the stubs share it. Objective-C, on macOS only. */

#ifndef RIG_METAL_STUBS_H
#define RIG_METAL_STUBS_H

#import <Metal/Metal.h>

#include <stdatomic.h>

#include "rig_metal.h"
#include "rig_metal_ring.h"

/* The command buffers a device's queue holds, and its ring's slots. A
   queue holds 64 by default; a fill that splits once per kernel, as one
   profiling its kernels does, would then wait after 64 kernels. 1,024 lets
   such work run ahead of the host, at 32 bytes of ring per command buffer;
   past it, a submission waits for the oldest to complete. */
enum { ring_slots = 1024 };

/* What a fill's [queue] points to: the open encoder first, as fills read
   it, then the command buffer that holds it and its slot (-1 for none). */
struct rig_metal_queue {
  id<MTLComputeCommandEncoder> encoder;
  id<MTLCommandBuffer> buffer;
  int slot;
  struct rig_metal *d;
};

/* What a launch of a function reads, the [launch] of its entry: its
   pipeline, the most threads a threadgroup of it holds, and the most
   threadgroup memory a launch may add to what the function declares, in
   bytes. Made by the function's first entry, freed by its image's
   unload. */
struct rig_metal_entry {
  id<MTLComputePipelineState> pipeline;
  uint32_t threads, shared;
};

/* A device: never freed, since a completion handler, another device or the
   C caller may reach it after its loss. Its residency set has a mutex of
   its own, which no completion handler takes; [changed] says whether the
   set holds additions it has not committed. */
struct rig_metal {
  const struct rig_driver *driver;
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLFence> fence;
  id<MTLResidencySet> set;
  pthread_mutex_t set_mutex;
  int changed;
  id<MTLBuffer> word;
  uint64_t last; /* the last value submit received */
  pthread_mutex_t open_mutex; /* held by a submit, a commit and a stop */
  struct rig_metal_queue open; /* the open command buffer, slot -1 for none */
  uint64_t open_v;             /* the last value whose work is encoded */
  _Atomic int open_values;     /* the values the open buffer ends */
  _Atomic int wanted;          /* a handler found the open mutex held */
  struct rig_metal_ring ring;
  struct rig_metal_slot slots[ring_slots];
};

/* The [split] of the device's capability record. */
int rig_metal_split(void *queue, uint64_t *start, uint64_t *end);

/* Drops the open command buffer uncommitted, for a stop. */
void rig_metal_drop(struct rig_metal *d);

#endif
