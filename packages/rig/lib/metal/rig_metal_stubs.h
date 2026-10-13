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

struct rig_metal_run;

/* What a fill's [queue] points to: the open encoder first, as fills read
   it, then the command buffer that holds it and its slot (-1 for none). */
struct rig_metal_queue {
  id<MTLComputeCommandEncoder> encoder;
  id<MTLCommandBuffer> buffer;
  int slot;
  struct rig_metal_run *r;
};

/* What a launch of a function reads, the [launch] of its entry: its
   pipeline, the most threads a threadgroup of it holds, and the most
   threadgroup memory a launch may add to what the function declares, in
   bytes; and the bytes of its parameters, -1 where Metal does not say.
   Made by the function's first entry, freed by its image's unload. */
struct rig_metal_entry {
  id<MTLComputePipelineState> pipeline;
  uint32_t threads, shared;
  int64_t params;
};

/* What a device runs work with, from its open to its stop and the end of
   the work committed before it: each committed command buffer holds it
   until its completion handler returns, and the device until its stop
   ([holds]). Its residency set has a mutex of its own, which no completion
   handler takes; [changed] says whether the set holds additions it has not
   committed. */
struct rig_metal_run {
  _Atomic int holds;
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLFence> fence;
  id<MTLResidencySet> set;
  pthread_mutex_t set_mutex;
  int changed;
  uint64_t last; /* the last value submit received */
  pthread_mutex_t open_mutex; /* held by a submit, a commit and a stop */
  struct rig_metal_queue open; /* the open command buffer, slot -1 for none */
  uint64_t open_v;             /* the last value whose work is encoded */
  _Atomic int open_values;     /* the values the open buffer ends */
  _Atomic int wanted;          /* a handler found the open mutex held */
  struct rig_metal_ring ring;
  struct rig_metal_slot slots[ring_slots];
};

/* A device, kept for the process with its word: rig never frees the word.
   Its run state goes at its stop, and rig calls the C functions no more. */
struct rig_metal {
  const struct rig_driver *driver;
  id<MTLBuffer> word;
  uint64_t *at; /* the word's contents */
  struct rig_metal_run *run; /* NULL once stopped */
};

/* Takes a hold of [r]'s, as a committed command buffer does. */
void rig_metal_hold(struct rig_metal_run *r);

/* Gives back a hold of [r]'s: the last frees it. */
void rig_metal_release(struct rig_metal_run *r);

/* The [split] of the device's capability record. */
int rig_metal_split(void *queue, uint64_t *start, uint64_t *end);

/* Drops the open command buffer uncommitted, for a stop. */
void rig_metal_drop(struct rig_metal_run *r);

#endif
