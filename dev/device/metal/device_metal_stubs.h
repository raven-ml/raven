/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* An open device, as the stubs share it. Objective-C, on macOS only. */

#ifndef DEVICE_METAL_STUBS_H
#define DEVICE_METAL_STUBS_H

#import <Metal/Metal.h>

#include "device_metal.h"
#include "device_metal_ring.h"

/* The command buffers a device's queue holds, and its ring's slots. */
enum { ring_slots = 1024 };

/* A device: never freed, since a completion handler, another device or the
   C caller may reach it after its loss. Its residency set has a mutex of
   its own, which no completion handler takes; [changed] says whether the
   set holds additions it has not committed. */
struct device_metal {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLFence> fence;
  id<MTLResidencySet> set;
  pthread_mutex_t set_mutex;
  int changed;
  id<MTLBuffer> word;
  uint64_t last; /* the last value submit received */
  struct device_metal_ring ring;
  struct device_metal_slot slots[ring_slots];
};

/* What a fill's [queue] points to: the open encoder first, as fills read
   it, then the command buffer that holds it and its slot (-1 for none). */
struct device_metal_queue {
  id<MTLComputeCommandEncoder> encoder;
  id<MTLCommandBuffer> buffer;
  int slot;
  struct device_metal *d;
};

/* The [split] of the device's capability record. */
int device_metal_split(void *queue, uint64_t *start, uint64_t *end);

#endif
