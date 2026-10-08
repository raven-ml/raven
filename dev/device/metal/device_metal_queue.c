/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to a device: everything that runs without the OCaml runtime.

   A submission runs its fills in order, each in a serial compute encoder
   that waits for the device's fence and updates it, so each runs after the
   encoders before it. Its command buffers take ring slots in commit order;
   the last carries the submission's value. A failure drops the open
   command buffer uncommitted and completes its slot as failed, so the word
   stops before the value. */

#define _GNU_SOURCE

#include "device_metal.h"

#ifdef __APPLE__

#include <math.h>
#include <stdio.h>

#include "device_metal_stubs.h"

static uint64_t host_ns(CFTimeInterval seconds) {
  return (uint64_t)llround(seconds * 1e9);
}

/* Completes the open command buffer's slot as failed with [why], taking a
   slot if none is open, and drops the buffer; the device's first failure. */
static const char *fail(struct device_metal_queue *q, const char *why) {
  if (q->encoder) [q->encoder endEncoding];
  if (q->slot < 0) q->slot = device_metal_ring_take(&q->d->ring);
  device_metal_ring_complete(&q->d->ring, q->slot, why, 0, 0);
  *q = (struct device_metal_queue){nil, nil, -1, q->d};
  return device_metal_ring_failure(&q->d->ring);
}

static const char *open_encoder(struct device_metal_queue *q) {
  q->encoder =
      [q->buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  if (q->encoder == nil) return fail(q, "Metal made no compute encoder");
  [q->encoder waitForFence:q->d->fence];
  return NULL;
}

static void close_encoder(struct device_metal_queue *q) {
  [q->encoder updateFence:q->d->fence];
  [q->encoder endEncoding];
  q->encoder = nil;
}

/* Starts a command buffer, after taking its slot. Its completion handler
   runs on a thread of Metal's and completes the slot. */
static const char *begin(struct device_metal_queue *q) {
  struct device_metal *d = q->d;
  int slot = device_metal_ring_take(&d->ring);
  q->slot = slot;
  q->buffer = [d->queue commandBuffer];
  if (q->buffer == nil) return fail(q, "Metal made no command buffer");
  [q->buffer addCompletedHandler:^(id<MTLCommandBuffer> b) {
    const char *why = NULL;
    if (b.status == MTLCommandBufferStatusError)
      why = b.error.localizedDescription.UTF8String
                ?: "Metal reported the command buffer failed";
    device_metal_ring_complete(&d->ring, slot, why, host_ns(b.GPUStartTime),
                               host_ns(b.GPUEndTime));
  }];
  return NULL;
}

/* Commits the open command buffer as [v]'s, with its times written at
   [start] and [end]. A buffer without an encoder, such as the one of a
   submission of no parts, never reaches the GPU: the ring orders its
   completion after the buffers before it. */
static void commit(struct device_metal_queue *q, uint64_t v, uint64_t *start,
                   uint64_t *end) {
  struct device_metal_slot *s = &q->d->ring.slots[q->slot];
  s->v = v;
  s->start = start;
  s->end = end;
  [q->buffer commit];
  *q = (struct device_metal_queue){nil, nil, -1, q->d};
}

int device_metal_split(void *queue, uint64_t *start, uint64_t *end) {
  struct device_metal_queue *q = queue;
  close_encoder(q);
  commit(q, 0, start, end);
  return begin(q) || open_encoder(q) ? 1 : 0;
}

int device_metal_room(void *self, const struct nx_part *parts, int n) {
  (void)self;
  for (int i = 0; i < n; i++) {
    const struct nx_part *p = &parts[i];
    if (p->queue != 0 || p->fill == NULL || p->n != 0 || p->ring_units != 0 ||
        p->segment_bytes != 0 || p->copy_bytes != 0)
      return NX_NEVER;
  }
  return NX_FITS;
}

/* Commits the residency set's additions, so that the work of this
   submission finds every allocation made before it resident. */
static void commit_residency(struct device_metal *d) {
  pthread_mutex_lock(&d->set_mutex);
  if (d->changed) [d->set commit];
  d->changed = 0;
  pthread_mutex_unlock(&d->set_mutex);
}

/* Runs the fills, each in an encoder of its own. A fill that returns 0
   without an open command buffer broke its contract after a failed split. */
static const char *run(struct device_metal *d, uint64_t v,
                       const struct nx_part *parts, int n) {
  struct device_metal_queue q = {nil, nil, -1, d};
  const char *why = NULL;
  @try {
    commit_residency(d);
    why = begin(&q);
    for (int i = 0; i < n && why == NULL; i++) {
      why = open_encoder(&q);
      if (why) break;
      int rc = parts[i].fill(&q, parts[i].arg, v);
      if (rc == 0 && q.slot >= 0) {
        close_encoder(&q);
        continue;
      }
      char text[64];
      snprintf(text, sizeof text, "a fill failed with %d", rc);
      why = fail(&q, text);
    }
    if (why == NULL) commit(&q, v, NULL, NULL);
  } @catch(NSException * e) {
    why = fail(&q, e.reason.UTF8String ?: "Metal raised an exception");
  }
  return why;
}

int device_metal_submit(void *self, uint64_t v, const struct nx_wait *waits,
                        int nwaits, const struct nx_part *parts, int nparts,
                        const uint64_t *handles, int nhandles,
                        const char **failure) {
  (void)waits, (void)nwaits, (void)handles, (void)nhandles;
  struct device_metal *d = self;
  d->last = v;
  const char *why = device_metal_ring_failure(&d->ring);
  if (why == NULL) {
    @autoreleasepool {
      why = run(d, v, parts, nparts);
    }
  }
  if (why == NULL) return NX_OK;
  *failure = why;
  return NX_FAILED;
}

#else

int device_metal_room(void *self, const struct nx_part *parts, int n) {
  (void)self, (void)parts, (void)n;
  return NX_NEVER;
}

int device_metal_submit(void *self, uint64_t v, const struct nx_wait *waits,
                        int nwaits, const struct nx_part *parts, int nparts,
                        const uint64_t *handles, int nhandles,
                        const char **failure) {
  (void)self, (void)v, (void)waits, (void)nwaits, (void)parts, (void)nparts,
      (void)handles, (void)nhandles;
  *failure = "Metal exists on macOS only";
  return NX_FAILED;
}

#endif
