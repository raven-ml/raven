/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to a device: everything that runs without the OCaml runtime.

   A submission runs its fills in order in one serial compute encoder per
   command buffer, which waits for the device's fence and updates it, so
   each command buffer runs after the ones before. The fills of a command
   buffer share its encoder: on the M1 Max a fence between two encoders
   costs about 25 µs, a dependent dispatch in one encoder about 2 µs. Its
   command buffers take ring slots in commit order; the last carries the
   submission's value. A failure drops the open command buffer uncommitted
   and completes its slot as failed, so the word stops before the value. */

#define _GNU_SOURCE

#include "rig_metal.h"

#ifdef __APPLE__

#include <math.h>
#include <stdio.h>

#include "rig_metal_stubs.h"

static uint64_t host_ns(CFTimeInterval seconds) {
  return (uint64_t)llround(seconds * 1e9);
}

/* The open command buffer

   A device's work goes into its open command buffer, which stays open
   across submits, under the device's open mutex, until one of four commits:
   a commit of a value in it, a submit that finds fewer than [backlog]
   committed command buffers uncompleted, the handler of a command buffer
   that completes while one is open, and a buffer that holds [open_bound]
   values. Metal runs nothing before a commit, and one round trip costs tens
   of dispatches, so the buffer carries as many values as the GPU's own
   backlog allows. The buffer and its encoder outlive the autorelease pools
   of the submits that fill them: each is retained once made and released
   once ended. A buffer's slot carries the last value whose work it ends. */

/* The values an open command buffer holds before it is committed. */
static const int open_bound = 256;

/* The committed command buffers below which a submit commits its own: the
   one running and two queued behind it. A completion handler runs about
   100 µs after the GPU ends its buffer, so a finished buffer still counts;
   with two, buffers shorter than that leave the GPU idle until the handler
   commits the next (replay/metal/kernel-pipelined-100 8.1 -> 9.9 ms). */
static const int backlog = 3;

/* Ends [q]'s encoder and command buffer, uncommitted, completing the slot as
   failed with [why]; the device's first failure. */
static const char *drop(struct rig_metal_queue *q, const char *why) {
  if (q->encoder) {
    [q->encoder endEncoding];
    [q->encoder release];
  }
  if (q->buffer) [q->buffer release];
  if (q->slot < 0) q->slot = rig_metal_ring_take(&q->d->ring);
  rig_metal_ring_complete(&q->d->ring, q->slot, why, 0, 0);
  *q = (struct rig_metal_queue){nil, nil, -1, q->d};
  return rig_metal_ring_failure(&q->d->ring);
}

static const char *fail(struct rig_metal_queue *q, const char *why) {
  q->d->open_values = 0;
  return drop(q, why);
}

static const char *open_encoder(struct rig_metal_queue *q) {
  q->encoder =
      [q->buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  if (q->encoder == nil) return fail(q, "Metal made no compute encoder");
  [q->encoder retain];
  [q->encoder waitForFence:q->d->fence];
  return NULL;
}

static void close_encoder(struct rig_metal_queue *q) {
  [q->encoder updateFence:q->d->fence];
  [q->encoder endEncoding];
  [q->encoder release];
  q->encoder = nil;
}

static void unlock_open(struct rig_metal *d);

/* Starts a command buffer, after taking its slot. Its completion handler
   runs on a thread of Metal's: it completes the slot, then commits the
   open command buffer if it holds work and no submit holds it, so the GPU
   finds work queued behind the one it ended. A handler that finds no work
   takes no mutex: a submit waiting for a handler's mutex took 4 µs more on
   the M1 Max. A submit that adds work after the handler looked finds the
   slot completed, and commits unless buffers whose handlers come later are
   in flight. */
static const char *begin(struct rig_metal_queue *q) {
  struct rig_metal *d = q->d;
  int slot = rig_metal_ring_take(&d->ring);
  q->slot = slot;
  q->buffer = [d->queue commandBuffer];
  if (q->buffer == nil) return fail(q, "Metal made no command buffer");
  [q->buffer retain];
  [q->buffer addCompletedHandler:^(id<MTLCommandBuffer> b) {
    int failed = b.status == MTLCommandBufferStatusError;
    char why[512];
    if (failed)
      snprintf(why, sizeof why, "the GPU's work failed: %s",
               b.error.localizedDescription.UTF8String ?: "no reason given");
    rig_metal_ring_complete(&d->ring, slot, failed ? why : NULL,
                               host_ns(b.GPUStartTime), host_ns(b.GPUEndTime));
    if (atomic_load_explicit(&d->open_values, memory_order_relaxed) == 0)
      return;
    atomic_store(&d->wanted, 1);
    if (pthread_mutex_trylock(&d->open_mutex) == 0) unlock_open(d);
  }];
  return NULL;
}

/* Commits [q]'s command buffer as the end of [v]'s work (0: of none), with
   its times written at [start] and [end]. A buffer without an encoder, such
   as one of values with no parts, never reaches the GPU: the ring orders
   its completion after the buffers before it. */
static void commit(struct rig_metal_queue *q, uint64_t v, uint64_t *start,
                   uint64_t *end) {
  struct rig_metal_slot *s = &q->d->ring.slots[q->slot];
  s->v = v;
  s->start = start;
  s->end = end;
  if (q->encoder) close_encoder(q);
  [q->buffer commit];
  [q->buffer release];
  *q = (struct rig_metal_queue){nil, nil, -1, q->d};
}

/* Commits the device's open command buffer. The open mutex is held. */
static void commit_open(struct rig_metal *d) {
  if (d->open.slot < 0) return;
  @autoreleasepool {
    commit(&d->open, d->open_v, NULL, NULL);
  }
  d->open_values = 0;
}

/* Releases the open mutex, then commits the open command buffer for each
   handler that found the mutex held: a handler marks its want before it
   tries the mutex, and every holder looks at the mark after releasing it,
   so no handler's commit is lost. */
static void unlock_open(struct rig_metal *d) {
  pthread_mutex_unlock(&d->open_mutex);
  while (atomic_exchange(&d->wanted, 0)) {
    pthread_mutex_lock(&d->open_mutex);
    if (d->open_values > 0) commit_open(d);
    pthread_mutex_unlock(&d->open_mutex);
  }
}

int rig_metal_split(void *queue, uint64_t *start, uint64_t *end) {
  struct rig_metal_queue *q = queue;
  commit(q, q->d->open_v, start, end);
  q->d->open_values = 0;
  return begin(q) || open_encoder(q) ? 1 : 0;
}

int rig_metal_room(void *self, const struct rig_part *parts, int n,
                   const uint8_t *args) {
  (void)args;
  (void)self;
  for (int i = 0; i < n; i++) {
    const struct rig_part *p = &parts[i];
    if (p->queue != 0 || p->kind != RIG_FILL || p->fill.ring_units != 0 ||
        p->fill.segment_bytes != 0)
      return RIG_NEVER;
  }
  return RIG_FITS;
}

/* Commits the residency set's additions, so that the work of this
   submission finds every allocation made before it resident. */
static void commit_residency(struct rig_metal *d) {
  pthread_mutex_lock(&d->set_mutex);
  if (d->changed) [d->set commit];
  d->changed = 0;
  pthread_mutex_unlock(&d->set_mutex);
}

/* Runs the fills of [v] in the open command buffer's encoder, opening
   either if none is, and commits the buffer if its work would otherwise wait:
   whether it did. A fill that returns 0 without an open command buffer broke
   its contract after a failed split. The open mutex is held. */
static const char *run(struct rig_metal *d, uint64_t v,
                       const struct rig_part *parts, int n, int *committed) {
  struct rig_metal_queue *q = &d->open;
  const char *why = NULL;
  @try {
    commit_residency(d);
    if (q->slot < 0) why = begin(q);
    if (why == NULL && n > 0 && q->encoder == nil) why = open_encoder(q);
    for (int i = 0; i < n && why == NULL; i++) {
      int rc = parts[i].fill.fn(q, parts[i].fill.arg, v);
      if (rc == 0 && q->slot >= 0) continue;
      char text[64];
      snprintf(text, sizeof text, "running a fill: it returned %d", rc);
      why = fail(q, text);
    }
    if (why == NULL) {
      d->open_v = v;
      d->open_values++;
      *committed = d->open_values >= open_bound ||
                   rig_metal_ring_taken(&d->ring) - 1 < backlog;
      if (*committed) commit_open(d);
    }
  } @catch(NSException * e) {
    char text[512];
    snprintf(text, sizeof text, "Metal raised an exception: %s",
             e.reason.UTF8String ?: "no reason given");
    why = fail(q, text);
  }
  return why;
}

int rig_metal_submit(void *self, uint64_t v, const struct rig_wait *waits,
                        int nwaits, const struct rig_part *parts, int nparts,
                        const uint8_t *args, const uint64_t *slots, int nslots,
                        const uint64_t *handles, int nhandles,
                        const char **failure) {
  (void)args;
  (void)slots;
  (void)nslots;
  (void)waits, (void)nwaits, (void)handles, (void)nhandles;
  struct rig_metal *d = self;
  int committed = 0;
  d->last = v;
  const char *why = rig_metal_ring_failure(&d->ring);
  if (why == NULL) {
    pthread_mutex_lock(&d->open_mutex);
    @autoreleasepool {
      why = run(d, v, parts, nparts, &committed);
    }
    unlock_open(d);
  }
  if (why != NULL) {
    *failure = why;
    return RIG_FAILED;
  }
  return committed ? RIG_COMMITTED : RIG_OK;
}

int rig_metal_commit(void *self, uint64_t v, const char **failure) {
  struct rig_metal *d = self;
  (void)v;
  pthread_mutex_lock(&d->open_mutex);
  commit_open(d);
  unlock_open(d);
  const char *why = rig_metal_ring_failure(&d->ring);
  if (why == NULL) return RIG_OK;
  *failure = why;
  return RIG_FAILED;
}

/* Drops the open command buffer uncommitted, completing its slot as failed,
   so that the ring drains. */
void rig_metal_drop(struct rig_metal *d) {
  pthread_mutex_lock(&d->open_mutex);
  @autoreleasepool {
    if (d->open.slot >= 0) drop(&d->open, "the device stopped");
  }
  d->open_values = 0;
  unlock_open(d);
}

#else

int rig_metal_room(void *self, const struct rig_part *parts, int n,
                   const uint8_t *args) {
  (void)args;
  (void)self, (void)parts, (void)n;
  return RIG_NEVER;
}

int rig_metal_submit(void *self, uint64_t v, const struct rig_wait *waits,
                        int nwaits, const struct rig_part *parts, int nparts,
                        const uint8_t *args, const uint64_t *slots, int nslots,
                        const uint64_t *handles, int nhandles,
                        const char **failure) {
  (void)args;
  (void)slots;
  (void)nslots;
  (void)self, (void)v, (void)waits, (void)nwaits, (void)parts, (void)nparts,
      (void)handles, (void)nhandles;
  *failure = "Metal exists on macOS only";
  return RIG_FAILED;
}

int rig_metal_commit(void *self, uint64_t v, const char **failure) {
  (void)self, (void)v;
  *failure = "Metal exists on macOS only";
  return RIG_FAILED;
}

#endif
