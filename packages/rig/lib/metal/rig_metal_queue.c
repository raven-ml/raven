/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Submitting to a device: everything that runs without the OCaml runtime,
   on macOS only.

   A submission runs its parts, fills and launches, in order in one serial
   compute encoder per command buffer, which waits for the device's fence
   and updates it, so each command buffer runs after the ones before. The
   parts of a command buffer share its encoder: on the M1 Max a fence
   between two encoders costs about 25 µs, a dependent dispatch in one
   encoder about 2 µs. Its command buffers take ring slots in commit order;
   the last carries the submission's value. A failure drops the open
   command buffer uncommitted and completes its slot as failed, so the word
   stops before the value. */

#define _GNU_SOURCE

#include "rig_metal.h"

#ifdef __APPLE__

#include <math.h>
#include <stdio.h>
#include <string.h>

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
  if (q->slot < 0) q->slot = rig_metal_ring_take(&q->r->ring);
  rig_metal_ring_complete(&q->r->ring, q->slot, why, 0, 0);
  *q = (struct rig_metal_queue){nil, nil, -1, q->r};
  return rig_metal_ring_failure(&q->r->ring);
}

static const char *fail(struct rig_metal_queue *q, const char *why) {
  q->r->open_values = 0;
  return drop(q, why);
}

static const char *open_encoder(struct rig_metal_queue *q) {
  q->encoder =
      [q->buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  if (q->encoder == nil) return fail(q, "Metal made no compute encoder");
  [q->encoder retain];
  [q->encoder waitForFence:q->r->fence];
  return NULL;
}

static void close_encoder(struct rig_metal_queue *q) {
  [q->encoder updateFence:q->r->fence];
  [q->encoder endEncoding];
  [q->encoder release];
  q->encoder = nil;
}

static void unlock_open(struct rig_metal_run *r);

/* Starts a command buffer, after taking its slot. Its completion handler
   runs on a thread of Metal's: it completes the slot, then commits the
   open command buffer if it holds work and no submit holds it, so the GPU
   finds work queued behind the one it ended, and last gives back the hold
   its commit took. A handler that finds no work takes no mutex: a submit
   waiting for a handler's mutex took 4 µs more on the M1 Max. A submit
   that adds work after the handler looked finds the slot completed, and
   commits unless buffers whose handlers come later are in flight. */
static const char *begin(struct rig_metal_queue *q) {
  struct rig_metal_run *r = q->r;
  int slot = rig_metal_ring_take(&r->ring);
  q->slot = slot;
  atomic_store_explicit(&r->ring.slots[slot].arena_end, r->arena_put,
                        memory_order_relaxed);
  q->buffer = [r->queue commandBuffer];
  if (q->buffer == nil) return fail(q, "Metal made no command buffer");
  [q->buffer retain];
  [q->buffer addCompletedHandler:^(id<MTLCommandBuffer> b) {
    int failed = b.status == MTLCommandBufferStatusError;
    char why[512];
    if (failed)
      snprintf(why, sizeof why, "the GPU's work failed: %s",
               b.error.localizedDescription.UTF8String ?: "no reason given");
    rig_metal_ring_complete(&r->ring, slot, failed ? why : NULL,
                               host_ns(b.GPUStartTime), host_ns(b.GPUEndTime));
    if (atomic_load_explicit(&r->open_values, memory_order_relaxed) > 0) {
      atomic_store(&r->wanted, 1);
      if (pthread_mutex_trylock(&r->open_mutex) == 0) unlock_open(r);
    }
    rig_metal_release(r);
  }];
  return NULL;
}

/* Commits [q]'s command buffer as the end of [v]'s work (0: of none), with
   its times written at [start] and [end]. A buffer without an encoder, such
   as one of values with no parts, never reaches the GPU: the ring orders
   its completion after the buffers before it. */
static void commit(struct rig_metal_queue *q, uint64_t v, uint64_t *start,
                   uint64_t *end) {
  struct rig_metal_slot *s = &q->r->ring.slots[q->slot];
  s->v = v;
  s->start = start;
  s->end = end;
  if (q->encoder) close_encoder(q);
  /* The hold comes first, since the handler may run before [commit]
     returns; a commit that raises gives it back, as no handler will. */
  rig_metal_hold(q->r);
  @try {
    [q->buffer commit];
  } @catch (NSException *e) {
    rig_metal_release(q->r);
    @throw;
  }
  [q->buffer release];
  *q = (struct rig_metal_queue){nil, nil, -1, q->r};
}

/* Commits the device's open command buffer. The open mutex is held. */
static void commit_open(struct rig_metal_run *r) {
  if (r->open.slot < 0) return;
  @autoreleasepool {
    commit(&r->open, r->open_v, NULL, NULL);
  }
  r->open_values = 0;
}

/* Releases the open mutex, then commits the open command buffer for each
   handler that found the mutex held: a handler marks its want before it
   tries the mutex, and every holder looks at the mark after releasing it,
   so no handler's commit is lost. */
static void unlock_open(struct rig_metal_run *r) {
  pthread_mutex_unlock(&r->open_mutex);
  while (atomic_exchange(&r->wanted, 0)) {
    pthread_mutex_lock(&r->open_mutex);
    if (r->open_values > 0) commit_open(r);
    pthread_mutex_unlock(&r->open_mutex);
  }
}

int rig_metal_split(void *queue, uint64_t *start, uint64_t *end) {
  struct rig_metal_queue *q = queue;
  commit(q, q->r->open_v, start, end);
  q->r->open_values = 0;
  return begin(q) || open_encoder(q) ? 1 : 0;
}

/* A threadgroup's memory length, which Metal takes in multiples of 16
   bytes. */
static uint64_t shared_length(uint32_t shared) {
  return ((uint64_t)shared + 15) & ~(uint64_t)15;
}

/* Whether the launch [p], its block in [args], has a grid and threadgroups
   of no empty axis, at most its entry's threads in a threadgroup, and at
   most its threadgroup memory. Each axis is at most the bound before the
   product is taken, so the product of three fits 64 bits. */
static int launch_fits(const struct rig_part *p, const uint8_t *args) {
  const struct rig_metal_entry *e = p->launch.launch;
  const struct rig_block *b = (const void *)(args + p->launch.args.block);
  uint64_t threads = 1;
  for (int k = 0; k < 3; k++) {
    if (b->groups[k] == 0 || b->threads[k] == 0 || b->threads[k] > e->threads)
      return 0;
    threads *= b->threads[k];
  }
  return threads <= e->threads && shared_length(b->shared) <= e->shared;
}

static int fits(const struct rig_part *p, const uint8_t *args) {
  if (p->queue != 0) return 0;
  if (p->kind == RIG_LAUNCH) return launch_fits(p, args);
  return p->kind == RIG_FILL;
}

/* The most bytes [setBytes:] copies: "Maximum length of inlined buffer
   contents using setBytes", 4 KB in every family of Apple's Metal Feature
   Set Tables. */
#define SET_BYTES_MAX 4096

/* The parameter arena

   A launch's parameters past SET_BYTES_MAX go into the device's arena, a
   shared buffer whose bytes the launches take in turn, ARENA_ALIGN apart
   and none across its end. Bytes are counted from the device's open:
   [arena_put] are taken, and a count from [arena_base] on lies at its
   offset from it, modulo the arena's size. A command buffer's slot holds
   the count its launches took up to, and the ring gives those bytes back
   as it releases the slot. Metal bounds a buffer's contents by its length
   alone (the Feature Set Tables), so the device's longest buffer bounds a
   launch. Room answers RIG_GROW for parts the empty arena does not hold,
   and the growth (caml_rig_metal_grow) makes an arena that holds twice
   them, which the next room installs: the command buffers that use the
   old one keep it until their work completes. */

_Thread_local uint64_t rig_metal_asked;

/* The arena bytes part [p] takes. */
static uint64_t arena_bytes(const struct rig_part *p) {
  uint64_t n = p->kind == RIG_LAUNCH ? p->launch.args.params : 0;
  if (n <= SET_BYTES_MAX) return 0;
  return (n + ARENA_ALIGN - 1) & ~(uint64_t)(ARENA_ALIGN - 1);
}

/* The count once [n] bytes are taken at [put]: from the arena's first byte
   where they would cross its end. */
static uint64_t take(const struct rig_metal_run *r, uint64_t put, uint64_t n) {
  uint64_t at = (put - r->arena_base) % r->arena_size;
  return (at + n > r->arena_size ? put + r->arena_size - at : put) + n;
}

/* Installs the arena a growth made, if it is larger, from the next byte
   taken on. */
static void install(struct rig_metal_run *r) {
  pthread_mutex_lock(&r->grown_mutex);
  id<MTLBuffer> g = r->grown;
  r->grown = nil;
  pthread_mutex_unlock(&r->grown_mutex);
  if (g == nil) return;
  if (g.length <= r->arena_size) {
    [g release];
    return;
  }
  [r->arena release];
  r->arena = g;
  r->arena_host = g.contents;
  r->arena_size = g.length;
  r->arena_base = r->arena_put;
}

/* Whether the arena holds the parts' parameters: RIG_FITS, RIG_LATER once
   the work of earlier values gives bytes back, RIG_GROW for more than the
   empty arena holds, and RIG_NEVER for more than the longest buffer. Once
   every byte taken is given back, the next is the arena's first. */
static int arena_room(struct rig_metal_run *r, const struct rig_part *parts,
                      int n) {
  uint64_t need = 0;
  for (int i = 0; i < n; i++) need += arena_bytes(&parts[i]);
  if (need == 0) return RIG_FITS;
  if (need > r->longest) return RIG_NEVER;
  install(r);
  if (need > r->arena_size) {
    rig_metal_asked = need;
    return RIG_GROW;
  }
  uint64_t freed =
      atomic_load_explicit(&r->ring.arena_freed, memory_order_acquire);
  if (freed >= r->arena_put) r->arena_base = r->arena_put;
  if (freed < r->arena_base) freed = r->arena_base;
  uint64_t put = r->arena_put;
  for (int i = 0; i < n; i++) {
    uint64_t b = arena_bytes(&parts[i]);
    if (b > 0) put = take(r, put, b);
  }
  return put - freed <= r->arena_size ? RIG_FITS : RIG_LATER;
}

int rig_metal_room(void *self, const struct rig_part *parts, int n,
                   void *scratch, const uint8_t *args) {
  (void)scratch;
  for (int i = 0; i < n; i++)
    if (!fits(&parts[i], args)) return RIG_NEVER;
  return arena_room(((struct rig_metal *)self)->run, parts, n);
}

/* Commits the residency set's additions, so that the work of this
   submission finds every allocation made before it resident. */
static void commit_residency(struct rig_metal_run *r) {
  pthread_mutex_lock(&r->set_mutex);
  if (r->changed) [r->set commit];
  r->changed = 0;
  pthread_mutex_unlock(&r->set_mutex);
}

/* Encodes the launch [p], its block in [args], into [q]'s encoder: its
   pipeline, its parameters as buffer 0, each ref's word plus its slot's
   address, its threadgroup memory, and its grid. Metal copies parameters
   of up to SET_BYTES_MAX bytes at [setBytes:], from the block's place, and
   longer ones lie in the arena, which room made hold them. Every state the
   dispatch reads is set here, as a fill sets its own. */
static void launch(struct rig_metal_queue *q, const struct rig_part *p,
                   uint8_t *args, const uint64_t *slots) {
  struct rig_metal_run *r = q->r;
  const struct rig_metal_entry *e = p->launch.launch;
  const struct rig_block *b = (const void *)(args + p->launch.args.block);
  uint32_t n = p->launch.args.params;
  id<MTLComputeCommandEncoder> c = q->encoder;
  [c setComputePipelineState:e->pipeline];
  uint64_t taken = arena_bytes(p);
  if (taken > 0) {
    uint64_t put = take(r, r->arena_put, taken);
    uint64_t at = (put - taken - r->arena_base) % r->arena_size;
    rig_params(r->arena_host + at, &p->launch.args, args, slots);
    [c setBuffer:r->arena offset:at atIndex:0];
    r->arena_put = put;
    atomic_store_explicit(&r->ring.slots[q->slot].arena_end, put,
                          memory_order_release);
  } else if (n > 0) {
    uint8_t *params = rig_place(&p->launch.args, args);
    rig_params(params, &p->launch.args, args, slots);
    [c setBytes:params length:n atIndex:0];
  }
  if (b->shared > 0)
    [c setThreadgroupMemoryLength:shared_length(b->shared) atIndex:0];
  [c dispatchThreadgroups:MTLSizeMake(b->groups[0], b->groups[1], b->groups[2])
      threadsPerThreadgroup:MTLSizeMake(b->threads[0], b->threads[1],
                                        b->threads[2])];
}

/* Runs the parts of [v] in the open command buffer's encoder, opening
   either if none is, and commits the buffer if its work would otherwise wait:
   whether it did. A fill that returns 0 without an open command buffer broke
   its contract after a failed split. With [times], [v]'s work gets command
   buffers of its own, committed at once, so the GPU times of the first's
   start and the last's end are [v]'s. The open mutex is held. */
static const char *run(struct rig_metal_run *r, uint64_t v,
                       const struct rig_part *parts, int n,
                       uint8_t *args, const uint64_t *slots,
                       uint64_t *times, int *committed) {
  struct rig_metal_queue *q = &r->open;
  const char *why = NULL;
  @try {
    commit_residency(r);
    if (times != NULL) commit_open(r);
    if (q->slot < 0) why = begin(q);
    if (why == NULL && times != NULL)
      r->ring.slots[q->slot].first = &times[0];
    if (why == NULL && n > 0 && q->encoder == nil) why = open_encoder(q);
    for (int i = 0; i < n && why == NULL; i++) {
      if (parts[i].kind == RIG_LAUNCH) {
        launch(q, &parts[i], args, slots);
        continue;
      }
      int rc = rig_fill(q, &parts[i], args, slots, v);
      if (rc == 0 && q->slot >= 0) continue;
      char text[64];
      snprintf(text, sizeof text, "running a fill: it returned %d", rc);
      why = fail(q, text);
    }
    if (why == NULL && times != NULL) {
      commit(q, v, NULL, &times[1]);
      r->open_values = 0;
      *committed = 1;
    } else if (why == NULL) {
      r->open_v = v;
      r->open_values++;
      *committed = r->open_values >= open_bound ||
                   rig_metal_ring_taken(&r->ring) - 1 < backlog;
      if (*committed) commit_open(r);
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
                     void *scratch, uint8_t *args,
                     const uint64_t *slots, int nslots, uint64_t *times,
                     const char **failure) {
  (void)waits, (void)nwaits, (void)scratch, (void)nslots;
  struct rig_metal_run *r = ((struct rig_metal *)self)->run;
  int committed = 0;
  r->last = v;
  const char *why = rig_metal_ring_failure(&r->ring);
  if (why == NULL) {
    pthread_mutex_lock(&r->open_mutex);
    @autoreleasepool {
      why = run(r, v, parts, nparts, args, slots, times, &committed);
    }
    unlock_open(r);
  }
  if (why != NULL) {
    *failure = why;
    return RIG_FAILED;
  }
  return committed ? RIG_COMMITTED : RIG_OK;
}

int rig_metal_commit(void *self, uint64_t v, const char **failure) {
  struct rig_metal_run *r = ((struct rig_metal *)self)->run;
  (void)v;
  pthread_mutex_lock(&r->open_mutex);
  commit_open(r);
  unlock_open(r);
  const char *why = rig_metal_ring_failure(&r->ring);
  if (why == NULL) return RIG_OK;
  *failure = why;
  return RIG_FAILED;
}

/* Drops the open command buffer uncommitted, completing its slot as failed,
   so that the ring drains. */
void rig_metal_drop(struct rig_metal_run *r) {
  pthread_mutex_lock(&r->open_mutex);
  @autoreleasepool {
    if (r->open.slot >= 0) drop(&r->open, "the device stopped");
  }
  r->open_values = 0;
  unlock_open(r);
}

#endif
