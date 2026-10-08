/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The ring of a device's command buffers, in commit order.

   Metal calls a command buffer's completion handler on one of its threads,
   in no documented order. The ring turns completions in any order into
   releases in commit order: each command buffer takes the next slot before
   it is made, and its completion releases every leading slot that
   completed. A released slot of value v > 0 writes v into the timeline
   word, unless a slot released before it failed: from then on the word
   stays. A slot's times are written when it completes, so before its
   release.

   The ring holds as many slots as the queue holds command buffers, and
   taking a slot waits while every slot is taken: that wait is the
   device's back-pressure.

   One mutex guards the ring. Completion holds it for a few stores and
   releases what was deferred to a slot after unlocking it. */

#ifndef DEVICE_METAL_RING_H
#define DEVICE_METAL_RING_H

#include <pthread.h>
#include <stdint.h>

/* A release to run once every command buffer committed before it was
   deferred completed. The caller owns the node until [run] is called. */
struct device_metal_release {
  void (*run)(struct device_metal_release *self);
  struct device_metal_release *next;
};

/* A command buffer between its take and its release. The taker sets [v]
   (0 for a buffer that is not its submission's last) and the [start] and
   [end] cells (NULL for none) before it commits the buffer. */
struct device_metal_slot {
  uint64_t v;
  uint64_t *start, *end;
  int state;
  struct device_metal_release *releases;
};

struct device_metal_ring {
  pthread_mutex_t mutex;
  pthread_cond_t changed;
  struct device_metal_slot *slots;
  uint64_t nslots, head, tail; /* slots [head, tail) are taken */
  uint64_t *word;
  int stopped;       /* a released slot failed: the word stays */
  char failure[512]; /* the first failure, "" if none */
};

/* Makes [r] a ring of the [n] slots at [slots], releasing into [word]. */
void device_metal_ring_init(struct device_metal_ring *r,
                            struct device_metal_slot *slots, int n,
                            uint64_t *word);

/* The index of the next slot, once one is free. */
int device_metal_ring_take(struct device_metal_ring *r);

/* Completes slot [i]: writes [start] and [end] into its cells, records
   [failure] (NULL if it succeeded) and releases the leading completed
   slots. */
void device_metal_ring_complete(struct device_metal_ring *r, int i,
                                const char *failure, uint64_t start,
                                uint64_t end);

/* Runs [rel] once the last slot taken is released, at once if none is
   taken. */
void device_metal_ring_defer(struct device_metal_ring *r,
                             struct device_metal_release *rel);

/* The first failure, or NULL. It lives as long as [r]. */
const char *device_metal_ring_failure(struct device_metal_ring *r);

/* Waits until the word differs from [seen], a failure is recorded, or
   [ms] milliseconds passed; then the first failure, or NULL. */
const char *device_metal_ring_sleep(struct device_metal_ring *r, uint64_t seen,
                                    int ms);

/* 1 after writing [last] into the word if no slot is taken, else 0. */
int device_metal_ring_stop(struct device_metal_ring *r, uint64_t last);

#endif
