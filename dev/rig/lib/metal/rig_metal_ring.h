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
   release. Once stopped, the ring writes the last value it was given when
   no slot is taken any more, whatever that work did.

   The ring holds as many slots as the queue holds command buffers, and
   taking a slot waits while every slot is taken: that wait is the
   device's back-pressure.

   One mutex guards the ring. Completion holds it for a few stores. */

#ifndef RIG_METAL_RING_H
#define RIG_METAL_RING_H

#include <pthread.h>
#include <stdint.h>

/* A command buffer between its take and its release. The taker sets [v]
   (0 for a buffer that is not its submission's last) and the [start] and
   [end] cells (NULL for none) before it commits the buffer. */
struct rig_metal_slot {
  uint64_t v;
  uint64_t *start, *end;
  int state;
};

struct rig_metal_ring {
  pthread_mutex_t mutex;
  pthread_cond_t changed;
  struct rig_metal_slot *slots;
  uint64_t nslots, head, tail; /* slots [head, tail) are taken */
  uint64_t *word;
  int held;          /* a released slot failed: the word stays */
  uint64_t drain;    /* written once no slot is taken, after stop; 0: none */
  char failure[512]; /* the first failure, "" if none */
};

/* Makes [r] a ring of the [n] slots at [slots], releasing into [word]. */
void rig_metal_ring_init(struct rig_metal_ring *r,
                            struct rig_metal_slot *slots, int n,
                            uint64_t *word);

/* The index of the next slot, once one is free. */
int rig_metal_ring_take(struct rig_metal_ring *r);

/* Completes slot [i]: writes [start] and [end] into its cells, records
   [failure] (NULL if it succeeded) and releases the leading completed
   slots. */
void rig_metal_ring_complete(struct rig_metal_ring *r, int i,
                                const char *failure, uint64_t start,
                                uint64_t end);

/* The first failure, or NULL. It lives as long as [r]. */
const char *rig_metal_ring_failure(struct rig_metal_ring *r);

/* Waits until the word differs from [seen], a failure is recorded, or
   [ms] milliseconds passed; then the first failure, or NULL. */
const char *rig_metal_ring_sleep(struct rig_metal_ring *r, uint64_t seen,
                                    int ms);

/* 1 after writing [last] into the word if no slot is taken. Else 0, and the
   completion that releases the last taken slot writes [last]. */
int rig_metal_ring_stop(struct rig_metal_ring *r, uint64_t last);

#endif
