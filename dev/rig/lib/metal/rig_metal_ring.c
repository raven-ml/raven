/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The ring of a device's command buffers. Plain C: no Metal and no OCaml,
   so that a test drives it in any completion order. */

#define _GNU_SOURCE

#include "rig_metal_ring.h"

#include <errno.h>
#include <stdio.h>
#include <time.h>

enum { taken, done, failed };

static const uint64_t ns_per_s = 1000000000;

static uint64_t monotonic_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * ns_per_s + (uint64_t)t.tv_nsec;
}

/* Waits on [r]'s condition until [until], monotonic nanoseconds, so a step
   of the wall clock neither lengthens nor cuts the wait. macOS has no clock
   attribute for conditions and waits for the time left; elsewhere the
   condition waits on CLOCK_MONOTONIC. */
static int wait_until(struct rig_metal_ring *r, uint64_t until) {
  uint64_t now = monotonic_ns();
  if (now >= until) return ETIMEDOUT;
#ifdef __APPLE__
  uint64_t left = until - now;
  struct timespec t = {(time_t)(left / ns_per_s), (long)(left % ns_per_s)};
  return pthread_cond_timedwait_relative_np(&r->changed, &r->mutex, &t);
#else
  struct timespec t = {(time_t)(until / ns_per_s), (long)(until % ns_per_s)};
  return pthread_cond_timedwait(&r->changed, &r->mutex, &t);
#endif
}

void rig_metal_ring_init(struct rig_metal_ring *r,
                            struct rig_metal_slot *slots, int n,
                            uint64_t *word) {
  pthread_mutex_init(&r->mutex, NULL);
#ifdef __APPLE__
  pthread_cond_init(&r->changed, NULL);
#else
  pthread_condattr_t monotonic;
  pthread_condattr_init(&monotonic);
  pthread_condattr_setclock(&monotonic, CLOCK_MONOTONIC);
  pthread_cond_init(&r->changed, &monotonic);
  pthread_condattr_destroy(&monotonic);
#endif
  r->slots = slots;
  r->nslots = (uint64_t)n;
  r->head = r->tail = 0;
  r->word = word;
  r->held = 0;
  r->drain = 0;
  r->failure[0] = '\0';
}

int rig_metal_ring_take(struct rig_metal_ring *r) {
  pthread_mutex_lock(&r->mutex);
  while (r->tail - r->head == r->nslots)
    pthread_cond_wait(&r->changed, &r->mutex);
  int i = (int)(r->tail++ % r->nslots);
  r->slots[i] = (struct rig_metal_slot){0, NULL, NULL, taken};
  pthread_mutex_unlock(&r->mutex);
  return i;
}

void rig_metal_ring_complete(struct rig_metal_ring *r, int i,
                                const char *failure, uint64_t start,
                                uint64_t end) {
  struct rig_metal_slot *s = &r->slots[i];
  pthread_mutex_lock(&r->mutex);
  if (s->start) *s->start = start;
  if (s->end) *s->end = end;
  s->state = failure ? failed : done;
  if (failure && r->failure[0] == '\0')
    snprintf(r->failure, sizeof r->failure, "%s", failure);
  while (r->head != r->tail) {
    struct rig_metal_slot *h = &r->slots[r->head % r->nslots];
    if (h->state == taken) break;
    if (h->state == failed) r->held = 1;
    if (h->v > 0 && !r->held) __atomic_store_n(r->word, h->v, __ATOMIC_RELEASE);
    r->head++;
  }
  if (r->drain && r->head == r->tail)
    __atomic_store_n(r->word, r->drain, __ATOMIC_RELEASE);
  pthread_cond_broadcast(&r->changed);
  pthread_mutex_unlock(&r->mutex);
}

int rig_metal_ring_taken(struct rig_metal_ring *r) {
  pthread_mutex_lock(&r->mutex);
  int n = (int)(r->tail - r->head);
  pthread_mutex_unlock(&r->mutex);
  return n;
}

const char *rig_metal_ring_failure(struct rig_metal_ring *r) {
  pthread_mutex_lock(&r->mutex);
  const char *failure = r->failure[0] ? r->failure : NULL;
  pthread_mutex_unlock(&r->mutex);
  return failure;
}

const char *rig_metal_ring_sleep(struct rig_metal_ring *r, uint64_t seen,
                                    int ms) {
  uint64_t until = monotonic_ns() + (uint64_t)ms * 1000000;
  pthread_mutex_lock(&r->mutex);
  while (*r->word == seen && r->failure[0] == '\0')
    if (wait_until(r, until) == ETIMEDOUT) break;
  const char *failure = r->failure[0] ? r->failure : NULL;
  pthread_mutex_unlock(&r->mutex);
  return failure;
}

int rig_metal_ring_stop(struct rig_metal_ring *r, uint64_t last) {
  pthread_mutex_lock(&r->mutex);
  int idle = r->head == r->tail;
  if (idle)
    __atomic_store_n(r->word, last, __ATOMIC_RELEASE);
  else
    r->drain = last;
  pthread_mutex_unlock(&r->mutex);
  return idle;
}
