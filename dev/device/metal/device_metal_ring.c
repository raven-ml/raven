/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The ring of a device's command buffers. Plain C: no Metal and no OCaml,
   so that a test drives it in any completion order. */

#define _GNU_SOURCE

#include "device_metal_ring.h"

#include <errno.h>
#include <stdio.h>
#include <time.h>

enum { taken, done, failed };

void device_metal_ring_init(struct device_metal_ring *r,
                            struct device_metal_slot *slots, int n,
                            uint64_t *word) {
  pthread_mutex_init(&r->mutex, NULL);
  pthread_cond_init(&r->changed, NULL);
  r->slots = slots;
  r->nslots = (uint64_t)n;
  r->head = r->tail = 0;
  r->word = word;
  r->stopped = 0;
  r->failure[0] = '\0';
}

int device_metal_ring_take(struct device_metal_ring *r) {
  pthread_mutex_lock(&r->mutex);
  while (r->tail - r->head == r->nslots)
    pthread_cond_wait(&r->changed, &r->mutex);
  int i = (int)(r->tail++ % r->nslots);
  r->slots[i] = (struct device_metal_slot){0, NULL, NULL, taken, NULL};
  pthread_mutex_unlock(&r->mutex);
  return i;
}

void device_metal_ring_complete(struct device_metal_ring *r, int i,
                                const char *failure, uint64_t start,
                                uint64_t end) {
  struct device_metal_slot *s = &r->slots[i];
  struct device_metal_release *releases = NULL;
  pthread_mutex_lock(&r->mutex);
  if (s->start) *s->start = start;
  if (s->end) *s->end = end;
  s->state = failure ? failed : done;
  if (failure && r->failure[0] == '\0')
    snprintf(r->failure, sizeof r->failure, "%s", failure);
  while (r->head != r->tail) {
    struct device_metal_slot *h = &r->slots[r->head % r->nslots];
    if (h->state == taken) break;
    if (h->state == failed) r->stopped = 1;
    if (h->v > 0 && !r->stopped)
      __atomic_store_n(r->word, h->v, __ATOMIC_RELEASE);
    /* Prepend the slot's list, whose order does not matter. */
    for (struct device_metal_release *l = h->releases, *next; l; l = next) {
      next = l->next;
      l->next = releases;
      releases = l;
    }
    r->head++;
  }
  pthread_cond_broadcast(&r->changed);
  pthread_mutex_unlock(&r->mutex);
  for (struct device_metal_release *next; releases; releases = next) {
    next = releases->next;
    releases->run(releases);
  }
}

void device_metal_ring_defer(struct device_metal_ring *r,
                             struct device_metal_release *rel) {
  pthread_mutex_lock(&r->mutex);
  int now = r->head == r->tail;
  if (!now) {
    struct device_metal_slot *last = &r->slots[(r->tail - 1) % r->nslots];
    rel->next = last->releases;
    last->releases = rel;
  }
  pthread_mutex_unlock(&r->mutex);
  if (now) rel->run(rel);
}

const char *device_metal_ring_failure(struct device_metal_ring *r) {
  pthread_mutex_lock(&r->mutex);
  const char *failure = r->failure[0] ? r->failure : NULL;
  pthread_mutex_unlock(&r->mutex);
  return failure;
}

const char *device_metal_ring_sleep(struct device_metal_ring *r, uint64_t seen,
                                    int ms) {
  struct timespec until;
  clock_gettime(CLOCK_REALTIME, &until);
  until.tv_sec += ms / 1000;
  until.tv_nsec += (long)(ms % 1000) * 1000000;
  if (until.tv_nsec >= 1000000000) {
    until.tv_sec++;
    until.tv_nsec -= 1000000000;
  }
  pthread_mutex_lock(&r->mutex);
  while (*r->word == seen && r->failure[0] == '\0')
    if (pthread_cond_timedwait(&r->changed, &r->mutex, &until) == ETIMEDOUT)
      break;
  const char *failure = r->failure[0] ? r->failure : NULL;
  pthread_mutex_unlock(&r->mutex);
  return failure;
}

int device_metal_ring_stop(struct device_metal_ring *r, uint64_t last) {
  pthread_mutex_lock(&r->mutex);
  int idle = r->head == r->tail;
  if (idle) __atomic_store_n(r->word, last, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&r->mutex);
  return idle;
}
