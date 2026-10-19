/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The domain lock as a thread holds it, timed in CPU time, which the
   scheduler's delays leave unchanged: the holding thread's for each
   stretch it holds the lock, the process's for the whole watch. And a job
   on the pool that lasts until a watched thread releases the lock. */

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <caml/fail.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include "rig_pool.h"

/* Only the runtime knows when a thread releases the domain lock, and it
   tells no public interface: these two hooks, which it calls on entering
   and leaving a blocking section and which systhreads sets the same way,
   are its internals (caml/signals.h under CAML_INTERNALS). They are
   declared here alone, so that this file reads no other internal. */
CAMLextern void (*caml_enter_blocking_section_hook)(void);
CAMLextern void (*caml_leave_blocking_section_hook)(void);

static void (*enter_before)(void);
static void (*leave_before)(void);

/* The watched thread's count of releases, its longest stretch holding the
   lock and the start of the current one, in its CPU time, and the start
   of the watch, in the process's. */
static _Thread_local int watching, holding;
static _Thread_local int64_t releases, longest, since, start;

/* Every watched thread's releases, which occupy's job waits for. */
static atomic_long watched_releases;

static int64_t clock_ns(clockid_t clock) {
  struct timespec t;
  clock_gettime(clock, &t);
  return (int64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}

static void held_until_now(void) {
  int64_t d = clock_ns(CLOCK_THREAD_CPUTIME_ID) - since;
  if (d > longest) longest = d;
}

/* Called holding the lock, before the runtime releases it. */
static void enter(void) {
  if (watching && holding) {
    held_until_now();
    releases++;
    holding = 0;
    atomic_fetch_add(&watched_releases, 1);
  }
  enter_before();
}

/* Called released, before the runtime reacquires the lock. */
static void leave(void) {
  leave_before();
  if (watching && !holding) {
    holding = 1;
    since = clock_ns(CLOCK_THREAD_CPUTIME_ID);
  }
}

/* At the module's initialisation, before another thread runs OCaml: the
   runtime reads the hooks unsynchronised. */
value nx_release_support_install(value unit) {
  (void)unit;
  enter_before = caml_enter_blocking_section_hook;
  leave_before = caml_leave_blocking_section_hook;
  caml_enter_blocking_section_hook = enter;
  caml_leave_blocking_section_hook = leave;
  return Val_unit;
}

value nx_release_support_start(value unit) {
  (void)unit;
  releases = 0;
  longest = 0;
  watching = 1;
  holding = 1;
  start = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
  since = clock_ns(CLOCK_THREAD_CPUTIME_ID);
  return Val_unit;
}

/* Ends the watch: answers its CPU time. */
value nx_release_support_stop(value unit) {
  (void)unit;
  if (holding) held_until_now();
  watching = 0;
  return Val_long(clock_ns(CLOCK_PROCESS_CPUTIME_ID) - start);
}

value nx_release_support_releases(value unit) {
  (void)unit;
  return Val_long(releases);
}

value nx_release_support_longest(value unit) {
  (void)unit;
  return Val_long(longest);
}

/* Occupying the pool */

/* How long occupy's job waits for a release at most: far past any
   scheduler's delay, so that only a call that never releases reaches
   it. */
#define PATIENCE_NS ((int64_t)10 * 1000000000)

static atomic_int occupied;

typedef struct {
  long from;
  int64_t deadline;
  atomic_int released;
} occupation;

/* A unit of occupy's job: waits until a watched thread releases the lock
   after the job began, or the deadline passes. */
static void wait_release(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  occupation *o = ctx;
  atomic_store(&occupied, 1);
  struct timespec pause = {0, 10000};
  while (atomic_load(&watched_releases) == o->from) {
    if (clock_ns(CLOCK_MONOTONIC) > o->deadline) return;
    nanosleep(&pause, NULL);
  }
  atomic_store(&o->released, 1);
}

value nx_release_support_occupy(value unit) {
  (void)unit;
  occupation o = {atomic_load(&watched_releases),
                  clock_ns(CLOCK_MONOTONIC) + PATIENCE_NS, 0};
  caml_enter_blocking_section();
  rig_pool_run(2, 2, 2, wait_release, &o);
  caml_leave_blocking_section();
  atomic_store(&occupied, 0);
  return Val_bool(atomic_load(&o.released));
}

value nx_release_support_occupied(value unit) {
  (void)unit;
  return Val_bool(atomic_load(&occupied));
}

/* The copy */

#define COPY_BYTES (1024 * 1024)

value nx_release_support_copy(value unit) {
  (void)unit;
  static uint8_t *from, *to;
  if (from == NULL) {
    /* Written once, so that the copy reads pages of its own: a system may
       map fresh pages read before any write to one zero page. */
    from = malloc(COPY_BYTES);
    if (from != NULL) memset(from, 1, COPY_BYTES);
  }
  if (to == NULL) to = malloc(COPY_BYTES);
  if (from == NULL || to == NULL) caml_failwith("Nx_release_support.copy");
  int64_t least = INT64_MAX;
  for (int i = 0; i < 5; i++) {
    int64_t t0 = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    memcpy(to, from, COPY_BYTES);
    __asm__ volatile("" : : "r"(to) : "memory");
    int64_t t = clock_ns(CLOCK_THREAD_CPUTIME_ID) - t0;
    if (t < least) least = t;
  }
  return Val_long(least);
}
