/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the pool's probe stubs share: a monotonic clock, and how long a
   probe waits for the pool before it gives up. */

#ifndef RIG_POOL_PROBE_H
#define RIG_POOL_PROBE_H

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

/* The nanoseconds a probe body waits for another call, which rig_pool.h
   forbids, before it gives up: half the test's limit, WINDTRAP_TIMEOUT
   seconds, read once, so that a pool that breaks a promise fails the test
   with the probe's answer, before the limit ends it. Far longer than any
   wakeup: only a pool that never runs the awaited chunk, or never ends the
   awaited job, reaches it. */
static inline int64_t patience(void) {
  /* windtrap's limit where WINDTRAP_TIMEOUT is unset, in seconds. */
  const double default_limit_s = 600.;
  static _Atomic int64_t cached;
  int64_t ns = atomic_load_explicit(&cached, memory_order_relaxed);
  if (ns == 0) {
    const char *limit = getenv("WINDTRAP_TIMEOUT");
    double s = limit == NULL ? 0. : strtod(limit, NULL);
    ns = (int64_t)((s > 0. ? s : default_limit_s) * 0.5e9);
    atomic_store_explicit(&cached, ns, memory_order_relaxed);
  }
  return ns;
}

static inline int64_t now_ns(void) {
#if defined(_WIN32)
  LARGE_INTEGER count, frequency;
  QueryPerformanceCounter(&count);
  QueryPerformanceFrequency(&frequency);
  int64_t c = count.QuadPart, f = frequency.QuadPart;
  return c / f * 1000000000 + c % f * 1000000000 / f;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#endif
}

#endif
