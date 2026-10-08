/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the pool's probe stubs share: a monotonic clock, and how long a
   probe waits for the pool before it gives up. */

#ifndef NX_POOL_PROBE_H
#define NX_POOL_PROBE_H

#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

/* How long a probe body waits for another call, which nx_pool.h forbids,
   before it gives up, so that a pool that breaks a promise fails the test
   instead of hanging it. Far longer than any wakeup: only a pool that never
   runs the awaited chunk, or never ends the awaited job, reaches it. */
static const int64_t patience = INT64_C(10000000000);

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
