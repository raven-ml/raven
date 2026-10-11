/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's clock, which rig, its drivers and their C share. */

#define _GNU_SOURCE

#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#include "rig_edge.h"

uint64_t rig_now_ns(void) {
#if defined(_WIN32)
  LARGE_INTEGER count, frequency;
  QueryPerformanceCounter(&count);
  QueryPerformanceFrequency(&frequency);
  uint64_t c = (uint64_t)count.QuadPart, f = (uint64_t)frequency.QuadPart;
  return c / f * 1000000000u + c % f * 1000000000u / f;
#elif defined(__APPLE__)
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
#endif
}

intnat caml_rig_edge_now(value unit) {
  (void)unit;
  return (intnat)rig_now_ns();
}

value caml_rig_edge_now_byte(value unit) {
  return Val_long(caml_rig_edge_now(unit));
}
