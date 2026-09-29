/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* C access to nx.device buffers and to the host clock. */

#ifndef NX_DEVICE_H
#define NX_DEVICE_H

#include <caml/mlvalues.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/* The host address of the first byte of the Nx_device.Buffer.t [b], which
   the host addresses, as every host buffer's memory is: its base's memory,
   plus its offset. It reads fields and allocates nothing, so it needs the
   runtime but no rooting, and the address stays valid while [b] is reachable.
   The fields are those of nx_device.ml's [Buffer.t], [base] and [memory],
   whose [host] is [Some] address. */
static inline void *nx_device_buffer_host(value b) {
  value base = Field(b, 0);
  value host = Field(Field(base, 1), 0);
  return (char *)Nativeint_val(Field(host, 0)) + Long_val(Field(b, 1));
}

/* The host clock: nanoseconds of the monotonic clock that
   Nx_device.Profile.now reads, and that the timestamps of a device of
   Nx_device.Host_clock are readings of. On macOS it is mach time, Metal's
   command buffer time base. */
static inline uint64_t nx_device_now_ns(void) {
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

#endif
