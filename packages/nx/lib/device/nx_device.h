/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* C access to nx.device buffers and to the host clock. */

#ifndef NX_DEVICE_H
#define NX_DEVICE_H

#include <caml/mlvalues.h>
#include <stdatomic.h>
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
   The fields are those of nx_device.ml's [Buffer.t], [base] and [region],
   whose [host] is [Some] address. */
static inline void *nx_device_buffer_host(value b) {
  value base = Field(b, 0);
  value host = Field(Field(base, 1), 0);
  return (char *)Nativeint_val(Field(host, 0)) + Long_val(Field(b, 1));
}

/* Whether the Nx_device.Buffer.t [b] is live: its memory was not consumed
   (Nx_device.Buffer.Claim.consume) since [b] was made. [b]'s generation (slot
   4) is then its memory's: slot 1 of the claim record in slot 9 of [b]'s base,
   which only a consumption replaces. */
static inline int nx_device_buffer_live(value b) {
  value claim = Field(Field(b, 0), 9);
  value generation = atomic_load_explicit((_Atomic value *)&Field(claim, 1),
                                          memory_order_acquire);
  return Field(b, 4) == generation;
}

/* Why the dead Nx_device.Buffer.t [b] was consumed: the reason of its memory's
   generation, valid until the next allocation. */
static inline const char *nx_device_buffer_why(value b) {
  return String_val(Field(Field(Field(Field(b, 0), 9), 1), 0));
}

/* The host's thread pool, which nx.cpu's kernels and the blocks of
   Nx_device.Program.call share, so that they never oversubscribe the cores.
   Nx_device hands it out as a nativeint (the primitive caml_nx_device_pool),
   so that another library's stubs hold no reference to a symbol of
   nx.device's, which a bytecode program loads as a shared library of its
   own.

   [workers ()] is the pool's threads, the caller included: the CPUs the
   process may use. [compute_workers ()] is those that pay for compute-bound
   work: the performance cores where the host has slower ones. [run nthreads
   total nchunks body ctx] cuts [0, total) into [nchunks] contiguous chunks
   and has at most [nthreads] threads, the caller as worker 0, claim them
   until none remain, calling [body lo hi worker ctx] for each. [worker] is in
   [0, nthreads) and stable across the chunks a thread claims. One region runs
   at a time: a second caller waits for the first to finish. With nthreads >
   1 the caller has released the OCaml runtime, and [body] never touches it. */
typedef void (*nx_device_range_body)(int64_t lo, int64_t hi, int worker,
                                     void *ctx);

typedef struct {
  int (*workers)(void);
  int (*compute_workers)(void);
  void (*run)(int nthreads, int64_t total, int64_t nchunks,
              nx_device_range_body body, void *ctx);
} nx_device_pool;

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
