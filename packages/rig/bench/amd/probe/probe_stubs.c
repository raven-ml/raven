/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#define _GNU_SOURCE
#include <caml/mlvalues.h>
#include <stdint.h>
#include <time.h>
#ifdef __linux__
#include <sched.h>
#endif

value probe_now(value u) {
  (void)u;
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return Val_long((intnat)t.tv_sec * 1000000000 + t.tv_nsec);
}

value probe_cpu(value u) {
  (void)u;
#ifdef __linux__
  return Val_int(sched_getcpu());
#else
  return Val_int(-1);
#endif
}

/* Spins until [ns] nanoseconds passed. */
value probe_spin(value v_ns) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  int64_t until = (int64_t)t.tv_sec * 1000000000 + t.tv_nsec + Long_val(v_ns);
  for (;;) {
    clock_gettime(CLOCK_MONOTONIC, &t);
    if ((int64_t)t.tv_sec * 1000000000 + t.tv_nsec >= until) break;
  }
  return Val_unit;
}
