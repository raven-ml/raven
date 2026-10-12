/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Stores value as the 64-bit word at out, then calls the ready function
   with counts and count. Its parameters are out, counts, ready, value and
   count. */

#include <stdint.h>

/* The ready function is the process's, of the platform's convention: on
   x86_64 Windows, ms_abi (gen.sh defines RIG_WINDOWS for that object). */
#ifdef RIG_WINDOWS
#define PLATFORM __attribute__((ms_abi))
#else
#define PLATFORM
#endif

typedef void(PLATFORM *ready_fn)(void *counts, uint64_t c);

struct params {
  int64_t *out;
  void *counts;
  ready_fn ready;
  int64_t value, count;
};

void ready(const struct params *p, int64_t first, int64_t end) {
  (void)first;
  (void)end;
  p->out[0] = p->value;
  p->ready(p->counts, (uint64_t)p->count);
}
