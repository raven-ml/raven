/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Stores value as the 64-bit word at out, then calls the ready function
   with counts and count. Its parameters are out, counts, ready, value and
   count. */

#include <stdint.h>

typedef void (*ready_fn)(void *counts, uint64_t c);

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
