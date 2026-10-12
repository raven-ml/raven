/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A launch whose stack frame spans 256 KiB, which it writes first at its far
   end, 64 pages below the stack pointer: code for an ELF target touches no
   page between. Its parameters are out and zero, which is 0 and keeps the
   compiler from shrinking the frame to the bytes it names. Each thread i of
   its range stores the sum of the frame's first and last bytes, 2, in
   out[i]. */

#include <stdint.h>

struct params {
  int64_t *out;
  int64_t zero;
};

void deep(const struct params *p, int64_t first, int64_t end) {
  volatile char frame[256 * 1024];
  int64_t last = (int64_t)sizeof frame - 1 - p->zero;
  frame[p->zero] = 1;
  frame[last] = 1;
  for (int64_t i = first; i < end; i++)
    p->out[i] = frame[p->zero] + frame[last];
}
