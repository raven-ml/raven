/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* 2,000 calls of a function of its own, one relocation each. With the buffer
   out: out[0] = 0 + 1 + ... + 1999. */

#include <stdint.h>

__attribute__((noinline)) void step(int64_t *out, int64_t k) { out[0] += k; }

#define C1(k) step(out, k);
#define C10(k)                                                                 \
  C1(k) C1(k + 1) C1(k + 2) C1(k + 3) C1(k + 4) C1(k + 5) C1(k + 6) C1(k + 7)  \
      C1(k + 8) C1(k + 9)
#define C100(k)                                                                \
  C10(k) C10(k + 10) C10(k + 20) C10(k + 30) C10(k + 40) C10(k + 50)           \
      C10(k + 60) C10(k + 70) C10(k + 80) C10(k + 90)
#define C1000(k)                                                               \
  C100(k) C100(k + 100) C100(k + 200) C100(k + 300) C100(k + 400)              \
      C100(k + 500) C100(k + 600) C100(k + 700) C100(k + 800) C100(k + 900)

void many(void **b, const int64_t *v) {
  (void)v;
  int64_t *out = b[0];
  out[0] = 0;
  C1000(0) C1000(1000)
}
