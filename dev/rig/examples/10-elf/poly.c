/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* out[i] = p(in[i]) for i below n, p the polynomial of the first d
   coefficients of the read-only table coeffs (d at most 8), evaluated by a
   helper the entry calls: code, read-only data and the relocations between
   them. Buffers out and in; values n and d. */

#include <stdint.h>

static const int64_t coeffs[8] = {1, -2, 3, 5, -8, 13, -21, 34};

__attribute__((noinline)) int64_t horner(int64_t x, int64_t d) {
  int64_t y = 0;
  for (int64_t k = d - 1; k >= 0; k--) y = y * x + coeffs[k];
  return y;
}

void poly(void **b, const int64_t *v) {
  int64_t *out = b[0];
  const int64_t *in = b[1];
  for (int64_t i = 0; i < v[0]; i++) out[i] = horner(in[i], v[1]);
}
