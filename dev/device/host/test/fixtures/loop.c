/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Calls the program f count times. Its values are f, count, how (0:
   through device_host_call, unsplit; 1: through it, split by the four
   values that follow; 2: directly), the split's four values, the number n
   of f's values, then f's n values. Its buffers are f's. */

#include <stdint.h>

typedef void (*program)(void **, const int64_t *);

void device_host_call(program f, void **buffers, const int64_t *values,
                      int64_t n, const int64_t *split);

void loop(void **b, const int64_t *v) {
  program f = (program)v[0];
  int64_t count = v[1], how = v[2], n = v[7];
  const int64_t *values = v + 8;
  for (int64_t i = 0; i < count; i++)
    if (how == 2)
      f(b, values);
    else
      device_host_call(f, b, values, n, how == 1 ? v + 3 : 0);
}
