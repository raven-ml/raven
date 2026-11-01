/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's load, which the gate and probe report beside their numbers. */

#define _GNU_SOURCE

#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/mlvalues.h>

/* The host's load average over the last minute; 0 on Windows, which has
   none. */
value nx_metal_bench_loadavg(value unit) {
  (void)unit;
  double l[1] = {0};
#if !defined(_WIN32)
  getloadavg(l, 1);
#endif
  return caml_copy_double(l[0]);
}
