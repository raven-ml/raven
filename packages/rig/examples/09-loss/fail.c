/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A fill that fails. It returns 1, and any value other than 0 is a
   failure, which loses the device. */

#define _GNU_SOURCE

#include <stdint.h>

#include <caml/alloc.h>
#include <caml/mlvalues.h>

static int fail(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)arg;
  (void)v;
  return 1;
}

/* Keeps the runtime, since it only returns an address. */
value caml_rig_example_fail(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&fail);
}
