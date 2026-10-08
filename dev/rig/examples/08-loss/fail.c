/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A fill that fails: it answers 1 where success is 0, as compiled work does
   when the device refuses what it encodes. */

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

/* Does not release the runtime: it returns an address. */
value caml_rig_example_fail(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&fail);
}
