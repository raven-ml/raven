/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A fill: the C function a device calls to run a part of a submission. Its
   argument holds the addresses of the step's buffers and its length. It
   computes out[i] = k[i] * in[i] and answers 0, its success. */

#define _GNU_SOURCE

#include <stdint.h>

#include <caml/alloc.h>
#include <caml/mlvalues.h>

struct args {
  int32_t *out;
  const int32_t *in;
  const int32_t *k;
  int64_t n;
};

static int scale(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  const struct args *a = arg;
  for (int64_t i = 0; i < a->n; i++) a->out[i] = a->k[i] * a->in[i];
  return 0;
}

/* Does not release the runtime: it returns an address. */
value caml_rig_example_scale(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&scale);
}
