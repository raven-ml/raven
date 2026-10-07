/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Process memory to map, and a transport over a buffer of its own that fails
   on request. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/fail.h>
#include <caml/mlvalues.h>

#include "device_pci.h"

value smoke_buffer(value n) {
  void *p = calloc(1, Long_val(n));
  if (p == NULL) caml_raise_out_of_memory();
  return Val_long((intnat)p);
}

static uint8_t far[4096];
static int broken;

static int far_read(void *ctx, uint64_t a, void *dst, size_t n) {
  (void)ctx;
  if (broken || a + n > sizeof far) return -1;
  memcpy(dst, far + a, n);
  return 0;
}

static int far_write(void *ctx, uint64_t a, const void *src, size_t n) {
  (void)ctx;
  if (broken || a + n > sizeof far) return -1;
  memcpy(far + a, src, n);
  return 0;
}

static const char *far_failed(void *ctx) {
  (void)ctx;
  return broken ? "far: the link broke" : NULL;
}

static const struct device_pci_transport transport = {
    NULL, far_read, far_write, far_failed};

value smoke_transport(value unit) {
  (void)unit;
  return Val_long((intnat)&transport);
}

value smoke_break(value unit) {
  (void)unit;
  broken = 1;
  return Val_unit;
}
