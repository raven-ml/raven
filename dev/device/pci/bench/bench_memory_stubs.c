/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The transport of the memory bench's machine. Its windows are mapped, so
   nothing reaches it: an access fails and the machine never does. */

#define _GNU_SOURCE

#include <stdint.h>
#include <stddef.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

#include "device_pci.h"

static int none_access(void *ctx, uint64_t a, void *p, size_t n) {
  (void)ctx, (void)a, (void)p, (void)n;
  return -1;
}

static int none_read(void *ctx, uint64_t a, void *dst, size_t n) {
  return none_access(ctx, a, dst, n);
}

static int none_write(void *ctx, uint64_t a, const void *src, size_t n) {
  return none_access(ctx, a, (void *)src, n);
}

static const char *none_failed(void *ctx) {
  (void)ctx;
  return NULL;
}

static const struct device_pci_transport none = {NULL, none_read, none_write,
                                                 none_failed};

value bench_memory_transport(value unit) {
  (void)unit;
  return Val_long((intnat)&none);
}
