/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A driver's submission, reduced to its stores: 1024 stores through
   device_pci.h at successive aligned offsets of a 4 KiB window, and the same
   stores made through a bare pointer, which is the floor. Each returns the
   number of stores that failed.

   The transport copies to and from a buffer of its own and does nothing
   else, so its rows are the window's cost above the wire. */

#define _GNU_SOURCE

#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

#include "device_pci.h"

#define STORES 1024
#define SPAN 4096

static uint8_t far[SPAN];

static int far_read(void *ctx, uint64_t a, void *dst, size_t n) {
  (void)ctx;
  memcpy(dst, far + a, n);
  return 0;
}

static int far_write(void *ctx, uint64_t a, const void *src, size_t n) {
  (void)ctx;
  memcpy(far + a, src, n);
  return 0;
}

static const char *far_failed(void *ctx) {
  (void)ctx;
  return NULL;
}

static const struct device_pci_transport transport = {NULL, far_read,
                                                      far_write, far_failed};

value device_pci_bench_far(value unit) {
  (void)unit;
  return Val_long((intnat)&transport);
}

value device_pci_bench_store32(value w) {
  struct device_pci_window win;
  int failed = 0;
  device_pci_window_of(w, &win);
  for (size_t i = 0; i < STORES; i++)
    failed -= device_pci_store32(&win, (i * 4) % SPAN, (uint32_t)i);
  return Val_int(failed);
}

value device_pci_bench_store64(value w) {
  struct device_pci_window win;
  int failed = 0;
  device_pci_window_of(w, &win);
  for (size_t i = 0; i < STORES; i++)
    failed -= device_pci_store64(&win, (i * 8) % SPAN, (uint64_t)i);
  return Val_int(failed);
}

value device_pci_bench_store32_bare(value a) {
  volatile uint8_t *p = (volatile uint8_t *)Long_val(a);
  for (size_t i = 0; i < STORES; i++)
    *(volatile uint32_t *)(p + (i * 4) % SPAN) = (uint32_t)i;
  return Val_int(0);
}

static uint8_t page[SPAN];

value device_pci_bench_write(value w) {
  struct device_pci_window win;
  device_pci_window_of(w, &win);
  return Val_int(-device_pci_write(&win, 0, page, SPAN));
}
