/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A driver's submission, reduced to its stores: 1024 stores through
   device_pci.h at successive aligned offsets of a 4 KiB window, and the same
   stores made through a bare pointer, which is the floor. Each returns the
   number of stores that failed. */

#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

#include "device_pci.h"

#define STORES 1024
#define SPAN 4096

value bench_store32(value w) {
  struct device_pci_window win;
  int failed = 0;
  device_pci_window_of(w, &win);
  for (size_t i = 0; i < STORES; i++)
    failed -= device_pci_store32(&win, (i * 4) % SPAN, (uint32_t)i);
  return Val_int(failed);
}

value bench_store64(value w) {
  struct device_pci_window win;
  int failed = 0;
  device_pci_window_of(w, &win);
  for (size_t i = 0; i < STORES; i++)
    failed -= device_pci_store64(&win, (i * 8) % SPAN, (uint64_t)i);
  return Val_int(failed);
}

value bench_store32_bare(value a) {
  volatile uint8_t *p = (volatile uint8_t *)Long_val(a);
  for (size_t i = 0; i < STORES; i++)
    *(volatile uint32_t *)(p + (i * 4) % SPAN) = (uint32_t)i;
  return Val_int(0);
}

static uint8_t page[SPAN];

value bench_write(value w) {
  struct device_pci_window win;
  device_pci_window_of(w, &win);
  return Val_int(-device_pci_write(&win, 0, page, SPAN));
}
