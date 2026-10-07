/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The clock of a machine's poll loop. */

#define _GNU_SOURCE
#include <time.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

/* Monotonic nanoseconds. Holds the runtime. */
intnat caml_device_pci_now_ns(value unit) {
  (void)unit;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (intnat)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

value caml_device_pci_now_ns_byte(value unit) {
  return Val_long(caml_device_pci_now_ns(unit));
}
