/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Machines: a transport's failure, and the clock of the poll loop. */

#include <time.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "device_pci.h"

value caml_device_pci_transport_failed(value tr) {
  CAMLparam1(tr);
  CAMLlocal1(why);
  const struct device_pci_transport *t =
      (const struct device_pci_transport *)Long_val(tr);
  const char *s = t->failed(t->ctx);
  if (s == NULL) CAMLreturn(Val_none);
  why = caml_copy_string(s);
  CAMLreturn(caml_alloc_some(why));
}

/* Monotonic milliseconds. */
intnat caml_device_pci_now_ms(value unit) {
  (void)unit;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (intnat)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

value caml_device_pci_now_ms_byte(value unit) {
  return Val_long(caml_device_pci_now_ms(unit));
}
