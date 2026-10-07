/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Machines: a transport's failure, and the clock of the poll loop. */

#define _GNU_SOURCE
#include <time.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "device_pci.h"

/* The reason the transport [tr] failed, if it did. This machine's
   transport is 0: nothing fails it. */
value caml_device_pci_transport_failed(value tr) {
  CAMLparam1(tr);
  CAMLlocal1(why);
  const struct device_pci_transport *t =
      (const struct device_pci_transport *)Long_val(tr);
  if (t == NULL) CAMLreturn(Val_none);
  const char *s = t->failed(t->ctx);
  if (s == NULL) CAMLreturn(Val_none);
  why = caml_copy_string(s);
  CAMLreturn(caml_alloc_some(why));
}

/* Monotonic nanoseconds. */
intnat caml_device_pci_now_ns(value unit) {
  (void)unit;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (intnat)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

value caml_device_pci_now_ns_byte(value unit) {
  return Val_long(caml_device_pci_now_ns(unit));
}
