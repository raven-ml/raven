/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The bytes of a device's host window, which the root suite fills and reads
   back. Neither releases the runtime: each touches at most a few pages. */

#define _GNU_SOURCE

#include <stdint.h>
#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

value rig_amd_pci_test_fill(value at, value n, value byte) {
  memset((void *)(uintptr_t)Long_val(at), Int_val(byte), Long_val(n));
  return Val_unit;
}

value rig_amd_pci_test_holds(value at, value n, value byte) {
  const volatile uint8_t *p = (const volatile uint8_t *)(uintptr_t)Long_val(at);
  for (long i = 0; i < Long_val(n); i++)
    if (p[i] != (uint8_t)Int_val(byte)) return Val_false;
  return Val_true;
}
