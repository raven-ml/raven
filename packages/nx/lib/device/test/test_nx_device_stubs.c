/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/mlvalues.h>
#include <stdatomic.h>
#include <stdint.h>

/* Stores [v] into the word at [addr], as a device signals its timeline. */
value test_nx_device_signal(value addr, value v) {
  atomic_store_explicit((_Atomic uint64_t *)Nativeint_val(addr),
                        (uint64_t)Long_val(v), memory_order_release);
  return Val_unit;
}
