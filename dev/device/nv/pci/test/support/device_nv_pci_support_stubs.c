/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The address of a bigarray's bytes, for windows on host memory. */

#define _GNU_SOURCE

#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/bigarray.h>
#include <caml/mlvalues.h>

/* Does not release the runtime: it reads a field. */
value caml_device_nv_pci_test_address(value b) {
  return Val_long((intptr_t)Caml_ba_data_val(b));
}
